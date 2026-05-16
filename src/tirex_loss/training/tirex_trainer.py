import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tirex_loss.training.lr_finder import LRFinder
from tirex_loss.training.utils import EarlyStopping
from tirex_loss.logger import TrainingLogger


class Tirex_Trainer():
    """Training loop for Torch model
    """
    
    def __init__(self, model:nn.Module,
                 criterion, optimizer,
                 model_path:str,
                 log_path:str,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 status_bar:bool=True
                 ):
        
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_path = model_path
        self.log_path = log_path
        self.scheduler = None
        self.train_loader = None
        self.test_loader = None
        self.lr_from_finder = None
        self.history = {"train_loss": [], "test_loss": []}
        self.status_bar = status_bar
    
        self.model_best_path = os.path.join(self.model_path, 'best_model.pth')
        self.model_last_path = os.path.join(self.model_path, 'last_model.pth')
        self.writer = TrainingLogger(log_dir=self.log_path)


    def reset(self):
        """Restores the model and optimizer to their initial states."""
        raise NotImplementedError('Reset not implemented!')
        # self.model.load_state_dict(self.state_cacher.retrieve("model"))
        # self.optimizer.load_state_dict(self.state_cacher.retrieve("optimizer"))
        # self.model.to(self.device)        
        
    def train_epoch(self, epoch_idx:int,
                    use_clipping:bool=False, clipping_max_norm:float=1.0,
                    fraction_log_interval:int=10):
        """train one epoch

        Args:
            train_loader (_type_): _description_
            test_loader (_type_): _description_
        """
        self.model.train()  # Set model to train mode
        train_loss = 0.0
        length_train_loader = len(self.train_loader)
        batch_bar_train = tqdm(self.train_loader, total=length_train_loader,
                               leave=False, position=1, desc='Batches Train',
                                disable=not self.status_bar)
        for i, (x_batch, y_batch, pred_length_batch) in enumerate(batch_bar_train):
            # put tensors on same device
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)
            pred_length = pred_length_batch[0].item()

            # zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            # using the _forecast_tensor will return the quantile losses for the given prediction length
            outputs = self.model._forecast_tensor(x, prediction_length=pred_length)
            # outputs shape: [batch, n_quantiles, seq_len] -> need to be transposed to [batch, seq_len, n_quantiles] for the loss function
            outputs = outputs.transpose(1, 2)
            loss = self.criterion(outputs, y)

            # Backward pass
            loss.backward()
            if use_clipping:
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_max_norm)
            self.optimizer.step()

            batch_bar_train.set_postfix(train_loss=loss.item())
            train_loss += loss.item()
            
            # log train losses inside batches
            if i%(int(length_train_loader/fraction_log_interval)) == 1: # use 1 to avoid division by 0
                global_step = epoch_idx*length_train_loader+i
                self.writer.log_batch_metrics(
                    epoch=epoch_idx,
                    batch=i,
                    global_step=global_step,
                    metrics={'train_loss': train_loss / i}
                )
        train_loss /= length_train_loader
        
        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Test loss
        self.model.eval()  # Set model to eval mode
        with torch.no_grad():
            test_loss = 0.0
            length_test_loader = len(self.test_loader)
            batch_bar_test = tqdm(self.test_loader, total=length_test_loader,
                                   leave=False, position=1, desc='Batches Test',
                                     disable=not self.status_bar)
            for i, (x_batch, y_batch, pred_length_batch) in enumerate(batch_bar_test):
                # put tensors on same device
                x = x_batch.to(self.device)
                y = y_batch.to(self.device)
                pred_length = pred_length_batch[0].item()

                outputs = self._forecast_tensor(x, prediction_length=pred_length)
                # outputs shape: [batch, n_quantiles, seq_len] -> need to be transposed to [batch, seq_len, n_quantiles] for the loss function
                outputs = outputs.transpose(1, 2)                
                loss = self.criterion(outputs, y)
                batch_bar_test.set_postfix(test_loss=loss.item())
                test_loss += loss.item()
            test_loss /= length_test_loader

        return train_loss, test_loss
    
    def train(self, train_dataloader, test_dataloader,
              batch_size=32,
              num_epochs=100,
              use_clipping:bool=False,
              clipping_max_norm:float=1.0,
              early_stop_patience:int=10,
              use_scheduler:bool=True,
              scheduler_exp_gamma:float=0.99,
              **kwargs):
        """ training the torch model

        Args:
            train_dataset (_type_): _description_
            test_dataset (_type_): _description_
            num_epochs (int, optional): _description_. Defaults to 100.
        """
        
        # set scheduler
        if use_scheduler:
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_exp_gamma)

        # save the model definition and hyperparameters to the start of the log file      
        hparams = {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'model_parameter_count': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'criterion': self.criterion.__class__.__name__,
            'optimizer': self.optimizer.__class__.__name__,
            'optimizer_parameters': str(self.optimizer.param_groups),
            'scheduler': self.scheduler.__class__.__name__ if self.scheduler is not None else None,
            'scheduler_parameters': str(self.scheduler.state_dict()) if self.scheduler is not None else None,
            'clipping': use_clipping,
            'clipping_max_norm': clipping_max_norm
        }
        self.writer.log_hyperparameters(hparams)
                                  
        # Dataloaders
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader         
        
        # training loop
        early_stopper = EarlyStopping(patience=early_stop_patience, min_delta=1e-4)
        best_test_loss = float('inf')
        epoch_bar = tqdm(range(num_epochs), total=num_epochs, leave=True, position=0, desc='Epochs')
        for epoch in epoch_bar:
            # train one epoch
            train_loss, test_loss = self.train_epoch(epoch, use_clipping, clipping_max_norm)
                                       
            # save best model
            if best_test_loss > test_loss:
                torch.save(self.model.state_dict(), self.model_best_path)
                best_test_loss = test_loss

            # log losses
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            epoch_bar.set_postfix(train_loss=train_loss, test_loss=test_loss,)
            self.writer.log_epoch_metrics(
                epoch=epoch + 1,
                metrics={
                    'train_loss': train_loss, 
                    'test_loss': test_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
            )
            # just in case, so we can continue training from the last epoch
            torch.save(self.model.state_dict(), self.model_last_path) # save last model

            if early_stopper(test_loss):
                print(f"Early stopping at epoch {epoch}")
                break

    def plot_loss(self, log_lr=True):
        """Plots the losses.

        Arguments:
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
        """

        # Get the data to plot from the history dictionary.
        train_losses = self.history["train_loss"]
        test_losses = self.history["test_loss"]
        epochs = list(range(len(train_losses)))

        # Plot loss as a function of the learning rate
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label='train')
        ax.plot(epochs, test_losses, label='test')
        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        #ax.grid()

        return fig

    def lr_range_test(self, train_dataset, test_dataset=None, batch_size=32, **kwargs):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, self.device, memory_cache=True)
        lr_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        if test_dataset is not None:
            lr_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        else:
            lr_test_loader = None
        # make the test
        lr_finder.range_test(lr_train_loader, val_loader=lr_test_loader, end_lr=100, num_iter=100, **kwargs)
        # create plot and find minimum lr
        lr_fig = lr_finder.plot()
        lr_idx_lowest = np.array(lr_finder.history["loss"]).argmin()
        lr_loss_low = lr_finder.history["loss"][lr_idx_lowest]
        self.lr_from_finder = lr_finder.history["lr"][lr_idx_lowest]
        print(f'Lowest loss at {lr_loss_low:.6f} at Learning Rate {self.lr_from_finder:.6f}')
        # dont forget to reset model and optimizer to initial states
        lr_finder.reset()
        
        plt.show()
        return lr_fig, self.lr_from_finder
    
    def lr_update(self, new_lrs=None):
        if new_lrs is None:
            if self.lr_from_finder is None:
                raise ValueError('New lr not set. Run lr_range_test or define new lr.')
            new_lrs = self.lr_from_finder
        
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )
        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr    
    
    

    