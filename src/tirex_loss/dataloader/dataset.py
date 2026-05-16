import torch
import polars as pl
from torch.utils.data import Dataset
from tirex_loss.dataloader.utils import create_windows


class TirexDataset(Dataset):
    """
    Dataset class used for dataloaders.
    """
    def __init__(self, 
        data: pl.DataFrame,
        context_length: int = 2048, 
        prediction_length_min: int = 32,
        prediction_length_max: int = 1024,
        ) -> None:
        
        super().__init__()
        self.sequences = None
        self.prediction_lengths = None
        self.context_length = context_length
        self.prediction_length_min = prediction_length_min
        self.prediction_length_max = prediction_length_max  

        # create the windows and prediction lengths for the dataset
        self.sequences, self.prediction_lengths = create_windows(data,
                                                                 n=context_length,
                                                                 s_min=prediction_length_min,
                                                                 s_max=prediction_length_max
                                                                 )
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        self.prediction_lengths = torch.tensor(self.prediction_lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        x = self.sequences[index, self.prediction_length_max-self.prediction_lengths[index]:-self.prediction_lengths[index]]
        y = self.sequences[index, -self.prediction_lengths[index]:]
        pred_length = self.prediction_lengths[index]

        return x, y, pred_length