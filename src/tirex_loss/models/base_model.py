import torch
import torch.nn as nn

from .mod_tirex import Mod_Tirex
from tirex.models.patcher import PatchedTokenizer

class Base_Model(nn.Module):
    """
    Base model

    """
    def __init__(self,
                 context_length: int,
                 quantiles: list[float],
                 patch_size: int = 32,
                 use_slstm:bool = True
                 ):
        super().__init__()
        
        self.context_length = context_length
        self.quantiles = quantiles
        self.patch_size = patch_size
        self.nan_mask_value = 0.

        # scaler and patcher for input and output transformation
        self.tokenizer = PatchedTokenizer(patch_size=self.patch_size)

        # here we define the model used
        self.forecast_model = Mod_Tirex(input_patch_size=patch_size,
                                        output_patch_size=patch_size,
                                        quantiles=quantiles,
                                        hidden_size=256,
                                        input_residual_h_dim=1024,
                                        output_residual_h_dim=1024,
                                        use_slstm=use_slstm)
    

    def _adjust_context_length(self, context: torch.Tensor, min_context: int, max_context: int):
        pad_len = 0
        if context.shape[-1] > max_context:
            context = context[..., -max_context:]
        if context.shape[-1] < min_context:
            pad_len = min_context - context.shape[-1]
            pad = torch.full(
                (context.shape[0], pad_len),
                fill_value=torch.nan,
                device=context.device,
                dtype=context.dtype,
            )
            context = torch.concat((pad, context), dim=1)
        return context, pad_len

    def _forecast_tensor(self, context, prediction_length=None, new_patch_count=1, autoregressive=False):
        return self(context, prediction_length, new_patch_count, autoregressive)

    def forward(self,
                context:torch.Tensor,
                prediction_length:int | None = None,
                new_patch_count:int = 1,
                autoregressive:bool = False,
                ) -> torch.Tensor:
        
        if prediction_length is None:
            prediction_length = self.tokenizer.patch_size
        if prediction_length <= 0:
            raise ValueError("prediction_length needs to be > 0")
        
        remaining = -(prediction_length // -self.tokenizer.patch_size)
        predictions = []
        context = context.to(dtype=torch.float32)
        while remaining > 0:
            new_patch_count = min(remaining, new_patch_count)
            prediction, tokenizer_state = self._forecast_single_step(context)

            predictions.append(prediction)
            remaining -= new_patch_count

            if remaining <= 0:
                break

            if autoregressive:
                mean = prediction[:, self.quantiles.index(0.5), :].squeeze(-1)
                context = torch.cat([context, mean], dim=-1)
            else:
                context = torch.cat([context, torch.full_like(prediction[:, 0, :].detach(), fill_value=torch.nan)], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32)        

        
    def _forecast_single_step(self, context:torch.Tensor,
                              new_patch_count: int = 1,
                              training:bool = False) -> torch.Tensor:

        # adjust context length, will take only last n time steps of the context
        context, _ = self._adjust_context_length(context, self.context_length, self.context_length)

        # scale the data and transform to patches
        input_token, tokenizer_state = self.tokenizer.input_transform(context)

        # mask null values
        input_mask = torch.isnan(input_token).logical_not().to(input_token.dtype)    
        input_token = torch.nan_to_num(input_token, nan=self.nan_mask_value)

        # model pass
        quantile_preds = self.forecast_model(input_token)

        # inverse transform predictions
        quantile_preds = torch.unflatten(
            quantile_preds, -1, (len(self.quantiles), self.patch_size)
        )
        quantile_preds = torch.transpose(quantile_preds, 1, 2)  # switch quantile and num_token_dimension
        predicted_token = quantile_preds[:, :, -new_patch_count:, :].to(input_token)  # predicted token

        # Shape: [bs, num_quantiles, num_predicted_token, output_patch_size]
        predicted_token = self.tokenizer.output_transform(predicted_token, tokenizer_state)
        return predicted_token, tokenizer_state