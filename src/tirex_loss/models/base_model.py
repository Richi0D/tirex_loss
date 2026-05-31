import torch
import torch.nn as nn

from .mod_tirex import Mod_Tirex
from tirex.models.tirex import _adjust_context_length
from tirex.models.patcher import PatchedTokenizer

class Base_Model(nn.Module):
    """
    Base model

    """
    def __init__(self,
                 context_length: int,
                 quantiles: List[float],
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
    
    def forward(self,
                context:torch.Tensor,
                prediction_length:int | None = None,
                new_patch_count:int = 1,
                autoregressive:bool = False
                ) -> torch.Tensor:
        
        if prediction_length is None:
            prediction_length = self.tokenizer.patch_size
        if prediction_length <= 0:
            raise ValueError("prediction_length needs to be > 0")
        
        predictions = []
        context = context.to(dtype=torch.float32)
        while remaining > 0:
            new_patch_count = min(remaining, new_patch_count)
            prediction = self._forecast_single_step(context, new_patch_count)

            predictions.append(prediction)
            remaining -= new_patch_count

            if remaining <= 0:
                break

            if autoregressive:
                mean = prediction[:, self.config.quantiles.index(0.5), :].squeeze(-1)
                context = torch.cat([context, mean], dim=-1)
            else:
                context = torch.cat([context, torch.full_like(prediction[:, 0, :].detach(), fill_value=torch.nan)], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32)        

        
    def _forecast_single_step(self, context:torch.Tensor) -> torch.Tensor:

        # adjust context length, will take only last n time steps of the context
        context, _ = _adjust_context_length(context, self.context_length, self.context_length)

        # scale the data and transform to patches
        input_token, tokenizer_state = self.tokenizer.input_transform(context)

        # mask null values
        input_mask = (
            input_mask.to(input_token.dtype)
            if input_mask is not None
            else torch.isnan(input_token).logical_not().to(input_token.dtype)
        )        
        input_token = torch.nan_to_num(input_token, nan=self.nan_mask_value)

        # model pass
        quantile_preds = self.forecast_model(input_token)

        # inverse transform predictions
        quantile_preds = torch.unflatten(
            quantile_preds, -1, (len(self.quantiles), self.patch_size)
        )
        quantile_preds = torch.transpose(quantile_preds, 1, 2)  # switch quantile and num_token_dimension
        # Shape: [bs, num_quantiles, num_predicted_token, output_patch_size]
        predicted_token = self.tokenizer.output_transform(predicted_token, tokenizer_state)