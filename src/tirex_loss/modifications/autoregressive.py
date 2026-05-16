import torch

# monkey patch for autoregressive forecasting loop
def autoregressive_mean_forecast_tensor(
    self,
    context: torch.Tensor,
    prediction_length: int | None = None,
    new_patch_count: int = 1,
) -> torch.Tensor:
    predictions = []
    if prediction_length is None:
        prediction_length = self.tokenizer.patch_size
    if prediction_length <= 0:
        raise ValueError("prediction_length needs to be > 0")

    remaining = -(prediction_length // -self.tokenizer.patch_size)

    context = context.to(dtype=torch.float32)
    while remaining > 0:
        new_patch_count = min(remaining, new_patch_count)
        prediction = self._forecast_single_step(context, new_patch_count)

        predictions.append(prediction)
        remaining -= new_patch_count

        if remaining <= 0:
            break
        
        # append mean instead of nan
        # context = torch.cat([context, torch.full_like(prediction[:, 0, :], fill_value=torch.nan)], dim=-1)
        mean = prediction[:, self.config.quantiles.index(0.5), :].squeeze(-1)
        context = torch.cat([context, mean], dim=-1)

    return torch.cat(predictions, dim=-1)[..., :prediction_length].to(dtype=torch.float32)