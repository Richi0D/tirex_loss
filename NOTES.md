# Personal Notes

# Inference TiRex

1. split input into batches (return generator)
   1. max length of a single batch? default=512, just split multiple sequences into batches. Does not split sequence!
2. _forecast_single_step
   1. _adjust_context_length: what happens at adjust context length? config is set for min and max at 2048
      1. Tirex process only a context of 2048 steps! if it is longer it get truncated otherwise padded with nan.
   2. input_transform: scale the data
      1. its uses per sample scaling!
         1. Use your scaler when:
                sequences vary wildly in scale
                you care about patterns, not magnitude
                time series / signals / trajectories
                Use standard scaler when:
                magnitude matters
                features must be comparable across dataset
                typical tabular ML
    3. unfold data into patches of size 32. (window size = 32)
    4. _forward_model_tokenized:
       1. create a mask for missing values (nan)
       2. set nan values based on mask
       3. _forward_model:
          1. input_patch_embedding: input residual block.
             1. input size = 2048/32=64 (64 patches), context length/window size
                each window(patch) get processed in parallel
             2. output size = 512
          2. xLSTM Block: Input is feeded into the 12x blocks
             1. sLSTM layer: 512 -> 512
             2. feedforward: 512 -> 1408 -> 512
          3. output_patch_embedding: 512 -> 2048 -> 288 (32*9, window size * quantile size)
          4. we get the same output.shape as input.shape with additional quantiles
    5. take the last patch as prediction (32 steps) (i guess for training the true data is shifted by the window size for computing the loss. but need to clarify this!)
    6. output_transform: scale back data
 3. repeat until full forecast horizon is predicted. 
    1. nan values are added to the context. (if horizon exceeds the max context length of 2048 only nan values are added as input and model will fail. so as longer the forecast horizon gets, as shorter gets the seen historic data!)
       1. TODO: add other methods instead of treading them as nan. (mean, median,...)



# Metrics
- MAPE explodes with zero values in time series!
  - Using sMAPE helps a bit, since we do 
      (np.abs(y_true) + np.abs(y_pred)) / 2.0
- MASE: scale free metric, but need some training data
- r2: just third option for some variance metric.
- Weighted Interval Score (WIS):
   The Weighted Interval Score is the primary metric for evaluating the overall accuracy and "sharpness" of the probabilistic forecast. It is a proper scoring rule that rewards the model for providing narrow uncertainty bounds while heavily penalizing any instances where the true value falls outside those bounds.
   
   Interpretation: 
   A lower WIS indicates a better model. It balances the desire for "sharp" (narrow) intervals with the need for coverage (capturing the true value). It can be viewed as a generalization of Mean Absolute Error (MAE) for distribution-based forecasts.
- Reliability Diagram (Calibration Plot):
   The Reliability Diagram evaluates the "honesty" of the model’s uncertainty estimates. It plots the predicted quantile levels (e.g., $0.1, 0.2, \dots, 0.9$) against the actual frequency with which the true values fall below those predictions.
   
   Interpretation:
   Perfect Calibration: 
   The data follows the $45^\circ$ dashed line.
   Over-forecasting: 
   The curve is below the diagonal (true values fall below the quantiles less often than expected).
   Under-forecasting: 
   The curve is above the diagonal (true values fall below the quantiles more often than expected).
   Over-confident: The curve has an "S" shape, meaning the predicted intervals are too narrow to capture the actual variance.


# idea forecasting with additional features
put learnable embedding before model. embedd all input features via this embedding.
We need two different embeddings: one for all features with known historic data, and one only with future known data.
we might need a linear layer head that squash this high dimensional embeddings into a single value. since tirex can only handle a single input feature.
then add this single value like residual style into the model
enriched = raw_value + projection_head(embeddings)
since the model has only a limtied context of 2048 we need to split this between historic and future input. example: 1024=historic, 1024=known future input.
The model has a mask. so set the mask for future values to zero.
output is then the predictions with quantiles.
So we can fine tune tirex model with additional features.
phase 1: only train your embedding + projection layers
for param in pretrained_model.parameters():
    param.requires_grad = False
phase 2: unfreeze and fine-tune everything together with small lr
for param in pretrained_model.parameters():
    param.requires_grad = True