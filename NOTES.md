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
             1. input size = 2048/32, context length/window size
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