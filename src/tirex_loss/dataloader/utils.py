import numpy as np

def create_windows(df, n, s_min, s_max, patch_size=32, seed=42):
    rng = np.random.default_rng(seed)
    
    all_windows = []
    all_s_values = []
    max_s = s_max
    total_len = n + max_s  # full padded row length

    for series_idx, group in df.group_by("series_index"):
        values = group["value"].to_numpy()
        
        i = 0
        while True:
            # random s for this chunk
            valid_s = np.arange(s_min, s_max + 1, patch_size)  # [32, 64, 96, 128, ...]
            valid_s = valid_s[valid_s >= s_min]
            s = rng.choice(valid_s)
            chunk_len = n + s
            
            if i + chunk_len > len(values):
                break  # not enough data, discard
            
            chunk = values[i : i + chunk_len]
            
            # pad front with NaN so every row has length n + max_s
            pad_size = max_s - s
            padded = np.concatenate([np.full(pad_size, np.nan), chunk])
            
            all_windows.append(padded)
            all_s_values.append(s)
            
            i += s  # shift by s

    windows = np.stack(all_windows, axis=0)  # [num_windows, n + max_s]
    s_values = np.array(all_s_values)         # [num_windows]
    return windows, s_values


def create_windows_fixed(df, n, s_min, s_max, patch_size=32, seed=42,
                         mode="shift", # "shift" pr "cpm"
                         cmax_mask=4,
                         pmax_mask=0.25,
                         cpm_stride=None,
                         ):
    assert mode in ("shift", "cpm"), f"unknown mode: {mode}"
    rng = np.random.default_rng(seed)
    
    all_windows = []
    all_s_values = []
    total_len = n + s_max  # full padded row length

    for series_idx, group in df.group_by("series_index"):
        values = group["value"].to_numpy()
        
        if mode == "shift":
            i = 0
            while True:
                # fixed s for this chunk
                s = s_max
                chunk_len = n + s
                
                if i + chunk_len > len(values):
                    break  # not enough data, discard
                
                chunk = values[i : i + chunk_len]
                            
                valid_s = np.arange(s_min, s_max + 1, patch_size)  # [32, 64, 96, 128, ...]
                valid_s = valid_s[valid_s >= s_min]
                for s in valid_s:
                    # pad front with NaN
                    pad_size = s - patch_size
                    shifted_input_chunk = chunk[pad_size : n]
                    target_chunk = chunk[n + s - patch_size : n + s]
                    assert not np.isnan(target_chunk).any(), "Target chunk contains NaN values"
                    padded = np.concatenate([shifted_input_chunk, np.full(pad_size, np.nan), target_chunk])

                    all_windows.append(padded)
                    all_s_values.append(patch_size) # always last patch in model
                
                i += s  # shift by s (non overlapping targets)

        else:  # mode == "cpm"
            assert n % patch_size == 0, "n must be a multiple of patch_size for cpm mode"
            num_patches = n // patch_size
            chunk_len = n + patch_size
            stride = cpm_stride if cpm_stride is not None else n
 
            i = 0
            while True:
                if i + chunk_len > len(values):
                    break  # not enough data, discard
 
                chunk = values[i : i + chunk_len]
 
                input_chunk = chunk[:n].copy()
                target_chunk = chunk[n : n + patch_size]
                assert not np.isnan(target_chunk).any(), "Target chunk contains NaN values"
 
                # sample CPM hyperparameters fresh for this sample
                c_mask = rng.integers(1, cmax_mask + 1)   # cmask ~ U(1, cmax_mask)
                p_mask = rng.uniform(0.0, pmax_mask)      # pmask ~ U(0, pmax_mask)
 
                num_blocks = num_patches // c_mask
                if num_blocks == 0:
                    # c_mask bigger than the whole input -> nothing maskable this sample
                    patch_mask = np.zeros(num_patches, dtype=bool)
                else:
                    block_mask = rng.random(num_blocks) < p_mask
                    patch_mask = np.repeat(block_mask, c_mask)
                    if patch_mask.size < num_patches:  # leftover from the floor()
                        patch_mask = np.concatenate(
                            [patch_mask, np.zeros(num_patches - patch_mask.size, dtype=bool)]
                        )
 
                step_mask = np.repeat(patch_mask, patch_size)
                input_chunk[step_mask] = np.nan
 
                padded = np.concatenate([input_chunk, target_chunk])
                all_windows.append(padded)
                all_s_values.append(patch_size)  # target is always exactly one patch
                i += stride

    windows = np.stack(all_windows, axis=0)  # [num_windows, n + max_s]
    s_values = np.array(all_s_values)         # [num_windows]
    return windows, s_values