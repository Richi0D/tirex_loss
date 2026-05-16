import numpy as np

def create_windows(df, n, s_min, s_max, seed=42):
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
            valid_s = np.arange(s_min, s_max + 1, 32)  # [32, 64, 96, 128, ...]
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