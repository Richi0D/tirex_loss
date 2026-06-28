from tirex_loss.dataloader.dataset import TirexDataset, TirexDataset_Fixed
from tirex_loss.dataloader.sampler import SameLengthBatchSampler
from torch.utils.data import DataLoader


def build_dataloader_from_dataset(dataset, batch_size, shuffle=True, **kwargs):
    sampler = SameLengthBatchSampler(dataset.prediction_lengths, batch_size, shuffle)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)

def build_dataloader(df, context_length, s_min, s_max, batch_size, shuffle=True, **kwargs):
    dataset = TirexDataset(df, context_length, s_min, s_max)
    sampler = SameLengthBatchSampler(dataset.prediction_lengths, batch_size, shuffle)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)

def build_dataloader_fixed(df, context_length, s_min, s_max, batch_size,
                           patch_size, mode, cmax_mask, pmax_mask, cpm_stride,
                            shuffle=True, **kwargs):
    dataset = TirexDataset_Fixed(df, context_length, s_min, s_max,
                                patch_size=patch_size,
                                mode=mode,
                                cmax_mask=cmax_mask,
                                pmax_mask=pmax_mask,
                                cpm_stride=cpm_stride)
    sampler = SameLengthBatchSampler(dataset.prediction_lengths, batch_size, shuffle)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)