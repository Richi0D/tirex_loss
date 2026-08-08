import torch
import torch.nn as nn
import torch.nn.functional as F

from tirex.models.slstm.layer import sLSTMBlockConfig, sLSTMLayer
from tirex.util import round_up_to_next_multiple_of


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        hid = F.relu(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out + res
        return out


class LSTMBlock(nn.Module):
    def __init__(self, config: sLSTMBlockConfig, hidden_size:int, num_layers:int):
        super().__init__()
        self.config = config
        self.norm_slstm = RMSNorm(config.embedding_dim)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.0
                            )
        self.norm_ffn = RMSNorm(config.embedding_dim)

        up_proj_dim = round_up_to_next_multiple_of(config.embedding_dim * config.ffn_proj_factor, 64)
        self.ffn = FeedForward(config.embedding_dim, up_proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out, _ = self.lstm(self.norm_slstm(x))
        x = x + x_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class sLSTMBlock(nn.Module):
    def __init__(self, config: sLSTMBlockConfig, backend: str):
        super().__init__()
        self.config = config
        self.norm_slstm = RMSNorm(config.embedding_dim)
        self.slstm_layer = sLSTMLayer(config, backend)
        self.norm_ffn = RMSNorm(config.embedding_dim)

        up_proj_dim = round_up_to_next_multiple_of(config.embedding_dim * config.ffn_proj_factor, 64)
        self.ffn = FeedForward(config.embedding_dim, up_proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.slstm_layer(self.norm_slstm(x), slstm_state=None)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, up_proj_dim: int):
        super().__init__()
        self.proj_up_gate = nn.Linear(embedding_dim, up_proj_dim, bias=False)
        self.proj_up = nn.Linear(embedding_dim, up_proj_dim, bias=False)
        self.proj_down = nn.Linear(up_proj_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.proj_up_gate(x)) * self.proj_up(x)
        x = self.proj_down(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._rms_normalize(x.float()).to(x.dtype)
        x = x * self.weight
        return x

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class Mod_Tirex(nn.Module):
    """
    Custom/Modified Tirex Model with slstm or lstm block

    """
    def __init__(self,
                 input_patch_size:int,
                 output_patch_size:int,
                 quantiles:list[float],
                 hidden_size:int=256,
                 input_residual_h_dim:int=1024,
                 output_residual_h_dim:int=1024,
                 use_slstm:bool=True  
                 ):
        super().__init__()
        self.use_slstm = use_slstm

        # input residual block
        self.input_block = ResidualBlock(
            in_dim=input_patch_size,
            h_dim=input_residual_h_dim,
            out_dim=hidden_size
            )
    
        # LSTM or slstm Block
        self.slstm_block_config = sLSTMBlockConfig(embedding_dim=hidden_size, num_heads=4)
        if use_slstm:
            # backend can be "torch" or "cuda". Only use cuda if xlstm with custom cuda kernel is installed!
            self.hidden_block = sLSTMBlock(config=self.slstm_block_config, backend='cuda')
        else:
            self.hidden_block = LSTMBlock(config=self.slstm_block_config, hidden_size=hidden_size, num_layers=4)

        # RMS Norm Layer
        self.out_norm = RMSNorm(hidden_size)
        
        # output residual block
        self.output_block = ResidualBlock(
            in_dim=hidden_size,
            h_dim=output_residual_h_dim,
            out_dim=output_patch_size*len(quantiles)
            )
                
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # input residual block
        hidden_states = self.input_block(x)

        # sLSTM Block, for now just use one to keep complexity low
        hidden_states = self.hidden_block(hidden_states)

        # RSM Norm Layer
        hidden_states = self.out_norm(hidden_states)

        # oputput layer is same as input residual block but with different size for output
        hidden_states = self.output_block(hidden_states)

        return hidden_states



