import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import MultiHeadAttention

class Conv_block(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(Conv_block, self).__init__()
        self.intial_conv = nn.Sequential(
                                         nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                                         nn.GroupNorm(4, out_channels),
                                         nn.SiLU()
                                        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.intial_conv(x)
        return out

"""_____________________________________________________________________________________________________________________________________________________________"""

class Resnet_Block(nn.Module):

    """
    This block comprises of two Convolutional blocks, followed by one Residual Connection.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super(Resnet_Block, self).__init__()
        self.time_linear = nn.Sequential(
                                         nn.SiLU(),
                                         nn.Linear(time_emb_dim, out_channels)
                                        )
        self.resnet_conv_1 = Conv_block(in_channels, out_channels)
        self.resnet_conv_2 = Conv_block(out_channels, out_channels)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
       
        time_emb = self.time_linear(t)[:, :, None, None]
        residue = x
        out = self.resnet_conv_1(x)
        out = out + time_emb
        out = self.resnet_conv_2(out)
        out = out + self.residual(residue)
        
        return out

"""_____________________________________________________________________________________________________________________________________________________________"""

class Downsample(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Downsample, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.down_conv(x)
        return out

"""_____________________________________________________________________________________________________________________________________________________________"""    

class Upsample(nn.Module):

    def __init__(self, in_channels: int):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size = 4, stride = 2, padding = 1)

    def forward(self, x: torch.Tensor):

        return self.conv(x)

"""_____________________________________________________________________________________________________________________________________________________________"""    

class UNET_Attention(nn.Module):

    def __init__(self, n_heads: int, emb_dim: int):
        super(UNET_Attention, self).__init__()
        self.n_heads = n_heads
        self.emd_dim = emb_dim
        self.channels = self.n_heads * self.emd_dim
        self.attention = MultiHeadAttention(self.n_heads, self.channels)
        self.layer_norm_1 = nn.LayerNorm(self.channels)
        self.layer_norm_2 = nn.LayerNorm(self.channels)
        self.linear_1 = nn.Linear(self.channels, 2 * self.channels, bias = False)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
    
        n, c, h, w = x.shape
        residue = x
        x = x.view(n, c, h * w) 
        x = x.transpose(-1, -2) # Height * Width is the Sequence, Channels are the features.
        residue_short = x

        x = self.layer_norm_1(x)
        out = self.attention(x)
        out += residue_short
        out = self.layer_norm_2(out)
        out, gate = self.linear_1(out).chunk(2, dim = -1)
        out = out * F.gelu(gate)

        out = x.transpose(-1, -2)
        out = out.view(n, c, h, w)
        out = out + residue
        
        return out
        
"""_____________________________________________________________________________________________________________________________________________________________"""

class Time_Encoding(nn.Module):

    """
    Similar to Transformer's Positional Encoding.
    """

    def __init__(self, time_emb_dim: int):
        super(Time_Encoding, self).__init__()
        self.time_emb_dim = time_emb_dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:

        freq = 10000 ** (torch.arange(0, self.time_emb_dim // 2, dtype = torch.float32) / (self.time_emb_dim // 2)).to(t.device)
        factor = t[:, None].repeat(1, self.time_emb_dim // 2) / freq
        pos_enc_a = torch.sin(factor).to(t.device)
        pos_enc_b = torch.cos(factor).to(t.device)
        time_enc = torch.cat([pos_enc_a, pos_enc_b], dim = -1).to(t.device)

        return time_enc

"""_____________________________________________________________________________________________________________________________________________________________"""

class UNET(nn.Module):

    """
    This UNET is conditioned and unconditioned to predict the noise based on the label. Based on the random probability, the model learns to acts both as
    conditioned and unconditioned model. This approach is known as "Classifier-Free Guidance."

    Encoder:
        1. Resenet Block
        2. Attention Block
        3. Downsample.

    Bottle Neck:
        1. Resenet Block
        2. Attention Block
        3. Resnet Block.

    Decoder:
        1. Resenet Block
        2. Attention Block
        3. Upsample.

    """

    def __init__(self, image_channel: int, time_emb_dim: int, num_labels: int):
        super(UNET, self).__init__()

        self.image_channel = image_channel
        self.time_emb_dim = time_emb_dim

        self.ini_conv = nn.Conv2d(self.image_channel, 16, kernel_size = 3, padding = 1)
        self.time_encoding =  Time_Encoding(self.time_emb_dim)
        self.time_embedding = nn.Sequential(
                                            nn.Linear(self.time_emb_dim, 2 * self.time_emb_dim),
                                            nn.SiLU(),
                                            nn.Linear(2 * self.time_emb_dim, self.time_emb_dim)
                                           )
        # Encoder Stack
        self.encoders = nn.ModuleList([
                                       Resnet_Block(16, 32, self.time_emb_dim), UNET_Attention(4, 8), Downsample(32, 32),
                                       Resnet_Block(32, 64, self.time_emb_dim), UNET_Attention(4, 16), Downsample(64, 64),
                                       Resnet_Block(64, 128, self.time_emb_dim), UNET_Attention(4, 32), nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
                                      ])
        # Bottleneck Stack
        self.bottleneck = nn.ModuleList([
                                         Resnet_Block(128, 128, self.time_emb_dim),
                                         UNET_Attention(4, 32),
                                         Resnet_Block(128, 64, self.time_emb_dim)
                                        ])
        # Decoder Stack
        self.decoders = nn.ModuleList([
                                       nn.ModuleList([Resnet_Block(64 + 128, 64, self.time_emb_dim), UNET_Attention(4, 16), nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 1, padding = 1)]),
                                       nn.ModuleList([Resnet_Block(64 + 64, 32, self.time_emb_dim), UNET_Attention(4, 8), Upsample(32)]),
                                       nn.ModuleList([Resnet_Block(32 + 32, 16, self.time_emb_dim), UNET_Attention(4, 4), Upsample(16)])
                                     ])
        self.group_norm = nn.GroupNorm(4, 16)
        self.activation = nn.SiLU()
        self.unet_out = nn.Conv2d(16, self.image_channel, kernel_size = 3, padding = 1)  
        # Label Embedding
        if num_labels is not None:
             self.label_emb = nn.Embedding(num_labels, self.time_emb_dim)

    def forward(self, x: torch.Tensor, label: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        time_emb = self.time_encoding(t)
        #print(time_emb.size())

        if label is not None:
            label_emb_ = self.label_emb(label)
            #print(label_emb_.size())
            time_emb += label_emb_
        
        time_emb = self.time_embedding(time_emb)
        out = x
        out = self.ini_conv(out)
        encoder_outputs = []

        # Encoder Functionality
        for layer in self.encoders:
            if isinstance(layer, Resnet_Block):
                out = layer(out, time_emb)
            elif isinstance(layer, Downsample):
                out = layer(out)
                encoder_outputs.append(out)
            elif isinstance(layer, UNET_Attention):
                out = layer(out)
            elif isinstance(layer, nn.Conv2d):
                out = layer(out)
                encoder_outputs.append(out)

        # Bottleneck Functionality         
        for layer in self.bottleneck:
            if isinstance(layer, Resnet_Block):
                out = layer(out, time_emb)
            else:
                out = layer(out)
        
        # Decoder Functionality
        for layers in self.decoders:
            out = torch.cat((out, encoder_outputs.pop()), dim = 1)
            for layer in layers:
                if isinstance(layer, UNET_Attention):
                    out = layer(out)
                elif  isinstance(layer, Upsample) or isinstance(layer, nn.ConvTranspose2d):
                    out = layer(out)
                elif isinstance(layer, Resnet_Block):
                    out = layer(out, time_emb)
        
        out = self.group_norm(out)
        out = self.activation(out)
        out = self.unet_out(out)

        return out

"""_____________________________________________________________________________________________________________________________________________________________"""

if __name__ == '__main__':

    unet = UNET(3, 128, 2)
    Num_of_parameters = sum(p.numel() for p in unet.parameters())
    print("Model Parameters : {:.3f} M".format(Num_of_parameters / 1e6))
    print(unet)