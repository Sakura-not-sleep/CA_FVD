import torch
import torch.nn as nn

class TransformerAlign(nn.Module):
    def __init__(self, embed_size=128, num_heads=2, num_layers=1):
        super(TransformerAlign, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        cls_token = torch.zeros(x.size(0), 1, self.embed_size).to(x.device)
        x = torch.cat((cls_token, x), dim=1)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x