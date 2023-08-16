import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D

p_enc_1d_model = PositionalEncoding1D(10)


class ResidualBlock(nn.Module):
    def __init__(self, layer, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return x + self.dropout(self.layer(x))


class TransformerEncoderWithMLP(nn.Module):
    def __init__(self, d_model, nhead, num_layers, mlp_hidden_dim, max_seq_len=125, dropout_prob=0.1):
        super(TransformerEncoderWithMLP, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   dim_feedforward=2048,
                                                   nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        residual_blocks = [ResidualBlock(encoder_layer, dropout_prob) for _ in range(num_layers)]
        self.residual_layers = nn.ModuleList(residual_blocks)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 2)  # Output shape: [batch_size, 2]
        )

    def forward(self, src):
        pos = p_enc_1d_model(src)
        src = src + pos
        out = src  # Initial input to the residual blocks

        for residual_layer in self.residual_layers:
            out = residual_layer(out)

        out = self.mlp(out.mean(dim=1))  # Taking the mean over the sequence dimension
        return out












