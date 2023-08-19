import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
import torch
# from resnet import MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
import logging

logger = logging.getLogger(__name__)

p_enc_1d_model = PositionalEncoding1D(10)


# class ResidualBlock(nn.Module):
#     def __init__(self, layer, dropout_prob):
#         super(ResidualBlock, self).__init__()
#         self.layer = layer
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, x):
#         return x + self.dropout(self.layer(x))
#
#
# class TransformerEncoderWithMLP(nn.Module):
#     def __init__(self, d_model, nhead, num_layers, mlp_hidden_dim, max_seq_len=125, dropout_prob=0.1):
#         super(TransformerEncoderWithMLP, self).__init__()
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
#                                                    dim_feedforward=2048,
#                                                    nhead=nhead)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         residual_blocks = [ResidualBlock(encoder_layer, dropout_prob) for _ in range(num_layers)]
#         self.residual_layers = nn.ModuleList(residual_blocks)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(mlp_hidden_dim, 2)  # Output shape: [batch_size, 2]
#         )
#
#     def forward(self, src):
#         bsz = src.shape[0]
#         src = torch.reshape(src, (bsz, 125, 5))
#         pos = p_enc_1d_model(src)
#         src = src + pos
#         out = src  # Initial input to the residual blocks
#
#         for residual_layer in self.residual_layers:
#             out = residual_layer(out)
#
#         out = self.mlp(out.mean(dim=1))  # Taking the mean over the sequence dimension
#         return out
#

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class MLP2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 256)
        self.fc8 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x


# class ResNet1D(nn.Module):
#     """
#
#     Input:
#         X: (n_samples, n_channel, n_length)
#         Y: (n_samples)
#
#     Output:
#         out: (n_samples)
#
#     Pararmetes:
#         in_channels: dim of input, the same as n_channel
#         base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
#         kernel_size: width of kernel
#         stride: stride of kernel moving
#         groups: set larget to 1 as ResNeXt
#         n_block: number of blocks
#         n_classes: number of classes
#
#         param_model:
#
#
#     """
#
#     def __init__(self, in_channels, base_filters, first_kernel_size, kernel_size, stride,
#                  groups, n_block, output_size, is_se=False, se_ch_low=4, downsample_gap=2,
#                  increasefilter_gap=2, use_bn=True, use_do=True, verbose=False):
#         super(ResNet1D, self).__init__()
#
#         self.verbose = verbose
#         self.n_block = n_block
#         self.first_kernel_size = first_kernel_size
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.groups = groups
#         self.use_bn = use_bn
#         self.use_do = use_do
#         self.is_se = is_se
#         self.se_ch_low = se_ch_low
#
#         self.downsample_gap = downsample_gap  # 2 for base model
#         self.increasefilter_gap = increasefilter_gap  # 4 for base model
#
#         # first block
#         self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters,
#                                                 kernel_size=self.first_kernel_size, stride=1)
#         self.first_block_bn = nn.BatchNorm1d(base_filters)
#         self.first_block_relu = nn.ReLU()
#         self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
#         out_channels = base_filters
#
#         # residual blocks
#         self.basicblock_list = nn.ModuleList()
#         for i_block in range(self.n_block):
#             # is_first_block
#             if i_block == 0:
#                 is_first_block = True
#             else:
#                 is_first_block = False
#             # downsample at every self.downsample_gap blocks
#             if i_block % self.downsample_gap == 1:
#                 downsample = True
#             else:
#                 downsample = False
#             # in_channels and out_channels
#             if is_first_block:
#                 in_channels = base_filters
#                 out_channels = in_channels
#             else:
#                 # increase filters at every self.increasefilter_gap blocks
#                 in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
#                 if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
#                     out_channels = in_channels * 2
#                 else:
#                     out_channels = in_channels
#
#             tmp_block = BasicBlock(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 groups=self.groups,
#                 downsample=downsample,
#                 use_bn=self.use_bn,
#                 use_do=self.use_do,
#                 is_first_block=is_first_block,
#                 is_se=self.is_se,
#                 se_ch_low=self.se_ch_low)
#             self.basicblock_list.append(tmp_block)
#
#         # final prediction
#         self.final_bn = nn.BatchNorm1d(out_channels)
#         self.final_relu = nn.ReLU(inplace=True)
#
#         # Classifier
#         self.main_clf = nn.Linear(out_channels, output_size)
#
#     # def forward(self, x):
#     def forward(self, x):
#         # x = x['ppg']
#         x = torch.unsqueeze(x, dim=1)
#         assert len(x.shape) == 3
#
#         # skip batch norm if batchsize<4:
#         if x.shape[0] < 4:    self.use_bn = False
#
#         # first conv
#         if self.verbose:
#             logger.info('input shape', x.shape)
#         out = self.first_block_conv(x)
#         if self.verbose:
#             logger.info('after first conv', out.shape)
#         if self.use_bn:
#             out = self.first_block_bn(out)
#         out = self.first_block_relu(out)
#         out = self.first_block_maxpool(out)
#
#         # residual blocks, every block has two conv
#         for i_block in range(self.n_block):
#             net = self.basicblock_list[i_block]
#             if self.verbose:
#                 logger.info('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block,
#                                                                                                         net.in_channels,
#                                                                                                         net.out_channels,
#                                                                                                         net.downsample))
#             out = net(out)
#             if self.verbose:
#                 logger.info(out.shape)
#
#         # final prediction
#         if self.use_bn:
#             out = self.final_bn(out)
#         h = self.final_relu(out)
#         h = h.mean(-1)  # (n_batch, out_channels)
#         # logger.info('final pooling', h.shape)
#
#         # ===== Concat x_demo
#         out = self.main_clf(h)
#         return out
#

class ModifiedMLP1(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.5):
        super(ModifiedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.batchnorm1(self.relu(self.fc1(x)))
        x = self.batchnorm2(self.relu(self.fc2(x)))
        x = self.batchnorm3(self.relu(self.fc3(x)))
        x = self.batchnorm4(self.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class ModifiedMLP(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.05):
        super(ModifiedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.batchnorm1(self.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.batchnorm2(self.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.batchnorm3(self.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.batchnorm4(self.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.batchnorm5(self.relu(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc6(x)
        return x


if __name__ == '__main__':
    model = MLP2(625, 2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
