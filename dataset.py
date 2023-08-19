import torch
from torch.utils.data import Dataset
import scipy.io
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_path, status='train'):
        self.status = status
        self.df = scipy.io.loadmat(data_path)
        self.signal = self.df['signal']
        self.sp = self.df['SP']
        self.dp = self.df['DP']

    def __len__(self):
        return len(self.sp)

    def __getitem__(self, idx):
        signal = torch.tensor(self.df['signal'][idx])

        sp = self.df['SP'][idx]
        dp = self.df['DP'][idx]

        if self.status == 'train' or self.status == 'valid':
            sp, dp = self.normalizer(sp, dp)
        else:
            pass

        # Convert the arrays to torch tensors
        sp_tensor = torch.tensor(sp)
        dp_tensor = torch.tensor(dp)
        combined_tensor = torch.stack([sp_tensor, dp_tensor])
        combined_tensor = torch.squeeze(combined_tensor)
        return signal.to(torch.float32), combined_tensor.to(torch.float32)

    @staticmethod
    def normalizer(sp, dp):
        sp = (sp - 80) / (190 - 80)
        dp = (dp - 50) / (119 - 50)
        return sp, dp


def denormalizer(input):
    # SP: input[0][0]
    # DP: input[0][1]
    input[0][0] = input[0][0] * (190 - 80) + 80
    input[0][1] = input[0][1] * (119 - 50) + 50

    return input


if __name__ == '__main__':
    mat_path = r'C:\python_project\prjs\bp-benchmark\datasets\splitted\uci2_dataset_ourPre\signal_fold_0.mat'
    dataset = CustomDataset(mat_path)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_idx, data in enumerate(dataloader):
        input = data[0]
        target = data[1]
        print(input.dtype)
        print(target.dtype)
        break

        # Set gradients to zero

