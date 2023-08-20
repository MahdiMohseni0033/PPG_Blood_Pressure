import torch
import argparse
from box import Box
import yaml
from model import MLP ,ModifiedMLP
from dataset import CustomDataset, denormalizer
from torch.utils.data import DataLoader
from tqdm import tqdm


# Load your model
def load_model(model_path):
    # Create an instance from your model
    input_size = 625
    output_size = 2
    # model = MLP(input_size, output_size).to('cuda')
    model = ModifiedMLP(input_size, output_size).to('cuda')

    # model = ResNet1D(in_channels=1,
    #                  base_filters=32,
    #                  first_kernel_size=13,
    #                  kernel_size=5,
    #                  stride=4,
    #                  groups=2,
    #                  n_block=8,
    #                  output_size=2,
    #                  is_se=True,
    #                  se_ch_low=4).to('cuda')

    # Load the checkpoint
    checkpoint = torch.load(model_path)

    # Load the state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Inference")

    parser.add_argument("--model", type=str,
                        default="/media/mmohseni/ubuntu/projects/BP_custom/train/result/2023_08_20__13_04_06/best_checkpoint.pth",
                        help="Path to the model file (default: model.pth)")

    parser.add_argument("--input", type=str,
                        default=r"/media/mmohseni/ubuntu/projects/BP_custom/train/train_data.mat",
                        help="Path to the input data file (signal_fold_0.mat)")

    args = parser.parse_args()
    device = 'cuda'

    # Load the model
    model = load_model(str(args.model))
    model.eval()
    test_data = CustomDataset(str(args.input), status='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    t_5 = 0
    t_10 = 0
    t_15 = 0
    print(len(test_data))
    with torch.no_grad():
        for batch_idx, (tst_src, test_target) in tqdm(enumerate(test_loader)):
            tst_src = tst_src.to(device)
            test_target = test_target.to(device)

            test_output = model(tst_src)
            test_output = denormalizer(test_output)
            Error = abs(test_target - test_output)[0][1].item()
            # print(Error)

            if Error <= 5:
                t_5 += 1

            if Error <= 10:
                t_10 += 1

            if Error <= 15:
                t_15 += 1

        print(f'{t_5 / len(test_data)} , {t_10 / len(test_data)} , {t_15 / len(test_data)}')
