import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score
from model import *
from model import *
from torch.utils.data import DataLoader
from dataset import CustomDataset
import shutil
import datetime
import os
import yaml
from pathlib import Path
from box import Box


def r2_metric(y_true, y_pred):
    return r2_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), multioutput='variance_weighted')

def plot_metrics(tmp, losses):
    tmp = np.array(tmp)
    losses = np.array(losses)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot learning rate in the first subplot
    ax1.plot(tmp, color='red')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate')

    # Plot loss in the second subplot
    ax2.plot(losses, color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')

    plt.tight_layout()
    plt.show()


def logger_setup(path):
    saved_path = os.path.join(path, 'training.log')
    # Set up logging
    logging.basicConfig(filename=saved_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Create a stream handler to display logs in the terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the stream handler to the logger
    logger = logging.getLogger()
    logger.addHandler(stream_handler)

    return logger


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        config = Box(config)
    return config
def create_result_dir(result_path,config_path):
    Path(result_path).mkdir(parents=True, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    result_path = os.path.join(result_path, run_id)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, result_path)
    return result_path


def prepare_model(logging, device):
    # Define the input and output sizes
    input_size = 625
    output_size = 2

    # Create an instance of the MLP model
    # model = MLP(input_size, output_size).to(device)
    model = ModifiedMLP(input_size, output_size).to(device)

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

    # Calculate the number of learnable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total learnable parameters: {total_params}")
    return model


def prepare_data(logging, train_data_path, valid_data_path, batch_size):
    train_dataset = CustomDataset(train_data_path, status='train')
    val_dataset = CustomDataset(valid_data_path, status='valid')

    logging.info(f'number of training data --- >  {len(train_dataset)}')
    logging.info(f'number of validation data --- >  {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader

