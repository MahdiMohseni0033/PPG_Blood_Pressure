import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import *
import os
import argparse


def main(config):
    # ======================= initialization ==========================

    learning_rate = float(config.train.learning_rate)
    batch_size = int(config.train.batch_size)
    num_epoch = int(config.train.num_epoch)
    do_val = int(config.train.do_val)

    device = str(config.train.device)
    result_path = str(config.train.result)
    config_path = 'config.yaml'

    train_data_path = str(config.dataset.train_path)
    valid_data_path = str(config.dataset.valid_path)

    # ============= Initialize Result path ========
    result_path = create_result_dir(result_path, config_path)

    # ============= Initialize Tensorboard writer ========
    writer = SummaryWriter(os.path.join(result_path, 'runs'))

    # ============= Setup Logger =======================
    logging = logger_setup(result_path)

    # ============= Prepare model =======================
    model = prepare_model(logging, device)

    train_loader, valid_loader = prepare_data(logging, train_data_path, valid_data_path, batch_size)

    num_step = len(train_loader) * num_epoch
    logging.info(f'number of step for each epoch --- > {len(train_loader)}')

    # Loss function / Optimizer / Learning Rate Scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_step)

    best_loss = float('inf')

    for epoch in range(num_epoch):  # Training loop
        logging.info(f"====================== Epoch {epoch} ==========================")
        model.train()
        running_loss = 0.0
        for batch_idx, (src, target) in enumerate(train_loader):
            src = src.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            running_loss += loss.item()
            step = len(train_loader) * epoch + batch_idx
            writer.add_scalar('Training Loss', loss.item(), step)
            writer.add_scalar('Learning Rate', scheduler.get_lr()[0], step)

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Epoch Loss', epoch_loss, epoch)
        logging.info(f'Loss = {epoch_loss}')

        # Validation
        if epoch % do_val == 0:
            model.eval()
            validation_loss = 0.0

            with torch.no_grad():
                for batch_idx, (val_src, val_target) in enumerate(valid_loader):
                    val_src = val_src.to(device)
                    val_target = val_target.to(device)

                    val_output = model(val_src)
                    val_loss = criterion(val_output, val_target)
                    validation_loss += val_loss.item()

            validation_loss /= len(valid_loader)

            logging.info(f"Validation Loss: {validation_loss}")
            writer.add_scalar('Validation Loss:', validation_loss, epoch)

            # Save the best checkpoint
            if validation_loss < best_loss:
                logging.info(f"Best checkpoint saved in Epoch {epoch}  with Validation Loss: {validation_loss}")
                best_loss = validation_loss
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss
                }
                best_checkpoint_path = os.path.join(result_path, 'best_checkpoint.pth')
                torch.save(best_checkpoint, best_checkpoint_path)

        last_checkpoint = {
            'epoch': num_epoch - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        last_checkpoint_path = os.path.join(result_path, 'last_checkpoint.pth')
        torch.save(last_checkpoint, last_checkpoint_path)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your Project Description")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file (default: config.yaml)")
    args = parser.parse_args()

    # Load the configuration file
    config = load_config(args.config)

    # Call the main function with the loaded configuration
    main(config)
