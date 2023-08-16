import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import TransformerEncoderWithMLP

# ========================================= MODEL =======================================================
# Model hyperparameters
d_model = 5
nhead = 1
num_layers = 10
mlp_hidden_dim = 20
dropout_prob = 0

# Create the model
model = TransformerEncoderWithMLP(d_model, nhead, num_layers, mlp_hidden_dim, dropout_prob=dropout_prob).to('cuda')
# Calculate the number of learnable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params}")

# ==================================== Train Hyperparameters ==============================================

# Train hyperparameters
learning_rate = 0.01
num_epoch = 1000
batch_size = 10

# Loss function / Optimizer / Learning Rate Scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)


# ========================================= Train_Loop =====================================================

tmp = []
losses = []

for epoch in range(1000):  # Training loop
    optimizer.zero_grad()
    output = model(src)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    scheduler.step()  # Update learning rate

    tmp.append(scheduler.get_lr()[0])
    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item():.8f}")
