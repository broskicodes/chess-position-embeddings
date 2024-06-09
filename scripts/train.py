import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import torch.nn as nn
from tqdm.auto import tqdm
from model_v1 import PositionAutoEncoder
from helpers import get_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")
print(f"using {device}")

# set hyperparams
hyperparams = {
    "batch_size": 128,
    "n_epochs": 50,
    "learning_rate": 10e-4,
    "dropout_rate": 0,
    "position_channels": 15,
    "n_embed": 64,
    "filters": 32,
    "fc_size": 256,
    "version": 4
}

batch_size = hyperparams["batch_size"]
n_epochs = hyperparams["n_epochs"]
learning_rate = hyperparams["learning_rate"]
# ------------------------------

# create data loaders
positions = torch.load("../positions/positions.pt")["positions"]
train_dataset, val_dataset, test_dataset = get_datasets(positions, device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"# training batches: {len(train_loader)}")
print(f"# val batches: {len(val_loader)}")
print(f"# test batches: {len(test_loader)}")
# -------------------------------

# initialize model
model = PositionAutoEncoder(hyperparams)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_params = sum(p.numel() for p in model.parameters())/1e6
print(f"{num_params:.2f}M parameters")

num_training_steps = n_epochs * len(train_loader)
scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate, total_steps=num_training_steps)
print(f"training iterations: {num_training_steps}")
# --------------------------------

# training loop
progress_bar = tqdm(range(num_training_steps))
criterion = nn.MSELoss()

for epoch in range(n_epochs):
    model.train() # switch model to training mode

    for batch in train_loader:
        batch = batch.to(device)
        outputs = model(batch)
        loss = criterion(outputs, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.update(1)

        # if (progress_bar.n % 4000 == 0):
        #     print(f"erratic train loss: {loss.item()} lr: {optimizer.param_groups[0]['lr']}")
    
    print(f"epcoh: {epoch}")
    with torch.no_grad():
        # evaluate validation loss
        model.eval() # switch model to evaluation mode
        losses = torch.zeros(len(val_loader), device=device)
        k = 0
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
                
            losses[k] = loss.item()
            k += 1

        avg_val_loss = losses.mean()
        # -----------------------------
        
        # evaluate training loss
        losses =  torch.zeros(len(train_loader), device=device)
        k = 0
        for batch in train_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
                
            losses[k] = loss.item()
            k += 1
            
            if(k == len(train_loader)):
                break
        
        avg_train_loss = losses.mean()
        # ------------------------------
        print(f"lr: {optimizer.param_groups[0]['lr']}")
        print(f"val loss: {avg_val_loss}")
        print(f"train loss: {avg_train_loss}")

        # save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "hyperparameters": hyperparams,
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
            }
            torch.save(checkpoint, f"../checkpoints/v{hyperparams['version']}-checkpoint-{epoch // 10}.pt")
        # -------------------------------
# --------------------------------

# save model
weights = {
    "model": model.state_dict(),
    "train_set": train_dataset,
    "val_set": val_dataset,
    "test_set": test_dataset,
    "hyperparameters": hyperparams
}
torch.save(weights, f"../models/v{hyperparams['version']}.pt")
# -------------------------------