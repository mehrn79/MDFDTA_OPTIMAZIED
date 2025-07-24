import os
import time
import torch
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from utils_dta import DTADataModule, calculate_metrics
from train_test import Trainer, Tester
from src.model import proSeqEncoder

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
data_module = DTADataModule(batch_size=BATCH_SIZE)
data_module.prepare_data()
data_module.setup()


print("Initializing model...")
model = proSeqEncoder() 
model.to(DEVICE)

trainer = Trainer(model, LEARNING_RATE)
tester = Tester(model, BATCH_SIZE)

best_model = None
min_val_loss = float('inf')
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    train_loss, y_train, y_pred_train = trainer.train(train_loader, DEVICE, len(train_loader))
    train_losses.append(train_loss)

    val_loss, y_val, y_pred_val = tester.test(val_loader, DEVICE, len(val_loader))
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < min_val_loss:
        best_model = copy.deepcopy(model)
        min_val_loss = val_loss
        print(f"New best model at epoch {epoch}!")

loss_data = pd.DataFrame({'epoch': range(1, EPOCHS + 1),
                          'train_loss': train_losses,
                          'val_loss': val_losses})
loss_data.to_csv("loss_curve.csv", index=False)
print("Training finished. Best validation loss:", min_val_loss)

print("\nTesting the best model...")
tester = Tester(best_model, BATCH_SIZE)
test_loader = data_module.test_dataloader()
test_loss, y_test, y_pred_test = tester.test(test_loader, DEVICE, len(test_loader))

mse_test, ci_test, rm2_test, pearson_test, spearman_test = calculate_metrics(y_test, y_pred_test)
print(f"Test MSE: {mse_test:.4f}, CI: {ci_test:.4f}, Pearson: {pearson_test:.4f}, Spearman: {spearman_test:.4f}")
