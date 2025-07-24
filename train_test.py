
import os
import time
import torch
import pandas as pd
from train_test import Tester
from utils_dta import DTADataModule, calculate_metrics
from src.model import proSeqEncoder
from src.utils import set_random_seed, get_featurizer
from src.data import get_task_dir

EPOCHS = 200
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
set_random_seed(SEED)

task = "Davis"
task_dir = get_task_dir(task)
drug_feat = get_featurizer("Mol2Vec", save_dir=task_dir)
target_feat = get_featurizer("ProtBERT", save_dir=task_dir)

data_module = DTADataModule(task_dir, drug_feat, None, target_feat,
                            tops="pockets.csv",
                            device=DEVICE,
                            seed=SEED,
                            use_test=True,
                            cold=["Drug", "target_key"],
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4)
data_module.prepare_data()
data_module.setup()

model = SimplePocketModel()  
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(DEVICE)


tester = Tester(model, BATCH_SIZE)
test_loader, test_len = data_module.test_dataloader(domain=True)
loss_test, y_true, y_pred = tester.test(test_loader, DEVICE, test_len)

mse, ci, rm2, pearson, spearman = calculate_metrics(y_true, y_pred)
print(f"MSE: {mse:.4f}, CI: {ci:.4f}, Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")
