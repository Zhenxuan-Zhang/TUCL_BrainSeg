import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from utils.dataset_utils import split_dataset, get_dataloader
from utils.loss_utils import seg_loss
from utils.scheduler_utils import cosine_scheduler
from models import get_model

# ------------------------
# PARSE COMMAND LINE ARGUMENTS
# ------------------------
parser = argparse.ArgumentParser(description="Train a segmentation model.")
parser.add_argument("--model", type=str, required=True, choices=[
    "tucl"
], help="Model type")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=0.003, help="Initial learning rate")
parser.add_argument("--min_lr", type=float, default=0.0005, help="Minimum learning rate")
parser.add_argument("--warmup_epochs", type=int, default=8, help="Number of warmup epochs")
parser.add_argument("--save_path", type=str, required=True, help="Directory to save model checkpoints")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
parser.add_argument("--use_train_loop_sp", action="store_true", help="Use train_loop_sp for TUCL model")
parser.add_argument("--use_train_loop_sp_uc", action="store_true", help="Use train_loop_sp_uc for TUCL model")
args = parser.parse_args()

# ------------------------
# SET DEVICE
# ------------------------
args.save_path = os.path.abspath(args.save_path)
os.makedirs(args.save_path, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

# ------------------------
# LOAD DATA
# ------------------------
full_dataset = get_dataloader(args.data_dir, batch_size=args.batch_size)
train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}, Test images: {len(test_dataset)}")

# ------------------------
# IMPORT TRAINING FUNCTIONS BASED ON MODEL
# ------------------------
if args.model == "tucl":
    from utils.train_utils_tucl import train_loop, train_loop_sp, train_loop_sp_uc, val_loop

# ------------------------
# MODEL SETUP
# ------------------------
model = get_model(args.model).to(device)
criterion = seg_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
scheduler = cosine_scheduler(args.lr, args.min_lr, args.epochs, args.warmup_epochs)

# ------------------------
# SELECT TRAINING FUNCTION
# ------------------------
if args.model == "tucl":
    if args.use_train_loop_sp_uc:
        train_loop_fn = train_loop_sp_uc
    elif args.use_train_loop_sp:
        train_loop_fn = train_loop_sp
    else:
        train_loop_fn = train_loop
else:
    train_loop_fn = train_loop

# ------------------------
# LOAD CHECKPOINT IF EXISTS
# ------------------------
weight_path = os.path.join(args.save_path, f"{args.model}.pth")
if os.path.exists(weight_path):
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Successfully loaded checkpoint.")

# ------------------------
# TRAINING
# ------------------------
train_log_path = os.path.join(args.save_path, f"{args.model}_log.txt")
train_loop_fn(model, optimizer, scheduler, criterion, train_loader, device, args.epochs)

# ------------------------
# VALIDATION & TESTING
# ------------------------
metrics_val = val_loop(model, criterion, val_loader, device)
metrics_test = test_loop(model, criterion, test_loader, device)

print(f"Valid -- Dice: ET {metrics_val['dice1']:.3f} WT {metrics_val['dice2']:.3f} TC {metrics_val['dice3']:.3f}")
print(f"Test -- Dice: ET {metrics_test['dice1']:.3f} WT {metrics_test['dice2']:.3f} TC {metrics_test['dice3']:.3f}")