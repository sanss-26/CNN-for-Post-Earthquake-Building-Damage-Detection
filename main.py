import argparse
import random
import os
import time

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score

from dataset import EarthquakeDataset, MyAug
from model import M3ICNet, MODEL_MM, MODEL_SAR, MODEL_OPT
from metrics import compute_imagewise_retrieval_metrics, compute_imagewise_f1_metrics

# speedups
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--root", default="data", type=str)
parser.add_argument("--val_split", default="fold-1.txt", type=str)
parser.add_argument("--checkpoints", default="checkpoints_all/fold1", type=str)
parser.add_argument("--sar_pretrain", default=None, type=str)
parser.add_argument("--opt_pretrain", default=None, type=str)
parser.add_argument("--use_shadow", action="store_true")
parser.add_argument("--model_type", default="m3ic", choices=["m3ic","all","sar","opt"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare splits
folds = [f"fold-{i}.txt" for i in range(1,6)]
train_splits = [f for f in folds if f != args.val_split]
val_splits   = [args.val_split]

train_ds = EarthquakeDataset(args.root, train_splits)
val_ds   = EarthquakeDataset(args.root, val_splits)
print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

# weighted sampler
y = train_ds.labels
counts = np.array([np.sum(y == t) for t in np.unique(y)])
weights = 1.0 / counts
sample_weights = np.array([weights[t] for t in y])
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=args.batch_size, sampler=sampler,
    num_workers=args.num_workers, pin_memory=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=1, shuffle=False,
    num_workers=args.num_workers, pin_memory=True
)

# select model
if args.model_type == "m3ic":
    model = M3ICNet(
        sar_pretrain=args.sar_pretrain,
        opt_pretrain=args.opt_pretrain,
        use_shadow=args.use_shadow
    )
elif args.model_type == "sar":
    model = MODEL_SAR(args.sar_pretrain)
elif args.model_type == "opt":
    model = MODEL_OPT(args.opt_pretrain, use_shadow=args.use_shadow)
else:
    model = MODEL_MM(args.sar_pretrain, args.opt_pretrain, use_shadow=args.use_shadow)
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

augment = MyAug().to(device)
os.makedirs(args.checkpoints, exist_ok=True)

# track best by F1
best_f1            = 0.0
best_auroc_at_f1   = 0.0
best_epoch         = 0
start = time.time()

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.to(device)
        sar    = images["sar"].to(device)
        sarftp = images["sarftp"].to(device)
        key    = "opt_with_shadow" if args.use_shadow else "opt"
        opt    = images[key].to(device)
        optftp = images["optftp"].to(device)

        sar, sarftp, opt, optftp = augment(sar, sarftp, opt, optftp)

        optimizer.zero_grad()
        if args.model_type == "sar":
            out = model(sar, sarftp)
        elif args.model_type == "opt":
            out = model(opt, optftp)
        else:
            out = model(sar, sarftp, opt, optftp)
        loss = criterion(out.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch} Step {i} Loss {loss.item():.4f}")

    scheduler.step()

    # validation
    model.eval()
    all_out, all_gt = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.to(device)
            sar    = images["sar"].to(device)
            sarftp = images["sarftp"].to(device)
            opt    = images[key].to(device)
            optftp = images["optftp"].to(device)
            if args.model_type == "sar":
                out = model(sar, sarftp)
            elif args.model_type == "opt":
                out = model(opt, optftp)
            else:
                out = model(sar, sarftp, opt, optftp)
            all_out.append(out.squeeze(1).cpu())
            all_gt.append(labels.cpu())

    all_out = torch.sigmoid(torch.cat(all_out))
    all_gt  = torch.cat(all_gt)
    val_loss = criterion(all_out, all_gt.float()).item()
    f1_metrics = compute_imagewise_f1_metrics(all_out.numpy(), all_gt.numpy())
    au_metrics = compute_imagewise_retrieval_metrics(all_out.numpy(), all_gt.numpy())

    epoch_f1 = f1_metrics["best_f1"]
    epoch_au = au_metrics["auroc"]
    thresh   = f1_metrics["best_threshold"]

    preds    = (all_out.numpy() >= thresh).astype(int)
    gts      = all_gt.numpy().astype(int)
    precision = precision_score(gts, preds)
    recall    = recall_score(gts, preds)

    print(f"Epoch {epoch} "
          f"TrainLoss {running_loss/len(train_loader):.4f} "
          f"ValLoss {val_loss:.4f} "
          f"AUROC {epoch_au:.4f} "
          f"F1 {epoch_f1:.4f} "
          f"Prec {precision:.4f} "
          f"Rec {recall:.4f} "
          f"Thr {thresh:.2f}")

    # save if this epoch has better F1
    if epoch_f1 > best_f1:
        best_f1          = epoch_f1
        best_auroc_at_f1 = epoch_au
        best_epoch       = epoch
        ckpt_path        = os.path.join(args.checkpoints, f"checkpoint_ep{epoch:02d}.pth")
        torch.save(model.state_dict(), ckpt_path)

elapsed = time.time() - start
print(f"Best Epoch {best_epoch} F1 {best_f1:.4f} AUROC {best_auroc_at_f1:.4f} Time {elapsed:.1f}s")
