"""Tasks B5, B6 & B7: Reusable evaluation pipeline (lesson -> exam -> score).

Usage examples:
    python evaluate.py --model imagenet
    python evaluate.py --model dinov2
    python evaluate.py --model dino_custom --checkpoint path/to/student_a_weights.pth --backbone resnet50
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import timm
from timm.data import resolve_model_data_config, create_transform
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
from tqdm import tqdm

DATA_DIR = "SMART-OM"
RESULTS_DIR = "results"
CLASS_NAMES = ["Normal", "Variation", "OPMD", "Oral Cancer"]
FRACTION_FILES = [
    ("1pct", "splits/train_1pct.csv"),
    ("10pct", "splits/train_10pct.csv"),
    ("25pct", "splits/train_25pct.csv"),
    ("100pct", "splits/train.csv"),
]


class OralDataset(Dataset):
    def __init__(self, csv_path: str, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(DATA_DIR, row["filepath"])).convert("RGB")
        return self.transform(img), int(row["label"])


def load_encoder(model_name: str, checkpoint: str = None, backbone_name: str = "resnet50"):
    if model_name == "imagenet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        encoder = models.resnet50(weights=weights)
        encoder.fc = torch.nn.Identity()
        transform = weights.transforms()
        return encoder, transform

    if model_name == "dinov2":
        encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        cfg = resolve_model_data_config(encoder)
        transform = create_transform(**cfg, is_training=False)
        return encoder, transform

    if model_name == "dino_custom":
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint introuvable pour dino_custom : {checkpoint}")
        encoder = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Support common DINO checkpoint dict layouts
        if isinstance(state, dict):
            for key in ("teacher", "student", "model", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
        print(f"Loaded {checkpoint} | missing={len(missing)}, unexpected={len(unexpected)}")
        cfg = resolve_model_data_config(encoder)
        transform = create_transform(**cfg, is_training=False)
        return encoder, transform

    raise ValueError(f"Modèle inconnu : {model_name}")


@torch.no_grad()
def extract_features(encoder, loader, device: str):
    encoder.eval().to(device)
    feats, labels = [], []
    for imgs, y in tqdm(loader, desc="Extracting features"):
        f = encoder(imgs.to(device)).cpu().numpy()
        feats.append(f)
        labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)


def evaluate_fraction(model_name: str, frac_name: str, X_tr, y_tr, X_te, y_te, test_df: pd.DataFrame):
    clf = make_pipeline(
        Normalizer(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)

    bal_acc = balanced_accuracy_score(y_te, y_pred)
    auroc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
    recalls = recall_score(y_te, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)

    # Save per-image exam predictions (needed by Student A for task A8)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = test_df.copy()
    out["pred_label"] = y_pred
    for c in range(4):
        out[f"prob_class_{c}"] = y_prob[:, c]
    out["max_prob"] = y_prob.max(axis=1)
    out["entropy"] = -(y_prob * np.log(np.clip(y_prob, 1e-12, 1.0))).sum(axis=1)

    preds_path = os.path.join(RESULTS_DIR, f"preds_{model_name}_{frac_name}.csv")
    out.to_csv(preds_path, index=False)

    return {
        "model": model_name,
        "fraction": frac_name,
        "n_train": len(y_tr),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_auroc": round(float(auroc), 4),
        "recall_normal": round(float(recalls[0]), 4),
        "recall_variation": round(float(recalls[1]), 4),
        "recall_opmd": round(float(recalls[2]), 4),
        "recall_cancer": round(float(recalls[3]), 4),
        "preds_file": preds_path,
    }


def upsert_metrics(rows: list[dict], metrics_path: str):
    new_df = pd.DataFrame(rows)
    if os.path.exists(metrics_path):
        old_df = pd.read_csv(metrics_path)
        keys = set(zip(new_df["model"], new_df["fraction"]))
        old_df = old_df[~old_df.apply(lambda r: (r["model"], r["fraction"]) in keys, axis=1)]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(metrics_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an encoder across label fractions.")
    parser.add_argument("--model", required=True, choices=["imagenet", "dinov2", "dino_custom"])
    parser.add_argument("--checkpoint", default=None, help="Path to Student A checkpoint (for dino_custom)")
    parser.add_argument("--backbone", default="resnet50", help="timm backbone name for dino_custom")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Model: {args.model} | Device: {device} ===")

    encoder, transform = load_encoder(args.model, args.checkpoint, args.backbone)

    # Extract features ONCE for full train and test, then slice per fraction
    full_train_ds = OralDataset("splits/train.csv", transform)
    test_ds = OralDataset("splits/test.csv", transform)
    train_loader = DataLoader(full_train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    X_train_all, y_train_all = extract_features(encoder, train_loader, device)
    X_test, y_test = extract_features(encoder, test_loader, device)

    # Map filename -> index in full train array
    fname_to_idx = {f: i for i, f in enumerate(full_train_ds.df["filename"])}

    rows = []
    for frac_name, csv_path in FRACTION_FILES:
        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} introuvable (lance d'abord make_fractions.py)")
            continue
        sub_df = pd.read_csv(csv_path)
        idx = [fname_to_idx[f] for f in sub_df["filename"]]
        row = evaluate_fraction(
            args.model, frac_name,
            X_train_all[idx], y_train_all[idx],
            X_test, y_test,
            test_ds.df,
        )
        rows.append(row)
        print(
            f"[{args.model} | {frac_name:6s} | n={row['n_train']:4d}] "
            f"BalAcc={row['balanced_accuracy']:.4f} | AUROC={row['macro_auroc']:.4f} | "
            f"Recalls (N/V/O/C)={row['recall_normal']:.2f}/{row['recall_variation']:.2f}/"
            f"{row['recall_opmd']:.2f}/{row['recall_cancer']:.2f}"
        )

    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    upsert_metrics(rows, metrics_path)
    print(f"\nRésultats enregistrés dans {metrics_path}")


if __name__ == "__main__":
    main()
