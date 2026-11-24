import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report

DATA_DIR = "SMART-OM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Dataset simple à partir des CSV de B2
class OralDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(DATA_DIR, row["filepath"])).convert("RGB")
        return self.transform(img), int(row["label"])

# 2. Prétraitement ImageNet standard
weights = models.ResNet50_Weights.IMAGENET1K_V2
transform = weights.transforms()  # Resize + CenterCrop(224) + Normalize ImageNet
train_ds = OralDataset("splits/train.csv", transform)
test_ds = OralDataset("splits/test.csv", transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

# 3. Encodeur ImageNet gelé (sans la tête de classification)
backbone = models.resnet50(weights=weights)
backbone.fc = torch.nn.Identity()
backbone.eval().to(DEVICE)
@torch.no_grad()
def extract_features(loader):
    feats, labels = [], []
    for imgs, y in loader:
        f = backbone(imgs.to(DEVICE)).cpu().numpy()
        feats.append(f)
        labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)
X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# 4. Linear probe avec gestion du déséquilibre
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
)
clf.fit(X_train, y_train)

# 5. Évaluation sur l'exam (test set)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
bal_acc = balanced_accuracy_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
print(f"Balanced Accuracy : {bal_acc:.4f}")
print(f"Macro AUROC (OvR) : {auroc:.4f}")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Variation", "OPMD", "Oral Cancer"],
))

# 6. Sauvegarder les prédictions (utile dès maintenant et pour B5/A8)
os.makedirs("results", exist_ok=True)
out = test_ds.df.copy()
out["pred_label"] = y_pred
for c in range(4):
    out[f"prob_class_{c}"] = y_prob[:, c]
out.to_csv("results/b3_imagenet_preds.csv", index=False)
