# Fine-tuning MomentFM on LSST dataset for time series classification
# We load the pretrained MOMENT model, replace its head with a small MLP,
# and unfreeze the last 4 encoder blocks so the model can adapt to LSST.
# Preprocessing is the same as in generalization.py so results are comparable.
#ran for around 8hours
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from momentfm import MOMENTPipeline

warnings.filterwarnings("ignore")

# ------- config -------
EPOCHS        = 200
WARMUP_EPOCHS = 15
PATIENCE      = 20
N_UNFREEZE    = 4
HEAD_LR       = 3e-4
BACKBONE_LR   = 5e-5
BATCH_SIZE    = 32
WEIGHT_DECAY  = 1e-3
HEAD_DROPOUT  = 0.5
GRAD_CLIP     = 1.0
MIN_DELTA     = 1e-4
NORM_MODE     = "global"
VAL_SIZE      = 0.2
D_MODEL       = 1024    # MOMENT-large hidden size
HEAD_HIDDEN   = 256
PATCH_SIZE    = 8       # MOMENT default, sequence length must be a multiple of this
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
import os
os.makedirs("results", exist_ok=True)
RESULTS_PATH  = "results/moment_finetune_results_v2.csv"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ------- weighted logloss (same function as in generalization.py) -------
def weighted_multi_logloss(true_classes, predictions):
    if not isinstance(true_classes, pd.Series):
        true_classes = pd.Series(true_classes)
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    object_loglosses = pd.Series(1e10 * np.ones(len(true_classes)), index=true_classes.index)
    sum_class_weights = 0.0

    for class_name in np.unique(true_classes):
        class_mask  = true_classes == class_name
        class_count = int(np.sum(class_mask))
        class_object_weights = np.ones(class_count)
        class_weight = 1

        if class_name not in predictions.columns:
            raise ValueError(f"No predictions for class {class_name}")

        class_predictions = np.clip(predictions[class_name][class_mask].astype(float), 1e-15, 1.0)
        class_loglosses   = (
            -class_weight * class_object_weights * np.log(class_predictions)
            / np.sum(class_object_weights)
        )
        object_loglosses[class_mask] = class_loglosses
        sum_class_weights += class_weight

    object_loglosses /= float(sum_class_weights)
    return float(np.sum(object_loglosses))


def compute_metrics(y_true, y_pred, y_proba):
    acc      = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average="macro")
    proba_df = pd.DataFrame(y_proba, columns=list(range(y_proba.shape[1])))
    wll      = weighted_multi_logloss(pd.Series(y_true), proba_df)
    return float(acc), float(f1), float(wll)


# ------- scaler (same as in generalization.py) -------
class TimeSeriesScaler:
    def __init__(self, mode="global"):
        self.mode   = mode
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        N, C, L = X.shape
        if self.mode == "global":
            X_flat = X.reshape(-1, 1)
            return self.scaler.fit_transform(X_flat).reshape(N, C, L)
        elif self.mode == "channel-wise":
            X_r = X.transpose(0, 2, 1).reshape(-1, C)
            return self.scaler.fit_transform(X_r).reshape(N, L, C).transpose(0, 2, 1)
        elif self.mode == "instance-wise":
            means = np.mean(X, axis=2, keepdims=True)
            stds  = np.std(X, axis=2, keepdims=True)
            return (X - means) / (stds + 1e-8)

    def transform(self, X):
        N, C, L = X.shape
        if self.mode == "global":
            X_flat = X.reshape(-1, 1)
            return self.scaler.transform(X_flat).reshape(N, C, L)
        elif self.mode == "channel-wise":
            X_r = X.transpose(0, 2, 1).reshape(-1, C)
            return self.scaler.transform(X_r).reshape(N, L, C).transpose(0, 2, 1)
        elif self.mode == "instance-wise":
            means = np.mean(X, axis=2, keepdims=True)
            stds  = np.std(X, axis=2, keepdims=True)
            return (X - means) / (stds + 1e-8)


# ------- data loading (same split and scaler as generalization.py) -------
def load_data():
    from tslearn.datasets import UCR_UEA_datasets
    ds = UCR_UEA_datasets()
    X_temp, y_temp, X_test, y_test = ds.load_dataset("LSST")

    # channel-first format (N, C, T)
    X_temp = np.transpose(X_temp, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    le     = LabelEncoder()
    y_temp = le.fit_transform(y_temp)
    y_test = le.transform(y_test)
    num_classes = len(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED, stratify=y_temp
    )

    scaler  = TimeSeriesScaler(mode=NORM_MODE)
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes


# ------- padding: MOMENT needs seq_len divisible by patch_size -------
# LSST has T=36 so we pad to 40 (next multiple of 8)
def pad_sequences(X):
    T         = X.shape[2]
    remainder = T % PATCH_SIZE
    if remainder == 0:
        return X, T, T
    pad_len  = PATCH_SIZE - remainder
    X_padded = np.pad(X, ((0,0),(0,0),(0, pad_len)), constant_values=0.0)
    return X_padded, T, T + pad_len

def make_mask(batch_size, original_T, padded_T):
    # 1 = real data, 0 = padding
    mask = torch.zeros(batch_size, padded_T)
    mask[:, :original_T] = 1.0
    return mask.to(DEVICE)


# ------- model: load MOMENT then modify it -------
class MLPHead(nn.Module):
    def __init__(self, num_classes, in_dim=6144):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, num_classes),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, input_mask=None):
        x = x.mean(dim=1)
        return self.net(x)


def build_model(num_classes):
    # load pretrained MOMENT for classification
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name" : "classification",
            "n_channels": 6,            # LSST has 6 channels
            "num_class" : num_classes,
        },
    )
    model.init()

    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # replace the classification head with our MLP
    model.head = MLPHead(num_classes, in_dim=6 * D_MODEL)

    # unfreeze last N_UNFREEZE encoder blocks so the backbone can also adapt
    try:
        # path for MOMENT-large (T5-based encoder)
        blocks = model.encoder.encoder.block
    except AttributeError:
        blocks = model.encoder.block

    for block in blocks[-N_UNFREEZE:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model.to(DEVICE)


# ------- class-weighted loss (same logic as generalization.py) -------
def get_loss(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    weights = counts.sum() / (len(classes) * counts)
    w = torch.ones(int(classes.max()) + 1)
    w[classes] = torch.tensor(weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=w.to(DEVICE))


# ------- get logits from model output (handles different output formats) -------
def get_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "prediction_logits"):
        return output.prediction_logits
    return output if isinstance(output, torch.Tensor) else output[0]


# ------- training -------
def train(model, train_data, val_data, run_label=""):
    X_tr, y_tr = train_data
    X_vl, y_vl = val_data if val_data is not None else (None, None)

    X_tr_pad, orig_T, pad_T = pad_sequences(X_tr)

    criterion    = get_loss(y_tr)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_pad, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    if X_vl is not None:
        X_vl_pad, _, _ = pad_sequences(X_vl)
        X_vl_t  = torch.tensor(X_vl_pad, dtype=torch.float32).to(DEVICE)
        y_vl_t  = torch.tensor(y_vl, dtype=torch.long).to(DEVICE)
        val_mask = make_mask(len(X_vl_pad), orig_T, pad_T)

    best_val_loss = None
    best_state    = None
    best_epoch    = 0
    patience_left = PATIENCE

    # phase 1: only head params; phase 2: head + unfrozen blocks
    head_params = list(model.head.parameters())
    try:
        blocks = model.encoder.encoder.block
    except AttributeError:
        blocks = model.encoder.block
    backbone_params = [p for b in blocks[-N_UNFREEZE:] for p in b.parameters() if p.requires_grad]

    optimizer = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    phase     = "warmup"

    print(f"\nTraining {run_label}  (device={DEVICE})")

    for epoch in range(1, EPOCHS + 1):

        # switch optimizer after warmup
        if epoch == WARMUP_EPOCHS + 1 and phase == "warmup":
            optimizer = optim.AdamW(
                [{"params": head_params,     "lr": HEAD_LR},
                 {"params": backbone_params, "lr": BACKBONE_LR}],
                weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
            patience_left = PATIENCE
            phase = "finetune"
            print(f"  Epoch {epoch}: switching to fine-tune phase")

        model.train()
        total_loss = 0.0

        for batch_X, batch_y in tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False):
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            mask    = make_mask(batch_X.shape[0], orig_T, pad_T)

            optimizer.zero_grad()
            loss = criterion(get_logits(model(x_enc=batch_X, input_mask=mask)), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
            )
            optimizer.step()
            total_loss += loss.item()

        print(f"  Ep {epoch:2d}/{EPOCHS}  train={total_loss/len(train_loader):.4f}  [{phase}]", end="")

        if X_vl is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(get_logits(model(x_enc=X_vl_t, input_mask=val_mask)), y_vl_t).item()
            print(f"  val={val_loss:.4f}")

            if best_val_loss is None or val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                best_epoch    = epoch
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = PATIENCE
                torch.save(best_state, f"results/moment_best_{run_label.replace('->','_')}.pt")
                print(f"    -> best model saved (val_loss={val_loss:.4f})")
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            print()  
        scheduler.step()   

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best weights from epoch {best_epoch}")

    return model


# ------- inference -------
def predict_proba(model, X):
    X_pad, orig_T, pad_T = pad_sequences(X)
    loader = DataLoader(TensorDataset(torch.tensor(X_pad, dtype=torch.float32)),
                        batch_size=BATCH_SIZE, shuffle=False)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for (bx,) in loader:
            bx   = bx.to(DEVICE)
            mask = make_mask(bx.shape[0], orig_T, pad_T)
            probs = torch.softmax(get_logits(model(x_enc=bx, input_mask=mask)), dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def evaluate(model, X, y, split="test"):
    proba  = predict_proba(model, X)
    y_pred = np.argmax(proba, axis=1)
    acc, f1, wll = compute_metrics(y, y_pred, proba)
    print(f"  [{split}]  Acc={acc:.4f}  F1={f1:.4f}  WLogLoss={wll:.4f}")
    return acc, f1, wll


# ------- main -------
def main():
    print("=== MOMENT Fine-Tuning on LSST ===")

    train_data, val_data, test_data, num_classes = load_data()

    results = []

    # run 1: train on train split, pick best checkpoint with val
    model = build_model(num_classes)
    t0    = time.time()
    model = train(model, train_data, val_data, run_label="train->val")
    print(f"Training time: {(time.time()-t0)/60:.1f} min")

    val_acc,  val_f1,  val_wll  = evaluate(model, val_data[0],  val_data[1],  "val")
    test_acc, test_f1, test_wll = evaluate(model, test_data[0], test_data[1], "test")

    tag = f"unfreeze{N_UNFREEZE}blocks_MLP{HEAD_HIDDEN}_headLR{HEAD_LR}_warmup{WARMUP_EPOCHS}ep"

    results.append({
        "Model": "MOMENT_FineTune", "Config": tag,
        "Val_Accuracy": val_acc, "Val_Macro_F1": val_f1, "Val_WLogLoss": val_wll,
        "Test_Accuracy": test_acc, "Test_Macro_F1": test_f1, "Test_Weighted_LogLoss": test_wll,
    })

    # run 2: retrain on train+val, same protocol as generalization.py
    X_all = np.concatenate([train_data[0], val_data[0]])
    y_all = np.concatenate([train_data[1], val_data[1]])

    model2 = build_model(num_classes)
    t0     = time.time()
    model2 = train(model2, (X_all, y_all), val_data= None , run_label="trainval->test")
    print(f"Retraining time: {(time.time()-t0)/60:.1f} min")

    final_acc, final_f1, final_wll = evaluate(model2, test_data[0], test_data[1], "test")

    results.append({
        "Model": "MOMENT_FineTune", "Config": tag + " (retrain train+val)",
        "Val_Accuracy": float("nan"), "Val_Macro_F1": val_f1, "Val_WLogLoss": float("nan"),
        "Test_Accuracy": final_acc, "Test_Macro_F1": final_f1, "Test_Weighted_LogLoss": final_wll,
    })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
