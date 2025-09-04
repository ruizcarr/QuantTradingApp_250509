import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os

# ---- Parameters ----
SEQ_LEN = 55     # months lookback
BATCH_SIZE = 32
HIDDEN = 64
EPOCHS = 10 #10
LR = 1e-4
ROLLING_WINDOW = 60  # rolling window for volatility


#More parameterers
training_columns = ["ES=F", "NQ=F", "GC=F", "CL=F"]
#training_columns = ["ES=F", "NQ=F"]
training_start = '2000-01-01'
n_classes = 3
# label_map = {0: "Strong Down", 1: "Down", 2: "Almost Zero", 3: "Up", 4: "Strong Up"}
label_map = {0: "Down", 1: "Almost Zero", 2: "Up"}

training = True
test = True

# ---- Model folder ----
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)  # create folder if not exists
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_model.pth")

# ---- Volatility-based labeling ----
def categorize_vol_scaled_5labels(series, window=ROLLING_WINDOW):
    vol = series.rolling(window).std()
    labels = []
    for r, s in zip(series, vol):
        if np.isnan(s) or s == 0:
            labels.append(2)  # default Almost Zero
            continue
        if r < -1.5 * s:
            labels.append(0)  # Strong Down
        elif r < -0.5 * s:
            labels.append(1)  # Down
        elif r <= 0.5 * s:
            labels.append(2)  # Almost Zero
        elif r <= 1.5 * s:
            labels.append(3)  # Up
        else:
            labels.append(4)  # Strong Up
    return labels

def categorize_vol_scaled(series, window=55):
    """
    Label returns into 3 classes using volatility-scaled thresholds.
    0 = Down, 1 = Neutral, 2 = Up
    """
    # std threshold
    upper_std_mult=0.5
    lower_std_mult=-0.2

    vol = series.rolling(window).std()
    labels = []
    for r, s in zip(series, vol):
        if np.isnan(s) or s == 0:
            labels.append(1)  # default Neutral
            continue
        if r < -lower_std_mult * s:
            labels.append(0)  # Down
        elif r > upper_std_mult * s:
            labels.append(2)  # Up
        else:
            labels.append(1)  # Neutral
    return labels

def categorize_quantiles(series, n_classes=3):
    # compute quantile edges per asset
    quantiles = np.quantile(series, np.linspace(0, 1, n_classes+1))
    # np.digitize returns 1..n_classes, subtract 1 to get 0..n_classes-1
    labels = np.digitize(series, quantiles[1:-1], right=True)
    # safety: clip any possible overflow
    labels = np.clip(labels, 0, n_classes-1)
    return labels.astype(int)

# ---- Dataset ----
class FinancialDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN, window=ROLLING_WINDOW):
        self.samples = []
        self.seq_len = seq_len

        for col in df.columns:
            #labels = categorize_vol_scaled(df[col])
            labels = categorize_quantiles(df[col])
            features = torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(-1)  # single asset
            labels = torch.tensor(labels, dtype=torch.long)
            for i in range(len(df) - seq_len):
                x = features[i:i+seq_len]  # shape [seq_len, 1]
                y = labels[i+seq_len]
                if not torch.isnan(y):
                    self.samples.append((x, y, col))  # keep asset name
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ---- Transformer Model ----
class TransformerPredictor(nn.Module):
    def __init__(self, n_features, hidden_dim=HIDDEN, n_heads=4, n_layers=2, n_classes=3):
        super().__init__()
        self.embedding = nn.Linear(n_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True,
            dropout=0.2# 20% dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # take last token
        return self.fc_out(x)

# ---- Training & Evaluation ----
def train_and_evaluate(model, train_loader, test_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        for x, y, _ in train_loader:
            # 🟢 new: unpack asset column
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        acc = correct / total

        # Evaluate on test set
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for x, y, _ in test_loader:
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct_test += (preds == y).sum().item()
                total_test += y.size(0)
        test_acc = correct_test / total_test

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}, "
              f"Train Acc: {acc:.4f}, Test Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ---- Inference ----
def predict(model, seq_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(seq_tensor.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
    return pred_class, probs

def get_data(training_columns, training_start):

    import Market_Data_Feed as mdf

    # Get SETTINGS
    from config import settings, utils
    settings = settings.get_settings()  # Edit Settings Dict at file config/settings.py

    # DATA & INDICATORS
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, indicators_dict = data_ind
    tickers_returns = data.tickers_returns

    # data filtered
    df = tickers_returns[training_columns][training_start:]

    # Monthly Data: Assuming df has daily pct changes
    #df = df.resample('M').agg(lambda x: (x + 1.0).prod() - 1.0)

    # Weekly Data: Assuming df has daily pct changes
    df = df.resample('W-FRI').agg(lambda x: (x + 1.0).prod() - 1.0)

    return df

# ---- Example Usage ----
if __name__ == "__main__":

    #Get data

    df = get_data(training_columns, training_start)

    print("tickers_returns",df)
    for col in df.columns:
        #labels = categorize_vol_scaled(df[col])
        labels = categorize_quantiles(df[col])
        print(col, np.bincount(labels))



    if training:

        dataset = FinancialDataset(df)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # ---- Train and save ----
        model = TransformerPredictor(n_features=1)
        train_and_evaluate(model, train_loader, test_loader)

    if test:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load model from sub-folder ----
        loaded_model = TransformerPredictor(n_features=1)
        loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        loaded_model.to(device)
        loaded_model.eval()
        print("Loaded model ready for inference.")

        # ---- Rolling prediction metrics for last 30 days per asset ----

        rolling_days = SEQ_LEN

        asset_metrics = {}

        for col in df.columns:
            #actual_labels = categorize_vol_scaled(df[col])
            actual_labels = categorize_quantiles(df[col])
            preds = []

            for i in range(-rolling_days, 0):
                seq_array = df[col].values[i - SEQ_LEN:i]

                # 🟢 Fix: ensure correct shape (1, SEQ_LEN, 1)
                seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                pred_class, _ = predict(loaded_model, seq_tensor)  # 🟢 integer label
                preds.append(pred_class)

            print(col, preds)

            # Exact-match accuracy
            exact_acc = sum([p == a for p, a in zip(preds, actual_labels)]) / rolling_days

            # Soft correctness (squared difference)
            correctness_list = [1 - ((p - a) ** 2) / ((n_classes - 1) ** 2) for p, a in zip(preds, actual_labels)]
            soft_correctness = np.mean(correctness_list)

            asset_metrics[col] = {"Exact Accuracy": exact_acc, "Soft Correctness": soft_correctness}

        print(f"Metrics over last {rolling_days} days per asset:")
        for col, metrics in asset_metrics.items():
            print(f"{col}: Exact Acc = {metrics['Exact Accuracy']:.3f}, Soft Correctness = {metrics['Soft Correctness']:.3f}")