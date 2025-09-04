import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.set_num_threads(os.cpu_count())
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# ---- Parameters ----
SEQ_LEN = 60       # lookback
BATCH_SIZE = 64
HIDDEN = 64
EPOCHS = 10
LR = 1e-4
ROLLING_WINDOW = 44  # rolling window for volatility
training_columns = ["ES=F", "NQ=F", "GC=F"]
training_start = '2015-01-01'
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_model_regression.pth")

# ---- Options ----
TRAIN = False
TEST = True

# ---- Settings flags for transformations ----
normalize_inputs = True   # apply z-score normalization
tanh_inputs = True       # apply tanh after                                                                                  z-score
clip_tanh_eps = 1e-6      # avoid exact -1 or 1 for arctanh

# ---- Dataset ----
class FinancialDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        self.samples = []
        self.seq_len = seq_len
        self.mean_std_dict = {}
        for col in df.columns:
            series = df[col].values.astype(np.float32)
            # ---- Normalization ----
            mean, std = 0.0, 1.0
            if normalize_inputs:
                mean = np.mean(series)
                std = np.std(series) + 1e-8
                series = (series - mean) / std
            if tanh_inputs:
                series = np.tanh(series)
            self.mean_std_dict[col] = (mean, std)

            features = torch.tensor(series, dtype=torch.float32).reshape(-1, 1)
            for i in range(len(series) - seq_len):
                x = features[i:i+seq_len]
                y = features[i+seq_len]
                if not np.isnan(y):
                    self.samples.append((x, y, col))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# ---- Transformer Model ----
class TransformerPredictor(nn.Module):
    def __init__(self, n_features, hidden_dim=HIDDEN, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(n_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(hidden_dim, 1)  # regression output
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # last token
        return self.fc_out(x)

# ---- Training ----
def train_model(model, train_loader, test_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Test loss
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for x, y, _ in test_loader:
                logits = model(x)
                loss = criterion(logits.squeeze(), y)
                total_test_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.6f}, "
              f"Test Loss: {total_test_loss/len(test_loader):.6f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ---- Predict ----
def predict(model, seq_tensor, mean=0.0, std=1.0):
    model.eval()
    with torch.no_grad():
        logits = model(seq_tensor)
        pred = logits.squeeze().cpu().numpy()
        # ---- Inverse tanh if applied ----
        if tanh_inputs:
            pred = np.clip(pred, -1+clip_tanh_eps, 1-clip_tanh_eps)
            pred = np.arctanh(pred)
        # ---- Denormalize if applied ----
        if normalize_inputs:
            pred = pred * std + mean
        return float(pred)

# ---- Data ----
def get_data(training_columns, training_start):
    import Market_Data_Feed as mdf
    from config import settings
    settings = settings.get_settings()
    data_ind = mdf.Data_Ind_Feed(settings).data_ind
    data, _ = data_ind
    tickers_returns = data.tickers_returns
    df = tickers_returns[training_columns][training_start:]
    # Uncomment for weekly data if needed
    # df = df.resample('W-FRI').agg(lambda x: (x + 1.0).prod() - 1.0)
    return df

# ---- Main ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = get_data(training_columns, training_start)
    print("tickers_returns", df.head())

    if TRAIN:
        dataset = FinancialDataset(df)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        num_workers = min(8, os.cpu_count())  # 4 is safe starting point
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,pin_memory=False)
        model = TransformerPredictor(n_features=1).to(device)
        train_model(model, train_loader, test_loader)

    # ---- TEST BLOCK ----
    if TEST:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        loaded_model = TransformerPredictor(n_features=1).to(device)
        loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        loaded_model.eval()
        print("Loaded model ready for inference.")

        # Prepare dataset for mean/std info
        dataset = FinancialDataset(df)
        mean_std_dict = dataset.mean_std_dict

        seq_len = SEQ_LEN
        n_assets = len(df.columns)
        n_samples = len(df) - seq_len

        # Initialize arrays
        predicted_array = np.zeros((n_samples, n_assets), dtype=np.float32)
        actual_array = np.zeros((n_samples, n_assets), dtype=np.float32)

        series_matrix = df.values.astype(np.float32)
        dates = df.index[seq_len:]

        for col_idx, col in enumerate(df.columns):
            series = series_matrix[:, col_idx]
            mean, std = mean_std_dict[col]

            # Build sliding windows
            seq_matrix = np.lib.stride_tricks.sliding_window_view(series, window_shape=seq_len)[:n_samples]

            # Apply normalization/tanh if needed
            if normalize_inputs:
                seq_matrix = (seq_matrix - mean) / (std + 1e-8)
            if tanh_inputs:
                seq_matrix = np.tanh(seq_matrix)

            # Batch predict
            batch_tensor = torch.tensor(seq_matrix, dtype=torch.float32).unsqueeze(-1).to(device)
            # shape will be (n_samples, seq_len, 1)
            with torch.no_grad():
                preds = loaded_model(batch_tensor).squeeze().cpu().numpy()

            # Inverse transformations
            if tanh_inputs:
                preds = np.clip(preds, -1 + clip_tanh_eps, 1 - clip_tanh_eps)
                preds = np.arctanh(preds)
            if normalize_inputs:
                preds = preds * std + mean

            predicted_array[:, col_idx] = preds
            actual_array[:, col_idx] = series[seq_len:]

        # Create DataFrames
        predicted_df = pd.DataFrame(predicted_array, columns=df.columns, index=dates)
        actual_df = pd.DataFrame(actual_array, columns=df.columns, index=dates)

        print("Normalized / transformed dataset preview:")
        print(predicted_df.head())

        # ---- Compute metrics ----
        rmse = np.sqrt(((predicted_array - actual_array) ** 2).mean(axis=0))
        corr = np.array([np.corrcoef(predicted_array[:, i], actual_array[:, i])[0, 1] for i in range(n_assets)])

        metrics_df = pd.DataFrame({"RMSE": rmse, "Corr": corr}, index=df.columns)
        print("Metrics:")
        print(metrics_df)

        # ---- Compute weights ----
        #pred_min = predicted_array.min(axis=0)
        #pred_max = predicted_array.max(axis=0)
        #weights_array = predicted_array.copy()
        #pred_scaled = (predicted_array - pred_min) / (pred_max - pred_min + 1e-8)# values 0 to 1
        #pred_scaled = pred_scaled**2 #Smaller low values
        #pred_scaled = pred_scaled*4 # now values 0 to k
        #weights_array = np.clip(pred_scaled, 0.0, 1.0) # clipped  values 0 to 1
        #weights_array *=1.5
        #equivolat = np.std(actual_df.values, axis=0) / np.std(actual_df.values * weights_array, axis=0)
        #equivolat = (.12/16) / np.std(actual_df.values * weights_array, axis=0)
        #weights_array =weights_array*equivolat

        #predicted_df = pd.DataFrame(predicted_array, columns=df.columns, index=dates)
        predicted_std = predicted_df.std(axis=0)  # pandas Series, aligned to columns
        predicted_mean = predicted_df.mean(axis=0)
        predicted_norm= (predicted_df-predicted_mean) / predicted_std

        weights_df = predicted_norm.clip(upper=1.5,lower=-0)
        #weights_df = (weights_df+1)/2
        #weights_df =weights_df*1.5
        #weights_df =weights_df.rolling(22).mean()

        #print("equivolat ratio",equivolat)
        print("weights_df",weights_df.tail(10))

        # ---- Strategy returns ----
        strategy_returns = actual_df * weights_df
        cum_strategy = (1 + strategy_returns).cumprod() - 1
        cum_actual = (1 + actual_df).cumprod() - 1

        # ---- Plots ----
        #weights_df.plot(title="Predicted Weights per Asset", figsize=(12, 6))



        for col in df.columns:
                plt.figure(figsize=(12,6))
                plt.plot(cum_strategy[col], label='Strategy (weighted)')
                plt.plot(cum_actual[col], label='Equal-weighted actual')
                plt.plot(weights_df[col], label='Weights')
                plt.title(col + " Accumulated Returns: Strategy vs Actual")
                plt.xlabel("Date")
                plt.ylabel("Accumulated Return")
                plt.legend()
                plt.grid(True)

    plt.show()


