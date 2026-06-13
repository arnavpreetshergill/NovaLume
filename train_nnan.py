import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

BATCH_SIZE = 256
EPOCHS = 50

THRESHOLDS = {
    "fridge": 15,
    "dishwasher": 10,
    "microwave": 50
}

PENALTIES = {
    "fridge": 20.0,
    "dishwasher": 20.0,
    "microwave": 8.0  
}

MSE_WEIGHTS = {
    "fridge": 2.0,
    "dishwasher": 2.0,
    "microwave": 15.0 
}

# ==========================================
# 1. Architecture Components
# ==========================================

class InceptionLSTMBlock(nn.Module):
    def __init__(self, seq_len):
        super(InceptionLSTMBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1, padding='same')
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding='same')
        self.conv_dilated = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, dilation=2, padding='same')
        
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm1d(25)
        
        self.lstm = nn.LSTM(input_size=25, hidden_size=30, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc = nn.Linear(60, 1)
        nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        c1 = self.conv1(x)
        c5 = self.conv5(x)
        cd = self.conv_dilated(x)
        p = self.pool(x)
        
        concat = torch.cat([c1, c5, cd, p], dim=1)
        concat = self.batch_norm(concat)
        
        lstm_in = concat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        
        last_hidden = lstm_out[:, -1, :] 
        out = F.softplus(self.fc(self.dropout(last_hidden)))
        return out.squeeze()

class HybridAsymmetricLoss(nn.Module):
    def __init__(self, on_penalty=5.0, mse_weight=5.0):
        super(HybridAsymmetricLoss, self).__init__()
        self.on_penalty = on_penalty
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        base_l1 = torch.abs(pred - actual)
        
        weight = torch.where(actual > 0.05, self.on_penalty, 1.0)
        asym_l1_loss = torch.mean(weight * base_l1)
        
        mse_loss = self.mse(pred, actual)
        
        return asym_l1_loss + (self.mse_weight * mse_loss)

# ==========================================
# 2. Evaluation Metrics
# ==========================================

def calc_da(pred, actual):
    num = torch.sum(torch.abs(pred - actual))
    den = 2 * torch.sum(actual)
    return (1 - (num / (den + 1e-8))).item() * 100

def calc_nrmse(pred, actual):
    rmse = torch.sqrt(torch.mean((pred - actual)**2))
    norm = actual.max() - actual.min()
    return (rmse / (norm + 1e-8)).item()

def calc_f1_macro(pred, actual, threshold):
    p_bool = (pred > threshold).cpu().numpy()
    a_bool = (actual > threshold).cpu().numpy()
    return f1_score(a_bool, p_bool, average='macro', zero_division=0) * 100

# ==========================================
# 3. Training Logic
# ==========================================

def train_subnetwork(model, X_train, Y_train, on_penalty, mse_weight, epochs=EPOCHS, lr=1e-3):
    device = next(model.parameters()).device
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    criterion = HybridAsymmetricLoss(on_penalty=on_penalty, mse_weight=mse_weight) 
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    loss_history = []
    total_batches = len(loader)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch+1}/{epochs}] Starting - LR: {current_lr:.6f}")

        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 5000 == 0 or (batch_idx + 1) == total_batches:
                avg_interval_loss = running_loss / (batch_idx + 1)
                percent_complete = 100. * (batch_idx + 1) / total_batches
                print(f"   -> Batch {batch_idx+1:>6}/{total_batches} ({percent_complete:>5.1f}%) | "
                      f"Running Loss: {avg_interval_loss:.6f}")

        avg_loss = running_loss / total_batches
        print(f"[Epoch {epoch+1}/{epochs}] Complete | Final Avg Loss {avg_loss:.6f}")
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

    return loss_history

def build_and_train_nnan(X_dict, targets, scales):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    models, histories, predictions, actuals_raw = {}, {}, {}, {}

    for idx, (app_name, Y_app) in enumerate(targets.items()):
        print(f"\n==============================================")
        print(f"--- Training Stage {idx+1}: {app_name.upper()} ---")
        print(f"==============================================")
        
        X_base = torch.tensor(X_dict[app_name], dtype=torch.float32).unsqueeze(1)
        Y_tensor = torch.tensor(Y_app, dtype=torch.float32)
        seq_len = X_base.shape[2]
        
        app_penalty = PENALTIES.get(app_name, 5.0)
        app_mse_weight = MSE_WEIGHTS.get(app_name, 5.0)
        print(f"Sequence Length: {seq_len} | Penalty: {app_penalty}x | MSE Weight: {app_mse_weight}")
        
        model = InceptionLSTMBlock(seq_len).to(device)
        hist = train_subnetwork(model, X_base, Y_tensor, on_penalty=app_penalty, mse_weight=app_mse_weight, epochs=EPOCHS)

        print(f"\nGenerating final predictions for {app_name}...")
        preds = []
        model.eval()

        with torch.no_grad():
            for i in range(0, len(X_base), 1024):
                batch = X_base[i:i+1024].to(device)
                pred = model(batch)
                pred = torch.clamp(pred, min=0)
                preds.append(pred.cpu())

        Y_pred = torch.cat(preds)
        app_max = scales[app_name]

        Y_pred_raw = Y_pred * app_max
        Y_tensor_raw = Y_tensor * app_max
        app_thresh = THRESHOLDS.get(app_name, 10)

        # Baseline clamp only. Output Calibration has been deleted.
        Y_pred_clean = torch.where(Y_pred_raw < app_thresh, torch.tensor(0.0, dtype=torch.float32), Y_pred_raw)

        da = calc_da(Y_pred_clean, Y_tensor_raw)
        nrmse = calc_nrmse(Y_pred_clean, Y_tensor_raw)
        f1 = calc_f1_macro(Y_pred_clean, Y_tensor_raw, threshold=app_thresh)

        print(f"--- {app_name.upper()} RESULTS ---")
        print(f"DA: {da:.2f}% | F1: {f1:.2f}% | NRMSE: {nrmse:.4f}")

        models[app_name] = model
        histories[app_name] = hist
        predictions[app_name] = Y_pred_clean.numpy()
        actuals_raw[app_name] = Y_tensor_raw.numpy()

        del X_base
        torch.cuda.empty_cache()

    return models, histories, predictions, actuals_raw

# ==========================================
# 4. Smart Visualization
# ==========================================

def plot_results(histories, predictions, actuals, time_steps=1000):
    fig, axs = plt.subplots(len(predictions), 2, figsize=(15, 4 * len(predictions)))
    
    for idx, app_name in enumerate(predictions.keys()):
        axs[idx, 0].plot(histories[app_name], color='blue')
        axs[idx, 0].set_title(f'{app_name.capitalize()} Training Loss')
        axs[idx, 0].set_ylabel('Loss')
        axs[idx, 0].set_xlabel('Epoch')
        axs[idx, 0].grid(True)
        
        threshold = THRESHOLDS.get(app_name, 10)
        app_actual = actuals[app_name]
        
        on_indices = np.where(app_actual > threshold)[0]
        start_idx = 0
        
        if len(on_indices) > 0:
            start_idx = max(0, on_indices[0] - 200)
            
        end_idx = start_idx + time_steps
        
        pred = predictions[app_name][start_idx:end_idx]
        actual = app_actual[start_idx:end_idx]
        
        axs[idx, 1].plot(actual, label='Actual Values', color='green', alpha=0.5, linewidth=2)
        axs[idx, 1].plot(pred, label='Pred', color='blue', linewidth=1)
        axs[idx, 1].set_title(f'{app_name.capitalize()} Disaggregation (Steps {start_idx:,} to {end_idx:,})')
        axs[idx, 1].set_ylabel('Power (W)')
        axs[idx, 1].set_xlabel('Time Steps')
        axs[idx, 1].legend()
        axs[idx, 1].grid(True)

    plt.tight_layout()
    plt.savefig('nnan_performance.png')
    print("\nVisualizations saved to nnan_performance.png")

if __name__ == "__main__":
    data = np.load("processed_ukdale.npz", allow_pickle=True)
    scales = data["scales"][0]

    X_dict = {
        "fridge": data["X_fridge"],
        "dishwasher": data["X_dishwasher"],
        "microwave": data["X_microwave"]
    }
    
    targets = {
        "fridge": data["Y_fridge"],
        "dishwasher": data["Y_dishwasher"],
        "microwave": data["Y_microwave"]
    }

    models, histories, predictions, actuals_raw = build_and_train_nnan(
        X_dict, targets, scales
    )

    plot_results(histories, predictions, actuals_raw)