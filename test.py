import torch
import torch.nn as nn
import numpy as np

print("========== CUDA TEST ==========")
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

print("\n========== DATA TEST ==========")

data = np.load("processed_ukdale.npz")

X = data["X"]

print("X shape:", X.shape)

targets = {
    "fridge": data["Y_fridge"],
    "dishwasher": data["Y_dishwasher"],
    "microwave": data["Y_microwave"]
}

for name, arr in targets.items():
    print(name, arr.shape)

seq_len = X.shape[1]

print("\n========== MODEL TEST ==========")

class InceptionLSTMBlock(nn.Module):

    def __init__(self, seq_len):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, 8,
            kernel_size=1,
            padding="same"
        )

        self.conv5 = nn.Conv1d(
            1, 8,
            kernel_size=5,
            padding="same"
        )

        self.conv9 = nn.Conv1d(
            1, 8,
            kernel_size=9,
            padding="same"
        )

        self.pool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.batch_norm = nn.BatchNorm1d(25)

        self.lstm = nn.LSTM(
            input_size=25,
            hidden_size=30,
            batch_first=True
        )

        self.fc = nn.Linear(
            seq_len * 30,
            1
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        c1 = self.conv1(x)
        c5 = self.conv5(x)
        c9 = self.conv9(x)
        p = self.pool(x)

        x = torch.cat(
            [c1, c5, c9, p],
            dim=1
        )

        x = self.batch_norm(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.relu(
            self.fc(x)
        )

        return x.squeeze()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = InceptionLSTMBlock(seq_len).to(device)

print("Model moved to:", device)

print("\n========== FORWARD PASS TEST ==========")

batch_size = 32

sample = torch.tensor(
    X[:batch_size],
    dtype=torch.float32
).unsqueeze(1).to(device)

with torch.no_grad():
    out = model(sample)

print("Input shape :", sample.shape)
print("Output shape:", out.shape)

if torch.cuda.is_available():
    print(
        "GPU Memory Used:",
        round(torch.cuda.memory_allocated()/1024**3, 2),
        "GB"
    )

print("\nSUCCESS")