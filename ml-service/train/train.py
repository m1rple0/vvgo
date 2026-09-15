import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.data_processing import load_data, compute_rfm, compute_clv, get_segment_thresholds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Online Retail.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_OUTPUT = os.path.join(BASE_DIR, 'data', 'rfm_data.csv')

os.makedirs(MODELS_DIR, exist_ok=True)


def prepare_data():
    df = load_data(DATA_PATH)
    rfm = compute_rfm(df)
    rfm = compute_clv(rfm)
    thresholds = get_segment_thresholds(rfm)
    rfm.to_csv(DATA_OUTPUT)
    return rfm, thresholds


def train_model(rfm):
    feature_columns = ['Recency', 'Frequency', 'Monetary']
    X = rfm[feature_columns].values
    y = rfm['CLV_log'].values

    with open(os.path.join(MODELS_DIR, 'feature_order.json'), 'w') as f:
        json.dump(feature_columns, f, indent=4)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    class CLVPredictor(nn.Module):
        def __init__(self, input_dim=3):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.network(x)

    model = CLVPredictor(input_dim=3)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    epochs = 100
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': 3,
        'train_losses': train_losses
    }, os.path.join(MODELS_DIR, 'clv_model.pth'))

    model.eval()
    with torch.no_grad():
        predictions_log = model(X_test_t).numpy().reshape(-1)
        predictions = np.expm1(predictions_log)
        y_test_actual = np.expm1(y_test_t.numpy().reshape(-1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)
    y_safe = np.where(y_test_actual == 0, 1e-10, y_test_actual)
    mape = np.mean(np.abs((y_test_actual - predictions) / y_safe)) * 100

    metrics = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'MAPE': float(mape)
    }

    print(f"\nМетрики на тестовой выборке:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return model, metrics, train_losses


def main():
    rfm, thresholds = prepare_data()
    model, metrics, train_losses = train_model(rfm)

    with open(os.path.join(MODELS_DIR, 'segment_thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=4)

    print(f"\nВсе файлы сохранены:")
    print(f"  - Модель: {os.path.join(MODELS_DIR, 'clv_model.pth')}")
    print(f"  - Scaler: {os.path.join(MODELS_DIR, 'scaler.pkl')}")
    print(f"  - Метрики: {os.path.join(MODELS_DIR, 'metrics.json')}")
    print(f"  - Порядок признаков: {os.path.join(MODELS_DIR, 'feature_order.json')}")
    print(f"  - Пороги сегментов: {os.path.join(MODELS_DIR, 'segment_thresholds.json')}")


if __name__ == "__main__":
    main()