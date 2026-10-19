import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
rfm = pd.read_csv('data/rfm_data.csv', index_col=0)

# Подготовка данных
X = rfm[['Recency', 'Frequency', 'Monetary']].values
y = rfm['CLV_log'].values

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Загрузка модели
import sys
sys.path.append('ml-service')
from app.model import CLVPredictor

checkpoint = torch.load('models/clv_model.pth', map_location='cpu')
model = CLVPredictor(checkpoint['input_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Прогноз на тестовой выборке
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
with torch.no_grad():
    predictions_log = model(X_test_t).numpy().reshape(-1)
    predictions = np.expm1(predictions_log)
    y_test_actual = np.expm1(y_test)

# Создание графика
plt.figure(figsize=(10, 8))

# Диагональная линия (идеальный прогноз)
min_val = min(y_test_actual.min(), predictions.min())
max_val = max(y_test_actual.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальный прогноз')

# Точки (прогноз vs факт)
plt.scatter(y_test_actual, predictions, alpha=0.5, s=20, color='steelblue')

plt.xlabel('Фактическое CLV (фунты)', fontsize=12)
plt.ylabel('Прогнозируемое CLV (фунты)', fontsize=12)
plt.title('Сравнение прогнозируемых и фактических значений CLV', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300)
print("✅ График сохранён как 'prediction_vs_actual.png'")
plt.show()