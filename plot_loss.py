import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

# Загрузка сохранённых потерь из модели
checkpoint = torch.load('models/clv_model.pth', map_location='cpu')
train_losses = checkpoint['train_losses']

# Создание графика
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Значение функции ошибки (Loss)', fontsize=12)
plt.title('Изменение функции ошибки в процессе обучения', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)
print("✅ График сохранён как 'loss_plot.png'")
plt.show()