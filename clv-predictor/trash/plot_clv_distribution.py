import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка стиля для отчёта
plt.style.use('seaborn-v0_8-whitegrid')

# Загрузка данных
rfm = pd.read_csv('data/rfm_data.csv', index_col=0)

# Создание рисунка
fig, ax = plt.subplots(figsize=(10, 6))

# Гистограмма распределения CLV
n, bins, patches = ax.hist(
    rfm['CLV'], 
    bins=50, 
    color='steelblue', 
    edgecolor='black', 
    alpha=0.7,
    linewidth=0.8
)

# Оформление
ax.set_xlabel('CLV (фунты стерлингов)', fontsize=13, fontweight='bold')
ax.set_ylabel('Количество клиентов', fontsize=13, fontweight='bold')
ax.set_title('Распределение CLV в датасете Online Retail II', fontsize=15, fontweight='bold', pad=15)

# Сетка
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Добавление статистики
stats_text = f'Всего клиентов: {len(rfm):,}\n'
stats_text += f'Среднее CLV: {rfm["CLV"].mean():.2f} £\n'
stats_text += f'Медиана CLV: {rfm["CLV"].median():.2f} £\n'
stats_text += f'Максимальное CLV: {rfm["CLV"].max():.2f} £'

ax.text(
    0.95, 0.95, stats_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
)

# Подписи на осях
ax.tick_params(axis='both', labelsize=11)

# Компактное отображение
plt.tight_layout()

# Сохранение в высоком качестве
plt.savefig('clv_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Рисунок сохранён как 'clv_distribution.png'")
plt.show()
