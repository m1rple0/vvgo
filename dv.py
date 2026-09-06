import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Загрузка данных
rfm = pd.read_csv('data/rfm_data.csv', index_col=0)

# Функция для определения сегмента (на русском)
def get_segment(clv):
    if clv >= 337:
        return 'Максимально ценный'
    elif clv >= 110:
        return 'Ценный'
    elif clv >= 39:
        return 'Потенциальный'
    elif clv >= 12:
        return 'Требует внимания'
    else:
        return 'Не имеет ценности'

# Добавляем истинные сегменты
rfm['Segment_true'] = rfm['CLV'].apply(get_segment)

# Для демонстрации: добавляем предсказанные сегменты (с небольшим шумом)
np.random.seed(42)
rfm['Segment_pred'] = rfm['Segment_true'].copy()
mask = np.random.random(len(rfm)) < 0.15
rfm.loc[mask, 'Segment_pred'] = np.random.choice(
    ['Максимально ценный', 'Ценный', 'Потенциальный', 'Требует внимания', 'Не имеет ценности'],
    size=mask.sum()
)

# Порядок сегментов (от самого ценного к наименее ценному)
segments = ['Максимально ценный', 'Ценный', 'Потенциальный', 'Требует внимания', 'Не имеет ценности']

# Построение матрицы ошибок
cm = confusion_matrix(rfm['Segment_true'], rfm['Segment_pred'], labels=segments)

# Визуализация
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=segments, yticklabels=segments)
plt.xlabel('Предсказанный сегмент', fontsize=12)
plt.ylabel('Истинный сегмент', fontsize=12)
plt.title('Матрица ошибок сегментации клиентов', fontsize=14)
plt.xticks(rotation=15, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_segments.png', dpi=300)
print("✅ Матрица ошибок сохранена как 'confusion_matrix_segments.png'")
plt.show()