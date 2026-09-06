import torch
import torch.nn as nn
import joblib
import json
import numpy as np
import os
from pathlib import Path

# ===== КУРС ВАЛЮТ =====
# 1 фунт стерлингов ≈ 120 рублей (актуальный курс можно подставить)
GBP_TO_RUB = 120.0

# Пути к артефактам
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / 'models'

class CLVPredictor(nn.Module):
    """Класс нейросетевой модели"""
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

class CLVModel:
    """Класс для загрузки и использования модели"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_order = None
        self.segment_thresholds = None
        self.gbp_to_rub = GBP_TO_RUB
        self.load_artifacts()
    
    def load_artifacts(self):
        """Загрузка всех артефактов модели"""
        print("Загрузка модели...")
        
        # Загрузка модели
        model_path = MODELS_DIR / 'clv_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = CLVPredictor(checkpoint['input_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Загрузка scaler
        scaler_path = MODELS_DIR / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler не найден: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Загрузка порядка признаков
        feature_order_path = MODELS_DIR / 'feature_order.json'
        if feature_order_path.exists():
            with open(feature_order_path, 'r') as f:
                self.feature_order = json.load(f)
        
        # Загрузка порогов сегментов
        thresholds_path = MODELS_DIR / 'segment_thresholds.json'
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                self.segment_thresholds = json.load(f)
        
        print("✅ Модель загружена успешно!")
        print(f"   Курс: 1 £ = {self.gbp_to_rub} ₽")
    
    def predict(self, recency: int, frequency: int, monetary: float):
        """
        Предсказание CLV и сегмента клиента
        
        Args:
            recency: дней с последней покупки
            frequency: количество покупок
            monetary: общая сумма покупок (в рублях)
        
        Returns:
            dict: {
                'clv': float (в рублях),
                'segment': str,
                'clv_log': float,
                'raw_prediction': float
            }
        """
        # Конвертируем рубли в фунты для модели
        monetary_gbp = monetary / self.gbp_to_rub
        
        # Формируем входной вектор (в фунтах)
        input_vector = np.array([[recency, frequency, monetary_gbp]], dtype=np.float32)
        
        # Стандартизация
        input_scaled = self.scaler.transform(input_vector)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        # Инференс (модель считает в фунтах)
        with torch.no_grad():
            clv_log = self.model(input_tensor).item()
            clv_gbp = np.expm1(clv_log)  # CLV в фунтах
        
        # Конвертируем результат в рубли
        clv_rub = clv_gbp * self.gbp_to_rub
        
        # Определение сегмента (по CLV в фунтах, так как модель обучена на фунтах)
        segment = self.get_segment(clv_gbp)
        
        return {
            'clv': float(clv_rub),
            'clv_gbp': float(clv_gbp),
            'clv_log': float(clv_log),
            'segment': segment,
            'raw_prediction': float(clv_log)
        }
    
    def get_segment(self, clv_value):
        """Определение сегмента по значению CLV (в фунтах)"""
        if not self.segment_thresholds:
            return 'Unknown'
        
        for segment_name, info in self.segment_thresholds.items():
            if info['min'] <= clv_value <= info['max']:
                return segment_name
        
        if clv_value >= self.segment_thresholds.get('Champions', {}).get('max', float('inf')):
            return 'Champions'
        else:
            return 'At Risk'
    
    def predict_batch(self, data):
        """Пакетное предсказание для нескольких клиентов"""
        results = []
        for item in data:
            result = self.predict(
                recency=item['recency'],
                frequency=item['frequency'],
                monetary=item['monetary']
            )
            results.append(result)
        return results

# Создаем глобальный экземпляр модели
model_instance = None

def get_model():
    """Получение экземпляра модели (синглтон)"""
    global model_instance
    if model_instance is None:
        model_instance = CLVModel()
    return model_instance