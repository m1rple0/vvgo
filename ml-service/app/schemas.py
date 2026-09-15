from pydantic import BaseModel, Field, validator
from typing import Optional

class PredictRequest(BaseModel):
    #zapros
    recency: int = Field(..., ge=0, description="Дней с последней покупки (0-365)")
    frequency: int = Field(..., ge=1, description="Количество покупок за весь период")
    monetary: float = Field(..., ge=0, description="Общая сумма покупок (в рублях)")
    
    @validator('recency')
    def validate_recency(cls, v):
        if v > 730:
            raise ValueError('Recency не может быть больше 730 дней')
        return v
    
    @validator('frequency')
    def validate_frequency(cls, v):
        if v > 1000:
            raise ValueError('Frequency не может быть больше 1000')
        return v
    
    @validator('monetary')
    def validate_monetary(cls, v):
        if v > 100000000:  # 100 млн рублей
            raise ValueError('Monetary не может быть больше 100 000 000 рублей')
        return v

class PredictResponse(BaseModel):
    #otvet
    clv: float = Field(..., description="Прогнозируемая ценность клиента (CLV) в рублях")
    segment: str = Field(..., description="Сегмент клиента")
    clv_log: float = Field(..., description="Логарифмированное значение CLV")
    details: Optional[dict] = Field(None, description="Дополнительная информация")

class BatchPredictRequest(BaseModel):
    data: list[PredictRequest]

class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str