from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.model import get_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    print("Сервис запущен")
    yield

app = FastAPI(title="CLV Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>CLV Predictor</title>
    </head>
    <body>
        <h2>CLV Predictor</h2>
        <p>Прогнозирование ценности клиента</p>
        
        <form id="form">
            <p>
                <b>Когда покупал последний раз</b><br>
                <small>Дней с последней покупки</small><br>
                <input type="number" id="recency" required min="0" style="width:250px;">
            </p>
            
            <p>
                <b>Сколько покупал</b><br>
                <small>Количество покупок за год</small><br>
                <input type="number" id="frequency" required min="1" style="width:250px;">
            </p>
            
            <p>
                <b>На какую сумму</b><br>
                <small>Сумма покупок за год (рубли)</small><br>
                <input type="number" id="monetary" required min="0" step="0.01" style="width:250px;">
            </p>
            
            <button type="submit" style="background:green;color:white;border:none;padding:8px 30px;cursor:pointer;">Рассчитать</button>
        </form>
        
        <div id="result" style="border:1px solid #ccc;padding:15px;margin-top:20px;display:none;width:350px;"></div>
        <div id="error" style="color:red;margin-top:10px;display:none;"></div>
        
        <script>
            document.getElementById('form').onsubmit = async function(e) {
                e.preventDefault();
                
                const recency = parseInt(document.getElementById('recency').value);
                const frequency = parseInt(document.getElementById('frequency').value);
                const monetary = parseFloat(document.getElementById('monetary').value);
                
                const resultDiv = document.getElementById('result');
                const errorDiv = document.getElementById('error');
                resultDiv.style.display = 'none';
                errorDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ 
                            recency, 
                            frequency: frequency * 2, 
                            monetary: monetary * 2 
                        })
                    });
                    
                    if (!response.ok) {
                        const err = await response.json();
                        errorDiv.textContent = 'Ошибка: ' + (err.detail || 'Неизвестная ошибка');
                        errorDiv.style.display = 'block';
                        return;
                    }
                    
                    const data = await response.json();
                    
                    const clv_year = data.clv / 2;
                    
                    const segments = {
                        'Champions': 'Максимально ценный',
                        'Loyal Customers': 'Ценный',
                        'Potential': 'Потенциальный',
                        'Need Attention': 'Требует внимания',
                        'At Risk': 'Не имеет ценности'
                    };
                    
                    resultDiv.innerHTML = `
                        <b>CLV:</b> ${clv_year.toFixed(2)} руб.<br>
                        <b>Сегмент:</b> ${segments[data.segment] || data.segment}
                    `;
                    resultDiv.style.display = 'block';
                    
                } catch (err) {
                    errorDiv.textContent = 'Ошибка подключения к серверу';
                    errorDiv.style.display = 'block';
                }
            };
        </script>
    </body>
    </html>
    """)

@app.get("/health", response_model=HealthResponse)
async def health():
    model = get_model()
    return {"status": "ok", "model_loaded": model.model is not None, "version": "1.0.0"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        model = get_model()
        result = model.predict(
            recency=request.recency,
            frequency=request.frequency,
            monetary=request.monetary
        )
        return PredictResponse(
            clv=result['clv'],
            segment=result['segment'],
            clv_log=result['clv_log'],
            details={}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))