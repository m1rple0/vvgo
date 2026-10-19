FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY app/ ./app/
COPY train/ ./train/

# Открываем порт
EXPOSE 8000

# Запускаем сервис
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]