FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости Python
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

# Создаём директории для данных, модели и БД
RUN mkdir -p database model/joint_model data/raw/subcategory_images

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
