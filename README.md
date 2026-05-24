# Airsoft Multimodal API

Система multi-head классификации страйкбольного снаряжения по тексту и фотографиям объявлений (category + subcategory одновременно).

## Структура проекта

```
airsoft_api/
├── app/
│   ├── ml/
│   │   ├── multimodal_model.py   # Единая multi-head модель
│   │   ├── trainer.py            # Обучение joint классификатора
│   │   └── inference.py          # Инференс
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   └── db/
│       └── crud.py              # Операции с БД
├── data/
│   └── raw/
│       ├── posts.parquet
│       ├── photos.parquet
│       └── subcategory_images/  # Папки с именами классов
├── database/
│   └── users.db                 # SQLite база пользователей
├── model/
│   ├── category_model/
│   │   └── classifier.pkl
│   └── subcategory_model/
│       └── classifier.pkl
├── config.py
├── main.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Запуск

```bash
# Первый запуск (обучение + старт API)
docker-compose up --build

# Последующие запуски (только загрузка модели)
docker-compose up
```

## API Endpoints

- `POST /auth/register` — регистрация пользователя
- `POST /auth/login` — авторизация, получение JWT токена
- `POST /predict` — предсказание категории/подкатегории (требует JWT)

## Пример запроса к /predict

```json
{
  "post_id": "12345",
  "text": "Продаю страйкбольную винтовку AK и тактический жилет.",
  "photos": [
    {"photo_id": "1", "url": "https://example.com/photo1.jpg"}
  ]
}
```
