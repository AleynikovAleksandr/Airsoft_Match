# Airsoft Multimodal API

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.29.0-darkgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)
![TorchVision](https://img.shields.io/badge/TorchVision-0.17.2-red)
![Transformers](https://img.shields.io/badge/Transformers-4.41.1-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-f7931e)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.30-red)
![SQLite](https://img.shields.io/badge/SQLite-aiosqlite--0.20.0-blue)
![Pydantic](https://img.shields.io/badge/Pydantic-2.7.1-e92063)
![HTTPX](https://img.shields.io/badge/HTTPX-0.27.0-purple)
![Requests](https://img.shields.io/badge/Requests-2.31.0-lightgrey)
![Pillow](https://img.shields.io/badge/Pillow-10.3.0-yellow)
![PyArrow](https://img.shields.io/badge/PyArrow-16.1.0-brightgreen)
![JOSE](https://img.shields.io/badge/python--jose-3.3.0-black)
![Passlib](https://img.shields.io/badge/Passlib-bcrypt--1.7.4-grey)
![Multipart](https://img.shields.io/badge/python--multipart-0.0.9-blueviolet)
![Joblib](https://img.shields.io/badge/Joblib-1.4.0-teal)
![StopWords](https://img.shields.io/badge/stop--words-2018.7.23-lightgrey)
![ruBERT](https://img.shields.io/badge/ruBERT-ai--forever%2FruBert--base-gold)
![Vision_Transformer](https://img.shields.io/badge/ViT-google%2Fvit--base--patch16--224-gold)


## Current Features

### Multimodal AI Processing
- Automatic preprocessing pipeline  
- Text and image processing  
- Transformer-based embeddings  
- Multimodal fusion architecture  
- Semantic embedding generation  

### API & Backend
- FastAPI REST API  
- JWT-based authentication with secure endpoints  
- Swagger / OpenAPI documentation  
- Fully asynchronous API support  

## How the Model Works

The AstroMindNLP model uses a multimodal architecture that combines Natural Language Processing and Computer Vision techniques for automatic category and subcategory prediction. During inference, the input text is preprocessed and encoded using the `ai-forever/ruBert-base` transformer model, while the input image is processed through the `google/vit-base-patch16-224` Vision Transformer model. Both models generate dense embedding vectors representing semantic textual information and visual features of the object.

The generated text and image embeddings are concatenated into a single fused multimodal vector used for prediction tasks. Category prediction is performed using a supervised `RandomForestClassifier` trained on fused embeddings, while subcategory prediction uses a prototype similarity-based approach. The fused embedding of a new object is compared against precomputed prototype vectors of each subcategory using cosine similarity, and the most similar prototype is selected as the final subcategory prediction.

## API Endpoints

- `POST /auth/register` — user registration
- `POST /auth/login` — authentication and JWT token retrieval
- `POST /predict` — category/subcategory prediction (requires JWT authentication)
- `GET /docs` — Swagger UI (interactive API documentation)
- `GET /redoc` — ReDoc documentation

## Application Notes

- **Model Loading:** Multimodal models (ruBERT + ViT) are initialized on startup and may take a short time to load depending on CPU/GPU performance.  
- **Compatibility:** Designed to run on modern laptops, servers, and cloud environments with CPU or GPU support.  
- **Resource Usage:** Optimized for efficient CPU/GPU usage while handling both text and image processing pipelines.  
- **Reliability:** Stable for continuous API usage with FastAPI async architecture and controlled memory management.  
- **Deployment:** Supports containerized deployment (Docker) with scalable backend architecture for production and development environments.  

## License

Copyright (c) 2026 Aleynikov Aleksandr

Permission is hereby granted, free of charge, to any person obtaining a copy of this software (`Airsoft Multimodal API`) for educational or demonstration purposes only. The software may not be used for commercial purposes, redistributed, or modified without prior written permission from the copyright holder.

For permissions, contact: [aleynikov.aleksandr@icloud.com](mailto:aleynikov.aleksandr@icloud.com).

## Author

Developed by Aleynikov Aleksandr  
Contact: [aleynikov.aleksandr@icloud.com](mailto:aleynikov.aleksandr@icloud.com)
