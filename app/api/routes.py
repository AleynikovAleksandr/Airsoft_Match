"""
routes.py
=========
FastAPI роуты:
  POST /auth/register  — регистрация
  POST /auth/login     — авторизация → JWT токен
  POST /predict        — предсказание (требует Authorization: Bearer <token>)
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.api.auth import create_access_token, decode_access_token
from app.api.schemas import LoginRequest, PredictRequest, PredictResponse, RegisterRequest, TokenResponse
from app.db.crud import authenticate_user, create_user, get_user_by_username
from app.db.database import get_db
from app.ml.inference import run_inference

router = APIRouter()
security = HTTPBearer()


# --------------------------------------------------------------------------- #
#  Зависимость: получить текущего пользователя из JWT                          #
# --------------------------------------------------------------------------- #

def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Session = Depends(get_db),
):
    token = credentials.credentials
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный или просроченный токен",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный токен: нет поля sub",
        )
    user = get_user_by_username(db, username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
        )
    return user


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #

@router.post(
    "/auth/register",
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация нового пользователя",
)
async def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_username(db, body.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким именем уже существует",
        )
    user = create_user(db, body.username, body.password)
    return {"message": "Пользователь создан", "username": user.username}


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Авторизация — получение JWT токена",
)
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(access_token=token)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Предсказание категории и подкатегории снаряжения",
)
async def predict(
    body: PredictRequest,
    current_user=Depends(get_current_user),
):
    photos_dicts = [p.model_dump() for p in body.photos]
    result = run_inference(body.post_id, body.text, photos_dicts)
    return PredictResponse(**result)
