"""Pydantic-схемы для API запросов/ответов."""

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PhotoItem(BaseModel):
    photo_id: str
    url: str


class PredictRequest(BaseModel):
    post_id: str
    text: str
    photos: list[PhotoItem]


class ObjectPrediction(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    photo_ids: list[str]


class PredictResponse(BaseModel):
    post_id: str
    predictions: list[ObjectPrediction]
