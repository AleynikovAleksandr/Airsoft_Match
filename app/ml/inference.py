"""
inference.py
============
Обёртка для инференса: принимает входные данные запроса API
и возвращает структурированный ответ.
"""

import logging
from typing import Optional

from app.ml.multimodal_model import AirsoftMultimodalModel

logger = logging.getLogger(__name__)


class InferenceService:
    """Сервис для ленивой инициализации модели и запуска инференса."""

    def __init__(self):
        self._model: Optional[AirsoftMultimodalModel] = None

    def get_model(self) -> AirsoftMultimodalModel:
        """Возвращает глобальный экземпляр модели (ленивая инициализация)."""
        if self._model is None:
            logger.info("Инициализация модели (первый вызов)...")
            self._model = AirsoftMultimodalModel()
        return self._model

    def run_inference(self, post_id: str, text: str, photos: list[dict]) -> dict:
        """
        Запускает предсказание для одного запроса.

        Args:
            post_id: Идентификатор поста.
            text:    Текст объявления.
            photos:  Список {"photo_id": str, "url": str}.

        Returns:
            Словарь формата:
            {
                "post_id": "...",
                "predictions": [
                    {
                        "object_id": "...",
                        "category": "...",
                        "subcategory": "...",
                        "confidence": 0.9,
                        "photo_ids": [...]
                    }, ...
                ]
            }
        """
        model = self.get_model()

        if not model.is_ready():
            logger.warning(
                "Модель не обучена — возвращаются placeholder предсказания"
            )

        predictions = model.predict(text, photos)

        return {
            "post_id": post_id,
            "predictions": predictions,
        }


inference_service = InferenceService()


def get_model() -> AirsoftMultimodalModel:
    return inference_service.get_model()


def run_inference(post_id: str, text: str, photos: list[dict]) -> dict:
    return inference_service.run_inference(post_id, text, photos)
