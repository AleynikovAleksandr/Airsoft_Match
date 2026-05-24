"""Вспомогательные утилиты: загрузка изображений, конвертация."""

import io
import logging
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import ViTImageProcessor

logger = logging.getLogger(__name__)


def download_and_process_image(
    url: str,
    processor: ViTImageProcessor,
    timeout: int = 10,
) -> Optional[torch.Tensor]:
    """
    Скачивает изображение по URL и возвращает тензор для ViT.

    Args:
        url:       Прямая ссылка на изображение.
        processor: ViTImageProcessor (google/vit-base-patch16-224).
        timeout:   Тайм-аут HTTP-запроса в секундах.

    Returns:
        torch.Tensor формы (3, 224, 224) или None при ошибке.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)  # (3, 224, 224)

    except Exception as exc:
        logger.warning("Не удалось загрузить изображение %s: %s", url, exc)
        return None


def load_image_from_path(
    path: str,
    processor: ViTImageProcessor,
) -> Optional[torch.Tensor]:
    """
    Загружает изображение с диска и возвращает тензор для ViT.
    Используется при обучении на subcategory_images/.
    """
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)
    except Exception as exc:
        logger.warning("Не удалось прочитать файл %s: %s", path, exc)
        return None
