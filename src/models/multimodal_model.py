"""
AirsoftMultimodalModel
======================
Мультимодальная модель в стиле CLIP + Late Fusion:
  - Текст:       ai-forever/ruBert-base   → CLS-эмбеддинг 768d
  - Изображение: google/vit-base-patch16-224 → CLS-эмбеддинг 768d
  - Fusion:      concat → вектор 1536d
  - Classifier:  Random Forest (sklearn)

Для категорий используются пары (текст поста + каждое фото),
для подкатегорий — только изображения из subcategory_images/.
"""

import logging
import os

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel

from config import settings
from src.utils import download_and_process_image

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Списки классов                                                              #
# --------------------------------------------------------------------------- #

CATEGORIES = [
    "Страйкбольное оружие",
    "Аксессуары и Запчасти",
    "Снаряжение и защита",
]

SUBCATEGORIES = [
    "AK",
    "M Series",
    "HK",
    "Rifle",
    "Pistol",
    "Machinegun",
    "Shotgun",
    "Vest",
    "Helmet",
    "Backpack",
    "Pouch",
    "Other",
]


# --------------------------------------------------------------------------- #
#  Модель                                                                      #
# --------------------------------------------------------------------------- #

class AirsoftMultimodalModel:
    """
    Загружает предобученные энкодеры и обученные классификаторы.
    При первом запуске классификаторы отсутствуют — они будут обучены
    модулем trainer.py и сохранены в директорию model/.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Используется устройство: %s", self.device)

        # ── Текстовый энкодер ──────────────────────────────────────────── #
        logger.info("Загрузка текстовой модели %s ...", settings.TEXT_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_MODEL_NAME)
        self.text_model = AutoModel.from_pretrained(settings.TEXT_MODEL_NAME).to(
            self.device
        )
        self.text_model.eval()

        # ── Визуальный энкодер ─────────────────────────────────────────── #
        logger.info("Загрузка визуальной модели %s ...", settings.IMAGE_MODEL_NAME)
        self.image_processor = ViTImageProcessor.from_pretrained(
            settings.IMAGE_MODEL_NAME
        )
        self.image_model = ViTModel.from_pretrained(settings.IMAGE_MODEL_NAME).to(
            self.device
        )
        self.image_model.eval()

        # ── Классификаторы (Random Forest) ─────────────────────────────── #
        self.categories = CATEGORIES
        self.subcategories = SUBCATEGORIES

        cat_path = os.path.join(
            settings.MODEL_DIR, "category_model", "classifier.pkl"
        )
        sub_path = os.path.join(
            settings.MODEL_DIR, "subcategory_model", "classifier.pkl"
        )

        self.category_clf = joblib.load(cat_path) if os.path.exists(cat_path) else None
        self.subcategory_clf = (
            joblib.load(sub_path) if os.path.exists(sub_path) else None
        )

        if self.category_clf:
            logger.info("Классификатор категорий загружен из %s", cat_path)
        else:
            logger.warning("Классификатор категорий не найден — требуется обучение")

        if self.subcategory_clf:
            logger.info("Классификатор подкатегорий загружен из %s", sub_path)
        else:
            logger.warning(
                "Классификатор подкатегорий не найден — требуется обучение"
            )

    # ----------------------------------------------------------------------- #
    #  Получение эмбеддингов                                                   #
    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """CLS-токен ruBert-base → numpy (1, 768)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self.text_model(**inputs)
        # Усредняем все токены (mean pooling)
        emb = outputs.last_hidden_state.mean(dim=1)  # (1, 768)
        return emb.cpu().numpy()

    @torch.no_grad()
    def get_image_embedding(self, pixel_values: torch.Tensor) -> np.ndarray:
        """ViT CLS-токен → numpy (1, 768)."""
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        outputs = self.image_model(pixel_values=pixel_values)
        emb = outputs.last_hidden_state.mean(dim=1)  # (1, 768)
        return emb.cpu().numpy()

    def get_fused_embedding(
        self, text_emb: np.ndarray, img_emb: np.ndarray
    ) -> np.ndarray:
        """Late fusion: конкатенация → (1, 1536)."""
        return np.concatenate([text_emb, img_emb], axis=1)

    # ----------------------------------------------------------------------- #
    #  Предсказание                                                             #
    # ----------------------------------------------------------------------- #

    def predict(self, text: str, photo_urls: list[dict]) -> list[dict]:
        """
        Основной метод инференса.

        Args:
            text:       Текст поста.
            photo_urls: Список словарей {"photo_id": str, "url": str}.

        Returns:
            Список предсказаний по каждому фото-объекту.
        """
        predictions = []

        # Один эмбеддинг текста на весь пост
        text_emb = self.get_text_embedding(text)

        for photo in photo_urls:
            photo_id = photo.get("photo_id", "?")
            url = photo.get("url", "")

            img_tensor = download_and_process_image(url, self.image_processor)
            if img_tensor is None:
                logger.warning("Пропускаем фото %s (не удалось загрузить)", photo_id)
                continue

            img_emb = self.get_image_embedding(img_tensor)
            fused = self.get_fused_embedding(text_emb, img_emb)

            # Категория
            if self.category_clf:
                cat_idx = int(self.category_clf.predict(fused)[0])
                cat_conf = float(self.category_clf.predict_proba(fused).max())
            else:
                cat_idx, cat_conf = 0, 0.75

            # Подкатегория (только изображение → 768d вектор)
            if self.subcategory_clf:
                sub_idx = int(self.subcategory_clf.predict(img_emb)[0])
                sub_conf = float(self.subcategory_clf.predict_proba(img_emb).max())
            else:
                sub_idx, sub_conf = 0, 0.65

            predictions.append(
                {
                    "object_id": str(photo_id),
                    "category": self.categories[cat_idx],
                    "subcategory": self.subcategories[sub_idx],
                    "confidence": round((cat_conf + sub_conf) / 2, 3),
                    "photo_ids": [str(photo_id)],
                }
            )

        return predictions

    def is_ready(self) -> bool:
        """Возвращает True, если оба классификатора загружены."""
        return self.category_clf is not None and self.subcategory_clf is not None
