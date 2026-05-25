"""
AirsoftMultimodalModel
======================
Единая multi-head multiclass система:
  - Общий энкодер: ruBert + ViT -> fused embedding (1536)
  - Head #1: category
  - Head #2: subcategory

Технически используется один multi-output классификатор sklearn,
который предсказывает обе метки одновременно по одному входному вектору.
"""

import logging
import os

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel

from app.config import settings
from app.utils import download_and_process_image
from app.text_preprocessing import preprocessor

logger = logging.getLogger(__name__)


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


class AirsoftMultimodalModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Используется устройство: %s", self.device)

        logger.info("Загрузка текстовой модели %s ...", settings.TEXT_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_MODEL_NAME)
        self.text_model = AutoModel.from_pretrained(settings.TEXT_MODEL_NAME).to(self.device)
        self.text_model.eval()

        logger.info("Загрузка визуальной модели %s ...", settings.IMAGE_MODEL_NAME)
        self.image_processor = ViTImageProcessor.from_pretrained(settings.IMAGE_MODEL_NAME)
        self.image_model = ViTModel.from_pretrained(settings.IMAGE_MODEL_NAME).to(self.device)
        self.image_model.eval()

        self.categories = CATEGORIES
        self.subcategories = SUBCATEGORIES

        joint_path = os.path.join(settings.MODEL_DIR, "joint_model", "classifier.pkl")
        cat_le_path = os.path.join(settings.MODEL_DIR, "joint_model", "category_label_encoder.pkl")
        sub_le_path = os.path.join(settings.MODEL_DIR, "joint_model", "subcategory_label_encoder.pkl")

        self.joint_clf = joblib.load(joint_path) if os.path.exists(joint_path) else None
        self.category_le = joblib.load(cat_le_path) if os.path.exists(cat_le_path) else None
        self.subcategory_le = joblib.load(sub_le_path) if os.path.exists(sub_le_path) else None

        if self.joint_clf is None:
            logger.warning("Joint multi-head классификатор не найден — требуется обучение")

    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        clean_text = preprocessor.clean_text(text)
        model_text = clean_text if clean_text else text

        inputs = self.tokenizer(
            model_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self.text_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy()

    @torch.no_grad()
    def get_image_embedding(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        outputs = self.image_model(pixel_values=pixel_values)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy()

    def get_fused_embedding(self, text_emb: np.ndarray, img_emb: np.ndarray) -> np.ndarray:
        return np.concatenate([text_emb, img_emb], axis=1)

    def predict(self, text: str, photo_urls: list[dict]) -> list[dict]:
        predictions = []
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

            if self.joint_clf and self.category_le and self.subcategory_le:
                pred = self.joint_clf.predict(fused)[0]
                cat_idx, sub_idx = int(pred[0]), int(pred[1])

                proba_heads = self.joint_clf.predict_proba(fused)
                cat_conf = float(np.max(proba_heads[0][0]))
                sub_conf = float(np.max(proba_heads[1][0]))

                category = self.category_le.inverse_transform([cat_idx])[0]
                subcategory = self.subcategory_le.inverse_transform([sub_idx])[0]
            else:
                category = self.categories[0]
                subcategory = self.subcategories[0]
                cat_conf, sub_conf = 0.7, 0.7

            predictions.append(
                {
                    "object_id": str(photo_id),
                    "category": str(category),
                    "subcategory": str(subcategory),
                    "confidence": round((cat_conf + sub_conf) / 2, 3),
                    "photo_ids": [str(photo_id)],
                }
            )

        return predictions

    def is_ready(self) -> bool:
        return (
            self.joint_clf is not None
            and self.category_le is not None
            and self.subcategory_le is not None
        )
