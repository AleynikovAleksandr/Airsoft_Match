"""
AirsoftMultimodalModel
======================
Мультимодальная система:
  - category: классификатор по fused embedding
  - subcategory: similarity-based по prototype vectors (cosine)
"""

import logging
import os

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, ViTImageProcessor, ViTModel

from app.config import settings
from app.text_preprocessing import preprocessor
from app.utils import download_and_process_image

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

        cat_clf_path = os.path.join(settings.MODEL_DIR, "joint_model", "category_classifier.pkl")
        cat_le_path = os.path.join(settings.MODEL_DIR, "joint_model", "category_label_encoder.pkl")
        proto_path = os.path.join(settings.MODEL_DIR, "joint_model", "subcategory_prototypes.pkl")

        self.category_clf = joblib.load(cat_clf_path) if os.path.exists(cat_clf_path) else None
        self.category_le = joblib.load(cat_le_path) if os.path.exists(cat_le_path) else None
        self.subcategory_prototypes = joblib.load(proto_path) if os.path.exists(proto_path) else {}

        if self.category_clf is None:
            logger.warning("Category classifier не найден — требуется обучение")
        if not self.subcategory_prototypes:
            logger.warning("Subcategory prototypes не найдены — требуется построение")

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

    def _predict_subcategory_by_similarity(self, fused: np.ndarray) -> tuple[str, float]:
        if not self.subcategory_prototypes:
            return self.subcategories[0], 0.0

        v = fused.flatten()
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return self.subcategories[0], 0.0

        best_label = self.subcategories[0]
        best_score = -1.0
        for label, proto in self.subcategory_prototypes.items():
            p = np.asarray(proto).flatten()
            p_norm = np.linalg.norm(p)
            if p_norm == 0:
                continue
            score = float(np.dot(v, p) / (v_norm * p_norm))
            if score > best_score:
                best_score = score
                best_label = label

        return best_label, max(0.0, min(1.0, (best_score + 1.0) / 2.0))

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

            if self.category_clf is not None and self.category_le is not None:
                cat_idx = int(self.category_clf.predict(fused)[0])
                cat_probs = self.category_clf.predict_proba(fused)
                cat_conf = float(np.max(cat_probs[0])) if isinstance(cat_probs, list) else float(np.max(cat_probs))
                category = self.category_le.inverse_transform([cat_idx])[0]
            else:
                category = self.categories[0]
                cat_conf = 0.7

            subcategory, sub_conf = self._predict_subcategory_by_similarity(fused)

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
        return self.category_clf is not None and self.category_le is not None and bool(self.subcategory_prototypes)
