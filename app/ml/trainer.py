"""
trainer.py
==========
Обучение:
  1) category-классификатор по fused embeddings
  2) subcategory prototype vectors по папкам reference-изображений
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.ml.multimodal_model import AirsoftMultimodalModel
from app.utils import download_and_process_image, load_image_from_path

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model: AirsoftMultimodalModel):
        self.model = model
        self.text_cache = {}

    def _get_text_embedding_cached(self, text: str):
        text = str(text)
        if text not in self.text_cache:
            self.text_cache[text] = self.model.get_text_embedding(text)
        return self.text_cache[text]

    def _load_dataset(self):
        posts_df = pd.read_parquet(settings.POSTS_FILE)
        photos_df = pd.read_parquet(settings.PHOTOS_FILE)

        posts_cols = {c.lower(): c for c in posts_df.columns}
        photos_cols = {c.lower(): c for c in photos_df.columns}

        required_posts = ["id", "text", "categoryname"]
        required_photos = ["postid", "url"]

        for col in required_posts:
            if col not in posts_cols:
                raise ValueError(f"posts.parquet missing column: {col}")
        for col in required_photos:
            if col not in photos_cols:
                raise ValueError(f"photos.parquet missing column: {col}")

        merged = photos_df.merge(
            posts_df,
            left_on=photos_cols["postid"],
            right_on=posts_cols["id"],
            how="inner",
        )

        logger.info("Merged rows: %s", len(merged))
        logger.info("Found categories: %s", merged[posts_cols["categoryname"]].nunique())
        return merged, posts_cols, photos_cols

    def _build_subcategory_prototypes(self) -> dict[str, np.ndarray]:
        root = Path(settings.SUBCATEGORY_IMAGES_DIR)
        if not root.exists():
            raise ValueError(f"Subcategory dir not found: {root}")

        prototypes = {}
        for sub_dir in root.iterdir():
            if not sub_dir.is_dir():
                continue

            class_name = sub_dir.name
            class_embs = []
            for fp in sub_dir.iterdir():
                if not fp.is_file():
                    continue
                img_tensor = load_image_from_path(str(fp), self.model.image_processor)
                if img_tensor is None:
                    continue
                img_emb = self.model.get_image_embedding(img_tensor)
                class_embs.append(img_emb.flatten())

            if class_embs:
                proto = np.mean(np.array(class_embs), axis=0)
                prototypes[class_name] = proto
                logger.info("Prototype built: %s (%s images)", class_name, len(class_embs))

        if not prototypes:
            raise ValueError("No subcategory prototypes were built")

        self.model.subcategories = sorted(prototypes.keys())
        return prototypes

    def train_joint_model(self) -> None:
        logger.info("=== Обучение category classifier + subcategory prototypes ===")

        if not os.path.exists(settings.POSTS_FILE) or not os.path.exists(settings.PHOTOS_FILE):
            logger.error("Файлы данных не найдены: %s, %s", settings.POSTS_FILE, settings.PHOTOS_FILE)
            return

        try:
            merged, posts_cols, photos_cols = self._load_dataset()
            prototypes = self._build_subcategory_prototypes()
        except ValueError as exc:
            logger.error("Ошибка подготовки данных: %s", exc)
            return

        merged = merged.dropna(subset=[posts_cols["text"], posts_cols["categoryname"], photos_cols["url"]])
        if merged.empty:
            logger.error("После очистки merged-датасет пуст")
            return

        cat_le = LabelEncoder()
        cat_le.fit(sorted(merged[posts_cols["categoryname"]].astype(str).str.strip().unique().tolist()))
        self.model.categories = list(cat_le.classes_)
        logger.info("Category classes: %s", self.model.categories)

        tasks = []
        for _, row in merged.iterrows():
            text = str(row[posts_cols["text"]])
            url = str(row[photos_cols["url"]])
            category = str(row[posts_cols["categoryname"]]).strip()
            if not category or not url:
                continue
            text_emb = self._get_text_embedding_cached(text)
            y_cat = int(cat_le.transform([category])[0])
            tasks.append((url, text_emb, y_cat))

        X, y_cat_arr = [], []

        def load_one(task):
            url, text_emb, y_cat = task
            img_tensor = download_and_process_image(url, self.model.image_processor)
            if img_tensor is None:
                return None
            img_emb = self.model.get_image_embedding(img_tensor)
            fused = self.model.get_fused_embedding(text_emb, img_emb)
            return fused, y_cat

        with ThreadPoolExecutor(max_workers=settings.IMAGE_LOAD_WORKERS) as executor:
            futures = {executor.submit(load_one, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fused, y_cat = result
                    X.append(fused.flatten())
                    y_cat_arr.append(y_cat)

        if not X:
            logger.error("Не удалось загрузить данные для category classifier")
            return

        X = np.array(X)
        y_cat = np.array(y_cat_arr)

        clf = RandomForestClassifier(
            n_estimators=settings.N_ESTIMATORS,
            n_jobs=settings.N_JOBS,
            random_state=42,
            verbose=1,
        )
        clf.fit(X, y_cat)

        save_dir = os.path.join(settings.MODEL_DIR, "joint_model")
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, os.path.join(save_dir, "category_classifier.pkl"))
        joblib.dump(cat_le, os.path.join(save_dir, "category_label_encoder.pkl"))
        joblib.dump(prototypes, os.path.join(save_dir, "subcategory_prototypes.pkl"))

        self.model.category_clf = clf
        self.model.category_le = cat_le
        self.model.subcategory_prototypes = prototypes

        logger.info("Category classifier and subcategory prototypes сохранены в %s", save_dir)

    def train_all(self) -> None:
        self.train_joint_model()
        logger.info("=== Обучение завершено ===")
