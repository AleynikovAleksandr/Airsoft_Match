"""
trainer.py
==========
Обучение единого multi-head multiclass классификатора.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.ml.multimodal_model import AirsoftMultimodalModel
from app.utils import download_and_process_image

logger = logging.getLogger(__name__)


def _load_one_post_photo(args):
    model, url, text_emb, y_cat, y_sub = args
    img_tensor = download_and_process_image(url, model.image_processor)
    if img_tensor is None:
        return None
    img_emb = model.get_image_embedding(img_tensor)
    fused = model.get_fused_embedding(text_emb, img_emb)
    return fused, y_cat, y_sub


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

        required_posts = ["id", "text", "category", "subcategory"]
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
        logger.info(
            "Found labels: categories=%s, subcategories=%s",
            merged[posts_cols["category"]].nunique(),
            merged[posts_cols["subcategory"]].nunique(),
        )

        return merged, posts_cols, photos_cols

    def train_joint_model(self) -> None:
        logger.info("=== Обучение joint multi-head модели ===")

        if not os.path.exists(settings.POSTS_FILE) or not os.path.exists(settings.PHOTOS_FILE):
            logger.error("Файлы данных не найдены: %s, %s", settings.POSTS_FILE, settings.PHOTOS_FILE)
            return

        try:
            merged, posts_cols, photos_cols = self._load_dataset()
        except ValueError as exc:
            logger.error("Ошибка схемы датасета: %s", exc)
            return

        merged = merged.dropna(subset=[posts_cols["text"], posts_cols["category"], posts_cols["subcategory"], photos_cols["url"]])
        if merged.empty:
            logger.error("После очистки merged-датасет пуст")
            return

        cat_le = LabelEncoder()
        sub_le = LabelEncoder()
        cat_le.fit(sorted(merged[posts_cols["category"]].astype(str).str.strip().unique().tolist()))
        sub_le.fit(sorted(merged[posts_cols["subcategory"]].astype(str).str.strip().unique().tolist()))

        self.model.categories = list(cat_le.classes_)
        self.model.subcategories = list(sub_le.classes_)

        logger.info("Category classes: %s", self.model.categories)
        logger.info("Subcategory classes: %s", self.model.subcategories)

        tasks = []
        for _, row in merged.iterrows():
            text = str(row[posts_cols["text"]])
            url = str(row[photos_cols["url"]])
            category = str(row[posts_cols["category"]]).strip()
            subcategory = str(row[posts_cols["subcategory"]]).strip()

            if not category or not subcategory or not url:
                continue

            text_emb = self._get_text_embedding_cached(text)
            y_cat = int(cat_le.transform([category])[0])
            y_sub = int(sub_le.transform([subcategory])[0])
            tasks.append((self.model, url, text_emb, y_cat, y_sub))

        if not tasks:
            logger.error("Не сформированы задачи для обучения")
            return

        X, y_cat_arr, y_sub_arr = [], [], []
        with ThreadPoolExecutor(max_workers=settings.IMAGE_LOAD_WORKERS) as executor:
            futures = {executor.submit(_load_one_post_photo, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fused, y_cat, y_sub = result
                    X.append(fused.flatten())
                    y_cat_arr.append(y_cat)
                    y_sub_arr.append(y_sub)

        if not X:
            logger.error("Не удалось загрузить данные для joint модели")
            return

        X = np.array(X)
        Y = np.column_stack([np.array(y_cat_arr), np.array(y_sub_arr)])

        base_rf = RandomForestClassifier(
            n_estimators=settings.N_ESTIMATORS,
            n_jobs=settings.N_JOBS,
            random_state=42,
            verbose=1,
        )
        clf = MultiOutputClassifier(base_rf)
        clf.fit(X, Y)

        save_dir = os.path.join(settings.MODEL_DIR, "joint_model")
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, os.path.join(save_dir, "classifier.pkl"))
        joblib.dump(cat_le, os.path.join(save_dir, "category_label_encoder.pkl"))
        joblib.dump(sub_le, os.path.join(save_dir, "subcategory_label_encoder.pkl"))

        self.model.joint_clf = clf
        self.model.category_le = cat_le
        self.model.subcategory_le = sub_le

        logger.info("Joint multi-head модель обучена и сохранена в %s", save_dir)

    def train_all(self) -> None:
        self.train_joint_model()
        logger.info("=== Обучение завершено ===")
