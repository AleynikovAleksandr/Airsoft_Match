"""
trainer.py
==========
Обучение единого multi-head multiclass классификатора:

Вход: concat(ruBert текст, ViT фото) -> 1536d
Выход: две метки одновременно
  1) category
  2) subcategory
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

    def train_joint_model(self) -> None:
        logger.info("=== Обучение joint multi-head модели ===")

        posts_path = settings.POSTS_FILE
        photos_path = settings.PHOTOS_FILE

        if not os.path.exists(posts_path) or not os.path.exists(photos_path):
            logger.error("Файлы данных не найдены: %s, %s", posts_path, photos_path)
            return

        posts_df = pd.read_parquet(posts_path)
        photos_df = pd.read_parquet(photos_path)

        if "category" not in posts_df.columns:
            logger.error("В posts.parquet нет колонки 'category'")
            return

        if "subcategory" not in posts_df.columns:
            logger.error("В posts.parquet нет колонки 'subcategory' для multi-head обучения")
            return

        if "url" not in photos_df.columns:
            logger.error("В photos.parquet нет колонки 'url'")
            return

        post_id_col = None
        for col in ("post_id", "id", "owner_id"):
            if col in posts_df.columns and col in photos_df.columns:
                post_id_col = col
                break

        cat_le = LabelEncoder()
        sub_le = LabelEncoder()
        cat_le.fit(sorted(posts_df["category"].astype(str).unique().tolist()))
        sub_le.fit(sorted(posts_df["subcategory"].astype(str).unique().tolist()))

        self.model.categories = list(cat_le.classes_)
        self.model.subcategories = list(sub_le.classes_)

        tasks = []
        for _, post_row in posts_df.iterrows():
            category = str(post_row.get("category", "")).strip()
            subcategory = str(post_row.get("subcategory", "")).strip()
            if not category or not subcategory:
                continue

            y_cat = int(cat_le.transform([category])[0])
            y_sub = int(sub_le.transform([subcategory])[0])

            text = str(post_row.get("text", ""))
            text_emb = self.model.get_text_embedding(text)

            if post_id_col:
                pid = post_row[post_id_col]
                post_photos = photos_df[photos_df[post_id_col] == pid]["url"].tolist()
            else:
                post_photos = photos_df["url"].tolist()

            for url in post_photos:
                tasks.append((self.model, str(url), text_emb, y_cat, y_sub))

        if not tasks:
            logger.error("Не сформированы задачи для обучения joint модели")
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
