"""
trainer.py
==========
Обучение единого multi-head multiclass классификатора.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

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


def _build_subcategory_index(subcategory_images_dir: str) -> dict[str, str]:
    """Индекс: имя файла -> subcategory (по папке)."""
    mapping: dict[str, str] = {}
    root = Path(subcategory_images_dir)
    if not root.exists():
        return mapping

    for sub_dir in root.iterdir():
        if not sub_dir.is_dir():
            continue
        subcategory = sub_dir.name
        for fp in sub_dir.iterdir():
            if fp.is_file():
                mapping[fp.name] = subcategory
    return mapping


def _extract_filename_from_url(url: str) -> str:
    parsed = urlparse(str(url))
    return unquote(Path(parsed.path).name)


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

        # Нормализуем имена столбцов для устойчивости
        posts_cols = {c.lower(): c for c in posts_df.columns}
        photos_cols = {c.lower(): c for c in photos_df.columns}

        # case 1: старый формат с category/subcategory
        has_old = "category" in posts_cols and "subcategory" in posts_cols and "url" in photos_cols

        records = []

        if has_old:
            logger.info("Используется формат posts(category/subcategory) + photos(url)")
            post_id_col = None
            for col in ("post_id", "id", "owner_id"):
                if col in posts_cols and col in photos_cols:
                    post_id_col = (posts_cols[col], photos_cols[col])
                    break

            for _, post_row in posts_df.iterrows():
                category = str(post_row.get(posts_cols["category"], "")).strip()
                subcategory = str(post_row.get(posts_cols["subcategory"], "")).strip()
                text = str(post_row.get(posts_cols.get("text", "text"), ""))
                if not category or not subcategory:
                    continue

                if post_id_col:
                    pcol, phcol = post_id_col
                    pid = post_row[pcol]
                    urls = photos_df[photos_df[phcol] == pid][photos_cols["url"]].tolist()
                else:
                    urls = photos_df[photos_cols["url"]].tolist()

                for url in urls:
                    records.append((str(text), str(url), category, subcategory))
        else:
            logger.info("Используется новый формат: merge photos.PostId -> posts.Id и categoryname")
            # Ожидаемые столбцы нового формата
            photos_postid = photos_cols.get("postid")
            photos_id = photos_cols.get("id")
            photos_url = photos_cols.get("url")
            posts_id = posts_cols.get("id")
            posts_text = posts_cols.get("text")
            posts_category = posts_cols.get("categoryname")

            missing = []
            for name, col in {
                "photos.PostId": photos_postid,
                "photos.Url": photos_url,
                "posts.Id": posts_id,
                "posts.Text": posts_text,
                "posts.categoryname": posts_category,
            }.items():
                if col is None:
                    missing.append(name)

            if missing:
                logger.error("Не хватает столбцов для merge-обучения: %s", ", ".join(missing))
                return

            merged = photos_df.merge(
                posts_df,
                left_on=photos_postid,
                right_on=posts_id,
                how="inner",
                suffixes=("_photo", "_post"),
            )

            sub_map = _build_subcategory_index(settings.SUBCATEGORY_IMAGES_DIR)
            if not sub_map:
                logger.error("Не найден индекс subcategory из папок: %s", settings.SUBCATEGORY_IMAGES_DIR)
                return

            for _, row in merged.iterrows():
                text = str(row.get(posts_text, ""))
                url = str(row.get(photos_url, ""))
                category = str(row.get(posts_category, "")).strip()
                filename = _extract_filename_from_url(url)
                subcategory = sub_map.get(filename, "")

                if not category or not subcategory or not url:
                    continue
                records.append((text, url, category, subcategory))

        if not records:
            logger.error("Не сформированы записи для обучения")
            return

        cat_le = LabelEncoder()
        sub_le = LabelEncoder()
        cat_le.fit(sorted({r[2] for r in records}))
        sub_le.fit(sorted({r[3] for r in records}))

        self.model.categories = list(cat_le.classes_)
        self.model.subcategories = list(sub_le.classes_)

        tasks = []
        for text, url, category, subcategory in records:
            text_emb = self.model.get_text_embedding(text)
            y_cat = int(cat_le.transform([category])[0])
            y_sub = int(sub_le.transform([subcategory])[0])
            tasks.append((self.model, url, text_emb, y_cat, y_sub))

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
