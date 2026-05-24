"""
trainer.py
==========
Обучение Random Forest классификаторов:

1. category_model  — обучается на posts.parquet + photos.parquet
   Вход: concat(ruBert текст, ViT фото) → 1536d
   Метка: колонка `category` из posts.parquet

2. subcategory_model — обучается на subcategory_images/
   Вход: ViT фото → 768d
   Метка: имя папки (название подкатегории)

Оба классификатора сохраняются в model/
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from config import settings
from src.models.multimodal_model import AirsoftMultimodalModel
from src.utils import download_and_process_image, load_image_from_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Вспомогательные функции                                                    #
# --------------------------------------------------------------------------- #

def _load_one_post_photo(args):
    """
    Загружает одно фото поста и возвращает (fused_emb, label).
    Используется в ThreadPoolExecutor для параллельной загрузки.
    """
    model, url, text_emb, label = args
    img_tensor = download_and_process_image(url, model.image_processor)
    if img_tensor is None:
        return None
    img_emb = model.get_image_embedding(img_tensor)
    fused = model.get_fused_embedding(text_emb, img_emb)
    return fused, label


def _load_one_subcategory_image(args):
    """Загружает одно изображение подкатегории и возвращает (img_emb, label)."""
    model, img_path, label = args
    img_tensor = load_image_from_path(img_path, model.image_processor)
    if img_tensor is None:
        return None
    img_emb = model.get_image_embedding(img_tensor)
    return img_emb, label


# --------------------------------------------------------------------------- #
#  Основной класс тренера                                                     #
# --------------------------------------------------------------------------- #

class ModelTrainer:
    def __init__(self, model: AirsoftMultimodalModel):
        self.model = model

    # ----------------------------------------------------------------------- #
    #  1. Обучение классификатора категорий                                    #
    # ----------------------------------------------------------------------- #

    def train_category_model(self) -> None:
        """
        Строит обучающую выборку из posts.parquet + photos.parquet.
        Для каждой пары (пост, фото) создаёт 1536d вектор.
        Обучает RandomForest и сохраняет в model/category_model/classifier.pkl
        """
        logger.info("=== Обучение модели категорий ===")

        posts_path = settings.POSTS_FILE
        photos_path = settings.PHOTOS_FILE

        if not os.path.exists(posts_path) or not os.path.exists(photos_path):
            logger.error(
                "Файлы данных не найдены: %s, %s", posts_path, photos_path
            )
            return

        posts_df = pd.read_parquet(posts_path)
        photos_df = pd.read_parquet(photos_path)

        logger.info(
            "Загружено постов: %d, фото: %d", len(posts_df), len(photos_df)
        )

        # Ожидаемые колонки
        if "category" not in posts_df.columns:
            logger.error("В posts.parquet нет колонки 'category'")
            return
        if "url" not in photos_df.columns:
            logger.error("В photos.parquet нет колонки 'url'")
            return

        # Определяем колонку связи поста и фото
        # (обычно owner_id + post_id или просто post_id)
        post_id_col = None
        for col in ("post_id", "id", "owner_id"):
            if col in posts_df.columns and col in photos_df.columns:
                post_id_col = col
                break

        # Кодируем метки
        le = LabelEncoder()
        le.fit(self.model.categories)

        # Готовим задачи для многопоточной загрузки
        tasks = []
        for _, post_row in posts_df.iterrows():
            raw_label = str(post_row.get("category", ""))
            if raw_label not in self.model.categories:
                continue

            label = le.transform([raw_label])[0]
            text = str(post_row.get("text", ""))
            text_emb = self.model.get_text_embedding(text)

            if post_id_col:
                pid = post_row[post_id_col]
                post_photos = photos_df[photos_df[post_id_col] == pid]["url"].tolist()
            else:
                # Если нет общего ключа — берём все фото (для небольших датасетов)
                post_photos = photos_df["url"].tolist()

            for url in post_photos:
                tasks.append((self.model, str(url), text_emb, label))

        if not tasks:
            logger.warning("Не удалось построить ни одной задачи для category_model")
            return

        logger.info("Параллельная загрузка %d фото (workers=%d) ...", len(tasks), settings.IMAGE_LOAD_WORKERS)

        X, y = [], []
        with ThreadPoolExecutor(max_workers=settings.IMAGE_LOAD_WORKERS) as executor:
            futures = {executor.submit(_load_one_post_photo, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fused, label = result
                    X.append(fused.flatten())
                    y.append(label)

        if not X:
            logger.error("Не удалось загрузить ни одного изображения для category_model")
            return

        X = np.array(X)
        y = np.array(y)
        logger.info("Обучающая выборка: %s, классов: %d", X.shape, len(set(y)))

        clf = RandomForestClassifier(
            n_estimators=settings.N_ESTIMATORS,
            n_jobs=settings.N_JOBS,
            random_state=42,
            verbose=1,
        )
        clf.fit(X, y)
        logger.info("RandomForest для категорий обучен")

        # Сохраняем
        save_dir = os.path.join(settings.MODEL_DIR, "category_model")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "classifier.pkl")
        joblib.dump(clf, save_path)
        joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))
        logger.info("Модель категорий сохранена: %s", save_path)

        # Обновляем классификатор в модели
        self.model.category_clf = clf

    # ----------------------------------------------------------------------- #
    #  2. Обучение классификатора подкатегорий                                #
    # ----------------------------------------------------------------------- #

    def train_subcategory_model(self) -> None:
        """
        Строит выборку из subcategory_images/<class_name>/*.jpg|png ...
        Вход: ViT эмбеддинг 768d.
        Обучает RandomForest и сохраняет в model/subcategory_model/classifier.pkl
        """
        logger.info("=== Обучение модели подкатегорий ===")

        images_root = Path(settings.SUBCATEGORY_IMAGES_DIR)
        if not images_root.exists():
            logger.error("Директория с изображениями не найдена: %s", images_root)
            return

        class_dirs = [d for d in images_root.iterdir() if d.is_dir()]
        if not class_dirs:
            logger.error("В %s не найдено подпапок с классами", images_root)
            return

        # Строим маппинг имя→индекс
        class_names = sorted([d.name for d in class_dirs])
        # Обновляем список подкатегорий в модели
        self.model.subcategories = class_names
        le = LabelEncoder()
        le.fit(class_names)

        # Готовим задачи
        tasks = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for class_dir in class_dirs:
            label = le.transform([class_dir.name])[0]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    tasks.append((self.model, str(img_path), label))

        if not tasks:
            logger.error("Не найдено ни одного изображения в %s", images_root)
            return

        logger.info(
            "Параллельная загрузка %d изображений (workers=%d) ...",
            len(tasks),
            settings.IMAGE_LOAD_WORKERS,
        )

        X, y = [], []
        with ThreadPoolExecutor(max_workers=settings.IMAGE_LOAD_WORKERS) as executor:
            futures = {
                executor.submit(_load_one_subcategory_image, t): t for t in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    img_emb, label = result
                    X.append(img_emb.flatten())
                    y.append(label)

        if not X:
            logger.error("Не удалось загрузить изображения для subcategory_model")
            return

        X = np.array(X)
        y = np.array(y)
        logger.info("Обучающая выборка: %s, классов: %d", X.shape, len(set(y)))

        clf = RandomForestClassifier(
            n_estimators=settings.N_ESTIMATORS,
            n_jobs=settings.N_JOBS,
            random_state=42,
            verbose=1,
        )
        clf.fit(X, y)
        logger.info("RandomForest для подкатегорий обучен")

        # Сохраняем
        save_dir = os.path.join(settings.MODEL_DIR, "subcategory_model")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "classifier.pkl")
        joblib.dump(clf, save_path)
        joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))
        logger.info("Модель подкатегорий сохранена: %s", save_path)

        self.model.subcategory_clf = clf

    # ----------------------------------------------------------------------- #
    #  Полный цикл обучения                                                    #
    # ----------------------------------------------------------------------- #

    def train_all(self) -> None:
        self.train_category_model()
        self.train_subcategory_model()
        logger.info("=== Обучение завершено ===")
