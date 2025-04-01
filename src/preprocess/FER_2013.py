import os
import shutil
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories(base_dir, emotions):
    """
    Create necessary directories for train, validation, and test sets.
    """
    os.makedirs(base_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for emotion in emotions:
            emotion_dir = os.path.join(split_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)

def copy_images(src_dir, dst_dir, images, emotion):
    """
    Copy images from source to destination directory.
    """
    for img_name in images:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, emotion, img_name)
        shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.getcwd(), "data")
    FER_2013_DIR = os.path.join(DATA_DIR, "original_datasets", "FER_2013")
    PROCESSED_FER_2013_DIR = os.path.join(DATA_DIR, "preprocessed_datasets", "FER_2013")

    if os.path.exists(PROCESSED_FER_2013_DIR): shutil.rmtree(PROCESSED_FER_2013_DIR)

    emotions = os.listdir(os.path.join(FER_2013_DIR, "train"))
    logger.info(f"Emotions found: {emotions}")

    create_directories(PROCESSED_FER_2013_DIR, emotions)

    ORIG_TEST_DIR = os.path.join(FER_2013_DIR, "test")
    NEW_TEST_DIR = os.path.join(PROCESSED_FER_2013_DIR, "test")
    
    shutil.copytree(ORIG_TEST_DIR, NEW_TEST_DIR, dirs_exist_ok=True)
    logger.info(f"Test data copied to {NEW_TEST_DIR}")

    ORIG_TRAIN_DIR = os.path.join(FER_2013_DIR, "train")
    for emotion in emotions:
        emotion_train_dir = os.path.join(ORIG_TRAIN_DIR, emotion)
        all_images = os.listdir(emotion_train_dir)

        train_images, val_images = train_test_split(all_images, test_size=0.15, random_state=42)
        logger.info(f"Processing emotion: {emotion}")
        logger.info(f"Total images: {len(all_images)}, Training images: {len(train_images)}, Validation images: {len(val_images)}")

        copy_images(emotion_train_dir, os.path.join(PROCESSED_FER_2013_DIR, "train"), train_images, emotion)
        copy_images(emotion_train_dir, os.path.join(PROCESSED_FER_2013_DIR, "val"), val_images, emotion)

    logger.info("Completed")
