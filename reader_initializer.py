# reader_initializer.py
import logging
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
import torch


def initialize_data(validation_ratio=0.1, seed=1234):
    """Adatok beolvasása, csoportosítása és előkészítése a tanításhoz."""

    # Seed beállítása a reprodukálhatósághoz
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f"Validation_ratio : {validation_ratio}")

    # Képek és címkék betöltése
    train_image_files = glob.glob('../Neural_network_contest/train_data/*.png')
    test_image_files = glob.glob('../Neural_network_contest/test_data/*.png')

    # EXCEL beolvasás
    file_path = 'data_labels_train.csv'
    df = pd.read_csv(file_path)
    selected_data = df[['filename_id', 'defocus_label']]
    data_array = selected_data.to_numpy()
    logging.info(f"data_array.shape : {data_array.shape}")

    # Tárolók az adatokhoz
    train_data_dict = {}
    test_data_dict = {}

    # Train képek beolvasása és csoportosítása ID és típus alapján
    for image_path in train_image_files:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        if "__" in file_name:
            id_part = file_name.rsplit('_', 1)[0]
            type_part = file_name.split('_')[-1]
            if type_part in ["amp", "mask", "phase"]:
                if id_part not in train_data_dict:
                    train_data_dict[id_part] = {'amp': None, 'mask': None, 'phase': None}
                if train_data_dict[id_part][type_part] is None:
                    img = mpimg.imread(image_path)
                    if img.shape[:2] != (128, 128):
                        img = resize(img, (128, 128), anti_aliasing=True)
                    train_data_dict[id_part][type_part] = img

    # Test képek beolvasása és csoportosítása ID és típus alapján
    for image_path in test_image_files:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        if "__" in file_name:
            test_id_part = file_name.rsplit('_', 1)[0]
            test_type_part = file_name.split('_')[-1]
            if test_type_part in ["amp", "mask", "phase"]:
                if test_id_part not in test_data_dict:
                    test_data_dict[test_id_part] = {'amp': None, 'mask': None, 'phase': None}
                if test_data_dict[test_id_part][test_type_part] is None:
                    img = mpimg.imread(image_path)
                    if img.shape[:2] != (128, 128):
                        img = resize(img, (128, 128), anti_aliasing=True)
                    test_data_dict[test_id_part][test_type_part] = img

    # Adatok numpy tömbbe konvertálása
    train_image_list = []
    train_image_ids = []
    for id_key, img_types in train_data_dict.items():
        if img_types['amp'] is not None and img_types['mask'] is not None and img_types['phase'] is not None:
            image_stack = np.stack([img_types['amp'], img_types['mask'], img_types['phase']], axis=-1)
            train_image_list.append(image_stack)
            train_image_ids.append(id_key)

    test_image_list = []
    test_image_ids = []
    for test_id_part, test_type_part in test_data_dict.items():
        if test_type_part['amp'] is not None and test_type_part['mask'] is not None and test_type_part[
            'phase'] is not None:
            image_stack = np.stack([test_type_part['amp'], test_type_part['mask'], test_type_part['phase']], axis=-1)
            test_image_list.append(image_stack)
            test_image_ids.append(test_id_part)

    # Számláló a törölt elemekhez
    deleted_count_train = len(train_data_dict) - len(train_image_ids)
    deleted_count_test = len(test_data_dict) - len(test_image_ids)

    logging.info(f"Torolt elemek szama a train-ben: {deleted_count_train}  , Maradt : {len(train_image_ids)}")
    logging.info(f"Torolt elemek szama a test-ben:  {deleted_count_test}   , Maradt : {len(test_image_ids)}")

    return train_image_list, train_image_ids, test_image_list, test_image_ids, data_array
