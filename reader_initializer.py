# reader_initializer.py
import logging
import glob
import os
import pandas as pd
import numpy as np
from skimage.transform import resize
import torch
from skimage.io import imread

#                   train_f,         valid_f,        test_f     all_label      0.1             True / False
def initialize_data(train_folder, validation_folder, test_folder, label_file, validation_ratio, kerekitsen_labeleket):
    # initialize_data("./augmentation", "validation_data", "./test_data", "./data_labels_train.csv", 0.1, True)

    logging.info(f"initialize_data() -----------------------------------")

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Kép fájlok beolvasása
    train_image_files = glob.glob(f"{train_folder}/*.png")
    validation_image_files = glob.glob(f"{validation_folder}/*.png")
    test_image_files = glob.glob(f"{test_folder}/*.png")

    # logging.info(f"Train folder contains {len(train_image_files)} images.")
    # logging.info(f"Validation folder contains {len(validation_image_files)} images.")
    # logging.info(f"Test folder contains {len(test_image_files)} images.")

    # CSV fájl beolvasás
    df = pd.read_csv(label_file)
    selected_data = df[['filename_id', 'defocus_label']]

    # ------------------------------------ Kerekítés és tartomány korlátozás (opcionális)
    if kerekitsen_labeleket:
        df['defocus_label'] = np.clip(np.round(df['defocus_label']), -10, 10)
    logging.info(df.head())
    # ------------------------------------------------------------------------------------

    data_array = selected_data.to_numpy()

    # Tárolók az adatokhoz
    train_data_dict = {}
    validation_data_dict = {}
    test_data_dict = {}

    # Képek feldolgozása és csoportosítása
    def process_images(image_files, data_dict, set_name):
        for image_path in image_files:
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            if "__" in file_name:
                id_part, type_part = file_name.rsplit('_', 1)
                if type_part in ["amp", "mask", "phase"]:
                    if id_part not in data_dict:
                        data_dict[id_part] = {'amp': None, 'mask': None, 'phase': None}
                    if data_dict[id_part][type_part] is None:
                        img = imread(image_path, as_gray=True)
                        if img.shape != (128, 128):
                            img = resize(img, (128, 128), anti_aliasing=True)
                        img = np.expand_dims(img, axis=-1)
                        data_dict[id_part][type_part] = img
        # logging.info(f"{set_name}: Processed {len(data_dict)} unique groups.")

    process_images(train_image_files, train_data_dict, "Train")
    if validation_ratio != 0:
        process_images(validation_image_files, validation_data_dict, "Validation")
    process_images(test_image_files, test_data_dict, "Test")

    # Adatok numpy tömbbe konvertálása
    def convert_to_arrays(data_dict, set_name):
        image_list = []
        image_ids = []
        for id_key, img_types in data_dict.items():
            if all(img_types[k] is not None for k in ['amp', 'mask', 'phase']):
                image_stack = np.concatenate([img_types['amp'], img_types['mask'], img_types['phase']], axis=-1)
                image_list.append(image_stack)
                image_ids.append(id_key)
            else:
                logging.warning(f"{set_name}: Missing images for ID {id_key}: {img_types}")
        return image_list, image_ids

    train_image_list, train_image_ids = convert_to_arrays(train_data_dict, "Train")
    if validation_ratio != 0:
        validation_image_list, validation_image_ids = convert_to_arrays(validation_data_dict, "Validation")
    test_image_list, test_image_ids = convert_to_arrays(test_data_dict, "Test")

    # Összegző kiíratás
    logging.info(f"Train set:      {len(train_image_list)*3} images, {len(train_image_ids)} group.")
    logging.info(f"Validation set: {len(validation_image_list)*3}  images, {len(validation_image_ids)} group.")
    logging.info(f"Test set:       {len(test_image_list)*3}  images, {len(test_image_ids)} group.")

    if validation_ratio == 0:
        validation_image_list = []
        validation_image_ids = []

    return train_image_list, train_image_ids, validation_image_list, validation_image_ids, test_image_list, test_image_ids, data_array
