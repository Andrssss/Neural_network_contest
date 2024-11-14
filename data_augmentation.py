# data_augmentation.py

import os
import pandas as pd
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def perform_data_augmentation():
    input_csv = 'data_labels_train.csv'  # Cseréld le a helyes CSV fájl elérési útra
    input_folder = 'train_data'
    output_folder = 'train_data_2'
    output_csv = 'data_labels_train_2.csv'

    # Olvasd be a CSV fájlt
    df = pd.read_csv(input_csv)
    selected_data = df[['filename_id', 'defocus_label']]

    # Ellenőrizzük, hogy az output mappa létezik-e, ha nem, hozzuk létre
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Albumentations augmentációk definiálása
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),  # Véletlenszerű kivágás
        A.HorizontalFlip(p=0.5),  # Véletlenszerű vízszintes tükrözés
        A.Rotate(limit=40, p=0.9),  # Véletlenszerű elforgatás
        A.RandomBrightnessContrast(p=0.2),  # Véletlenszerű fényerő és kontraszt módosítás
        ToTensorV2(),  # PyTorch tensorba konvertálás
    ])

    # Új adat lista létrehozása az augmented képekhez
    augmented_data = []

    # Feldolgozzuk az összes képet az input mappában
    # todo --






    # Új CSV fájl létrehozása
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_csv, index=False)

    # Visszatérünk a mentési mappa és a CSV fájl elérési útjával
    return output_folder, output_csv
