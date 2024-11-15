import os
import csv
from PIL import Image
from torchvision import transforms
import re
import random
import torch

# Egyéni forgatás, amely 90, 180 vagy 270 fokos
def rotate_fixed_angles(img):
    """ Forgatás 90, 180 vagy 270 fokkal, véletlenszerűen """
    angle = random.choice([90, 180, 270])
    rotated = img.rotate(angle, resample=Image.BILINEAR)
    return rotated

# Transzformációk definiálása
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)  # 100% eséllyel tükröz
rotation = rotate_fixed_angles  # Rögzített szögek szerinti forgatás

data_augmentations = [
    ("Horizontal Flip", horizontal_flip),
    ("Rotation", rotation),
]

# Fájlok csoportosítása közös alapnév szerint
def group_images_by_basename(input_dir):
    grouped_files = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Alapnév kinyerése az "_amp", "_mask", "_phase" eltávolításával
            base_name = re.sub(r'(_amp|_phase|_mask)\.png$', '', filename)
            if base_name not in grouped_files:
                grouped_files[base_name] = []
            grouped_files[base_name].append(filename)
    return grouped_files

# Augmentált képek létrehozása és címke mentése
def augment_and_save_images(input_dir, output_dir, label_file, output_label_file, num_augments=5):
    # Címkék beolvasása
    labels = {}
    with open(label_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Fejléc átugrása
        for row in reader:
            image_id, class_label, defocus_label = row
            labels[image_id] = (class_label, defocus_label)

    # Output mappa létrehozása, ha nem létezik
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Új címkefájl megnyitása írásra
    with open(output_label_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename_id', 'class_label', 'defocus_label'])  # Fejléc írása

        # Képek csoportosítása közös alapnév szerint
        grouped_files = group_images_by_basename(input_dir)

        for base_name, file_list in grouped_files.items():
            for augment_idx in range(1, num_augments + 1):  # Több transzformált változat
                # Véletlenszerű transzformáció kiválasztása
                random.seed(f"{base_name}_{augment_idx}")
                torch.manual_seed(int.from_bytes(f"{base_name}_{augment_idx}".encode(), 'little') % (2 ** 32))
                transform_name, transform = data_augmentations[random.randint(0, len(data_augmentations) - 1)]

                print(f"\nFeldolgozás csoport: {base_name} (Transzformáció: {transform_name})")

                for filename in file_list:
                    img_path = os.path.join(input_dir, filename)
                    img = Image.open(img_path).convert("RGB")

                    # Transzformáció alkalmazása
                    if transform_name == "Rotation":
                        augmented_img = rotate_fixed_angles(img)
                    else:
                        augmented_img = transform(img)

                    # Új fájlnév generálása: augmentációs szám + eredeti fájlnév
                    new_filename = f"{augment_idx}_{filename}"
                    save_path = os.path.join(output_dir, new_filename)
                    augmented_img.save(save_path)

                # Egy bejegyzés készítése a CSV-hez az egész csoporthoz
                if base_name in labels:
                    class_label, defocus_label = labels[base_name]
                    writer.writerow([f"{augment_idx}_{base_name}", class_label, defocus_label])

                print(f"  Mentett fájlok: {[f'{augment_idx}_{fname}' for fname in file_list]}")

# Fájlazonosító tisztító függvény
def clean_filename(filename):
    filename = os.path.splitext(filename)[0]
    filename = re.sub(r'(_amp|_phase|_mask)$', '', filename)
    return filename

# Futtatás a megadott mappára
input_dir = "./train_data"  # Az eredeti képek mappája
output_dir = "./augmentation"  # Az augmentált képek mentési mappája
label_file = "./data_labels_train.csv"  # Eredeti címkefájl
output_label_file = "./data_labels_transformed.csv"  # Új címkefájl

augment_and_save_images(input_dir, output_dir, label_file, output_label_file, num_augments=5)
