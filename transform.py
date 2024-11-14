import os
import csv
from PIL import Image
from torchvision import transforms
import re
import random
import torch


# Egyéni forgatás tükrözéssel a széleken
def rotate_with_reflect(img, degrees):
    """ Forgatás tükrözött szélkitöltéssel """
    rotated = img.rotate(degrees, resample=Image.BILINEAR, fillcolor=None)
    img_width, img_height = img.size

    # A forgatás során keletkező fekete pixelek helyettesítése tükrözött szélekkel
    for x in range(img_width):
        for y in range(img_height):
            if rotated.getpixel((x, y)) == (0, 0, 0):
                if 0 <= x < img_width // 2:
                    rotated.putpixel((x, y), img.getpixel((x * 2, y % img_height)))
                else:
                    rotated.putpixel((x, y), img.getpixel(((img_width - x - 1) * 2, y % img_height)))
    return rotated


# Transzformációk definiálása
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)  # 100% eséllyel tükröz
rotation = lambda img: rotate_with_reflect(img, 15)  # Egyéni forgatás 15 fokkal
#translation = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Véletlen eltolás
#brightness_adjust = transforms.ColorJitter(brightness=0.2)  # Fényerő változtatás

data_augmentations = [
    ("Horizontal Flip", horizontal_flip),
    ("Rotation", rotation),
    #("Translation", translation),
    #("Brightness Adjust", brightness_adjust),
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
                        augmented_img = rotate_with_reflect(img, degrees=15)
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
