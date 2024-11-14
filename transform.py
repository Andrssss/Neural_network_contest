import os
import csv
from PIL import Image
from torchvision import transforms
import re

# Transzformációk definiálása
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)  # 100% eséllyel tükröz
rotation = transforms.RandomRotation(degrees=15)  # ±15 fokos forgatás
#translation = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Véletlen eltolás
#color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Fényerő és kontraszt változás
#resized_crop = transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))  # Véletlen crop és méretezés

data_augmentations = [
    ("Horizontal Flip", horizontal_flip),
    ("Rotation", rotation),
  #  ("Translation", translation),
  #  ("Color Jitter", color_jitter),
  #  ("Resized Crop", resized_crop),
]

# Utótag eltávolító függvény
def clean_filename(filename):
    # Eltávolítja a speciális utótagokat és a kiterjesztést
    filename = os.path.splitext(filename)[0]
    filename = re.sub(r'(_amp|_phase|_mask)$', '', filename)  # Speciális utótagok eltávolítása
    return filename

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

        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_id = clean_filename(filename)  # Fájlazonosító megtisztítása
                if img_id not in labels:
                    print(f"Warning: {img_id} not found in label file.")
                    continue

                class_label, defocus_label = labels[img_id]

                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert("RGB")

                print(f"\nEredeti fájl feldolgozása: {filename}")

                for i in range(num_augments):
                    transform_name, transform = data_augmentations[i % len(data_augmentations)]
                    augmented_img = transform(img)

                    # Új fájlnév generálása
                    new_filename = f"{img_id}_{transform_name.replace(' ', '_')}_{i}.png"
                    save_path = os.path.join(output_dir, new_filename)
                    augmented_img.save(save_path)

                    # Új címke hozzáadása a CSV-hez
                    writer.writerow([new_filename, class_label, defocus_label])

                    print(f"  Transzformáció: {transform_name}, Mentés: {save_path}")

# Futtatás a megadott mappára
input_dir = "./train_data"  # Az eredeti képek mappája
output_dir = "./augmentation"  # Az augmentált képek mentési mappája
label_file = "./data_labels_train.csv"  # Eredeti címkefájl
output_label_file = "./data_labels_transformed.csv"  # Új címkefájl

augment_and_save_images(input_dir, output_dir, label_file, output_label_file)
