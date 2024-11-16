import os
import shutil
from PIL import Image, ImageEnhance, ImageOps
import re
import random
import torch
from collections import Counter

# Forgatás fix szögekkel
def rotate_fixed_angles(img, angle):
    return img.rotate(angle, resample=Image.BILINEAR)

# Kis forgatás random + 90, 180, 270 fok hozzáadása kisebb zoommal
def rotate_small_angle_with_offset(img, base_angle, offset_angle, zoom_factor):
    total_angle = base_angle + offset_angle
    rotated = img.rotate(total_angle, resample=Image.BILINEAR)
    width, height = rotated.size
    crop_size_w = int(width * zoom_factor)
    crop_size_h = int(height * zoom_factor)

    left = (width - crop_size_w) // 2
    top = (height - crop_size_h) // 2
    right = left + crop_size_w
    bottom = top + crop_size_h

    cropped = rotated.crop((left, top, right, bottom))
    return cropped.resize((width, height), Image.LANCZOS)

# Kontraszt módosítása
def adjust_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

# Fájlok csoportosítása közös alapnév szerint
def group_images_by_basename(input_dir):
    grouped_files = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            base_name = re.sub(r'(_amp|_phase|_mask)\.png$', '', filename)
            if base_name not in grouped_files:
                grouped_files[base_name] = []
            grouped_files[base_name].append(filename)
    return grouped_files

# Dummy2 mappa tartalmának törlése
def clear_output_directory(output_dir):
    print(f"A {output_dir} mappa tartalma törölve.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

# Augmentált képek létrehozása
def augment_and_save_images(input_dir, output_dir, num_augments=5):
    clear_output_directory(output_dir)
    grouped_files = group_images_by_basename(input_dir)
    total_images_before = sum(len(files) for files in grouped_files.values())
    total_augmented_images = 0
    transform_stats = Counter()

    print(f"\n[INFO] Felismert csoportok száma: {len(grouped_files)}")
    print(f"[INFO] Eredeti képek száma: {total_images_before}")

    for base_name, file_list in grouped_files.items():
        for augment_idx in range(1, num_augments + 1):
            random.seed(f"{base_name}_{augment_idx}")
            torch.manual_seed(int.from_bytes(f"{base_name}_{augment_idx}".encode(), 'little') % (2 ** 32))

            transform_type = random.choices(
                ["Fixed Rotation", "Small Rotation with Offset"],
                weights=[1/2, 1/2]
            )[0]

            transform_stats[transform_type] += 1

            if transform_type == "Fixed Rotation":
                angle = random.choice([90, 180, 270])
                transform = lambda img: rotate_fixed_angles(img, angle)
            else:
                base_angle = random.uniform(-15, 15)
                offset_angle = random.choice([90, 180, 270])
                zoom_factor = 0.84   # A kép középső 84%-át tartja meg
                transform = lambda img: rotate_small_angle_with_offset(img, base_angle, offset_angle, zoom_factor)

            contrast_factor = random.uniform(0.8, 1.2)

            # Tükrözési döntések egy csoporton belül azonosak lesznek
            apply_x_flip = random.random() < 0.5
            apply_y_flip = random.random() < 0.5

            for filename in file_list:
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert("RGB")

                # Alkalmazzuk az augmentációt
                augmented_img = transform(img)

                # Tengely tükrözés (azonos minden csoporttagon)
                if apply_x_flip:
                    augmented_img = ImageOps.mirror(augmented_img)  # X tengely (vízszintes)
                if apply_y_flip:
                    augmented_img = ImageOps.flip(augmented_img)  # Y tengely (függőleges)

                # Kontraszt módosítása, ha nem _mask.png
                if not filename.endswith("_mask.png"):
                    augmented_img = adjust_contrast(augmented_img, contrast_factor)

                new_filename = f"{augment_idx}_{filename}"
                save_path = os.path.join(output_dir, new_filename)
                augmented_img.save(save_path)

            total_augmented_images += len(file_list)

    # Eredeti képek átmásolása
    total_original_images = 0
    for filename in os.listdir(input_dir):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        shutil.copy(src_path, dst_path)
        total_original_images += 1

    print(f"\n[INFO] Feldolgozott képek száma az augmentáció után: {total_augmented_images} + {total_original_images}")
    print("[INFO] Augmentációk eloszlása:", dict(transform_stats))

"""
# Futtatás a megadott mappára
input_dir = "./dummy"
output_dir = "./train_data_Augmented+original"
augment_and_save_images(input_dir, output_dir, num_augments=30)
"""