import os
from PIL import Image
import re
import matplotlib.pyplot as plt


def find_and_display_original_image(input_dir, filename):
    """
    Kikeresi és megjeleníti az eredeti fájlt az input mappából,
    amely nem kezdődik számmal, és kiírja az eredeti fájl nevét.
    """
    # Szám nélküli eredeti fájlnév kivonása
    original_filename_pattern = re.sub(r'^\d+_', '', filename)

    # Input mappa fájljainak listázása
    for file in os.listdir(input_dir):
        if re.match(re.escape(original_filename_pattern), file):
            original_img_path = os.path.join(input_dir, file)

            # Kép betöltése és megjelenítése
            img = Image.open(original_img_path)
            plt.imshow(img, cmap="gray")
            plt.title(f"Eredeti kép: {file}")
            plt.axis("off")
            plt.show()

            # Eredeti fájlnév kiírása
            print(f"[INFO] Az eredeti fájl neve: {file}")
            return

    print(f"[INFO] Nem található eredeti kép a {original_filename_pattern} minta alapján.")


# Példa futtatás
input_dir = "./train_data"  # Az eredeti képek mappája
filename = "4_holodetect_heamato_2024-10-14_13-27-39.950__hlg_81_idx_19_cnt_1512_648_dst_16131.34_mask.png"

find_and_display_original_image(input_dir, filename)
