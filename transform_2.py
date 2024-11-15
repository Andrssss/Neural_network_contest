import shutil
import os
import re
import random
import pandas as pd


def process_data(source_folder, train_folder, validation_folder, original_csv_file, train_csv_path, validation_csv_path,  validation_rate):
    """
    Átmásolja a fájlokat és szétosztja őket tanító és validációs adatokra.
    Frissíti a CSV-ket is, és kiírja az egyes mappákba került fájlok és sorok számát.
    """
    # Másoljuk át az eredeti CSV tartalmát az új fájlokba
    shutil.copy(original_csv_file, train_csv_path)
    shutil.copy(original_csv_file, validation_csv_path)
    print(f"A {original_csv_file} tartalma átmásolva a {train_csv_path} és {validation_csv_path} fájlokba.")

    # Eredeti mappa fájljainak számolása
    original_files = [file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file))]
    print(f"A forrás mappában ({source_folder}) eredetileg {len(original_files)} fájl található.")

    # Cél mappák tartalmának törlése és újrateremtése
    for folder in [train_folder, validation_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"A {folder} mappa tartalma törölve.")
        os.makedirs(folder, exist_ok=True)

    # Fájlok csoportosítása az alapnév alapján
    def group_files_by_name(input_dir):
        grouped_files = {}
        for filename in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, filename)):
                base_name = re.sub(r'(_amp|_phase|_mask)?\.(jpg|png)$', '', filename)
                if base_name not in grouped_files:
                    grouped_files[base_name] = []
                grouped_files[base_name].append(filename)
        return grouped_files

    # Csoportok szétosztása
    def split_groups(grouped_files, input_dir, train_dir, validation_dir, train_csv, validation_csv):
        df = pd.read_csv(original_csv_file)

        group_names = list(grouped_files.keys())
        random.seed(None)  # Véletlenszerűség biztosítása

        validation_group_count = int(len(group_names) * validation_rate)
        validation_groups = set(random.sample(group_names, validation_group_count))
        train_groups = set(group_names) - validation_groups

        train_files_count = 0
        validation_files_count = 0

        for group, file_list in grouped_files.items():
            target_folder = validation_dir if group in validation_groups else train_dir
            for file in file_list:
                shutil.copy(os.path.join(input_dir, file), os.path.join(target_folder, file))
                # print(f"{file} átmásolva a {target_folder} mappába.")
                if target_folder == train_dir:
                    train_files_count += 1
                else:
                    validation_files_count += 1

        # Frissítjük a CSV fájlokat
        train_df = df[df['filename_id'].apply(lambda x: any(group in x for group in train_groups))]
        validation_df = df[df['filename_id'].apply(lambda x: any(group in x for group in validation_groups))]

        train_df.to_csv(train_csv, index=False)
        validation_df.to_csv(validation_csv, index=False)

        print(f"Train CSV ({train_csv}) frissítve, {len(train_df)} sor maradt.")
        print(f"Validation CSV ({validation_csv}) frissítve, {len(validation_df)} sor maradt.")

        # Kiírjuk az adatok számát
        print(f"\nÖsszegzés:")
        print(f"{train_files_count} fájl került a tanító mappába ({train_dir}).")
        print(f"{validation_files_count} fájl került a validációs mappába ({validation_dir}).")
        print(f"{len(train_df)} sor található a tanító CSV-ben.")
        print(f"{len(validation_df)} sor található a validációs CSV-ben.")

    # Fő logika
    grouped_files = group_files_by_name(source_folder)
    split_groups(grouped_files, source_folder, train_folder, validation_folder, train_csv_path, validation_csv_path)


"""
process_data(
    train_folder="train_data_2",
    validation_folder="validation_data",
    train_csv_path="data_labels_train_2.csv",
    validation_csv_path="validation_data.csv",
    source_folder='./train_data',
    original_csv_file = './data_labels_train.csv',
    validation_rate=0.2  # A validációs adatok aránya
)
"""