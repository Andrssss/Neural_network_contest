from torch.optim import Adam

# EZEN A VERZIÓN - mask nélkül



validation_ratio     = 0.05   # 0.1 -> 10%, ha ezen változtatni szeretnél, akkor az alatta lévőt tedd TRUE-ra, első körben
hozzon_letre_uj_augmentalt_fileokat_e = False   # külön is futtatható
Augmentation_number    = 0
kerekitsen_labeleket = False
patience = 300
tolerance = 0.2

# Érdemes növekvősorrendbe rakni az olyan tanításokat, amiknél csak epoch külömböző.
configurations = [
    (300, 8, 1, "MobileNetV2Custom_2"),
    (300, 16, 1, "MobileNetV2Custom_2"),
    (300, 32, 1, "MobileNetV2Custom_2"),
    (300, 64, 1, "MobileNetV2Custom_2"),

    (300, 8, 1, "MobileNetV2Custom"),
    (300, 16, 1, "MobileNetV2Custom"),
    (300, 32, 1, "MobileNetV2Custom"),
    (300, 64, 1, "MobileNetV2Custom"),

    (300, 8, 1, "MobileNetV2Custom_3"),
    (300, 16, 1, "MobileNetV2Custom_3"),
    (300, 32, 1, "MobileNetV2Custom_3"),
    (300, 64, 1, "MobileNetV2Custom_3"),

    (300, 8, 1, "MobileNetV2Custom_4"),
    (300, 16, 1, "MobileNetV2Custom_4"),
    (300, 32, 1, "MobileNetV2Custom_4"),
    (300, 64, 1, "MobileNetV2Custom_4"),
]



# num_epochs       = 50
# train_batch_size = 8
# fel_le_kerekit   = 1     # ennek még nincs funkciója, butaság
# model_neve       = "MobileNetV2Custom"

# MobileNetV2Custom     - gyors, de 20% pontosság max,eddig
# ResNet34Custom        -
# EfficientNetB0Custom  - kurva lassú betanulás ( laptopomon esélytelen )
# SwinTransformerCustom - picsog
# ConvNeXtCustom        - kurva lassú betanulás, meg jó szar is. Logban van mérés.
# AlexNet               - megnezem mennyire pontos



# ----------------------------------------------------------------------------------------  DATA AUGMENTATION




# --------------------------------------   INICIALIZÁLÁS   -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
import csv
import logging
import os

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import MobileNetV2Custom
from evaluate_and_export import evaluate_model
# Logger létrehozás -------------------------------------
from logger import setup_logger
setup_logger()
logging.info(f"Validation_ratio : {validation_ratio}")
logging.info(f"Augmentation_number : {Augmentation_number}")

# DATA AUGMENTATION ------------------------------------- vad verzió
source_folder = "./train_data"
output_dir = "./train_data_Augmented+original"
original_csv_file = label_file = "./data_labels_train.csv"
test_folder = "./test_data"
train_folder = "train_data_2"
validation_folder = "validation_data"

def ensure_folder_and_process(folder_path, callback):
    """
    Ellenőrzi, hogy a mappa létezik-e és nem üres.
    Ha nem létezik vagy üres, létrehozza és meghívja a callback függvényt.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"A mappa létrehozva: {folder_path}")
        callback()
    elif not os.listdir(folder_path):  # Mappa létezik, de üres
        print(f"A mappa létezik, de üres: {folder_path}")
        callback()

def process_and_augment_data():
    from transform_2 import process_data
    # Szétválasztja az eredeti képeket, train és validációs mappába
    process_data(source_folder, train_folder, validation_folder, validation_ratio)
    from transform import augment_and_save_images
    # A széválasztott adatok közűl a train adatokat fogja Augmentálni.
    augment_and_save_images(train_folder, output_dir, num_augments=Augmentation_number)

if hozzon_letre_uj_augmentalt_fileokat_e :
    process_and_augment_data()

# Mappák ellenőrzése és a callback meghívása, ha szükséges
ensure_folder_and_process(output_dir, process_and_augment_data)
ensure_folder_and_process(train_folder, process_and_augment_data)
ensure_folder_and_process(validation_folder, process_and_augment_data)




# Fileok beolvasása -------------------------------------
from reader_initializer import initialize_data
# ö megkapja a :  initialize_data("./augmentation", "validation_data", "./test_data", "./data_labels_train.csv", 0.1, True) és ezekből csinál tenszort
train_image_list, train_image_ids, validate_image_list, validate_image_ids, test_image_list, test_image_ids, data_array = initialize_data(output_dir,validation_folder,test_folder,label_file,validation_ratio,kerekitsen_labeleket)




# GPU beállítása ha lehet -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
print(f"Graphic card: {torch.cuda.is_available()}")




test_images = np.array(test_image_list)
train_images = np.array(train_image_list)
validate_images = np.array(validate_image_list) if validate_image_list else None







# ----------------------------------------------------------------------------------------------------------------------
# Kép normalizálás: skálázás 0-1 közé ---> ezt meg kell nézni, h kell-e. Mert amúgy elég erős lehet.
# train_images = train_images / 255.0
# test_images = test_images / 255.0   test_images,train_images
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Átlag és szórás kiszámolása mindkét halmazra
train_mean = train_images.mean()
train_std = train_images.std()
test_mean = test_images.mean()
test_std = test_images.std()
validate_mean = validate_images.mean() if validate_images is not None else train_mean
validate_std = validate_images.std() if validate_images is not None else train_std



# Alkalmazzuk a transzformációkat: Normalizálás mindkét adathalmazra
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std]) ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[test_mean], std=[test_std])   ])
transform_validate = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[validate_mean], std=[validate_std])  ])


train_image_tensors = []
for img in train_images:
    transformed_img = transform_train(img).float()  # Konvertálás float32-re
    train_image_tensors.append(transformed_img)

train_image_tensors = torch.stack(train_image_tensors) # [db, type, x, y]
print(f"Train tensor dtype: {train_image_tensors.dtype}")


test_image_tensors = []
for img in test_images:
    transformed_img = transform_test(img).float()  # float32 biztosítás
    test_image_tensors.append(transformed_img)
test_image_tensors = torch.stack(test_image_tensors)
logging.info(f"test_image_tensors : {test_image_tensors.shape}")  # [db, type, x, y]

if validate_images is not None:
    validate_image_tensors = []
    for img in validate_images:
        transformed_img = transform_validate(img).float()  # float32 biztosítás
        validate_image_tensors.append(transformed_img)
    validate_image_tensors = torch.stack(validate_image_tensors)  # [count, channels, x, y]
else :
    validate_image_tensors = None
    validate_image_ids = None











# --------------------------------------   ALAP BEALLÍTÁS  -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# erre azért van szükség, mert CrossEntrophy 0-vmennyi számokat vár
logging.info(f"Eredeti címke: {data_array[:, 1]}")
unique_labels  = np.unique(data_array[:, 1])  # Eredeti címkék
label_map      = {label: idx for idx, label in enumerate(unique_labels)}  # Mappeljük 0-tól kezdve
# mapped_labels  = np.array([label_map[label] for label in data_array[:, 1]])
# data_array címkék frissítése a mapped_labels segítségével
# data_array[:, 1] = mapped_labels
# logging.info(f"Átalakított címke: {data_array[:, 1]}")




"""
def compute_class_weights(labels, num_classes):
        # A címkék előfordulásának megszámolása
        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        for label in labels:
            class_counts[label] += 1

        # Súlyok számítása (az osztályok relatív ritkasága alapján)
        total_samples = len(labels)
        class_weights = total_samples / (num_classes * class_counts)

        return class_weights

labels = data_array[:, 1].astype(int).tolist()
print(labels)
num_classes = 21
weights = compute_class_weights(labels, num_classes)
print("Osztálysúlyok:", weights)
scaled_weights = torch.log1p(weights)
"""




# --------------------------------------   BETANITAS  ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Augmentáció miatt, lecsícsi az elejéről a számot és így kereshető az ID
def normalize_id(img_id):
    parts = img_id.split('_', 1)  # Az első '_' után osztjuk ketté
    if parts[0].isdigit():  # Csak akkor vágjuk le, ha az első rész szám
        return parts[1]
    return img_id  # Ha nincs prefix, az eredeti ID-t adjuk vissza

def calculate_rms(model, test_loader, device):
    model.eval()
    mse_loss = 0.0
    count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            mse_loss += ((outputs - labels) ** 2).sum().item()
            count += labels.size(0)

    rms = (mse_loss / count) ** 0.5
    return rms




# Egyedi dataset osztály
class CustomImageDataset(Dataset):
    def __init__(self, images, image_ids, data_array, transform=None):
        self.images = images
        self.image_ids = image_ids
        # Az ID-k normalizálása a data_dict létrehozásakor
        self.data_dict = {normalize_id(row[0]): row[1] for row in data_array}
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_id = normalize_id(self.image_ids[idx])  # Normalizált ID használata
        label = self.data_dict[img_id]  # Címke hozzárendelése a normalizált ID alapján

        if self.transform:
            img = self.transform(img)

        return img, label


from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from torch_lr_finder import LRFinder

previous_config = None
t_loss_min = 99
max_acc = 0
val_accuracy = 0.0

# EARLY STOPPING
best_val_loss = float('inf')  # Legjobb validációs veszteség
early_stopping_counter = 0  # Megszakítás számláló

for num_epochs, train_batch_size, fel_le_kerekit, model_neve in configurations:


    cur_acc = 0.0
    if (previous_config != None and num_epochs > previous_config[0] and train_batch_size == previous_config[1] and fel_le_kerekit == previous_config[2] and model_neve == previous_config[3]):
        start_epoch = previous_config[0]  # Folytatás az aktuális num_epochs értéktől
    else:
        early_stopping_counter = 0
        t_loss_min = 99
        max_acc = 0
        val_accuracy = 0.0
        # Újrainicializáljuk az adatokat és a modellt
        # dataset = CustomImageDataset(  images=train_image_tensors,  data_array=data_array, ids=train_image_ids, transform=transform_train)
        dataset = CustomImageDataset(images=train_image_tensors, image_ids=train_image_ids, data_array=data_array)
        train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        # Validation dataset (if available)
        if validate_image_tensors is not None:
            #val_dataset = CustomImageDataset(images=validate_image_tensors, data_array=data_array, ids=validate_image_ids,   transform=transform_validate)
            val_dataset = CustomImageDataset(images=validate_image_tensors, image_ids=validate_image_ids,data_array=data_array)
            val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)


        # todo - ez mi a faszom, 21 kategória kell nem 4000
        num_classes = len(np.unique(data_array[:, 1]))

        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        if  model_neve == "MobileNetV2Custom":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
            criterion = nn.MSELoss()
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            logging.info("scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")
            logging.info("criterion = nn.MSELoss()")


        elif model_neve == "MobileNetV2Custom_2":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(1).to(device)  # Ha támogatott
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
            criterion = nn.MSELoss()
            logging.info("optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)")
            logging.info("scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)")
            logging.info("criterion = nn.MSELoss()")


        elif model_neve == "MobileNetV2Custom_3":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(1).to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")

            # LR Finder futtatása
            logging.info("Starting LR Finder...")
            criterion = nn.MSELoss()
            lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
            lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
            lr_finder.plot()  # Az optimális tanulási ráta grafikon megjelenítése
            lr_finder.reset()  # Modell és optimizer visszaállítása
            # Miután az LR Finder segítségével megtaláltad az optimális tanulási rátát, állítsd be
            optimal_lr = 1e-3  # Példa: ezt a grafikon alapján válaszd ki
            optimizer = optim.Adam(model.parameters(), lr=optimal_lr, weight_decay=1e-4)
            # CyclicLR Scheduler
            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
            logging.info( "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')" )

        elif model_neve == "MobileNetV2Custom_4":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(1).to(device)  # Ha támogatott
            optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            criterion = nn.MSELoss()
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)")
            logging.info("scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)")
            logging.info("criterion = nn.MSELoss()")





        else:
            raise ValueError(f"Hibás/nem létező modell név: {model_neve}")

        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using device:", device)
        model = model.to(dev)
        start_epoch = 0  # Újratöltés esetén a ciklus kezdőértéke

        output_file = 'log.csv'
        with open(output_file, mode='a') as file:
            file.write('\n')

    logging.info(f" [{start_epoch} - {num_epochs}] - {model_neve} - B={train_batch_size}")
    train_loss_history = []
    best_val_accuracy = 9999999.0  # Legjobb validációs pontosság nyilvántartása
     # Megengedett eltérés a helyes predikcióhoz

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0  # Helyes előrejelzések száma a train adatokon
        total_train = 0  # Teljes train minta száma

        mse_train = 0.0  # MSE metrika az edzéshez

        for train_images, labels in train_loader:
            train_images, labels = train_images.to(device), labels.to(device).float()
            optimizer.zero_grad()

            # Modell előrejelzés
            outputs = model(train_images)
            outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
            outputs = outputs.unsqueeze(0) if outputs.dim() == 0 else outputs
            labels = labels.squeeze(-1) if labels.dim() > 1 else labels
            labels = labels.unsqueeze(0) if labels.dim() == 0 else labels

            # Veszteség számítása
            assert outputs.shape == labels.shape, f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}"
            loss = criterion(outputs, labels)

            # Gradiens visszaterjesztés és lépés
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            mse_train += torch.mean((outputs - labels) ** 2).item()  # Négyzetes hiba hozzáadása

            # Pontosság számítása toleranciával
            correct_train += torch.sum(torch.abs(outputs - labels) < tolerance).item()
            total_train += labels.size(0)

        train_loss /= len(train_loader)
        mse_train /= len(train_loader)
        rmse_train = mse_train ** 0.5  # RMSE kiszámítása
        train_accuracy = 100. * correct_train / total_train  # Pontosság százalékban

        train_loss_history.append(train_loss)

        if len(train_loss_history) >= 10 and all(
                train_loss_history[i] < train_loss_history[i + 1] for i in range(-10, -1)):
            logging.warning("Train loss folyamatosan növekszik az utolsó 30 epochban. Következő konfiguráció...")
            break

        # Validáció
        if validate_image_tensors is not None:
            model.eval()
            val_loss = 0.0
            mse_val = 0.0  # MSE metrika a validációhoz
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for test_images, test_ids in zip(validate_image_tensors, validate_image_ids):
                    test_images = test_images.unsqueeze(0).to(device)

                    # Modell előrejelzés
                    outputs = model(test_images)
                    outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
                    outputs = outputs.unsqueeze(0) if outputs.dim() == 0 else outputs

                    # Validációs címkék előkészítése
                    original_label = data_array[data_array[:, 0] == test_ids, 1].item()
                    original_label_tensor = torch.tensor([original_label], dtype=torch.float).to(device)
                    original_label_tensor = original_label_tensor.unsqueeze(
                        0) if original_label_tensor.dim() == 0 else original_label_tensor

                    # Veszteség számítása
                    loss = criterion(outputs, original_label_tensor)
                    val_loss += loss.item()
                    mse_val += torch.mean((outputs - original_label_tensor) ** 2).item()

                    # Pontosság validációs toleranciával
                    if torch.abs(outputs - original_label_tensor) < tolerance:
                        correct_val += 1
                    total_val += 1

            val_loss /= len(validate_image_ids)
            mse_val /= len(validate_image_ids)
            rmse_val = mse_val ** 0.5  # RMSE kiszámítása
            val_accuracy = 100. * correct_val / total_val  # Validációs pontosság százalékban

            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Train RMSE: {rmse_train:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}, "
                f"Val RMSE: {rmse_val:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

            scheduler.step()  # Lépés a tanulási rátával
        else:
            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Train RMSE: {rmse_train:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early Stopping és LR csökkentés
        if train_loss < best_val_accuracy :
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0  # Reset counter
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logging.info("Early stopping triggered.")
                break

        if val_accuracy > 0.7 and scheduler.get_last_lr()[0] > 1e-4:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5  # Tanulási ráta csökkentése

        if 2 > rmse_train and val_accuracy >30 :
            evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size,
                           val_accuracy, 0, model_neve, t_loss_min, cur_acc)
            evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size,
                           val_accuracy, 1, model_neve, t_loss_min, cur_acc)
            # Modell értékelése és kiiratás

    # Loggolás, mentés és egyéb műveletek
    output_file = 'log.csv'
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model Name', 'Epochs', 'Batch Size', 'Validation Accuracy'])  # Fejléc
        writer.writerow([model_neve, num_epochs, train_batch_size, cur_acc])  # Modell név is bekerül

    # Előző epoch értékének frissítése
    previous_config = (num_epochs, train_batch_size, fel_le_kerekit, model_neve)
