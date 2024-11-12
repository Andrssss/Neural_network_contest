
# ITT lehet változtatni, hogy milyen konfigurációkkal fusson le egymás után.
# Ez azért sexy, mert teljesen autómata, mind a log, mind a file létrehozása a result mappába.
# Érdemes növekvősorrendbe rakni az olyan tanításokat, amiknél csak epoch külömböző

configurations = [
    (50, 8, 1, "SwinTransformerCustom"),
    (55, 8, 1, "SwinTransformerCustom"),
    (60, 8, 1, "SwinTransformerCustom"),
    (65, 8, 1, "SwinTransformerCustom"),
    (70, 8, 1, "SwinTransformerCustom"),
    (75, 8, 1, "SwinTransformerCustom"),
    (80, 8, 1, "SwinTransformerCustom"),
    (85, 8, 1, "SwinTransformerCustom"),
    (90, 8, 1, "SwinTransformerCustom"),
    (95, 8, 1, "SwinTransformerCustom"),

    (50, 8, 1, "ConvNeXtCustom"),
    (55, 8, 1, "ConvNeXtCustom"),
    (60, 8, 1, "ConvNeXtCustom"),
    (65, 8, 1, "ConvNeXtCustom"),
    (70, 8, 1, "ConvNeXtCustom"),
    (75, 8, 1, "ConvNeXtCustom"),
    (80, 8, 1, "ConvNeXtCustom"),
    (85, 8, 1, "ConvNeXtCustom"),
    (90, 8, 1, "ConvNeXtCustom"),
    (95, 8, 1, "ConvNeXtCustom"),
]

validation_ratio = 0.1
# 0.0 -> nincs validáció
# 0.1 -> 10%


# num_epochs = 50
# train_batch_size = 8
# fel_le_kerekit = 1     # ennek most nincs funkciója, butaság
# model_neve = "MobileNetV2Custom"

# MobileNetV2Custom -
# ResNet34Custom -
# EfficientNetB0Custom - kurva lassú betanulás
# SwinTransformerCustom -
# ConvNeXtCustom -


# --------------------------------------   INICIALIZÁLÁS   -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import glob
import csv
import os
import io
import logging
import sys
import codecs

from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from torchvision import transforms
from model import MobileNetV2Custom, ResNet34Custom, EfficientNetB0Custom, SwinTransformerCustom, ConvNeXtCustom
from evaluate_and_export import evaluate_model



# UTF-8 kimenet biztosítása konzolhoz
class Utf8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream=codecs.getwriter("utf-8")(stream.buffer))

# Logger konfigurálása fájlba és konzolra történő íráshoz
date_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_dir = "log"  # A mappa neve
log_file_path = f"{log_dir}/log_{date_str}.txt"

# Log konfiguráció (fájl és konzol kezelése)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),  # Fájl log
        Utf8StreamHandler()  # Konzol log
    ]
)


# Seed beállítása a reprodukálhatósághoz
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.info(f"Validation_ratio : {validation_ratio}")


train_image_files = glob.glob('../Neural_network_contest/train_data/*.png')
test_image_files = glob.glob('../Neural_network_contest/test_data/*.png')

# Tárolók az adatokhoz
train_data_dict = {}
test_data_dict = {}


# Fájlok beolvasása és csoportosítása ID és típus alapján
for image_path in train_image_files:
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    if "__" in file_name:
        id_part = file_name.rsplit('_', 1)[0]  # itt tárolja az ID-t
        type_part = file_name.split('_')[-1]  # itt tárolja a típusát : maszk, fázis , amplitúdó
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
        test_id_part = file_name.rsplit('_', 1)[0]  # ID-t tárolja
        test_type_part = file_name.split('_')[-1]   # Típust tárolja: maszk, fázis, amplitúdó
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
    if test_type_part['amp'] is not None and test_type_part['mask'] is not None and test_type_part['phase'] is not None:
        image_stack = np.stack([test_type_part['amp'], test_type_part['mask'], test_type_part['phase']], axis=-1)
        test_image_list.append(image_stack)
        test_image_ids.append(test_id_part)






# Set validation ratio (0.1 for 90-10 split; set to 0.0 for 100-0 split)
if validation_ratio > 0.0:
    train_image_list, validate_image_list, train_image_ids, validate_image_ids = train_test_split(
        train_image_list, train_image_ids, test_size=validation_ratio, random_state=42 )
else:
    validate_image_list = []
    validate_image_ids = []


test_images = np.array(test_image_list)
train_images = np.array(train_image_list)
validate_images = np.array(validate_image_list) if validate_image_list else None



# Számláló a törölt elemekhez ---> KITÖRLI AMIBEN NINCS : phase + amp + mask  !!!!!!!!!!!!!!!!!
deleted_count_train = 0
deleted_count_test = 0

# Az ID-k másolata az iterációhoz, hogy ne módosítsuk közben az eredeti dictionary-t
for id_key in list(train_data_dict.keys()):
    img_types = train_data_dict[id_key]
    if img_types['amp'] is None or img_types['mask'] is None or img_types['phase'] is None:
        del train_data_dict[id_key]
        deleted_count_train += 1

for test_id_part in list(test_data_dict.keys()):
    img_types = test_data_dict[test_id_part]
    if test_type_part['amp'] is None or test_type_part['mask'] is None or test_type_part['phase'] is None:
        del test_data_dict[test_id_part]
        deleted_count_test += 1

logging.info(f"Torolt elemek szama a train-ben: {deleted_count_train}")
logging.info(f"Torolt elemek szama a test-ben: {deleted_count_test}")
logging.info(f"Maradek teljes elemek szama a train-ben: {len(train_data_dict)}")
logging.info(f"Maradek teljes elemek szama a test-ben: {len(test_data_dict)}")
# ----------------------------------------------------------------------------------------------------------------------
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
    transformed_img = transform_train(img)
    train_image_tensors.append(transformed_img)
train_image_tensors = torch.stack(train_image_tensors)
logging.info(f"train_image_tensors : {train_image_tensors.shape}")  # [db, type, x, y]

test_image_tensors = []
for img in test_images:
    transformed_img = transform_test(img)
    test_image_tensors.append(transformed_img)
test_image_tensors = torch.stack(test_image_tensors)
logging.info(f"test_image_tensors : {test_image_tensors.shape}")  # [db, type, x, y]

if validate_images is not None:
    validate_image_tensors = []
    for img in validate_images:
        transformed_img = transform_validate(img)
        validate_image_tensors.append(transformed_img)
    validate_image_tensors = torch.stack(validate_image_tensors)
    logging.info(f"validate_image_tensors : {validate_image_tensors.shape}")  # [count, channels, x, y]



# EXCEL BEOLVASÁS --------------------------------------------------------------
file_path = 'data_labels_train.csv'
df = pd.read_csv(file_path)
selected_data = df[['filename_id', 'defocus_label']]
data_array = selected_data.to_numpy()
logging.info(f"data_array.shape : {data_array.shape}")






# --------------------------------------   Data augmentation   ---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------





# --------------------------------------   ALAP BEALLÍTÁS  -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
logging.info(f"Eredeti címke: {data_array[:, 1]}")
# erre azért van szükség, mert CrossEntrophy 0-vmennyi számokat vár
unique_labels = np.unique(data_array[:, 1])
label_map = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_map[label] for label in data_array[:, 1]])
# data_array címkék frissítése a mapped_labels segítségével
data_array[:, 1] = mapped_labels
logging.info(f"Átalakított címke: {data_array[:, 1]}")










# --------------------------------------   BETANITAS  ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Egyedi dataset osztály
class CustomImageDataset(Dataset):
    def __init__(self, images, image_ids, data_array, transform=None):
        self.images = images
        self.image_ids = image_ids
        self.data_dict = {row[0]: row[1] for row in data_array}  # ID-k és címkék
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_id = self.image_ids[idx]
        label = self.data_dict[img_id]  # címke hozzárendelése az ID alapján

        if self.transform:
            img = self.transform(img)

        return img, label




"""
from torch.utils.data import DataLoader, random_split
validation_split_ratio = 0.2  # 10%-os validáció
train_size = int((1 - validation_split_ratio) * len(train_image_ids))
val_size = len(train_image_ids) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
"""
# Dataset betöltés




previous_config = None
t_loss_min = 99
max_acc = 0
val_accuracy = 0.0


for num_epochs, train_batch_size, fel_le_kerekit, model_neve in configurations:

    if (previous_config != None and num_epochs > previous_config[0] and train_batch_size == previous_config[1] and fel_le_kerekit == previous_config[2] and model_neve == previous_config[3]):
        start_epoch = previous_config[0]  # Folytatás az aktuális num_epochs értéktől
    else:
        t_loss_min = 99
        max_acc = 0
        val_accuracy = 0.0
        # Újrainicializáljuk az adatokat és a modellt
        dataset = CustomImageDataset(images=train_image_tensors, image_ids=train_image_ids, data_array=data_array)
        train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        # Validation dataset (if available)
        if validate_image_tensors is not None:
            val_dataset = CustomImageDataset(images=validate_image_tensors, image_ids=validate_image_ids,data_array=data_array)
            val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)

        num_classes = len(np.unique(data_array[:, 1]))

        if model_neve == "EfficientNetB0Custom":
            model = EfficientNetB0Custom(num_classes=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
            criterion = nn.MSELoss()  # Regresszióhoz megfelelő

        elif model_neve == "MobileNetV2Custom":
            model = MobileNetV2Custom(num_classes=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.MSELoss()

        elif model_neve == "ResNet34Custom":
            model = ResNet34Custom(num_classes=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.MSELoss()

        elif model_neve == "SwinTransformerCustom":
            model = SwinTransformerCustom(num_classes=num_classes)
            optimizer = optim.AdamW(model.parameters(), lr=0.0005)  # AdamW optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,
                                                             eta_min=1e-6)  # Speciális LR csökkentés
            criterion = nn.MSELoss()

        elif model_neve == "ConvNeXtCustom":
            model = ConvNeXtCustom(num_classes=num_classes)
            optimizer = optim.AdamW(model.parameters(), lr=0.0005)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
            criterion = nn.MSELoss()

        else:
            raise ValueError(f"Hibás/nem létező modell név: {model_neve}")

        #model = MobileNetV2Custom(num_classes=num_classes)#-------------------------------------------
        # criterion = nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        start_epoch = 0  # Újratöltés esetén a ciklus kezdőértéke
    logging.info(f" [{start_epoch} - {num_epochs}] - {model_neve}")
    # Epoch ciklus a megadott start_epoch-tól num_epochs-ig
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for train_images, labels in train_loader:
            train_images, labels = train_images.to(dev), labels.to(dev).long()
            optimizer.zero_grad()
            outputs = model(train_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if train_loss < t_loss_min:
            t_loss_min = train_loss
        scheduler.step()

        # Validation evaluation
        # Validation evaluation
        if validate_image_tensors is not None:
            model.eval()  # Váltás kiértékelési módba

            results = []
            reverse_label_map = {idx: label for label, idx in label_map.items()}  # Címkék visszafejtése
            correct_count = 0  # Egyezések számlálása
            total_count = 0  # Összes validációs adat száma

            with torch.no_grad():  # Gradiensek nem szükségesek kiértékelés során
                for test_images, test_ids in zip(validate_image_tensors, validate_image_ids):
                    test_images = test_images.unsqueeze(0).to(dev)

                    outputs = model(test_images)
                    _, predicted = torch.max(outputs, 1)

                    predicted_label = reverse_label_map[predicted.item()]
                    predicted_label = int(abs(float(predicted_label)))

                    # Az eredeti címke kinyerése az ID alapján
                    original_label = reverse_label_map[data_array[data_array[:, 0] == test_ids, 1].item()]
                    original_label = int(abs(float(original_label)))

                    # Összehasonlítás és számlálás
                    #print(f"Predictions: {predicted_label} / Original: {original_label}")
                    if predicted_label == original_label:
                        correct_count += 1
                    total_count += 1

                    results.append([test_ids, predicted_label, original_label])

            # Pontosság kiszámítása
            # print(f"Correct Predictions: {original_label} / Total: {predicted_label}")
            val_accuracy = correct_count / total_count
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        else:
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        if max_acc < val_accuracy: max_acc = val_accuracy

    # Loggolás, mentés és egyéb műveletek
    output_file = 'log.csv'
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model Name', 'Epochs', 'Batch Size', 'Validation Accuracy'])  # Fejléc
        writer.writerow([model_neve, num_epochs, train_batch_size, val_accuracy])  # Modell név is bekerül

    # Modell értékelése és kiiratás
    evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size,val_accuracy, 0, model_neve, t_loss_min,max_acc)
    evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size,val_accuracy, 1, model_neve, t_loss_min,max_acc)


    # Előző epoch értékének frissítése
    previous_config = (num_epochs, train_batch_size, fel_le_kerekit, model_neve)





