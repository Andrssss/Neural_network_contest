
# ITT lehet változtatni, hogy milyen konfigurációkkal fusson le egymás után.
# Ez azért sexy, mert teljesen autómata, mind a log, mind a file létrehozása a result mappába.
# Érdemes növekvősorrendbe rakni az olyan tanításokat, amiknél csak epoch külömböző

# Start tensorboard. --> ez összehasonlítja a különböző modellek tanulását
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/


configurations = [

(80, 8, 1, "MobileNetV2Custom"),
(100, 8, 1, "MobileNetV2Custom"),
(120,8,1,"MobileNetV2Custom"),
(150, 8, 1, "MobileNetV2Custom"),

]

validation_ratio   = 0.05        # 0.0 -> nincs validáció   0.1 -> 10%
# num_epochs       = 50
# train_batch_size = 8
# fel_le_kerekit   = 1     # ennek most nincs funkciója, butaság
# model_neve       = "MobileNetV2Custom"

# MobileNetV2Custom     - gyors, de 20% pontosság max,eddig
# ResNet34Custom        -
# EfficientNetB0Custom  - kurva lassú betanulás ( laptopomon esélytelen )
# SwinTransformerCustom - picsog
# ConvNeXtCustom        - kurva lassú betanulás, meg jó szar is. Logban van mérés.
# AlexNet               - megnezem mennyire pontos

# --------------------------------------   INICIALIZÁLÁS   -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
import csv
import os
import logging
from torch import nn, optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import MobileNetV2Custom, ResNet34Custom, EfficientNetB0Custom, SwinTransformerCustom, ConvNeXtCustom
from evaluate_and_export import evaluate_model
# Logger létrehozás -------------------------------------
# from logger import setup_logger
# setup_logger()
# DATA AUGMENTATION -------------------------------------
from data_augmentation import perform_data_augmentation
saved_folder, saved_csv = perform_data_augmentation()
print(f"Az augmented képek mentési mappája: {saved_folder}")
print(f"A mentett címke CSV fájl elérési útja: {saved_csv}")
"""
# Fileok beolvasása -------------------------------------
from reader_initializer import initialize_data
train_image_list, train_image_ids, test_image_list, test_image_ids, data_array = initialize_data(validation_ratio,saved_csv,saved_folder)
# GPU beállítása ha lehet -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
print(f"Graphic card: {torch.cuda.is_available()}")






# Set validation ratio (0.1 for 90-10 split; set to 0.0 for 100-0 split)
if validation_ratio > 0.01:
    train_image_list, validate_image_list, train_image_ids, validate_image_ids = train_test_split( train_image_list, train_image_ids, test_size=validation_ratio, random_state=42 )
else:
    validate_image_list = []
    validate_image_ids = []

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
mapped_labels  = np.array([label_map[label] for label in data_array[:, 1]])
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





previous_config = None
t_loss_min = 99
max_acc = 0
val_accuracy = 0.0


for num_epochs, train_batch_size, fel_le_kerekit, model_neve in configurations:
    cur_acc = 0.0
    if (previous_config != None and num_epochs > previous_config[0] and train_batch_size == previous_config[1] and fel_le_kerekit == previous_config[2] and model_neve == previous_config[3]):
        start_epoch = previous_config[0]  # Folytatás az aktuális num_epochs értéktől
    else:
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


        if model_neve == "EfficientNetB0Custom":

            model = EfficientNetB0Custom(num_classes=num_classes)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
            criterion = nn.MSELoss()  # Regresszióhoz megfelelő

        elif model_neve == "MobileNetV2Custom":

            # 0–40 Epoch: A MobileNetV2 kisebb és gyorsabb modell, így gyorsabban konvergálhat. Érdemes 20-30 epochot kezdetben, majd figyelni a teljesítményt. Ha szükséges, lehet növelni akár 50 epochra is.
            # Batch Size : 32 vagy 64: A MobileNetV2 hatékonysága miatt nagyobb batch size-t is kezelhet, így érdemes 32-vel vagy 64-gyel kezdeni, hogy stabilabb gradienseket érj el. (Ha memória problémák lépnek fel, akkor 16-ra csökkenthető.)
            model = MobileNetV2Custom(num_classes=len(label_map))  # vagy az adott modellnek megfelelő
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # StepLR: 10 epochonként 0.1-es faktorral csökkenti a tanulási rátát. -> may 20%
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) # CosineAnnealingLR: Ha 20–50 epocra tervezel, a cosine annealing finoman csökkenti a tanulási rátát.
            criterion = nn.CrossEntropyLoss()

            logging.info(f" scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)")
            logging.info(f" optimizer = optim.Adam(model.parameters(), lr=0.001)")

        elif model_neve == "ResNet34Custom":
            model = ResNet34Custom(num_classes=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.MSELoss()

        elif model_neve == "SwinTransformerCustom":
            model = SwinTransformerCustom(num_classes=num_classes)
            optimizer = optim.AdamW(model.parameters(), lr=0.0005)  # AdamW optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # Speciális LR csökkentés
            criterion = nn.MSELoss()

        elif model_neve == "ConvNeXtCustom":
            # Epochs -  20-40 : A ConvNeXt hálózat jól konvergálhat 20-40 epoch alatt, de ha a tanulás lassabb, akár 50 epochot is használhatsz.
            # Batch Size - 32-64
            model = ConvNeXtCustom(num_classes=num_classes)
            optimizer = optim.AdamW(model.parameters(), lr=0.0005)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7) # Ha 20-40 epochig tervezed az edzést
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # ezt kell változtatni, ha picsog
        elif model_neve == "AlexNet":
            from torchvision.models import alexnet  # AlexNet importálása

            # Modell inicializálása a megfelelő osztályok számával
            model = alexnet(num_classes=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Gyorsabb konvergálás érdekében
            # Alternatíva: optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Hibás/nem létező modell név: {model_neve}")




        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        start_epoch = 0  # Újratöltés esetén a ciklus kezdőértéke

        output_file = 'log.csv'
        with open(output_file, mode='a') as file:
            file.write('\n')

    logging.info(f" [{start_epoch} - {num_epochs}] - {model_neve} - B={train_batch_size}")

    # Epoch ciklus a megadott start_epoch-tól num_epochs-ig
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for train_images, labels in train_loader:
            train_images, labels = train_images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(train_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if train_loss < t_loss_min:  t_loss_min = train_loss
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
                    test_images = test_images.unsqueeze(0).to(device)

                    outputs = model(test_images)
                    _, predicted = torch.max(outputs, 1)

                    predicted_label = reverse_label_map[predicted.item()]
                    predicted_label = int(abs(float(predicted_label)))

                    # Az eredeti címke kinyerése az ID alapján
                    original_label = reverse_label_map[data_array[data_array[:, 0] == test_ids, 1].item()]
                    original_label = int(abs(float(original_label)))

                    # Összehasonlítás és számlálás
                    # print(f"Predictions: {predicted_label} / Original: {original_label}")
                    if predicted_label == original_label:
                        correct_count += 1
                    total_count += 1

                    results.append([test_ids, predicted_label, original_label])

            # Pontosság kiszámítása
            # print(f"Correct Predictions: {original_label} / Total: {predicted_label}")
            val_accuracy = correct_count / total_count
            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        else:
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        if cur_acc < val_accuracy: cur_acc = val_accuracy
        if max_acc < val_accuracy: max_acc = val_accuracy

    # Loggolás, mentés és egyéb műveletek
    output_file = 'log.csv'
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model Name', 'Epochs', 'Batch Size', 'Validation Accuracy'])  # Fejléc
        writer.writerow([model_neve, num_epochs, train_batch_size, cur_acc])  # Modell név is bekerül

    # Modell értékelése és kiiratás
    evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size,val_accuracy, 0, model_neve, t_loss_min,cur_acc)
    evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size,val_accuracy, 1, model_neve, t_loss_min,cur_acc)


    # Előző epoch értékének frissítése
    previous_config = (num_epochs, train_batch_size, fel_le_kerekit, model_neve)
"""


