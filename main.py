# Start tensorboard. --> ez összehasonlítja a különböző modellek tanulását
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/


validation_ratio     = 0.1   # 0.1 -> 10%, ha ezen változtatni szeretnél, akkor az alatta lévőt tedd TRUE-ra, első körben
hozzon_letre_uj_augmentalt_fileokat_e = True   # külön is futtatható
Augmentation_number    = 3
kerekitsen_labeleket = True



# Érdemes növekvősorrendbe rakni az olyan tanításokat, amiknél csak epoch külömböző.
configurations = [
    # (100, 8, 1, "MobileNetV2Custom"),
    (100, 16, 1, "MobileNetV2Custom"),
    (100, 32, 1, "MobileNetV2Custom"),

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
from model import MobileNetV2Custom, ResNet34Custom, EfficientNetB0Custom, SwinTransformerCustom, ConvNeXtCustom
from evaluate_and_export import evaluate_model
# Logger létrehozás -------------------------------------
from logger import setup_logger
setup_logger()
logging.info(f"Validation_ratio : {validation_ratio}")
# DATA AUGMENTATION ------------------------------------- vad verzió
source_folder = "./train_data"
output_dir = "./train_data_Augmented+original"
original_csv_file = label_file = "./data_labels_train.csv"
test_folder = "./test_data"
train_folder = "train_data_2"
validation_folder = "validation_data"

def ensure_folder_and_process(folder_path, callback):
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

if kerekitsen_labeleket:



    logging.info(f"Eredeti címke: {data_array[:, 1]}")
    unique_labels = np.unique(data_array[:, 1])  # Eredeti egyedi címkék
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Leképezés 0-tól kezdve
    mapped_labels = np.array([label_map[label] for label in data_array[:, 1]])  # Minden címke újrakódolva
    data_array[:, 1] = mapped_labels  # Frissítés átalakított címkékkel
    logging.info(f"Átalakított címke: {data_array[:, 1]}")
else :
    logging.info(f"Eredeti címke: {data_array[:, 1]}")
    unique_labels = np.unique(data_array[:, 1])  # Eredeti címkék
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Mappeljük 0-tól kezdve
    mapped_labels = np.array([label_map[label] for label in data_array[:, 1]])
    data_array[:, 1] = mapped_labels
    logging.info(f"Átalakított címke: {data_array[:, 1]}")









# --------------------------------------   BETANITAS  ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Augmentáció miatt, lecsícsi az elejéről a számot és így kereshető az ID
def normalize_id(img_id):
    parts = img_id.split('_', 1)  # Az első '_' után osztjuk ketté
    if parts[0].isdigit():  # Csak akkor vágjuk le, ha az első rész szám
        return parts[1]
    return img_id  # Ha nincs prefix, az eredeti ID-t adjuk vissza




# Egyedi dataset osztály
class CustomImageDataset(Dataset):
    def __init__(self, images, image_ids, data_array_, transform=None):
        self.images = images
        self.image_ids = image_ids
        # Az ID-k normalizálása a data_dict létrehozásakor
        self.data_dict = {normalize_id(row[0]): row[1] for row in data_array_}
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




previous_config = None
t_loss_min = 99
max_acc = 0
val_accuracy = 0.0

# EARLY STOPPING
patience = 999999  # Hány epoch után álljon le, ha nincs javulás - GPT - 5 re állította
best_val_loss = float('inf')  # Legjobb validációs veszteség
early_stopping_counter = 0  # Megszakítás számláló

for num_epochs, train_batch_size, fel_le_kerekit, model_neve in configurations:
    cur_acc = 0.0
    if (previous_config != None and num_epochs > previous_config[0] and train_batch_size == previous_config[1] and fel_le_kerekit == previous_config[2] and model_neve == previous_config[3]):
        start_epoch = previous_config[0]  # Folytatás az aktuális num_epochs értéktől
    else:
        best_val_loss = float('inf')
        early_stopping_counter = 0
        t_loss_min = 99
        max_acc = 0
        val_accuracy = 0.0
        dataset = CustomImageDataset(images=train_image_tensors, image_ids=train_image_ids, data_array_=data_array)
        train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        # Validation dataset (if available)
        if validate_image_tensors is not None:
            #val_dataset = CustomImageDataset(images=validate_image_tensors, data_array=data_array, ids=validate_image_ids,   transform=transform_validate)
            val_dataset = CustomImageDataset(images=validate_image_tensors, image_ids=validate_image_ids,data_array_=data_array)
            val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)


        # todo - ez mi a faszom, 21 kategória kell nem 4000
        num_classes = len(np.unique(data_array[:, 1]))
        logging.info(f" num_classes = {num_classes}")

        best_val_accuracy = 0.0
        default_lr = 0.005

        if model_neve == "EfficientNetB0Custom":

            model = EfficientNetB0Custom(num_classes=num_classes)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
            original_lr = 0.001
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
            criterion = nn.CrossEntropyLoss()  # Regresszióhoz megfelelő

        elif model_neve == "MobileNetV2Custom": # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)

            # Egyben kell tesztelni !!!!!!!!!!!!!!!!!!!!
            optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            original_lr = 0.005
 #           from torch.optim.lr_scheduler import CosineAnnealingLR
#            scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 50 epoch után csökkentés
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

            # VAGY
            # from torch.optim.lr_scheduler import CyclicLR
            # scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular')


            from torch.nn.functional import cross_entropy
            def focal_loss(inputs, targets, alpha=0.5, gamma=1.5):
                ce_loss = cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = alpha * (1 - pt) ** gamma * ce_loss
                return focal_loss.mean()
            criterion = focal_loss
            #criterion = nn.CrossEntropyLoss() # EREDETI



        elif model_neve == "SwinTransformerCustom":
            model = SwinTransformerCustom(num_classes=num_classes)
            optimizer = optim.AdamW(model.parameters(), lr=0.0005)  # AdamW optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # Speciális LR csökkentés
            criterion = nn.CrossEntropyLoss()

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
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1  )
            criterion = nn.CrossEntropyLoss()

        elif model_neve == "ResNet34Custom":
            model = ResNet34Custom(num_classes=num_classes)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)

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

            # Gradiens vágás  ----> EZ LEHET, HOGY NAGYON HASZNOS
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if train_loss < t_loss_min:  t_loss_min = train_loss
        # Log  Current Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        # current_lr = scheduler.get_last_lr()[0]   ----> EZ LEHET GOND


        # Validation evaluation ----------------------------------------------------
        if validate_image_tensors is not None:
            model.eval()  # Váltás kiértékelési módba
            results = []
            reverse_label_map = {idx: label for label, idx in label_map.items()}  # Címkék visszafejtése
            val_loss = 0.0
            correct_count = 0  # Egyezések számlálása
            total_count = 0  # Összes validációs adat száma

            with torch.no_grad():  # Gradiensek nem szükségesek kiértékelés során
                for test_images, test_ids in zip(validate_image_tensors, validate_image_ids):
                    test_images = test_images.unsqueeze(0).to(device)

                    outputs = model(test_images)
                    _, predicted = torch.max(outputs, 1)
                    predicted_label = predicted
                    #predicted_label = reverse_label_map[predicted.item()]
                    #predicted_label = int(abs(float(predicted_label)))

                    # Az eredeti címke kinyerése az ID alapján
                    # original_label = reverse_label_map[data_array[data_array[:, 0] == test_ids, 1].item()]
                    # original_label original_label = int(abs(float(original_label)))

                    original_label = data_array[data_array[:, 0] == test_ids, 1].item()
                    original_label_tensor = torch.tensor([original_label], dtype=torch.long).to(device)
                    #
                    # # Veszteség számítása a valós címkékkel
                    loss = criterion(outputs, original_label_tensor)
                    val_loss += loss.item()


                    # Összehasonlítás és számlálás
                    predicted_label = predicted_label.item()
                    # print(f"Predictions: {predicted_label} / Original: {original_label}")
                    if predicted_label == original_label:
                        correct_count += 1
                    total_count += 1

                    results.append([test_ids, predicted_label, original_label])
            # Pontosság kiszámítása
            # print(f"Correct Predictions: {original_label} / Total: {predicted_label}")
            val_loss /= len(validate_image_ids)
            val_accuracy = correct_count / total_count
            logging.info(  f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val loss: {val_loss:.4f}, LR: {current_lr:.6f}")



            # scheduler.step()
            scheduler.step(val_loss)

        else:
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        if cur_acc < val_accuracy: cur_acc = val_accuracy
        if max_acc < val_accuracy: max_acc = val_accuracy


        #if current_lr <= 0.0001 and cur_acc <= 0.5:
        #    early_stopping_counter = 0
        #    logging.info("LR = 0.")
        #    break

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter == patience:
                logging.info("Early stopping triggered.")
                # Modell értékelése és kiiratás
                evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size, val_accuracy, 0, model_neve, t_loss_min, cur_acc)
                evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size, val_accuracy, 1, model_neve, t_loss_min, cur_acc)
                previous_config = (num_epochs, train_batch_size, fel_le_kerekit, model_neve)
                break
            elif early_stopping_counter > patience:
                logging.info("Early stopping triggered.")
                break

        if val_accuracy > best_val_accuracy: # TESZTELÉS ALATT
            best_val_accuracy = val_accuracy
            if val_accuracy > 0.7 and current_lr > 1e-4:  # Csak akkor csökkentsük, ha az LR még magasabb
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4  # Csökkentett tanulási ráta
                print("Tanulási ráta csökkentve finomhangoláshoz: 1e-4")

            # Ha az accuracy leesik és az LR-t korábban csökkentettük
        elif val_accuracy < best_val_accuracy - 0.05:
            if current_lr < default_lr:  # Csak akkor növeld, ha az LR csökkentett volt
                for param_group in optimizer.param_groups:
                    param_group['lr'] = default_lr  # Visszaállítja az eredeti tanulási rátát
                print(f"Val_accuracy csökkent, tanulási ráta visszaállítva: {default_lr}")


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
