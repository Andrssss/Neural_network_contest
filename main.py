# Start tensorboard. --> ez összehasonlítja a különböző modellek tanulását
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

# validation_ratio     = 0.014 # -> 50 kép
validation_ratio     = 0.05   # 0.1 -> 10%, ha ezen változtatni szeretnél, akkor az alatta lévőt tedd TRUE-ra, első körben
hozzon_letre_uj_augmentalt_fileokat_e = False   # külön is futtatható
Augmentation_number    = 2
kerekitsen_labeleket = True

patience = 50

configurations = [
(100, 16, 1, "MobileNetV2Custom_0",0.003),
(100, 32, 1, "MobileNetV2Custom_0",0.001),
(100, 64, 1, "MobileNetV2Custom_0",0.0005),
(100, 128, 1, "MobileNetV2Custom_0",0.0001),

(100, 16, 1, "MobileNetV2Custom_1",0.003),
(100, 32, 1, "MobileNetV2Custom_1",0.001),
(100, 64, 1, "MobileNetV2Custom_1",0.0005),
(100, 128, 1, "MobileNetV2Custom_1",0.0001),

(100, 16, 1, "MobileNetV2Custom_2",0.003),
(100, 32, 1, "MobileNetV2Custom_2",0.001),
(100, 64, 1, "MobileNetV2Custom_2",0.0005),
(100, 128, 1, "MobileNetV2Custom_2",0.0001),

(100, 16, 1, "MobileNetV2Custom_3",0.003),
(100, 32, 1, "MobileNetV2Custom_3",0.001),
(100, 64, 1, "MobileNetV2Custom_3",0.0005),
(100, 128, 1, "MobileNetV2Custom_3",0.0001),

(100, 16, 1, "MobileNetV2Custom_4",0.003),
(100, 32, 1, "MobileNetV2Custom_4",0.001),
(100, 64, 1, "MobileNetV2Custom_4",0.0005),
(100, 128, 1, "MobileNetV2Custom_4",0.0001),

(100, 16, 1, "MobileNetV2Custom_5",0.003),
(100, 32, 1, "MobileNetV2Custom_5",0.001),
(100, 64, 1, "MobileNetV2Custom_5",0.0005),
(100, 128, 1, "MobileNetV2Custom_5",0.0001),

(100, 16, 1, "MobileNetV2Custom_6",0.003),
(100, 32, 1, "MobileNetV2Custom_6",0.001),
(100, 64, 1, "MobileNetV2Custom_6",0.0005),
(100, 128, 1, "MobileNetV2Custom_6",0.0001),

(100, 16, 1, "MobileNetV2Custom_7",0.003),
(100, 32, 1, "MobileNetV2Custom_7",0.001),
(100, 64, 1, "MobileNetV2Custom_7",0.0005),
(100, 128, 1, "MobileNetV2Custom_7",0.0001),

(100, 16, 1, "MobileNetV2Custom_8",0.003),
(100, 32, 1, "MobileNetV2Custom_8",0.001),
(100, 64, 1, "MobileNetV2Custom_8",0.0005),
(100, 128, 1, "MobileNetV2Custom_8",0.0001),

(100, 16, 1, "MobileNetV2Custom_9",0.003),
(100, 32, 1, "MobileNetV2Custom_9",0.001),
(100, 64, 1, "MobileNetV2Custom_9",0.0005),
(100, 128, 1, "MobileNetV2Custom_9",0.0001),

(100, 16, 1, "MobileNetV2Custom_10",0.003),
(100, 32, 1, "MobileNetV2Custom_10",0.001),
(100, 64, 1, "MobileNetV2Custom_10",0.0005),
(100, 128, 1, "MobileNetV2Custom_10",0.0001),

(100, 16, 1, "MobileNetV2Custom_11",0.003),
(100, 32, 1, "MobileNetV2Custom_11",0.001),
(100, 64, 1, "MobileNetV2Custom_11",0.0005),
(100, 128, 1, "MobileNetV2Custom_11",0.0001),

(100, 16, 1, "MobileNetV2Custom_12",0.003),
(100, 32, 1, "MobileNetV2Custom_12",0.001),
(100, 64, 1, "MobileNetV2Custom_12",0.0005),
(100, 128, 1, "MobileNetV2Custom_12",0.0001),

(100, 16, 1, "MobileNetV2Custom_13",0.003),
(100, 32, 1, "MobileNetV2Custom_13",0.001),
(100, 64, 1, "MobileNetV2Custom_13",0.0005),
(100, 128, 1, "MobileNetV2Custom_13",0.0001),

(100, 16, 1, "MobileNetV2Custom_14",0.003),
(100, 32, 1, "MobileNetV2Custom_14",0.001),
(100, 64, 1, "MobileNetV2Custom_14",0.0005),
(100, 128, 1, "MobileNetV2Custom_14",0.0001),

(100, 16, 1, "MobileNetV2Custom_15",0.003),
(100, 32, 1, "MobileNetV2Custom_15",0.001),
(100, 64, 1, "MobileNetV2Custom_15",0.0005),
(100, 128, 1, "MobileNetV2Custom_15",0.0001),

(100, 16, 1, "MobileNetV2Custom_16",0.003),
(100, 32, 1, "MobileNetV2Custom_16",0.001),
(100, 64, 1, "MobileNetV2Custom_16",0.0005),
(100, 128, 1, "MobileNetV2Custom_16",0.0001)

]


# num_epochs       = 50
# train_batch_size = 8
# fel_le_kerekit   = 1     # ennek még nincs funkciója, butaság
# model_neve       = "MobileNetV2Custom"

# MobileNetV2Custom     - gyors, de 20% pontosság max,eddig
# AdvancedMobileNetV2        -





# ----------------------------------------------------------------------------------------  EDDIG LEGJOBBAK
"""  15 %
        
validation_ratio     = 0.05   # 0.1 -> 10%, ha ezen változtatni szeretnél, akkor az alatta lévőt tedd TRUE-ra, első körben
hozzon_letre_uj_augmentalt_fileokat_e = False   # külön is futtatható
Augmentation_number    = 2
kerekitsen_labeleket = True

        (100, 16, 1, "MobileNetV2Custom"),  

        elif model_neve == "MobileNetV2Custom":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR
            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
            logging.info("scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F

            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()

            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")

"""



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
from model import MobileNetV2Custom, AdvancedMobileNetV2
from evaluate_and_export import evaluate_model
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
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
max_acc = 0.2
val_accuracy = 0.0

# EARLY STOPPING
 # Hány epoch után álljon le, ha nincs javulás - GPT - 5 re állította
best_val_loss = float('inf')  # Legjobb validációs veszteség
early_stopping_counter = 0  # Megszakítás számláló

for num_epochs, train_batch_size, fel_le_kerekit, model_neve,default_lr in configurations:
    cur_acc = 0.0
    if (previous_config != None and num_epochs > previous_config[0] and train_batch_size == previous_config[1] and fel_le_kerekit == previous_config[2] and model_neve == previous_config[3]):
        start_epoch = previous_config[0]  # Folytatás az aktuális num_epochs értéktől
    else:
        best_val_loss = float('inf')
        early_stopping_counter = 0
        t_loss_min = 99
        max_acc = 0.2
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

        if model_neve == "MobileNetV2Custom_0":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")




        elif model_neve == "MobileNetV2Custom_1":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.0008
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
            logging.info("scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")







        elif model_neve == "MobileNetV2Custom_2":  # Epoch : 0–50 , Batch Size : 32 vagy 64
                model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
                model = model.to(device)
                default_lr = 0.0009
                optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
                logging.info("optimizer = optim.Adam(model.parameters(), lr=0.0009, weight_decay=1e-4)")
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
                from torch.optim.lr_scheduler import CyclicLR

                scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
                logging.info(
                    "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")

                # CrossEntropy loss label smoothing-gel
                import torch.nn.functional as F


                def label_smoothing_loss(inputs, targets, smoothing=0.1):
                    confidence = 1.0 - smoothing
                    log_probs = F.log_softmax(inputs, dim=-1)
                    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    loss = confidence * nll_loss + smoothing * smooth_loss
                    return loss.mean()


                criterion = label_smoothing_loss
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                logging.info("criterion = label_smoothing_loss")




        elif model_neve == "MobileNetV2Custom_3":  # Epoch : 0–50 , Batch Size : 32 vagy 64
                model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
                model = model.to(device)
                default_lr = 0.0007
                optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
                logging.info("optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-4)")
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
                from torch.optim.lr_scheduler import CyclicLR

                scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
                logging.info(
                    "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")

                # CrossEntropy loss label smoothing-gel
                import torch.nn.functional as F


                def label_smoothing_loss(inputs, targets, smoothing=0.1):
                    confidence = 1.0 - smoothing
                    log_probs = F.log_softmax(inputs, dim=-1)
                    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    loss = confidence * nll_loss + smoothing * smooth_loss
                    return loss.mean()


                criterion = label_smoothing_loss
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                logging.info("criterion = label_smoothing_loss")

        elif model_neve == "MobileNetV2Custom_4":  # Epoch : 0–50 , Batch Size : 32 vagy 64
                model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
                model = model.to(device)
                default_lr = 0.001
                optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
                logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
                from torch.optim.lr_scheduler import CyclicLR

                scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=20, mode='triangular2')
                logging.info(
                    "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=20, mode='triangular2')")

                # CrossEntropy loss label smoothing-gel
                import torch.nn.functional as F


                def label_smoothing_loss(inputs, targets, smoothing=0.1):
                    confidence = 1.0 - smoothing
                    log_probs = F.log_softmax(inputs, dim=-1)
                    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    loss = confidence * nll_loss + smoothing * smooth_loss
                    return loss.mean()


                criterion = label_smoothing_loss
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                logging.info("criterion = label_smoothing_loss")




        elif model_neve == "MobileNetV2Custom_5":  # Epoch : 0–50 , Batch Size : 32 vagy 64
                model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
                model = model.to(device)
                default_lr = 0.0007
                optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
                logging.info("optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-4)")
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
                from torch.optim.lr_scheduler import CyclicLR

                scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=30, mode='triangular2')
                logging.info(
                    "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=30, mode='triangular2')")

                # CrossEntropy loss label smoothing-gel
                import torch.nn.functional as F


                def label_smoothing_loss(inputs, targets, smoothing=0.1):
                    confidence = 1.0 - smoothing
                    log_probs = F.log_softmax(inputs, dim=-1)
                    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    loss = confidence * nll_loss + smoothing * smooth_loss
                    return loss.mean()


                criterion = label_smoothing_loss
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                logging.info("criterion = label_smoothing_loss")


        elif model_neve == "MobileNetV2Custom_6":  # Epoch : 0–50 , Batch Size : 32 vagy 64
                model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
                model = model.to(device)
                default_lr = 0.0007
                optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
                logging.info("optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-4)")
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
                from torch.optim.lr_scheduler import CyclicLR

                scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=40, mode='triangular2')
                logging.info(
                    "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=40, mode='triangular2')")

                # CrossEntropy loss label smoothing-gel
                import torch.nn.functional as F


                def label_smoothing_loss(inputs, targets, smoothing=0.1):
                    confidence = 1.0 - smoothing
                    log_probs = F.log_softmax(inputs, dim=-1)
                    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    loss = confidence * nll_loss + smoothing * smooth_loss
                    return loss.mean()


                criterion = label_smoothing_loss
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                logging.info("criterion = label_smoothing_loss")


        elif model_neve == "MobileNetV2Custom_7":  # Epoch : 0–50 , Batch Size : 32 vagy 64
                model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
                model = model.to(device)
                default_lr = 0.001
                optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-3)
                logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)")
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
                from torch.optim.lr_scheduler import CyclicLR

                scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
                logging.info(
                    "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")

                # CrossEntropy loss label smoothing-gel
                import torch.nn.functional as F


                def label_smoothing_loss(inputs, targets, smoothing=0.1):
                    confidence = 1.0 - smoothing
                    log_probs = F.log_softmax(inputs, dim=-1)
                    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    loss = confidence * nll_loss + smoothing * smooth_loss
                    return loss.mean()


                criterion = label_smoothing_loss
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                logging.info("criterion = label_smoothing_loss")

        elif model_neve == "MobileNetV2Custom_8":  # Epoch : 0–50 , Batch Size : 32 vagy 64
                model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
                model = model.to(device)
                default_lr = 0.001
                optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-5)
                logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)")
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
                from torch.optim.lr_scheduler import CyclicLR

                scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')
                logging.info(
                    "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular2')")

                # CrossEntropy loss label smoothing-gel
                import torch.nn.functional as F


                def label_smoothing_loss(inputs, targets, smoothing=0.1):
                    confidence = 1.0 - smoothing
                    log_probs = F.log_softmax(inputs, dim=-1)
                    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    loss = confidence * nll_loss + smoothing * smooth_loss
                    return loss.mean()


                criterion = label_smoothing_loss
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                logging.info("criterion = label_smoothing_loss")




        elif model_neve == "MobileNetV2Custom_9":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=10, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=10, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")


        elif model_neve == "MobileNetV2Custom_10":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=10, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=10, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")


        elif model_neve == "MobileNetV2Custom_11":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=100, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=100, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")

        elif model_neve == "MobileNetV2Custom_12":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=50, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=50, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")

        elif model_neve == "MobileNetV2Custom_13":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=150, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=150, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            criterion = label_smoothing_loss
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")


        elif model_neve == "MobileNetV2Custom_14":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=100, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=100, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            # criterion = label_smoothing_loss
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = nn.CrossEntropyLoss(label_smoothing=0.1)")

        elif model_neve == "MobileNetV2Custom_15":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=50, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=50, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            #criterion = label_smoothing_loss
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = nn.CrossEntropyLoss(label_smoothing=0.1)")

        elif model_neve == "MobileNetV2Custom_16":  # Epoch : 0–50 , Batch Size : 32 vagy 64
            model = MobileNetV2Custom(num_classes=len(label_map))  # Ha támogatott
            model = model.to(device)
            default_lr = 0.001
            optimizer = optim.Adam(model.parameters(), lr=default_lr, weight_decay=1e-4)
            logging.info("optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)")
            from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
            from torch.optim.lr_scheduler import CyclicLR

            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=150, mode='triangular2')
            logging.info(
                "scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1, step_size_up=150, mode='triangular2')")

            # CrossEntropy loss label smoothing-gel
            import torch.nn.functional as F


            def label_smoothing_loss(inputs, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(inputs, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = confidence * nll_loss + smoothing * smooth_loss
                return loss.mean()


            # criterion = label_smoothing_loss
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            logging.info("criterion = label_smoothing_loss")



        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        start_epoch = 0  # Újratöltés esetén a ciklus kezdőértéke

        output_file = 'log.csv'


    logging.info(f" [{start_epoch} - {num_epochs}] - {model_neve} - B={train_batch_size}")

    # Epoch ciklus a megadott start_epoch-tól num_epochs-ig
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0  # Helyes előrejelzések száma a train adatokon
        total_train = 0  # Teljes train minta száma

        for train_images, labels in train_loader:
            train_images, labels = train_images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(train_images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradiens vágás
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Pontosság kiszámítása a tanítás során
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100. * correct_train / total_train  # Tanítási pontosság százalékban

        if train_loss < t_loss_min:
            t_loss_min = train_loss

        # Log  Current Learning Rate
        current_lr = scheduler.get_last_lr()[0]

        # Validációs kiértékelés
        if validate_image_tensors is not None:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for test_images, test_ids in zip(validate_image_tensors, validate_image_ids):
                    test_images = test_images.unsqueeze(0).to(device)
                    outputs = model(test_images)
                    _, predicted = torch.max(outputs, 1)

                    original_label = data_array[data_array[:, 0] == test_ids, 1].item()
                    original_label_tensor = torch.tensor([original_label], dtype=torch.long).to(device)

                    # Veszteség számítása a valós címkékkel
                    loss = criterion(outputs, original_label_tensor)
                    val_loss += loss.item()

                    # Pontosság ellenőrzése
                    if predicted.item() == original_label:
                        correct_val += 1
                    total_val += 1

            val_loss /= len(validate_image_ids)
            val_accuracy = 100. * correct_val / total_val  # Validációs pontosság százalékban

            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

            scheduler.step()

        else:
            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"LR: {current_lr:.6f}")

        # Maximális pontosságok frissítése
        if cur_acc < val_accuracy:
            cur_acc = val_accuracy
        if max_acc < val_accuracy:
            max_acc = val_accuracy

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
                evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, epoch + 1, train_batch_size, val_accuracy, 0, model_neve, t_loss_min, cur_acc)
                evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, epoch + 1, train_batch_size, val_accuracy, 1, model_neve, t_loss_min, cur_acc)
                previous_config = (num_epochs, train_batch_size, fel_le_kerekit, model_neve)
                break
            elif early_stopping_counter > patience:
                logging.info("Early stopping triggered.")
                break

        if val_accuracy > 0.7 and current_lr > 1e-4:
            new_lr = max(current_lr * 0.5, 1e-4)  # Fokozatos csökkentés
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        elif val_accuracy < best_val_accuracy - 0.05:
            if current_lr < default_lr:  # Csak akkor növeld, ha az LR csökkentett volt
                for param_group in optimizer.param_groups:
                    param_group['lr'] = default_lr  # Visszaállítja az eredeti tanulási rátát
        if val_accuracy > 11:
            evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size,val_accuracy, 0, model_neve, t_loss_min, cur_acc)
            evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev, num_epochs, train_batch_size, val_accuracy, 1, model_neve, t_loss_min, cur_acc)
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
