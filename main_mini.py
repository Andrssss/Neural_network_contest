from transfrom_2_mini import process_data

validation_ratio     = 0.05   # 0.1 -> 10%, ha ezen változtatni szeretnél, akkor az alatta lévőt tedd TRUE-ra, első körben
hozzon_letre_uj_augmentalt_fileokat_e = True   # külön is futtatható
Augmentation_number    = 0
kerekitsen_labeleket = True


configurations = [
(1000, 4),
(1000, 8),
(1000, 16),
(1000, 32),
(1000, 64),
(1000, 128),
]




if hozzon_letre_uj_augmentalt_fileokat_e:
    process_data(source_folder='./train_data',train_folder="train_data_2",validation_folder="validation_data",validation_rate=validation_ratio)

# --------------------------------------   INICIALIZÁLÁS   -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import logging
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MobileNetV2Custom
from evaluate_and_export import evaluate_model
# Logger létrehozás -------------------------------------
from logger import setup_logger
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup_logger()
logging.info(f"Validation_ratio : {validation_ratio}")

# DATA AUGMENTATION ------------------------------------- vad verzió
source_folder = "./train_data"
# output_dir = "./train_data_Augmented+original"
original_csv_file = label_file = "./data_labels_train.csv"
test_folder = "./test_data"
train_folder = "train_data_2"
validation_folder = "validation_data"


best_val_accuracy = 0
best_val_acc_at = 0
best_val_acc_batch = 0

# Fileok beolvasása -------------------------------------
from reader_initializer import initialize_data
train_image_list, train_image_ids, validate_image_list, validate_image_ids, test_image_list, test_image_ids, data_array = initialize_data(train_folder,validation_folder,test_folder,label_file,validation_ratio,kerekitsen_labeleket)


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
import pandas as pd # ------------------------ TRAIN ----------------------------------------------------
import numpy as np

# Adatok betöltése és előfeldolgozása a TRAIN számára
df = pd.read_csv("data_labels_train_2.csv")
selected_data = df[['filename_id', 'defocus_label']]
# Kerekítés és tartomány korlátozás, ha szükséges
if kerekitsen_labeleket:
    selected_data.loc[:, 'defocus_label'] = np.clip(np.round(selected_data['defocus_label']), -10, 10)

# Eredeti címkék mappolása 0-tól kezdve
unique_labels = np.unique(data_array[:, 1])
label_map = {label: idx for idx, label in enumerate(unique_labels)}
selected_data.loc[:, 'defocus_label'] = selected_data['defocus_label'].map(label_map)
TRAIN_array = selected_data.to_numpy()
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# VALIDATION előfeldolgozása
df = pd.read_csv("validation_data.csv")
selected_data = df[['filename_id', 'defocus_label']]
# Kerekítés és tartomány korlátozás, ha szükséges
if kerekitsen_labeleket:
    selected_data.loc[:, 'defocus_label'] = np.clip(np.round(selected_data['defocus_label']), -10, 10)
selected_data.loc[:, 'defocus_label'] = selected_data['defocus_label'].map(label_map)
VALIDATION_array = selected_data.to_numpy()





# Előfeldolgozás: méretezés, tensor konverzió, normalizálás
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Átméretezi a képet
    transforms.ToTensor(),  # Tensor formátum
])


class CustomDataset(Dataset):
    def __init__(self, data_array, root_dir, transform=None):
        self.root_dir = root_dir
        self.data_dict = {row[0]: row[1] for row in data_array}
        self.image_ids = list(self.data_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        base_name = self.image_ids[idx]
        parts = base_name.split("_")
        clean_base_name = "_".join(parts[1:]) if parts[0].isdigit() else base_name

        # Load images as PIL.Image
        img1 = Image.open(os.path.join(self.root_dir, f"{clean_base_name}_phase.png")).convert("L")
        img2 = Image.open(os.path.join(self.root_dir, f"{clean_base_name}_mask.png")).convert("L")
        img3 = Image.open(os.path.join(self.root_dir, f"{clean_base_name}_amp.png")).convert("L")

        # Apply transformations if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Convert to tensor if not already transformed
            # Ensure tensors have the correct dimensions
            img1 = img1.unsqueeze(0) if len(img1.shape) == 2 else img1
            img2 = img2.unsqueeze(0) if len(img2.shape) == 2 else img2
            img3 = img3.unsqueeze(0) if len(img3.shape) == 2 else img3

        image = torch.cat([img1, img2, img3], dim=0)  # Combine the channels [3, H, W]

        label = self.data_dict[base_name]
        return image, torch.tensor(label, dtype=torch.long)

test_images = np.array(test_image_list)
test_mean = test_images.mean()
test_std = test_images.std()
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[test_mean], std=[test_std])   ])

train_data = CustomDataset(data_array=TRAIN_array,root_dir=train_folder,transform=transform)
val_data = CustomDataset(data_array=VALIDATION_array,root_dir=validation_folder,transform=transform)
#test_image_tensors = CustomDataset(data_array=test_image_ids ,root_dir='test_data',transform=transform)

test_image_tensors = []
for img in test_images:
    transformed_img = transform_test(img).float()  # float32 biztosítás
    test_image_tensors.append(transformed_img)
test_image_tensors = torch.stack(test_image_tensors)
logging.info(f"test_image_tensors : {test_image_tensors.shape}")  # [db, type, x, y]


# Modell definíció
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, output_dim):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)  # Input has 3 channels
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        # Fully connected layers with dropout
        self.fc_1 = nn.Linear(64 * 29 * 29, 512)
        self.fc_2 = nn.Linear(512, 120)
        self.fc_3 = nn.Linear(120, output_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.2)  # 50% dropout
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)  # [batch size, 32, H1, W1]
        x = self.pool(self.act(x))  # [batch size, 32, H1//2, W1//2]
        x = self.conv2(x)  # [batch size, 64, H2, W2]
        x = self.pool(self.act(x))  # [batch size, 64, H2//2, W2//2]

        # Flatten the data
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.act(x)
        x = self.dropout1(x)  # Apply dropout after first fully connected layer
        x = self.fc_2(x)
        x = self.act(x)
        x = self.dropout2(x)  # Apply dropout after second fully connected layer
        x = self.fc_3(x)
        return x


def compute_class_weights(labels, num_classes):
        # A címkék előfordulásának megszámolása
        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        for label in labels:
            class_counts[label] += 1
        total_samples = len(labels)
        class_weights = total_samples / (num_classes * class_counts)

        return class_weights


labels = data_array[:, 1].astype(int).tolist()
print(labels)
num_classes = 21
weights = compute_class_weights(labels, num_classes)
print("Osztálysúlyok:", weights)
scaled_weights = torch.log1p(weights)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  ---> szarabb






for num_epochs, batch_size in configurations:

    logging.info(f"batch_size : {batch_size}")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    model = CustomCNN(21)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=scaled_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    logging.info("optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)")

    train_loss_history = []
    for epoch in range(num_epochs):
        # Tanítási fázis
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        current_lr = 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()  # Gradiensek nullázása

            # Előrehaladás és veszteség kiszámítás
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Visszaterjesztés
            optimizer.step()  # Optimalizáció

            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(train_loss)
        train_accuracy = 100. * correct_train / total_train
        if len(train_loss_history) >= 30 and all(train_loss_history[i] < train_loss_history[i + 1] for i in range(-30, -1) ):
            logging.warning("Train loss folyamatosan növekszik az utolsó 30 epochban. Következő konfiguráció...")
            break



        # Validációs fázis
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100. * correct_val / total_val
        current_lr = optimizer.param_groups[0]['lr']
        if  val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_acc_batch  = batch_size
            best_val_acc_at  = epoch+1
        # Epoch eredmények kiíratása
        logging.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, LR: {current_lr:.6f}')

        if val_accuracy  > 15 :
            evaluate_model(model,test_image_ids, label_map, best_val_acc_at, best_val_accuracy,batch_size,test_image_tensors,dev )
    logging.info(f'Best ACC {best_val_accuracy}% at E = {best_val_acc_at}.4f where B = {best_val_acc_batch}')

print("Tanítás befejeződött!")

# evaluate_model(model,test_image_ids, label_map, best_val_acc_at, best_val_accuracy,batch_size,test_image_tensors,dev )