



ezeket kell figyelni :

    - Validation Loss (Val Loss) követése.
    Learning Rate (tanulási ráta) változások logolása.
    Early Stopping implementálása.
    Túlilleszkedés (Overfitting) felismerése és kezelése.




import time  # Training idő méréséhez
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # Confusion Matrix és F1-score
import seaborn as sns
import matplotlib.pyplot as plt

# Epoch ciklusban
for epoch in range(start_epoch, num_epochs):
    start_time = time.time()  # Epoch start time

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
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']  # Current Learning Rate

    # Validation loss kiszámítása
    val_loss = 0.0
    correct_count = 0
    total_count = 0

    if validate_image_tensors is not None:
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(dev), val_labels.to(dev).long()
                outputs = model(val_images)
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_count += (predicted == val_labels).sum().item()
                total_count += val_labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_count / total_count

    # Logolás
    logging.info(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {current_lr:.6f}"
    )

    epoch_time = time.time() - start_time
    logging.info(f"Training time for epoch {epoch + 1}: {epoch_time:.2f} seconds")

    # Confusion Matrix számítása
    if validate_image_tensors is not None:
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(dev)
                outputs = model(val_images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Confusion Matrix mentése
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Epoch {epoch + 1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'confusion_matrix_epoch_{epoch + 1}.png')
        plt.close()

    # Early Stopping feltétel
    if val_loss > t_loss_min:
        patience_counter += 1
        if patience_counter > patience_limit:
            logging.info("Early stopping triggered.")
            break
    else:
        t_loss_min = val_loss
        patience_counter = 0
