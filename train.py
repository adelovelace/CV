import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score

import wandb
import cv2

# --- NEW: Grad-CAM Imports ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import opt
from utils.data_loader import get_data_loaders, download_kaggle_datasets
from utils.training_utils import clip_gradient, adjust_lr
from models.mobilenetv2_rnn import MobileNetV2_RNN
from models.inceptionv3_rnn import InceptionV3_RNN
from models.custom_cnn_rnn import CustomCNN_RNN

# ==========================================
# Configuration & Hardware Setup
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODEL_CHOICE = opt.model
DATASET_CHOICE = opt.dataset

if MODEL_CHOICE == "mobilenet" and opt.trainsize != 224:
    opt.trainsize = 224
elif MODEL_CHOICE == "inception" and opt.trainsize != 299:
    opt.trainsize = 299
elif MODEL_CHOICE == "custom" and opt.trainsize != 48:
    opt.trainsize = 48

PATIENCE = getattr(opt, 'patience', 15)

# ==========================================
# Training and Evaluation Functions
# ==========================================
def train_model(model, dataloaders, class_names, criterion, optimizer, num_epochs, init_lr, log_dir, ckpt_dir):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    log_file_path = os.path.join(log_dir, f'training_{MODEL_CHOICE}_{DATASET_CHOICE}.log')
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Training for {num_epochs} Epochs...\n")
        log_file.write("-" * 50 + "\n")
    
    for epoch in range(num_epochs):
        current_lr = adjust_lr(optimizer, init_lr, epoch, decay_rate=opt.decay_rate, decay_epoch=opt.decay_epoch)
        
        print(f'\nEpoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}]')
        print('-' * 15)

        wandb_metrics = {"epoch": epoch + 1, "learning_rate": current_lr}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            epoch_preds = []
            epoch_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        clip_gradient(optimizer, opt.clip)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / total_samples
            epoch_acc = (running_corrects.double() / total_samples).item() 
            
            epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro', zero_division=0)
            epoch_recall = recall_score(epoch_labels, epoch_preds, average='macro', zero_division=0)

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                wandb_metrics["Train Loss"] = epoch_loss
                wandb_metrics["Train Accuracy"] = epoch_acc
                wandb_metrics["Train F1"] = epoch_f1
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                wandb_metrics["Val Loss"] = epoch_loss
                wandb_metrics["Val Accuracy"] = epoch_acc
                wandb_metrics["Val F1"] = epoch_f1
                wandb_metrics["Val Recall"] = epoch_recall

            log_msg = f"Phase: {phase.capitalize()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f} | Recall: {epoch_recall:.4f}\n"
            print(log_msg.strip())
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Epoch {epoch+1}/{num_epochs} | " + log_msg)

            if phase == 'val':
                
                # --- Dynamic Filename Logic ---
                if MODEL_CHOICE in ["custom", "scratch"]:
                    best_filename = f"{DATASET_CHOICE}_best.pth"
                    epoch_filename = f"{DATASET_CHOICE}_epoch{epoch+1}.pth"
                else:
                    best_filename = f"{DATASET_CHOICE}_frozen{opt.frozen}_best.pth"
                    # For fine-tuning, overwriting the exact frozen filename every epoch as requested
                    epoch_filename = f"{DATASET_CHOICE}_frozen{opt.frozen}.pth"
                # -----------------------------------

                # 1. Save Best Model (Based on Accuracy)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    best_path = os.path.join(ckpt_dir, best_filename)
                    torch.save(best_model_wts, best_path)
                    print(f"[*] New High Score! Best model saved to: {best_path}")
                
                # 2. Early Stopping Tracking (Based on Loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0  
                else:
                    epochs_no_improve += 1
                    
        wandb.log(wandb_metrics)

        # Save the current epoch (or overwrite the frozen file)
        epoch_ckpt_path = os.path.join(ckpt_dir, epoch_filename)
        torch.save(model.state_dict(), epoch_ckpt_path)

        if epochs_no_improve >= PATIENCE:
            stop_msg = f"\n[!] EARLY STOPPING TRIGGERED: Validation loss has not improved for {PATIENCE} epochs.\n"
            print(stop_msg)
            with open(log_file_path, "a") as log_file:
                log_file.write(stop_msg)
            break

    time_elapsed = time.time() - since
    summary_msg = f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\nBest Val Acc: {best_acc:4f}\n'
    print(summary_msg)
    with open(log_file_path, "a") as log_file:
        log_file.write(summary_msg + "-" * 50 + "\n")

    model.load_state_dict(best_model_wts)
    model.eval()
    
    print("[!] Generating Final Confusion Matrix and Evaluation Metrics...")
    final_preds = []
    final_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    report = classification_report(final_labels, final_preds, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n", report)
    with open(log_file_path, "a") as log_file:
        log_file.write("\nFinal Classification Report:\n" + report + "\n")

    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({MODEL_CHOICE.upper()})')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    cm_path = os.path.join(log_dir, f'confusion_matrix_{MODEL_CHOICE}_{DATASET_CHOICE}.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"[!] Confusion Matrix saved to: {cm_path}")

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='red', linewidth=2, linestyle='dashed')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='red', linewidth=2, linestyle='dashed')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(log_dir, f'learning_curves_{MODEL_CHOICE}_{DATASET_CHOICE}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    wandb.log({
        "Learning Curves (Plot)": wandb.Image(plot_path),
        "Confusion Matrix": wandb.Image(cm_path)
    })
    wandb.save(os.path.join(ckpt_dir, f'best_{MODEL_CHOICE}_{DATASET_CHOICE}.pth'))

    return model

# --- NEW: Grad-CAM Generation Function ---
def generate_and_log_gradcam(model, val_loader, class_names, device, log_dir, model_choice):
    print("\n[!] Generating Grad-CAM Heatmaps...")
    model.eval()
    
    # Identify the correct target layer based on the architecture
    target_layers = []
    if model_choice == "custom" or model_choice == "scratch":
        # The last Conv2d layer in your CustomCNN_RNN Sequential block is at index 17
        target_layers = [model.cnn[17]] 
    elif model_choice == "mobilenet":
        target_layers = [model.features[-1]]
    elif model_choice == "inception":
        target_layers = [model.features.Mixed_7c]
    else:
        print("[-] Grad-CAM not configured for this architecture yet.")
        return

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
    except Exception as e:
        print(f"[-] Could not initialize Grad-CAM: {e}")
        return
    
    # Grab a single batch of validation images
    inputs, labels = next(iter(val_loader))
    
    # Take the first 8 images from the batch to visualize
    inputs = inputs[:8].to(device)
    labels = labels[:8].to(device)
    
    wandb_images = []
    
    for i in range(inputs.size(0)):
        input_tensor = inputs[i].unsqueeze(0) # Shape: (1, C, H, W)
        true_label = labels[i].item()
        target = [ClassifierOutputTarget(true_label)]
        
        # Generate the grayscale Grad-CAM mask
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0, :]
        
        # Un-normalize the image for display purposes
        img = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Overlay the heatmap on the original image
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        # Get the model's prediction for the caption
        with torch.no_grad():
            output = model(input_tensor)
            pred_label = output.argmax(dim=1).item()
            
        caption = f"True: {class_names[true_label]} | Pred: {class_names[pred_label]}"
        
        # Save locally
        save_path = os.path.join(log_dir, f'gradcam_{model_choice}_{i}.png')
        cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        # Append to W&B list
        wandb_images.append(wandb.Image(visualization, caption=caption))
        
    if wandb_images:
        wandb.log({"Grad-CAM Heatmaps": wandb_images})
        print(f"[!] Saved 8 Grad-CAM visualizations to {log_dir} and W&B.")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    
    wandb.init(
        entity="usi-cv",
        project="sentiment-project",
        name=f"Run_{MODEL_CHOICE}_{DATASET_CHOICE}",
        config={
            "architecture": MODEL_CHOICE,
            "dataset": DATASET_CHOICE,
            "epochs": opt.epoch,
            "learning_rate": opt.lr,
            "batch_size": opt.batchsize,
            "image_size": opt.trainsize,
            "grad_clip": opt.clip,
            "decay_rate": opt.decay_rate,
            "decay_interval": opt.decay_epoch,
            "early_stopping_patience": PATIENCE,
        }
    )
    
    base_save_dir = opt.save_path if '*' not in opt.save_path else './outputs'
    log_dir = os.path.join(base_save_dir, 'logs')
    ckpt_dir = os.path.join(base_save_dir, 'checkpoints', MODEL_CHOICE)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"[!] Using device: {DEVICE} (GPU ID: {opt.gpu_id})")
    
    fer_path, ck_path = download_kaggle_datasets('./data')
    dataset_path = fer_path if DATASET_CHOICE == "fer2013" else ck_path
    
    img_size = opt.trainsize 
    batch_size = opt.batchsize
    
    train_loader, val_loader, class_names = get_data_loaders(dataset_path, img_size, batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}
    NUM_CLASSES = len(class_names)

    if MODEL_CHOICE == "inception":
        model = InceptionV3_RNN(num_classes=NUM_CLASSES)
    elif MODEL_CHOICE == "mobilenet":
        model = MobileNetV2_RNN(num_classes=NUM_CLASSES)
    elif MODEL_CHOICE == "custom":
        model = CustomCNN_RNN(num_classes=NUM_CLASSES)
        
    model = model.to(DEVICE)

    if opt.load and os.path.exists(opt.load):
        model.load_state_dict(torch.load(opt.load, map_location=DEVICE))

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[!] Starting training for {opt.epoch} epochs (Early Stopping Patience: {PATIENCE})...")
    best_model = train_model(
        model=model, 
        dataloaders=dataloaders, 
        class_names=class_names,  
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=opt.epoch, 
        init_lr=opt.lr,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir
    )
    
    # --- NEW: Generate Grad-CAM before finishing ---
    generate_and_log_gradcam(
        model=best_model, 
        val_loader=dataloaders['val'], 
        class_names=class_names, 
        device=DEVICE, 
        log_dir=log_dir, 
        model_choice=MODEL_CHOICE
    )
    
    print("[!] Training loop finished. Closing W&B run.")
    wandb.finish()