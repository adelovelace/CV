import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from datetime import datetime

# 1. Import the configuration arguments
from config import opt

# Import custom modules
from utils.data_loader import get_data_loaders, download_kaggle_datasets
from utils.training_utils import clip_gradient, adjust_lr
from models.mobilenetv2_rnn import MobileNetV2_RNN
from models.inceptionv3_rnn import InceptionV3_RNN

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


# Automatically fix image size if the user forgets to set it for MobileNet
if MODEL_CHOICE == "mobilenet" and opt.trainsize != 224:
    print("[!] MobileNetV2 selected: Automatically setting image size to 224x224")
    opt.trainsize = 224
elif MODEL_CHOICE == "inception" and opt.trainsize != 299:
    print("[!] InceptionV3 selected: Automatically setting image size to 299x299")
    opt.trainsize = 299

# ==========================================
# Training and Evaluation Functions
# ==========================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs, init_lr, log_dir, ckpt_dir):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Initialize the log file
    log_file_path = os.path.join(log_dir, f'training_{MODEL_CHOICE}_{DATASET_CHOICE}.log')
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Training for {num_epochs} Epochs...\n")
        log_file.write("-" * 50 + "\n")
    
    for epoch in range(num_epochs):
        current_lr = adjust_lr(optimizer, init_lr, epoch, decay_rate=opt.decay_rate, decay_epoch=opt.decay_epoch)
        
        print(f'\nEpoch {epoch+1}/{num_epochs} [LR: {current_lr:.6f}]')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

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

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            # Print to console and write to log file
            log_msg = f"Epoch {epoch+1}/{num_epochs} | Phase: {phase.capitalize()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}\n"
            print(log_msg.strip())
            with open(log_file_path, "a") as log_file:
                log_file.write(log_msg)

            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save the "best" model dynamically
                best_path = os.path.join(ckpt_dir, f'best_{MODEL_CHOICE}_{DATASET_CHOICE}.pth')
                torch.save(best_model_wts, best_path)

        # Save an explicit checkpoint at the end of every epoch
        epoch_ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}_{MODEL_CHOICE}.pth')
        torch.save(model.state_dict(), epoch_ckpt_path)

    time_elapsed = time.time() - since
    summary_msg = f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\nBest Val Acc: {best_acc:4f}\n'
    print(summary_msg)
    with open(log_file_path, "a") as log_file:
        log_file.write(summary_msg + "-" * 50 + "\n")

    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    
    # 1. Setup Directories for outputs, logs, and checkpoints
    base_save_dir = opt.save_path if '*' not in opt.save_path else './outputs'
    log_dir = os.path.join(base_save_dir, 'logs')
    ckpt_dir = os.path.join(base_save_dir, 'checkpoints')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"[!] Using device: {DEVICE} (GPU ID: {opt.gpu_id})")
    print(f"[!] Logs saving to: {log_dir}")
    print(f"[!] Checkpoints saving to: {ckpt_dir}")
    
    # Download/Verify Kaggle Data
    fer_path, ck_path = download_kaggle_datasets('./data')
    dataset_path = fer_path if DATASET_CHOICE == "fer2013" else ck_path
    
    img_size = opt.trainsize 
    batch_size = opt.batchsize
    
    train_loader, val_loader, class_names = get_data_loaders(dataset_path, img_size, batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}
    NUM_CLASSES = len(class_names)
    print(f"[!] Detected {NUM_CLASSES} classes. Image Size: {img_size}x{img_size}")

    # Initialize Model
    if MODEL_CHOICE == "inception":
        model = InceptionV3_RNN(num_classes=NUM_CLASSES)
    else:
        model = MobileNetV2_RNN(num_classes=NUM_CLASSES)
        
    model = model.to(DEVICE)

    # Load model from checkpoint if provided in config
    if opt.load and os.path.exists(opt.load):
        print(f"[!] Loading checkpoint weights from: {opt.load}")
        model.load_state_dict(torch.load(opt.load, map_location=DEVICE))

    # Setup Optimizer
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    # Train Model
    print(f"\n[!] Starting training for {opt.epoch} epochs...")
    best_model = train_model(
        model=model, 
        dataloaders=dataloaders, 
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=opt.epoch, 
        init_lr=opt.lr,
        log_dir=log_dir,          # Passed dynamically
        ckpt_dir=ckpt_dir         # Passed dynamically
    )
    
    print("[!] Training loop finished. The best weights are saved in the checkpoints directory.")