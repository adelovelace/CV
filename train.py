import torch
import torch.nn as nn
from utils.data_loader import get_data_loaders
from models.mobilenet_model import get_mobilenet_v3_custom # Suponiendo que lo moviste 
from config import get_options

# Configuración
DATA_DIR = './data'
IMG_SIZE = 224 # Cambiar a 299 si usas Inception
BATCH_SIZE = 32
NUM_CLASSES = 7
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, IMG_SIZE, BATCH_SIZE)
print(f"Clases detectadas: {class_names}")

model = get_mobilenet_v3_custom(num_classes=len(class_names))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


def train_one_epoch(model, loader, criterion, optimizer, is_inception=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if is_inception:
            # Inception devuelve (salida_principal, salida_auxiliar)
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, 100. * correct / total


if __name__ == "__main__":

    opt = get_options()
    print(f"Entrenando por {opt.epoch} épocas con un LR de {opt.lr}")
    
    MODELO_A_USAR = "mobilenet"  
    # ----------------------------------

    if MODELO_A_USAR == "inception":
        img_size = 299
        model = get_inception_v3_custom(NUM_CLASSES)
        is_inc = True
    else:
        img_size = 224
        model = get_mobilenet_v3_custom(NUM_CLASSES)
        is_inc = False

    # Filtramos solo los parámetros que NO están congelados para el optimizador
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Modelo: {MODELO_A_USAR}")
    print(f"Parámetros a entrenar: {sum(p.numel() for p in params_to_update)}")
    
    # Aquí cargarías tus datos reales:
    # train_loader = DataLoader(datasets.ImageFolder('path/train', transform=get_transforms(img_size)), batch_size=BATCH_SIZE, shuffle=True)
    
    print("\n[!] Listo para entrenar. Asegúrate de apuntar a la carpeta de tu dataset.")