import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from collections import Counter

# Import your models
from models.mobilenetv2_rnn import MobileNetV2_RNN
from models.inceptionv3_rnn import InceptionV3_RNN
from models.custom_cnn_rnn import CustomCNN_RNN

# ==========================================
# 1. Configuration
# ==========================================
MODEL_CHOICE = 'custom'      # Change to 'mobilenet' or 'inception' based on your best model
WEIGHTS_PATH = './outputs/checkpoints/custom/fer2013_best.pth' # Path to your weights
VIDEO_FOLDER = './data/depvidmood' # Path where you extracted the Kaggle video dataset

IMG_SIZE = 48                # 48 for custom, 224 for mobilenet, 299 for inception
NUM_CLASSES = 7
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Setup Model & Processing
# ==========================================
print(f"[!] Loading {MODEL_CHOICE.upper()} Model...")
if MODEL_CHOICE == "inception":
    model = InceptionV3_RNN(num_classes=NUM_CLASSES)
elif MODEL_CHOICE == "mobilenet":
    model = MobileNetV2_RNN(num_classes=NUM_CLASSES)
elif MODEL_CHOICE == "custom":
    model = CustomCNN_RNN(num_classes=NUM_CLASSES)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==========================================
# 3. Video Processing Pipeline
# ==========================================
if not os.path.exists(VIDEO_FOLDER):
    print(f"[-] Error: Could not find video folder at {VIDEO_FOLDER}")
    print("Please download the Kaggle dataset and extract it there.")
    exit()

video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
print(f"[!] Found {len(video_files)} videos to process.\n")

for video_file in video_files:
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)
    
    frame_predictions = []
    
    # Process every 5th frame to speed up inference significantly
    frame_count = 0 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 5 != 0:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        
        # If faces are found, take the largest one
        if len(faces) > 0:
            # Sort by area (w*h) and grab the biggest box (most likely the main subject)
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            face_crop = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                frame_predictions.append(CLASS_NAMES[preds.item()])

    cap.release()
    
    # Calculate the final emotion using a Majority Vote
    if len(frame_predictions) > 0:
        most_common_emotion = Counter(frame_predictions).most_common(1)[0][0]
        print(f"🎬 Video: {video_file: <25} | 🧠 Final Emotion: {most_common_emotion} (Based on {len(frame_predictions)} faces)")
    else:
        print(f"🎬 Video: {video_file: <25} | ⚠️ No faces clearly detected.")

print("\n[!] Video processing complete.")