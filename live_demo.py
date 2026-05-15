import os
import cv2
import torch
import pyttsx3
import threading
from torchvision import transforms
from PIL import Image

# Import your models
from models.mobilenetv2_rnn import MobileNetV2_RNN
from models.inceptionv3_rnn import InceptionV3_RNN
from models.custom_cnn_rnn import CustomCNN_RNN

# ==========================================
# 1. Configuration
# ==========================================
MODEL_CHOICE = 'custom'      # Change to 'mobilenet' or 'inception'
WEIGHTS_PATH = './outputs/checkpoints/custom/fer2013_best.pth' # Path to your best weights
IMG_SIZE = 48                # 48 for custom, 224 for mobilenet, 299 for inception
NUM_CLASSES = 7
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Text-To-Speech (TTS) Engine
# ==========================================
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speaking speed

is_speaking = False

def speak_emotion(emotion_text):
    """Runs in a background thread so the video doesn't freeze"""
    global is_speaking
    if not is_speaking:
        is_speaking = True
        engine.say(f"You look {emotion_text}")
        engine.runAndWait()
        is_speaking = False

# ==========================================
# 3. Model & Video Setup
# ==========================================
print("[!] Loading Model...")
if MODEL_CHOICE == "inception":
    model = InceptionV3_RNN(num_classes=NUM_CLASSES)
elif MODEL_CHOICE == "mobilenet":
    model = MobileNetV2_RNN(num_classes=NUM_CLASSES)
elif MODEL_CHOICE == "custom":
    model = CustomCNN_RNN(num_classes=NUM_CLASSES)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Same transform used during validation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load OpenCV's built-in face tracker
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open Webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)
print("[!] Starting Video Stream. Press 'q' to quit.")

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale just for the face detector (it's faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw a box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the face out of the frame
        face_crop = frame[y:y+h, x:x+w]
        
        # Convert cropped face to RGB (PyTorch models expect RGB)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        
        # Apply transforms and add batch dimension: shape becomes (1, 3, H, W)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        # Predict Emotion
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            emotion = CLASS_NAMES[preds.item()]
            
        # Put the text above the face
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Trigger the Voice if the emotion changes (and it's not currently talking)
        if emotion != last_emotion and not is_speaking:
            last_emotion = emotion
            threading.Thread(target=speak_emotion, args=(emotion,), daemon=True).start()

    # Show the video window
    cv2.imshow('Live Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()