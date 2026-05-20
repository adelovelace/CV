import argparse
import csv
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

try:
    import cv2
except ImportError:
    cv2 = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.custom_cnn_rnn import CustomCNN_RNN
from models.inceptionv3_rnn import InceptionV3_RNN
from models.mobilenetv2_rnn import MobileNetV2_RNN


FER2013_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CKPLUS_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Sadness", "Surprise"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate unlabelled videos with a trained FER model using frame-level inference and temporal aggregation."
    )
    parser.add_argument("--video-dir", type=str, default="./data/depvidmood", help="Folder containing videos.")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/best_inception_fer2013.pth")
    parser.add_argument("--model", type=str, default="inception", choices=["custom", "mobilenet", "inception"])
    parser.add_argument("--dataset", type=str, default="fer2013", choices=["fer2013", "ckplus"])
    parser.add_argument("--class-names", type=str, default="", help="Comma-separated labels. Overrides --dataset.")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])

    parser.add_argument("--frame-step", type=int, default=5, help="Process every Nth frame.")
    parser.add_argument("--max-videos", type=int, default=0, help="Optional limit. 0 processes all videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit per video. 0 means no limit.")
    parser.add_argument("--min-face-size", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--no-face-crop", action="store_true", help="Run inference on full frames instead of detected face crops.")

    parser.add_argument("--output-dir", type=str, default="./outputs/video_results")
    parser.add_argument("--save-timelines", action="store_true", help="Save one probability timeline plot per video.")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated videos with predicted emotions.")
    parser.add_argument("--pseudo-labels", action="store_true", help="Save high-confidence pseudo-label candidates.")
    parser.add_argument("--min-majority-ratio", type=float, default=0.60)
    parser.add_argument("--min-confidence", type=float, default=0.70)

    return parser.parse_args()


def require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV is required for video_evaluator.py. Install it with: pip install opencv-python")


def get_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_class_names(args):
    if args.class_names:
        return [name.strip() for name in args.class_names.split(",") if name.strip()]
    if args.dataset == "ckplus":
        return CKPLUS_CLASSES
    return FER2013_CLASSES


def get_image_size(model_name, override):
    if override is not None:
        return override
    if model_name == "mobilenet":
        return 224
    if model_name == "inception":
        return 299
    return 48


def build_model(model_name, num_classes):
    if model_name == "custom":
        return CustomCNN_RNN(num_classes=num_classes)
    if model_name == "mobilenet":
        return MobileNetV2_RNN(num_classes=num_classes)
    if model_name == "inception":
        return InceptionV3_RNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key.replace("module.", "")] = value

    model.load_state_dict(cleaned, strict=True)
    return model


def make_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def create_face_detector():
    require_cv2()
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade face detector.")
    return detector


def list_videos(video_dir, max_videos):
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if max_videos > 0:
        videos = videos[:max_videos]
    return videos


def detect_largest_face(frame_bgr, detector, min_face_size):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def crop_with_margin(frame_bgr, box, margin_ratio):
    if box is None:
        return frame_bgr, None

    x, y, w, h = box
    margin = int(max(w, h) * margin_ratio)
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, frame_bgr.shape[1])
    y2 = min(y + h + margin, frame_bgr.shape[0])
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict_frame(model, frame_bgr, transform, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    return int(np.argmax(probabilities)), probabilities


def annotate_frame(frame, box, label, confidence):
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        text_origin = (x1, max(y1 - 10, 24))
    else:
        text_origin = (10, 34)

    cv2.putText(
        frame,
        f"{label}: {confidence:.1%}",
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 220, 80),
        2,
        cv2.LINE_AA,
    )
    return frame


def save_timeline_plot(video_name, frame_indices, probability_history, class_names, majority_label, output_dir):
    if len(probability_history) == 0:
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = np.vstack(probability_history)
    plt.figure(figsize=(12, 5))
    for class_idx, class_name in enumerate(class_names):
        plt.plot(frame_indices, probabilities[:, class_idx], label=class_name, linewidth=1.8)

    plt.title(f"Frame-level emotion probabilities: {video_name}\nMajority vote: {majority_label}")
    plt.xlabel("Frame index")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", ncol=2, fontsize=8)

    safe_name = Path(video_name).stem.replace(" ", "_")
    output_path = output_dir / f"{safe_name}_timeline.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)


def summarize_video(video_path, predictions, confidences, probability_history, class_names, frame_count, processed_count, face_count):
    empty_probs = np.zeros(len(class_names), dtype=float)
    if not predictions:
        return {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "face_frames": face_count,
            "face_detection_rate": 0.0,
            "majority_emotion": "No face detected",
            "majority_count": 0,
            "majority_ratio": 0.0,
            "average_confidence": 0.0,
            **{f"avg_prob_{name.lower()}": float(empty_probs[i]) for i, name in enumerate(class_names)},
        }

    counts = Counter(predictions)
    majority_emotion, majority_count = counts.most_common(1)[0]
    majority_ratio = majority_count / len(predictions)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    avg_probs = np.mean(np.vstack(probability_history), axis=0)
    detection_rate = face_count / processed_count if processed_count else 0.0

    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "total_frames": frame_count,
        "processed_frames": processed_count,
        "face_frames": face_count,
        "face_detection_rate": detection_rate,
        "majority_emotion": majority_emotion,
        "majority_count": majority_count,
        "majority_ratio": majority_ratio,
        "average_confidence": avg_confidence,
        **{f"avg_prob_{name.lower()}": float(avg_probs[i]) for i, name in enumerate(class_names)},
    }


def evaluate_video(video_path, model, transform, detector, class_names, device, args, annotated_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        out_path = annotated_dir / f"{video_path.stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    processed_count = 0
    face_count = 0
    predictions = []
    confidences = []
    probability_history = []
    frame_indices = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break

        should_process = frame_count % max(args.frame_step, 1) == 0
        if not should_process:
            if writer is not None:
                writer.write(frame)
            continue

        processed_count += 1
        box = None if args.no_face_crop else detect_largest_face(frame, detector, args.min_face_size)
        face_crop, expanded_box = crop_with_margin(frame, box, args.margin)

        if box is not None or args.no_face_crop:
            face_count += 1
            pred_idx, probabilities = predict_frame(model, face_crop, transform, device)
            label = class_names[pred_idx]
            confidence = float(probabilities[pred_idx])

            predictions.append(label)
            confidences.append(confidence)
            probability_history.append(probabilities)
            frame_indices.append(frame_count)

            if writer is not None:
                annotate_frame(frame, expanded_box, label, confidence)
        elif writer is not None:
            cv2.putText(frame, "No face detected", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    summary = summarize_video(
        video_path=video_path,
        predictions=predictions,
        confidences=confidences,
        probability_history=probability_history,
        class_names=class_names,
        frame_count=frame_count,
        processed_count=processed_count,
        face_count=face_count,
    )

    return summary, frame_indices, probability_history


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    require_cv2()
    output_dir = Path(args.output_dir)
    timeline_dir = output_dir / "timelines"
    annotated_dir = output_dir / "annotated"

    device = get_device(args.device)
    class_names = get_class_names(args)
    img_size = get_image_size(args.model, args.img_size)
    transform = make_transform(img_size)
    detector = None if args.no_face_crop else create_face_detector()

    print(f"[!] Device: {device}")
    print(f"[!] Model: {args.model}")
    print(f"[!] Checkpoint: {args.checkpoint}")
    print(f"[!] Video directory: {args.video_dir}")
    print(f"[!] Classes: {class_names}")
    print(f"[!] Frame step: {args.frame_step}")

    model = build_model(args.model, num_classes=len(class_names))
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    videos = list_videos(args.video_dir, args.max_videos)
    print(f"[!] Found {len(videos)} videos.")

    if len(videos) == 0:
        print("[!] No videos found. Check --video-dir and file extensions.")
        return

    rows = []
    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] Processing {video_path.name}")
        try:
            summary, frame_indices, probability_history = evaluate_video(
                video_path=video_path,
                model=model,
                transform=transform,
                detector=detector,
                class_names=class_names,
                device=device,
                args=args,
                annotated_dir=annotated_dir,
            )

            if args.save_timelines and summary["majority_count"] > 0:
                timeline_path = save_timeline_plot(
                    video_name=video_path.name,
                    frame_indices=frame_indices,
                    probability_history=probability_history,
                    class_names=class_names,
                    majority_label=summary["majority_emotion"],
                    output_dir=timeline_dir,
                )
                summary["timeline_path"] = timeline_path
            else:
                summary["timeline_path"] = ""

            rows.append(summary)
            print(
                f"    majority={summary['majority_emotion']} "
                f"ratio={summary['majority_ratio']:.2f} "
                f"avg_conf={summary['average_confidence']:.2f} "
                f"faces={summary['face_frames']}/{summary['processed_frames']}"
            )
        except Exception as exc:
            print(f"    [!] Skipped due to error: {exc}")

    if not rows:
        print("[!] No videos were successfully processed.")
        return

    fieldnames = list(rows[0].keys())
    summary_path = output_dir / "video_summary.csv"
    write_csv(summary_path, rows, fieldnames)
    print(f"\n[!] Saved summary CSV to: {summary_path.resolve()}")

    if args.pseudo_labels:
        pseudo_rows = [
            row for row in rows
            if row["majority_ratio"] >= args.min_majority_ratio
            and row["average_confidence"] >= args.min_confidence
            and row["majority_count"] > 0
        ]
        pseudo_path = output_dir / "pseudo_labels.csv"
        write_csv(pseudo_path, pseudo_rows, fieldnames)
        print(
            f"[!] Saved {len(pseudo_rows)} high-confidence pseudo-label candidates to: "
            f"{pseudo_path.resolve()}"
        )

    emotion_counts = Counter(row["majority_emotion"] for row in rows)
    print("\nPredicted video-level emotion distribution:")
    for emotion, count in emotion_counts.most_common():
        print(f"  {emotion:>16s}: {count}")

    print("\n[!] Reminder: DepVidMood is treated as unlabelled here, so these are predictions/generalization results, not accuracy.")


if __name__ == "__main__":
    main()
