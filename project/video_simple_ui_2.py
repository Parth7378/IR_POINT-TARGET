import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
import os
import subprocess

# Load models
G1 = tf.keras.models.load_model("G_MD.h5", compile=False)
G2 = tf.keras.models.load_model("G_FA.h5", compile=False)

# Resize with padding (to preserve aspect ratio)
def resize_with_padding(img, target_size=(128, 128)):
    old_size = img.shape[:2]  # (height, width)
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size])

    img_resized = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h - delta_h//2
    left, right = delta_w//2, delta_w - delta_w//2

    new_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_img

# Preprocess each frame
def preprocess_frame(frame):
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    resized = resize_with_padding(gray, target_size=(128, 128))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))  # Shape: (1, 128, 128, 1)

# Postprocess: overlay GAN output on original frame
def postprocess_frame(original_frame, output, original_size):
    mask = cv2.resize((output * 255).astype(np.uint8), original_size)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    if len(original_frame.shape) == 2:
        original_bgr = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_frame

    blended = cv2.addWeighted(original_bgr, 0.7, heatmap, 0.3, 0)
    return blended

# Process the video file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = preprocess_frame(frame)
        pred1 = G1(input_frame, training=False)
        pred2 = G2(input_frame, training=False)
        final_pred = (pred1 + pred2) / 2.0
        final_pred_np = final_pred[0, ..., 0].numpy()

        output_frame = postprocess_frame(frame, final_pred_np, (frame_width, frame_height))
        out.write(output_frame)

    cap.release()
    out.release()

    # Automatically play the video (Windows only)
    abs_path = os.path.abspath(output_path)
    subprocess.run(['start', '', abs_path], shell=True)

# GUI file picker
def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if filepath:
        process_video(filepath)

# UI setup
root = tk.Tk()
root.title("IR Target Video Processor")

label = tk.Label(root, text="Select an IR Video File", font=("Arial", 14))
label.pack(pady=10)

btn = tk.Button(root, text="Browse Video", command=open_file, font=("Arial", 12))
btn.pack(pady=20)

root.mainloop()
