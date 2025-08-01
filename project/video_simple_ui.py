import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
import os
import platform
import subprocess

# Load both generators
G1 = tf.keras.models.load_model("G_MD.h5", compile=False)
G2 = tf.keras.models.load_model("G_FA.h5", compile=False)

frame_width = 128
frame_height = 128

# Resize with padding to keep aspect ratio
def resize_with_padding(img, target_size=(128, 128)):
    old_size = img.shape[:2]  # (height, width)
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size])

    img_resized = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

# Frame pre-processing
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    padded = resize_with_padding(gray, target_size=(128, 128))
    normalized = padded.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

# Resize back to original with padding removed
def postprocess_frame(output, original_size):
    output = (output * 255.0).astype(np.uint8)
    resized_back = cv2.resize(output, original_size)
    return resized_back

# Process the video frame by frame
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    global frame_width, frame_height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Save as .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess_frame(frame)
        pred1 = G1(inp, training=False)
        pred2 = G2(inp, training=False)
        final_pred = ((pred1 + pred2) / 2.0)[0, ..., 0].numpy()

        output_frame = postprocess_frame(final_pred, (frame_width, frame_height))
        out.write(output_frame)

    cap.release()
    out.release()

# File selection and video processing with auto-play
def open_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if filepath:
        output_path = os.path.splitext(filepath)[0] + "_processed.mp4"
        process_video(filepath, output_path)
        messagebox.showinfo("Success", f"Processed video saved as:\n{output_path}")

        # Automatically play the video on Windows
        try:
            if platform.system() == "Windows":
                os.startfile(output_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_path])
            else:  # Linux
                subprocess.run(["xdg-open", output_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video:\n{e}")

# Tkinter GUI setup
root = tk.Tk()
root.title("IR Target Prediction from Video")

label = tk.Label(root, text="Select a Video File", font=("Arial", 14))
label.pack(pady=10)

btn = tk.Button(root, text="Browse Video", command=open_video, font=("Arial", 12))
btn.pack(pady=20)

root.mainloop()
