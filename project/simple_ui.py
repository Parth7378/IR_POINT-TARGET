import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load both generators
G1 = tf.keras.models.load_model("G_MD.h5", compile=False)
G2 = tf.keras.models.load_model("G_FA.h5", compile=False)

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def run_model(image_path):
    image = preprocess_image(image_path)
    pred1 = G1(image, training=False)
    pred2 = G2(image, training=False)
    final_pred = (pred1 + pred2) / 2.0
    return pred1[0, ..., 0].numpy(), pred2[0, ..., 0].numpy(), final_pred[0, ..., 0].numpy()

def display_results(img_path):
    pred1, pred2, final = run_model(img_path)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(pred1, cmap='gray')
    axs[0].set_title("G1 Output")
    axs[1].imshow(pred2, cmap='gray')
    axs[1].set_title("G2 Output")
    axs[2].imshow(final, cmap='gray')
    axs[2].set_title("Final Prediction")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
    if filepath:
        display_results(filepath)

# UI setup
root = tk.Tk()
root.title("IR Target Prediction")

label = tk.Label(root, text="Select an IR Image (.png)", font=("Arial", 14))
label.pack(pady=10)

btn = tk.Button(root, text="Browse Image", command=open_file, font=("Arial", 12))
btn.pack(pady=20)

root.mainloop()
