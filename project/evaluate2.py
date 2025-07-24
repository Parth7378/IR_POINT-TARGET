from model_gan import build_generator
import tensorflow as tf
from test_dataset import load_test_data
import matplotlib.pyplot as plt

# Load test data
test_ds = load_test_data('./test_org', './test_gt')

# Load trained generators
G_MD = tf.keras.models.load_model('G_MD.h5')
G_FA = tf.keras.models.load_model('G_FA.h5')

# Predict and visualize
for imgs, masks in test_ds.take(1):
    pred1 = G_MD(imgs)
    pred2 = G_FA(imgs)
    combined = (pred1 + pred2) / 2.0

    for i in range(imgs.shape[0]):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].imshow(imgs[i, ..., 0], cmap='gray')
        axs[0].set_title("Image")
        axs[1].imshow(masks[i, ..., 0], cmap='gray')
        axs[1].set_title("GT")
        axs[2].imshow(pred1[i, ..., 0], cmap='gray')
        axs[2].set_title("G_MD")
        axs[3].imshow(combined[i, ..., 0], cmap='gray')
        axs[3].set_title("Final Pred")
        plt.show()
