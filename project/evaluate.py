from dataset import load_test_data
from model import build_model
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model and test data
model = tf.keras.models.load_model('ir_segmentor.h5')
test_ds = load_test_data('./test_org', './test_gt')

# Predict and visualize
for images, masks in test_ds.take(1):
    preds = model.predict(images)

    for i in range(images.shape[0]):
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].imshow(images[i, ..., 0], cmap='gray')
        ax[0].set_title("IR Image")
        ax[1].imshow(masks[i, ..., 0], cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(preds[i, ..., 0], cmap='gray')
        ax[2].set_title("Predicted Mask")
        plt.show()
