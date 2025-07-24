import tensorflow as tf
import os

# ==== 1. ACTUAL LOADER FUNCTION ====
def load_train_image_mask(image_filename, train_dir='./training', resize=(128, 128)):
    image_filename = image_filename.numpy().decode()  # e.g., 000000_1.png
    image_path = os.path.join(train_dir, image_filename)
    mask_path = image_path.replace('_1.png', '_2.png')

    # Read and preprocess image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, resize)
    image = tf.cast(image, tf.float32) / 255.0

    # Read and preprocess mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, resize)
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask

# ==== 2. WRAPPER FOR tf.py_function + SET SHAPES ====
def wrap_py_function(filename):
    image, mask = tf.py_function(
        func=load_train_image_mask,
        inp=[filename],
        Tout=[tf.float32, tf.float32]
    )
    image.set_shape([128, 128, 1])
    mask.set_shape([128, 128, 1])
    return image, mask

# ==== 3. MAIN DATASET FUNCTION ====
def tf_load_train_image_mask(train_dir='./trainning', batch_size=4, resize=(128, 128)):
    filenames = sorted([f for f in os.listdir(train_dir) if f.endswith('_1.png')])
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(wrap_py_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(50).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
