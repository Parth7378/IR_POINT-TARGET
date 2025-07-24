import tensorflow as tf
import os

def load_image_mask(image_path, mask_dir, resize_to=(128, 128)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, resize_to)
    image = tf.image.convert_image_dtype(image, tf.float32)

    filename = tf.strings.split(image_path, os.sep)[-1]
    mask_path = tf.strings.join([mask_dir, filename])
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, resize_to)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    return image, mask

def build_dataset(image_dir, mask_dir, batch_size=8, resize_to=(128, 128)):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: load_image_mask(x, mask_dir, resize_to), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).shuffle(50).prefetch(tf.data.AUTOTUNE)
    return dataset
