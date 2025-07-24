import tensorflow as tf
import os

def get_image_mask_pairs(train_dir):
    all_files = sorted(os.listdir(train_dir))
    image_files = [f for f in all_files if f.endswith('_1.png')]
    mask_files = [f.replace('_1.png', '_2.png') for f in image_files]
    return image_files, mask_files

def load_train_pair(image_file, mask_file, train_dir, resize_to=(128, 128)):
    image_path = tf.strings.join([train_dir, image_file], separator=os.sep)
    mask_path = tf.strings.join([train_dir, mask_file], separator=os.sep)

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, resize_to)
    image = tf.image.convert_image_dtype(image, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, resize_to)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    return image, mask

def load_train_dataset(train_dir, batch_size=8, resize_to=(128, 128)):
    train_dir = tf.convert_to_tensor(train_dir)
    image_files, mask_files = get_image_mask_pairs(train_dir.numpy().decode())

    image_ds = tf.data.Dataset.from_tensor_slices(image_files)
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_files)

    dataset = tf.data.Dataset.zip((image_ds, mask_ds))
    dataset = dataset.map(lambda i, m: load_train_pair(i, m, train_dir, resize_to),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
