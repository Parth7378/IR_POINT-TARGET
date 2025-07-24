import tensorflow as tf
import os

def load_test_image_mask(img_filename, org_dir, gt_dir, resize=(128, 128)):
    img_filename = img_filename.numpy().decode('utf-8')
    org_dir = org_dir.numpy().decode('utf-8')
    gt_dir = gt_dir.numpy().decode('utf-8')

    org_path = os.path.join(org_dir, img_filename)
    gt_path = os.path.join(gt_dir, img_filename)

    # Load test image
    image = tf.io.read_file(org_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, resize)
    image = tf.cast(image, tf.float32) / 255.0

    # Load ground truth mask
    mask = tf.io.read_file(gt_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, resize)
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask

def wrap_test_loader(filename, org_dir, gt_dir):
    image, mask = tf.py_function(
        func=load_test_image_mask,
        inp=[filename, org_dir, gt_dir],
        Tout=[tf.float32, tf.float32]
    )
    image.set_shape([128, 128, 1])
    mask.set_shape([128, 128, 1])
    return image, mask


def load_test_data(org_dir='./test_org', gt_dir='./test_gt', batch_size=4):
    filenames = sorted(os.listdir(org_dir))
    filenames = [f for f in filenames if f.endswith('.png')]

    org_dir_tensor = tf.constant(org_dir)
    gt_dir_tensor = tf.constant(gt_dir)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda f: wrap_test_loader(f, org_dir_tensor, gt_dir_tensor),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
