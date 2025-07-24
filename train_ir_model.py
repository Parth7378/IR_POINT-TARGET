import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# ======== LOADERS =========
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

def load_train_dataset(train_dir, batch_size=4, resize_to=(128, 128)):
    image_files, mask_files = get_image_mask_pairs(train_dir)
    image_ds = tf.data.Dataset.from_tensor_slices(image_files)
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_files)
    dataset = tf.data.Dataset.zip((image_ds, mask_ds))
    dataset = dataset.map(lambda i, m: load_train_pair(i, m, train_dir, resize_to),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(50).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_test_dataset(test_img_dir, test_mask_dir, resize_to=(128, 128), batch_size=4):
    image_paths = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.png')])
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_test_pair(image_path):
        filename = tf.strings.split(image_path, os.sep)[-1]
        mask_path = tf.strings.join([test_mask_dir, filename], separator=os.sep)

        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(image, resize_to)
        image = tf.image.convert_image_dtype(image, tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, resize_to)
        mask = tf.image.convert_image_dtype(mask, tf.float32)

        return image, mask

    dataset = dataset.map(load_test_pair).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ======== MODEL =========
def build_can(dilations):
    inputs = tf.keras.Input(shape=(128, 128, 1))
    x = inputs
    for d in dilations:
        x = tf.keras.layers.Conv2D(32, 3, padding='same', dilation_rate=d, activation='relu')(x)
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, x)

def build_generator_md():
    return build_can([1, 2, 4, 8])

def build_generator_fa():
    return build_can([8, 16, 32, 64])

def build_discriminator():
    img = tf.keras.Input(shape=(128, 128, 1))
    seg = tf.keras.Input(shape=(128, 128, 1))
    x = tf.keras.layers.Concatenate()([img, seg])
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model([img, seg], x)

# ======== LOSSES =========
def adversarial_loss(real, fake):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real), real)) + \
           tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake), fake))

def generator_adv_loss(fake):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake), fake))

def data_loss(pred, gt, mode='md'):
    pred_bin = tf.cast(pred > 0.5, tf.float32)
    md = tf.reduce_mean(tf.square((gt - pred_bin) * gt))
    fa = tf.reduce_mean(tf.square((pred_bin - gt) * (1 - gt)))
    return md + 0.1 * fa if mode == 'md' else fa + 0.1 * md

# ======== TRAINING =========
def train_model():
    train_dir = './training'
    test_img_dir = './test_org'
    test_mask_dir = './test_gt'

    train_ds = load_train_dataset(train_dir)
    test_ds = load_test_dataset(test_img_dir, test_mask_dir)

    G1 = build_generator_md()
    G2 = build_generator_fa()
    D = build_discriminator()

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-5)

    @tf.function
    def train_step(imgs, masks):
        with tf.GradientTape(persistent=True) as tape:
            s1 = G1(imgs, training=True)
            s2 = G2(imgs, training=True)
            s_avg = (s1 + s2) / 2.0

            real_output = D([imgs, masks], training=True)
            fake_output_g1 = D([imgs, s1], training=True)
            fake_output_g2 = D([imgs, s2], training=True)

            adv_loss = adversarial_loss(real_output, fake_output_g1) + adversarial_loss(real_output, fake_output_g2)
            d_loss = adv_loss

            g_loss = data_loss(s1, masks, 'md') + data_loss(s2, masks, 'fa') + \
                     generator_adv_loss(fake_output_g1) + generator_adv_loss(fake_output_g2)

        gen_vars = G1.trainable_variables + G2.trainable_variables
        disc_vars = D.trainable_variables

        gen_grads = tape.gradient(g_loss, gen_vars)
        disc_grads = tape.gradient(d_loss, disc_vars)

        gen_opt.apply_gradients(zip(gen_grads, gen_vars))
        disc_opt.apply_gradients(zip(disc_grads, disc_vars))

        return g_loss, d_loss

    print("Starting training...\n")
    for epoch in range(1, 6):  # run 5 epochs for now
        print(f"\nEpoch {epoch}")
        for step, (imgs, masks) in enumerate(train_ds):
            g_loss, d_loss = train_step(imgs, masks)
            if step % 10 == 0:
                print(f"Step {step} | Generator Loss: {g_loss.numpy():.4f} | Discriminator Loss: {d_loss.numpy():.4f}")

    print("\nTraining complete. Saving model...")
    G1.save("generator_md.h5")
    G2.save("generator_fa.h5")
    print("Models saved.\n")

    # Visualize test outputs
    for imgs, masks in test_ds.take(1):
        s1_out = G1(imgs, training=False)
        s2_out = G2(imgs, training=False)
        avg_out = (s1_out + s2_out) / 2.0

        for i in range(imgs.shape[0]):
            fig, ax = plt.subplots(1, 4, figsize=(10, 3))
            ax[0].imshow(imgs[i, ..., 0], cmap='gray')
            ax[0].set_title("Test Image")
            ax[1].imshow(masks[i, ..., 0], cmap='gray')
            ax[1].set_title("GT Mask")
            ax[2].imshow(s1_out[i, ..., 0], cmap='gray')
            ax[2].set_title("G1 Output")
            ax[3].imshow(avg_out[i, ..., 0], cmap='gray')
            ax[3].set_title("Averaged Output")
            plt.show()

if __name__ == "__main__":
    train_model()
