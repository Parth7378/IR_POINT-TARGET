import tensorflow as tf

# Generator model (simple U-Net style)
def build_generator(name):
    inputs = tf.keras.Input(shape=(128, 128, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, output, name=name)

# Discriminator model
def build_discriminator():
    inputs = tf.keras.Input(shape=(128, 128, 1))
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, x, name='Discriminator')
