import tensorflow as tf

# Simple segmentation model
def build_model():
    inputs = tf.keras.Input(shape=(128, 128, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, x)
