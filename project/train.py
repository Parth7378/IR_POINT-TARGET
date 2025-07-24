from dataset import tf_load_train_image_mask
from model import build_model
import tensorflow as tf

# Load training data
train_ds = tf_load_train_image_mask('./training')

# Build model
model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_ds, epochs=5)

# Save
model.save('ir_segmentor.h5')
