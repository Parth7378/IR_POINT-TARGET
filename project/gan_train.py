import tensorflow as tf
from dataset import tf_load_train_image_mask
from model_gan import build_generator, build_discriminator

# Load dataset
train_ds = tf_load_train_image_mask('./training', batch_size=4)

# Create models
G_MD = build_generator("G_MD")
G_FA = build_generator("G_FA")
D = build_discriminator()

# Define optimizers
g_md_optimizer = tf.keras.optimizers.Adam(1e-4)
g_fa_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Loss functions
bce = tf.keras.losses.BinaryCrossentropy()

def generator_loss(pred_mask, gt_mask, disc_fake):
    semantic = bce(gt_mask, pred_mask)
    adversarial = bce(tf.ones_like(disc_fake), disc_fake)
    return semantic + 0.001 * adversarial

def discriminator_loss(disc_real, disc_fake):
    real_loss = bce(tf.ones_like(disc_real), disc_real)
    fake_loss = bce(tf.zeros_like(disc_fake), disc_fake)
    return real_loss + fake_loss

# Training step
@tf.function
def train_step(images, masks):
    with tf.GradientTape(persistent=True) as tape:
        # Generators
        pred_md = G_MD(images, training=True)
        pred_fa = G_FA(images, training=True)

        # Average prediction
        avg_pred = (pred_md + pred_fa) / 2.0

        # Discriminator
        disc_real = D(masks, training=True)
        disc_fake = D(avg_pred, training=True)

        # Losses
        g_md_loss = generator_loss(pred_md, masks, disc_fake)
        g_fa_loss = generator_loss(pred_fa, masks, disc_fake)
        d_loss = discriminator_loss(disc_real, disc_fake)

    # Apply gradients
    g_md_grads = tape.gradient(g_md_loss, G_MD.trainable_variables)
    g_fa_grads = tape.gradient(g_fa_loss, G_FA.trainable_variables)
    d_grads = tape.gradient(d_loss, D.trainable_variables)

    g_md_optimizer.apply_gradients(zip(g_md_grads, G_MD.trainable_variables))
    g_fa_optimizer.apply_gradients(zip(g_fa_grads, G_FA.trainable_variables))

    d_optimizer.apply_gradients(zip(d_grads, D.trainable_variables))

    return g_md_loss, g_fa_loss, d_loss

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for step, (img_batch, mask_batch) in enumerate(train_ds):
        g1_loss, g2_loss, d_loss = train_step(img_batch, mask_batch)
        if step % 50 == 0:
            print(f"Step {step} | G1 Loss: {g1_loss:.4f}, G2 Loss: {g2_loss:.4f}, D Loss: {d_loss:.4f}")

# Save models
G_MD.save('G_MD.h5')
G_FA.save('G_FA.h5')
D.save('Discriminator.h5')
