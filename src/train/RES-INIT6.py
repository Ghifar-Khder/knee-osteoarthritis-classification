import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# Residual block
# -------------------------
def residual_block(x, filters, stride=1):
    shortcut = x
    in_channels = x.shape[-1]

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)

    if stride != 1 or in_channels != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)

    return layers.Add()([x, shortcut])

# -------------------------
# Residual CNN
# -------------------------
def build_model(input_shape=(224, 224, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(2)(x)

    # Stage 1
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    # Stage 3
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # Stage 4
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

# -------------------------
# Main training
# -------------------------
def main():
    base_dir = r"Final-DATA\KNEE-final-initialization"
    train_dir = os.path.join(base_dir, "train2")
    val_dir = os.path.join(base_dir, "validation2")

    train_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15]
    )

    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = build_model(num_classes=train_data.num_classes)

    model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

    os.makedirs("results", exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("results", "RES-INITIAL.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=100,
        callbacks=[checkpoint],
        verbose=1
    )

if __name__ == "__main__":
    main()
