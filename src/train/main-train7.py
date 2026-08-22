import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# Main
# -------------------------
def main():

    # -------------------------
    # Paths
    # -------------------------
    initial_model_path = r"results\final-main6.keras"

    main_train_dir = r"Final-DATA\KNEE-main-images-stablized\train"
    main_val_dir   = r"Final-DATA\KNEE-main-images-stablized\val"

    init_train_dir = r"Final-DATA\KNEE-final-initialization\train2"
    init_val_dir   = r"Final-DATA\KNEE-final-initialization\validation2"

    os.makedirs("results", exist_ok=True)

    # -------------------------
    # Load model
    # -------------------------
    model = keras.models.load_model(initial_model_path)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------
    # Data generators
    # -------------------------
    train_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15]
    )

    val_gen = ImageDataGenerator(rescale=1./255)

    # Main dataset
    main_train = train_gen.flow_from_directory(
        main_train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True
    )

    # Initialization datasets
    init_train = train_gen.flow_from_directory(
        init_train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True
    )

    init_val = val_gen.flow_from_directory(
        init_val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True
    )

    main_val = val_gen.flow_from_directory(
        main_val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    # -------------------------
    # Combine with weights
    # -------------------------
    def combined_generator():
        while True:
            x1, y1 = next(main_train)
            w1 = tf.ones(len(x1))              # weight = 1.0

            x2, y2 = next(init_train)
            w2 = tf.ones(len(x2)) * 0.1        # weight = 0.1

            x3, y3 = next(init_val)
            w3 = tf.ones(len(x3)) * 0.1        # weight = 0.1

            x = tf.concat([x1, x2, x3], axis=0)
            y = tf.concat([y1, y2, y3], axis=0)
            w = tf.concat([w1, w2, w3], axis=0)

            yield x, y, w

    steps_per_epoch = (
        main_train.samples // main_train.batch_size
    )

    # -------------------------
    # Callbacks
    # -------------------------
    best_ckpt = keras.callbacks.ModelCheckpoint(
        filepath="results/best-main.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # -------------------------
    # Train 10
    # -------------------------
    model.fit(
        combined_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=main_val,
        callbacks=[best_ckpt],
        verbose=1
    )

    # -------------------------
    # Save final model of this stage
    # -------------------------
    model.save("results/final-main7.keras")

if __name__ == "__main__":
    main()
