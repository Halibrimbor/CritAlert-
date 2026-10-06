import os
import tensorflow as tf
from tensorflow.keras import layers, models


DATASET_DIR = r"D:\PROJECTS\Critalert\Dataset\Lung_Xray_dataset" 
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 16
EPOCHS = 10

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found!")
        print("Please create this folder and add 'Normal' and 'Critical' subfolders inside it with your report images.")
        return

    print(f"Loading dataset from {DATASET_DIR}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # ─── Build CNN Model ──────────────────────────────────────
    print("Building Convolutional Neural Network...")
    num_classes = len(class_names)

    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # ─── Train Model ────────────────────────────────────
    print(f"Training for {EPOCHS} epochs...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )
        
        # Save model
        model.save("image_model.h5")
        print("Training complete! Model saved perfectly as 'image_model.h5'")
    except Exception as e:
        print(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()