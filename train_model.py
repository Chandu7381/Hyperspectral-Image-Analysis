import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import argparse

def main(args):
    # --- Load data ---
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_flow = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical"
    )

    val_flow = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical"
    )

    print(f"✅ Detected classes: {train_flow.class_indices}")

    # --- Build model ---
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(train_flow.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # --- Training setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, mode="max"
    )
    early = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

    # --- Train model ---
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs,
        callbacks=[checkpoint, early]
    )

    # --- Save final model ---
    model.save(os.path.join(args.output_dir, "final_model.keras"))
    print("🎉 Model training complete and saved to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/valid")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    main(args)
