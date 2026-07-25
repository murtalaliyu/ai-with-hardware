# Binary image classification (cats vs dogs) with a from-scratch CNN.
# Run once to train and save best_model.keras, then choose option 2 to skip retraining.
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Larger images keep more detail than 50x50, at the cost of speed/memory
IMG_SIZE = (128, 128)
# Checkpoint file written during training; option 2 loads this instead of retraining
MODEL_PATH = 'best_model.keras'


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def build_model():
    """Build and compile the CNN. Filters increase as spatial size shrinks."""
    model = Sequential()

    # Block 1: low-level features (edges, colors, textures)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(*IMG_SIZE, 3), activation='relu'))
    model.add(BatchNormalization())  # Stabilize activations for faster/smoother training
    model.add(MaxPool2D(pool_size=(2, 2)))  # Downsample: keep strong signals, cut size

    # Block 2: mid-level features
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Drop 25% of units during training to reduce overfitting

    # Block 3: higher-level features (ears, eyes, legs, etc.)
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 4: richest feature maps before flattening
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classifier head: map learned features to a cat/dog probability
    model.add(Flatten())  # Turn 2D feature maps into a 1D vector
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Stronger dropout here — dense layers overfit easily
    model.add(Dense(units=1, activation='sigmoid'))  # Output in [0, 1]: P(dog) vs cat

    # Adam adapts the learning rate per parameter; 0.0005 is a stable middle ground
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Standard loss for binary classification
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# Training path (option 1): data → fit → save best checkpoint → report val accuracy
# ---------------------------------------------------------------------------
def train_model():
    """Load data, train from scratch, save best_model.keras, return the trained model."""

    # Training generator: rescale pixels to [0, 1] + random transforms (augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,           # Normalize RGB values from 0-255 to 0-1
        validation_split=0.2,       # Hold out 20% of training_set for validation
        rotation_range=20,          # Random rotation up to ±20 degrees
        width_shift_range=0.15,     # Horizontal shift up to 15% of width
        height_shift_range=0.15,    # Vertical shift up to 15% of height
        shear_range=0.15,           # Shear (slant) transform
        zoom_range=0.2,             # Zoom in/out up to 20%
        horizontal_flip=True,       # Random left-right flips (OK for cats/dogs)
        fill_mode='nearest'         # Fill empty pixels after warp with nearest neighbor
    )

    # Validation generator: rescale only — no augmentation for an honest score
    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    # 80% of training_set/cats and training_set/dogs
    training_set = train_datagen.flow_from_directory(
        './ch3-dataset/training_set',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',        # Two classes → single sigmoid output
        subset='training'
    )

    # Remaining 20%; shuffle=False keeps evaluation order consistent
    validation_set = val_datagen.flow_from_directory(
        './ch3-dataset/training_set',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    model = build_model()
    model.summary()

    # Callbacks: stop early, save best weights, shrink LR when stuck
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,                 # Wait 8 epochs with no val_loss improvement
        verbose=1,
        restore_best_weights=True   # After stop, reload weights from the best epoch
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,        # Only overwrite when val_accuracy improves
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,                 # New LR = old LR × 0.5
        patience=3,                 # Reduce sooner than EarlyStopping gives up
        min_lr=1e-6,
        verbose=1
    )

    # epochs=40 is a ceiling; EarlyStopping usually finishes sooner
    model.fit(
        training_set,
        epochs=40,
        validation_data=validation_set,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    # Held-out score from the same folder split (not the official test_set)
    val_loss, val_accuracy = model.evaluate(validation_set)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    return model


# ---------------------------------------------------------------------------
# Inference path (option 2): reuse weights already saved on disk
# ---------------------------------------------------------------------------
def load_saved_model():
    """Load best_model.keras if it exists; otherwise tell the user to retrain first."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model at '{MODEL_PATH}'. Choose retrain (1) first."
        )
    print(f"Loading existing model from '{MODEL_PATH}'...")
    return load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_single_image(model, image_path):
    """Preprocess one image and return 'Dog' or 'Cat'."""
    # Resize and scale to [0, 1] so inputs match training
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    # Keras expects a batch: (batch_size, height, width, channels) → (1, H, W, 3)
    img_array = img_array.reshape(1, *IMG_SIZE, 3)
    # Sigmoid output is a probability; > 0.5 → class 1 (dog), else class 0 (cat)
    # predict returns shape (1, 1), so index [0][0] for the scalar label
    result = (model.predict(img_array) > 0.5).astype("int32")
    label = 'Dog' if result[0][0] == 1 else 'Cat'
    return label


def predict_and_display_images(model, image_files):
    """Show a 3x3 grid of images with predicted labels as titles."""
    fig = plt.figure(figsize=(10, 10))
    for i, img_name in enumerate(image_files):
        # Same preprocessing pipeline as predict_single_image
        img_ori = load_img(img_name, target_size=IMG_SIZE)
        img_array = img_to_array(img_ori) / 255.0
        img_array = img_array.reshape(1, *IMG_SIZE, 3)

        result = (model.predict(img_array) > 0.5).astype("int32")
        label = 'dog' if result[0][0] == 1 else 'cat'

        # Larger size for display only — prediction still used IMG_SIZE above
        img_display = load_img(img_name, target_size=(250, 250))
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_display)
        plt.title(f'predict: {label}')
    plt.show()


# ---------------------------------------------------------------------------
# Main: choose retrain vs load, then always evaluate test set + sample predictions
# ---------------------------------------------------------------------------
print("Choose an option:")
print("  1 - Retrain the model (saves to best_model.keras)")
print("  2 - Load existing model and run tests/predictions only")
choice = input("Enter 1 or 2: ").strip()

if choice == '1':
    model = train_model()
elif choice == '2':
    model = load_saved_model()
else:
    raise SystemExit("Invalid choice. Please run again and enter 1 or 2.")

# Test set: images never used in training/validation (rescale only, no augmentation)
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(
    './ch3-dataset/test_set',
    target_size=IMG_SIZE,       # Must match the size the model was trained on
    batch_size=32,
    class_mode='binary',
    shuffle=False               # Stable order for evaluation / debugging
)

# Final honesty check on the full test folder
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Demo on local photos in the project folder (paths relative to cwd)
print("Dog Image Prediction:", predict_single_image(model, 'cutie1.jpg'))
print("Cat Image Prediction:", predict_single_image(model, 'cutie2.jpg'))

# Uncomment to show a 3x3 grid (place 1.jpg ... 8.jpg in the working directory)
# image_files = [f"{i}.jpg" for i in range(1, 9)]
# predict_and_display_images(model, image_files)
