# Binary image classification (cats vs dogs) with a from-scratch CNN.
# Interactive menu loop: retrain, load, upload an image to predict, or evaluate test set.
import os
import tkinter as tk
from tkinter import filedialog

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
    """Preprocess one image; return (label, confidence) where confidence is in [0, 1]."""
    # Resize and scale to [0, 1] so inputs match training
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    # Keras expects a batch: (batch_size, height, width, channels) → (1, H, W, 3)
    img_array = img_array.reshape(1, *IMG_SIZE, 3)
    # Sigmoid output is P(dog); > 0.5 → Dog, else Cat
    dog_prob = float(model.predict(img_array, verbose=0)[0][0])
    if dog_prob > 0.5:
        return 'Dog', dog_prob
    return 'Cat', 1.0 - dog_prob


def choose_image_file():
    """
    Open a file-picker dialog so the user can upload/select an image on the fly.
    Falls back to typing a path if the dialog is cancelled or unavailable.
    """
    # Hide the empty root Tk window; only show the native file dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    image_path = filedialog.askopenfilename(
        title='Select a cat or dog image',
        filetypes=[
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.gif *.webp'),
            ('All files', '*.*'),
        ]
    )
    root.destroy()

    # If the user closes the dialog, allow a manual path instead
    if not image_path:
        image_path = input('No file selected. Enter image path (or press Enter to cancel): ').strip()
        image_path = image_path.strip('"').strip("'")

    return image_path or None


def predict_uploaded_image(model):
    """Option 3: pick an image via dialog (or path) and print the prediction."""
    image_path = choose_image_file()
    if not image_path:
        print('Prediction cancelled.')
        return
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return

    try:
        label, confidence = predict_single_image(model, image_path)
        print(f"\nFile: {image_path}")
        print(f"Prediction: {label}  (confidence: {confidence * 100:.1f}%)\n")
    except Exception as exc:
        print(f"Could not predict on that image: {exc}")


def evaluate_test_set(model):
    """Run accuracy/loss on the held-out test_set folder."""
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(
        './ch3-dataset/test_set',
        target_size=IMG_SIZE,       # Must match the size the model was trained on
        batch_size=32,
        class_mode='binary',
        shuffle=False               # Stable order for evaluation / debugging
    )
    test_loss, test_accuracy = model.evaluate(test_set)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


def predict_and_display_images(model, image_files):
    """Show a 3x3 grid of images with predicted labels as titles."""
    fig = plt.figure(figsize=(10, 10))
    for i, img_name in enumerate(image_files):
        label, _confidence = predict_single_image(model, img_name)
        img_display = load_img(img_name, target_size=(250, 250))
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_display)
        plt.title(f'predict: {label.lower()}')
    plt.show()


def require_model(model):
    """Return True if a model is loaded; otherwise print guidance and return False."""
    if model is None:
        print('No model loaded yet. Choose option 1 (retrain) or 2 (load) first.')
        return False
    return True


def print_menu(model):
    status = 'ready' if model is not None else 'not loaded'
    print('\n' + '=' * 56)
    print(' Cat vs Dog Classifier')
    print(f' Model status: {status}')
    print('=' * 56)
    print('  1 - Retrain the model (saves to best_model.keras)')
    print('  2 - Load existing model from best_model.keras')
    print('  3 - Upload an image and predict cat/dog')
    print('  4 - Evaluate on the test set')
    print('  5 - Quit')
    print('=' * 56)


# ---------------------------------------------------------------------------
# Main loop: keep running until the user quits
# ---------------------------------------------------------------------------
model = None

# If a checkpoint already exists, offer a faster start by auto-loading it
if os.path.exists(MODEL_PATH):
    auto = input(
        f"Found '{MODEL_PATH}'. Load it now? [Y/n]: "
    ).strip().lower()
    if auto in ('', 'y', 'yes'):
        try:
            model = load_saved_model()
            print('Model loaded. You can upload images (option 3) right away.')
        except Exception as exc:
            print(f'Could not auto-load model: {exc}')

while True:
    print_menu(model)
    choice = input('Enter 1-5: ').strip()

    if choice == '1':
        try:
            model = train_model()
            print('Training complete. Model is ready for predictions.')
        except Exception as exc:
            print(f'Training failed: {exc}')

    elif choice == '2':
        try:
            model = load_saved_model()
            print('Model loaded. Ready for predictions.')
        except Exception as exc:
            print(f'Load failed: {exc}')

    elif choice == '3':
        if require_model(model):
            predict_uploaded_image(model)

    elif choice == '4':
        if require_model(model):
            try:
                evaluate_test_set(model)
            except Exception as exc:
                print(f'Test evaluation failed: {exc}')

    elif choice == '5':
        print('Goodbye!')
        break

    else:
        print('Invalid choice. Enter a number from 1 to 5.')
