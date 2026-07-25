# Binary image classification (cats vs dogs) with a from-scratch CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Larger images keep more detail than 50x50, at the cost of speed/memory
IMG_SIZE = (128, 128)

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
# Training: rescale pixels to [0, 1] and apply random transforms so the model
# sees varied views of each image and generalizes better (data augmentation).
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

# Validation: rescale only — no augmentation, so the score reflects real images
val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

# Load training subset (80%) from folder structure: training_set/cats, training_set/dogs
training_set = train_datagen.flow_from_directory(
    './ch3-dataset/training_set',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',        # Two classes → single sigmoid output
    subset='training'
)

# Load validation subset (20%); shuffle=False keeps evaluation order consistent
validation_set = val_datagen.flow_from_directory(
    './ch3-dataset/training_set',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ---------------------------------------------------------------------------
# CNN model: filters increase as spatial size shrinks (edges → parts → objects)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
# Adam adapts the learning rate per parameter; 0.0005 is a stable middle ground
optimizer = Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',  # Standard loss for binary classification
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------------------------------
# Callbacks: stop early, save best weights, shrink LR when stuck
# ---------------------------------------------------------------------------
# Stop if validation loss does not improve for 8 epochs; restore best weights
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    verbose=1,
    restore_best_weights=True
)

# Save the model whenever validation accuracy reaches a new best
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# If val_loss plateaus for 3 epochs, halve the learning rate (down to min_lr)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ---------------------------------------------------------------------------
# Train (epochs=40 is a ceiling; EarlyStopping usually finishes sooner)
# ---------------------------------------------------------------------------
history = model.fit(
    training_set,
    epochs=40,
    validation_data=validation_set,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Final score on the held-out validation images
val_loss, val_accuracy = model.evaluate(validation_set)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# ---------------------------------------------------------------------------
# Test set: unseen images for a final honesty check (rescale only, no aug)
# ---------------------------------------------------------------------------
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(
    './ch3-dataset/test_set',
    target_size=IMG_SIZE,       # Must match the size the model was trained on
    batch_size=32,
    class_mode='binary',
    shuffle=False               # Stable order for evaluation / debugging
)

# Overall loss and accuracy on the full test folder
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------
def predict_single_image(model, image_path):
    # Load and resize; divide by 255 so pixels match training scale [0, 1]
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    # Keras expects a batch: (batch_size, height, width, channels) → (1, H, W, 3)
    img_array = img_array.reshape(1, *IMG_SIZE, 3)
    # Sigmoid output is a probability; > 0.5 → class 1 (dog), else class 0 (cat)
    # predict returns shape (1, 1), so index [0][0] for the scalar label
    result = (model.predict(img_array) > 0.5).astype("int32")
    label = 'Dog' if result[0][0] == 1 else 'Cat'
    return label

# Demo on local photos in the project folder (paths relative to cwd)
print("Dog Image Prediction:", predict_single_image(model, 'cutie1.jpg'))
print("Cat Image Prediction:", predict_single_image(model, 'cutie2.jpg'))

# ---------------------------------------------------------------------------
# Batch prediction + grid visualization
# ---------------------------------------------------------------------------
def predict_and_display_images(model, image_files):
    # 3x3 grid of up to 9 images with predicted labels as titles
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

# Uncomment to show a 3x3 grid (place 1.jpg ... 8.jpg in the working directory)
# image_files = [f"{i}.jpg" for i in range(1, 9)]
# predict_and_display_images(model, image_files)
