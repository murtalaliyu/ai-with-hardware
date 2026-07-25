from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    validation_split = 0.2
)

# Load training set
training_set = train_datagen.flow_from_directory(
    './ch3-dataset/training_set',
    target_size = (50, 50),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'training'
)

# Load validation set
validation_set = train_datagen.flow_from_directory(
    './ch3-dataset/training_set',
    target_size = (50, 50),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'validation'
)

# Build CNN model
model = Sequential()

# Add convolutional layers with increasing filters
model.add(Conv2D(64, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))    # Add dropout to avoid overfitting

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Fully connected layers
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.5)) # Increase dropout to prevent overfitting
model.add(Dense(units = 1, activation = 'sigmoid')) # Binary output

# Compile the model with lower learning rate for more precision
optimizer = Adam(learning_rate = 0.0001)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Model summary
model.summary()

# Callbacks: Early Stopping and Model Checkpoint
early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    verbose = 1,
    restore_best_weights = True
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor = 'val_accuracy',
    save_best_only = True,
    verbose = 1
)

# Train the model with validation set
history = model.fit(
    training_set,
    epochs = 10,
    validation_data = validation_set,
    callbacks = [early_stop, checkpoint]
)

# Evaluate on the validation set
val_loss, val_accuracy = model.evaluate(validation_set)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

