import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
data_dir = 'C:\\Users\\USER\\OneDrive\\Desktop\\model\\train'  # Parent directory containing Fight and NonFight folders
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
FRAMES_PER_VIDEO = 10  # Number of frames to extract from each video

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame)
    
    cap.release()
    return frames

# Load dataset
def load_dataset(data_dir):
    images = []
    labels = []
    
    # Violence class (1)
    violence_dir = os.path.join(data_dir, 'Fight')
    for video_file in os.listdir(violence_dir):
        if video_file.endswith(('.avi', '.mp4', '.mov')):  # Add more video formats if needed
            video_path = os.path.join(violence_dir, video_file)
            try:
                frames = extract_frames(video_path)
                images.extend(frames)
                labels.extend([1] * len(frames))  # 1 for violence
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
    
    # Non-violence class (0)
    non_violence_dir = os.path.join(data_dir, 'NonFight')
    for video_file in os.listdir(non_violence_dir):
        if video_file.endswith(('.avi', '.mp4', '.mov')):  # Add more video formats if needed
            video_path = os.path.join(non_violence_dir, video_file)
            try:
                frames = extract_frames(video_path)
                images.extend(frames)
                labels.extend([0] * len(frames))  # 0 for non-violence
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
    
    return np.array(images), np.array(labels)

# Load and preprocess data
print("Loading dataset...")
X, y = load_dataset(data_dir)

if len(X) == 0:
    raise ValueError("No valid video frames were loaded. Please check your video files and paths.")

# Normalize pixel values to [0, 1]
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=2)

# Split dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Model architecture
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    return model

# Create model
model = create_model((IMG_SIZE[0], IMG_SIZE[1], 3))
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=10, 
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=5, 
                              min_lr=0.00001)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Evaluate the model
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.show()

plot_history(history)

# Load best model
from tensorflow.keras.models import load_model
best_model = load_model('best_model.h5')

# Evaluate on test set
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# Save the model for future use
best_model.save('violence_detection_model.h5')
print("Model saved as 'violence_detection_model.h5'")

# Confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Non-Violence', 'Violence']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Violence', 'Violence'], 
            yticklabels=['Non-Violence', 'Violence'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()