import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
import pickle

warnings.filterwarnings("ignore")

# -------------------- Parameters --------------------
# CSV file containing data with "file_path" and "label" columns
dataset_csv = r"E:\bishe\mixed_dataset\multi_task_dataset_info.csv"  # Modify as needed
target_length = 174    # fixed time steps
n_mfcc = 40            # number of MFCC features
sr = 22050             # sampling rate
duration = 4.0         # duration in seconds

# -------------------- Audio Parameter Calculation Function --------------------
def compute_sound_levels(y, sr):
    """
    Compute sound level parameters using short-time energy (RMS).
    Returns: Leq, Lmax, Lmin, Lpeak (in dB)
    """
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    Leq = 10 * np.log10(np.mean(rms**2) + 1e-9)
    Lmax = 10 * np.log10(np.max(rms**2) + 1e-9)
    Lmin = 10 * np.log10(np.min(rms**2) + 1e-9)
    Lpeak = 10 * np.log10(np.max(np.abs(y)**2) + 1e-9)
    return Leq, Lmax, Lmin, Lpeak

# -------------------- Data Loading and Feature Extraction --------------------
def load_audio(file_path, sr=22050, duration=4.0):
    """Load an audio file."""
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    return y

def extract_mfcc(y, sr, n_mfcc, target_length):
    """
    Extract MFCC features from an audio signal.
    Pads or truncates to a fixed number of time steps.
    Returns a feature matrix of shape (target_length, n_mfcc).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    # Zero-pad or truncate to fixed time steps
    if mfcc.shape[1] < target_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :target_length]
    return mfcc.T  # (target_length, n_mfcc)

def prepare_multitask_dataset(csv_file, sr=22050, duration=4.0, target_length=174, n_mfcc=40):
    """
    Prepare dataset for multi-task learning:
      - Input: mixed audio MFCC features.
      - Outputs: classification label and sound level parameters.
    Assumes the CSV file has columns "file_path" and "label".
    """
    df = pd.read_csv(csv_file)
    X_list = []
    labels = []
    levels_list = []  # Sound level parameters
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
        file_path = row["file_path"]
        label = row["label"]
        try:
            y = load_audio(file_path, sr=sr, duration=duration)
            mfcc = extract_mfcc(y, sr, n_mfcc, target_length)
            Leq, Lmax, Lmin, Lpeak = compute_sound_levels(y, sr)
            
            X_list.append(mfcc)
            labels.append(label)
            levels_list.append([Leq, Lmax, Lmin, Lpeak])
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            continue

    X = np.array(X_list)
    levels = np.array(levels_list)
    
    # Expand channel dimension: final shape (samples, target_length, n_mfcc, 1)
    X = X[..., np.newaxis]
    
    # Encode labels (e.g., 'traffic', 'industrial', 'human', 'animal')
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_cat = to_categorical(y_encoded)
    
    return X, y_cat, levels, le.classes_

print("Preparing multi-task dataset...")
X, y_cat, levels, class_names = prepare_multitask_dataset(dataset_csv, sr=sr, duration=duration,
                                                          target_length=target_length, n_mfcc=n_mfcc)
print("Dataset shapes:", X.shape, y_cat.shape, levels.shape)
print("Classes:", class_names)

# Split into training and validation sets
X_train, X_val, y_train, y_val, levels_train, levels_val = train_test_split(
    X, y_cat, levels, test_size=0.2, random_state=42
)

# -------------------- Build Multi-task Model --------------------
def build_multitask_model(input_shape, num_classes):
    """
    Build a multi-task model with a shared CNN encoder and two branches:
      - Classification branch for noise category (e.g., traffic, industrial, human, animal)
      - Regression branch for audio parameters (Leq, Lmax, Lmin, Lpeak)
    """
    inputs = Input(shape=input_shape)
    
    # Shared CNN Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    
    x = GlobalAveragePooling2D()(x)
    x_shared = Dense(128, activation="relu")(x)
    x_shared = Dropout(0.5)(x_shared)
    
    # Classification Branch
    class_output = Dense(num_classes, activation="softmax", name="classification")(x_shared)
    
    # Regression Branch for sound level parameters (4 parameters)
    reg_output = Dense(64, activation="relu")(x_shared)
    reg_output = Dense(4, activation="linear", name="regression")(reg_output)
    
    model = Model(inputs=inputs, outputs=[class_output, reg_output])
    model.compile(optimizer=Adam(0.001),
                  loss={"classification": "categorical_crossentropy", "regression": "mean_squared_error"},
                  loss_weights={"classification": 1.0, "regression": 0.5},
                  metrics={"classification": "accuracy", "regression": "mse"})
    return model

num_classes = y_cat.shape[1]
input_shape = (target_length, n_mfcc, 1)
model_mt = build_multitask_model(input_shape, num_classes)
model_mt.summary()

# -------------------- Train the Model --------------------
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("multitask_noise_model_best.h5", save_best_only=True, monitor="val_classification_accuracy", mode="max")

history = model_mt.fit(
    X_train, {"classification": y_train, "regression": levels_train},
    validation_data=(X_val, {"classification": y_val, "regression": levels_val}),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model_mt.save("multitask_noise_model.h5")
print("Multi-task training complete. Model saved.")
# 保存LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved.")

# -------------------- Save Training History --------------------
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# -------------------- Plot Training History --------------------
history_dict = history.history
epochs = range(1, len(history_dict['classification_loss']) + 1)

plt.figure(figsize=(16, 6))

# Plot Classification Loss (Training & Validation)
plt.subplot(1, 3, 1)
plt.plot(epochs, history_dict['classification_loss'], 'r-', label='Train Classification Loss')
plt.plot(epochs, history_dict.get('val_classification_loss', []), 'r--', label='Val Classification Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Classification Loss')
plt.legend()
plt.grid(True)

# Plot Regression (Noise Parameter) Loss (Training & Validation)
plt.subplot(1, 3, 2)
plt.plot(epochs, history_dict['regression_loss'], 'b-', label='Train Regression Loss')
plt.plot(epochs, history_dict.get('val_regression_loss', []), 'b--', label='Val Regression Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Regression Loss (Noise Parameters)')
plt.legend()
plt.grid(True)

# Plot Classification Accuracy (Training & Validation)
plt.subplot(1, 3, 3)
plt.plot(epochs, history_dict['classification_accuracy'], 'g-', label='Train Accuracy')
plt.plot(epochs, history_dict.get('val_classification_accuracy', []), 'g--', label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
