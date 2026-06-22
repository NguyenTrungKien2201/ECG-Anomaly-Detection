
# %% [markdown]
# # ECG Anomaly Detection
# 
# This notebook trains models for ECG anomaly detection using the MIT-BIH CSV files from the Kaggle ECG Heartbeat Categorization Dataset.
# 
# Target task: binary classification.
# 
# - `0`: Normal heartbeat
# - `1`: Abnormal heartbeat
# 
# Required files:
# 
# - `mitbih_train.csv`
# - `mitbih_test.csv`
# 
# If you uploaded `.zip` files instead, the notebook will automatically extract them.
# %%
# Import libraries
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

print('TensorFlow version:', tf.__version__)

# %% [markdown]
# ## 1. Extract ZIP files if needed
# 
# Run this cell if your uploaded files are `mitbih_train.csv.zip` and `mitbih_test.csv.zip`.
# %%
zip_files = ['mitbih_train.csv.zip', 'mitbih_test.csv.zip']

for zip_file in zip_files:
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f'Extracted: {zip_file}')

print('Current files:')
print(os.listdir('.'))

# %% [markdown]
# ## 2. Load dataset
# 
# Upload `mitbih_train.csv` and `mitbih_test.csv` to the Colab working directory. If you use local Jupyter, place both files in the same folder as this notebook.
# %%
train_path = 'mitbih_train.csv'
test_path = 'mitbih_test.csv'

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print('CSV files were not found. If you are using Colab, upload the files below.')
    try:
        from google.colab import files
        uploaded = files.upload()
    except Exception as e:
        print('Not running in Colab or files have not been uploaded:', e)

train_df = pd.read_csv(train_path, header=None)
test_df = pd.read_csv(test_path, header=None)

print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)
train_df.head()

# %% [markdown]
# ## 3. Prepare features and labels
# 
# The original label is stored in the last column. Label `0` is normal. All other labels are grouped into abnormal for binary classification.
# %%
X_train = train_df.iloc[:, :-1].values
y_train_multi = train_df.iloc[:, -1].values.astype(int)

X_test = test_df.iloc[:, :-1].values
y_test_multi = test_df.iloc[:, -1].values.astype(int)

# Binary labels: 0 = Normal, 1 = Abnormal
y_train = (y_train_multi != 0).astype(int)
y_test = (y_test_multi != 0).astype(int)

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)

print('\nTraining label distribution:')
print(pd.Series(y_train).value_counts().rename(index={0: 'Normal', 1: 'Abnormal'}))

print('\nTest label distribution:')
print(pd.Series(y_test).value_counts().rename(index={0: 'Normal', 1: 'Abnormal'}))

# %% [markdown]
# ## 4. Visualize sample ECG signals
# %%
normal_idx = np.where(y_train == 0)[0][0]
abnormal_idx = np.where(y_train == 1)[0][0]

plt.figure(figsize=(10, 4))
plt.plot(X_train[normal_idx])
plt.title('Sample Normal ECG Heartbeat')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig('sample_normal_ecg.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(X_train[abnormal_idx])
plt.title('Sample Abnormal ECG Heartbeat')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig('sample_abnormal_ecg.png', dpi=150)
plt.show()

# %% [markdown]
# ## 5. Baseline model: Random Forest
# 
# Random Forest is used as a baseline model. It is simple, stable, and useful for comparison with the 1D CNN.
# %%
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_results = {
    'Model': 'Random Forest Baseline',
    'Accuracy': accuracy_score(y_test, rf_pred),
    'Precision': precision_score(y_test, rf_pred, zero_division=0),
    'Recall': recall_score(y_test, rf_pred, zero_division=0),
    'F1-score': f1_score(y_test, rf_pred, zero_division=0)
}

pd.DataFrame([rf_results])

# %% [markdown]
# ## 6. Stratified validation split for 1D CNN
# 
# The dataset can be ordered by class. Using `validation_split=0.2` directly may create a biased validation set. This cell creates a stratified validation split and computes class weights to handle class imbalance.
# %%
# Reshape for Conv1D: samples x time_steps x channels
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_cnn,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_tr),
    y=y_tr
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

print('Class weights:', class_weights)
print('\nTraining distribution:')
print(pd.Series(y_tr).value_counts().rename(index={0: 'Normal', 1: 'Abnormal'}))
print('\nValidation distribution:')
print(pd.Series(y_val).value_counts().rename(index={0: 'Normal', 1: 'Abnormal'}))

# %% [markdown]
# ## 7. Proposed model: 1D CNN
# %%
model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),

    Conv1D(32, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# %% [markdown]
# ## 8. Train 1D CNN
# %%
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# %% [markdown]
# ## 9. Tune decision threshold on validation set
# 
# Because this is an imbalanced classification task, the default threshold `0.5` may not always produce the best F1-score. This cell selects the threshold that gives the highest F1-score on the validation set.
# %%
val_prob = model.predict(X_val).ravel()

thresholds = np.arange(0.10, 0.91, 0.01)
threshold_scores = []

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)
    threshold_scores.append({
        'Threshold': threshold,
        'Precision': precision_score(y_val, val_pred, zero_division=0),
        'Recall': recall_score(y_val, val_pred, zero_division=0),
        'F1-score': f1_score(y_val, val_pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_scores)
best_row = threshold_df.loc[threshold_df['F1-score'].idxmax()]
best_threshold = float(best_row['Threshold'])

print('Best threshold based on validation F1-score:', best_threshold)
threshold_df.sort_values('F1-score', ascending=False).head(10)

# %% [markdown]
# ## 10. Evaluate 1D CNN on test set
# %%
cnn_prob = model.predict(X_test_cnn).ravel()
cnn_pred_default = (cnn_prob >= 0.5).astype(int)
cnn_pred = (cnn_prob >= best_threshold).astype(int)

cnn_results = {
    'Model': f'1D CNN, threshold={best_threshold:.2f}',
    'Accuracy': accuracy_score(y_test, cnn_pred),
    'Precision': precision_score(y_test, cnn_pred, zero_division=0),
    'Recall': recall_score(y_test, cnn_pred, zero_division=0),
    'F1-score': f1_score(y_test, cnn_pred, zero_division=0)
}

cnn_default_results = {
    'Model': '1D CNN, threshold=0.50',
    'Accuracy': accuracy_score(y_test, cnn_pred_default),
    'Precision': precision_score(y_test, cnn_pred_default, zero_division=0),
    'Recall': recall_score(y_test, cnn_pred_default, zero_division=0),
    'F1-score': f1_score(y_test, cnn_pred_default, zero_division=0)
}

results_df = pd.DataFrame([rf_results, cnn_default_results, cnn_results])
results_df

# %% [markdown]
# ## 11. Classification report and confusion matrix
# %%
print('Classification Report: 1D CNN')
print(classification_report(y_test, cnn_pred, target_names=['Normal', 'Abnormal'], zero_division=0))

cm = confusion_matrix(y_test, cnn_pred)
print('Confusion Matrix:')
print(cm)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title('Confusion Matrix - 1D CNN')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1], ['Normal', 'Abnormal'])
plt.yticks([0, 1], ['Normal', 'Abnormal'])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar()
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# %% [markdown]
# ## 12. Training curves
# %%
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_curve.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()

# %% [markdown]
# ## 13. Save model and results
# %%
model.save('ecg_1d_cnn_model.keras')
results_df.to_csv('model_results.csv', index=False)
threshold_df.to_csv('threshold_tuning_results.csv', index=False)

print('Saved files:')
print('- ecg_1d_cnn_model.keras')
print('- model_results.csv')
print('- threshold_tuning_results.csv')
print('- sample_normal_ecg.png')
print('- sample_abnormal_ecg.png')
print('- confusion_matrix.png')
print('- accuracy_curve.png')
print('- loss_curve.png')
