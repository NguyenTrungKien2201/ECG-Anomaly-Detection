# ECG-Anomaly-Detection
# ECG Anomaly Detection

This project builds a machine learning system for ECG anomaly detection. The goal is to classify ECG heartbeat signals as normal or abnormal using the MIT-BIH dataset from the Kaggle ECG Heartbeat Categorization Dataset.

## Project Overview

Electrocardiogram (ECG) signals are widely used to monitor heart activity. Abnormal ECG patterns may indicate arrhythmia or other heart-related conditions. This project applies machine learning and deep learning methods to detect abnormal ECG heartbeat segments automatically.

## Dataset

Dataset used:

- ECG Heartbeat Categorization Dataset
- Source: Kaggle
- Files used:
  - `mitbih_train.csv`
  - `mitbih_test.csv`

The original dataset contains multiple heartbeat classes. In this project, the task is converted into binary classification:

- `0`: Normal heartbeat
- `1`: Abnormal heartbeat

Dataset shape used in the experiment:

- Training set: 87,554 samples
- Test set: 21,892 samples
- Features per sample: 187 ECG signal values

Class distribution:

Training set:

- Normal: 72,471
- Abnormal: 15,083

Test set:

- Normal: 18,118
- Abnormal: 3,774

## Methods

Two models were implemented:

1. Random Forest Baseline
2. 1D Convolutional Neural Network

The 1D CNN was selected as the final model because it achieved better recall and F1-score for abnormal heartbeat detection.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest Baseline | 0.976658 | 0.993049 | 0.870694 | 0.927855 |
| 1D CNN | 0.982916 | 0.954789 | 0.945681 | 0.950213 |

The 1D CNN achieved the best overall performance with an accuracy of 98.29%, precision of 95.48%, recall of 94.57%, and F1-score of 95.02%.

## Output Files

- `ECG_Anomaly_Detection_Model_Fixed_English.ipynb`: Main notebook
- `ECG_Anomaly_Detection_Model_Fixed_English.py`: Python script version
- `model_results.csv`: Evaluation metrics
- `confusion_matrix.png`: Confusion matrix
- `accuracy_curve.png`: Training and validation accuracy curve
- `loss_curve.png`: Training and validation loss curve
- `ecg_1d_cnn_model.keras`: Trained 1D CNN model

## How to Run

1. Download the dataset from Kaggle.
2. Upload `mitbih_train.csv` and `mitbih_test.csv` to Google Colab.
3. Open and run `ECG_Anomaly_Detection_Model_Fixed_English.ipynb`.
4. The notebook will train the models and save the output files.

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## References

- Kaggle ECG Heartbeat Categorization Dataset
- MIT-BIH Arrhythmia Database
- PhysioNet
