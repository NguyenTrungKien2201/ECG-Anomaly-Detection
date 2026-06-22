# ECG-Anomaly-Detection

This project builds a machine learning system for ECG anomaly detection. The goal is to classify ECG heartbeat segments as normal or abnormal using public ECG data derived from the MIT-BIH Arrhythmia Database.

## Project Overview

Electrocardiogram (ECG) signals are used to record the electrical activity of the heart and check heart rate and rhythm. Abnormal ECG patterns may indicate arrhythmia or other heart-related conditions. This project applies machine learning and deep learning methods to classify ECG heartbeat signals automatically.

## Dataset
The dataset is not included in this repository because the CSV files are large.

Please download the dataset from Kaggle:

ECG Heartbeat Categorization Dataset  
https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Required files:

- `mitbih_train.csv`
- `mitbih_test.csv`

After downloading, place both CSV files in the same folder as the notebook before running it.

Dataset used:

- ECG Heartbeat Categorization Dataset by Shayan Fazeli
- Source: Kaggle
- Original sources: MIT-BIH Arrhythmia Database and PTB Diagnostic ECG Database
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

- `ECG_Anomaly_Detection_Model.ipynb`: Main notebook
- `ECG_Anomaly_Detection_Model.py`: Python script version
- `model_results.csv`: Evaluation metrics
- `confusion_matrix.png`: Confusion matrix
- `accuracy_curve.png`: Training and validation accuracy curve
- `loss_curve.png`: Training and validation loss curve
- `ecg_1d_cnn_model.keras`: Trained 1D CNN model

## How to Run

1. Download the ECG Heartbeat Categorization Dataset from Kaggle.
2. Upload `mitbih_train.csv` and `mitbih_test.csv` to Google Colab.
3. Open and run `ECG_Anomaly_Detection_Model_Fixed_English.ipynb`.
4. The notebook trains the models and saves the output files.

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## Supporting Sources

- Cleveland Clinic. Electrocardiogram (EKG/ECG) Test. https://my.clevelandclinic.org/health/diagnostics/16953-electrocardiogram-ekg
- PhysioNet. MIT-BIH Arrhythmia Database v1.0.0. https://physionet.org/content/mitdb/1.0.0/
- Kaggle. ECG Heartbeat Categorization Dataset. https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- Kachuee, M., Fazeli, S., & Sarrafzadeh, M. ECG Heartbeat Classification: A Deep Transferable Representation. https://arxiv.org/abs/1805.00794
- TensorFlow. Conv1D documentation. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
- Keras. Conv1D layer documentation. https://keras.io/api/layers/convolution_layers/convolution1d/
- Scikit-learn. RandomForestClassifier documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
