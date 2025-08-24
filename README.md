# Multivariate Time Series Anomaly Detection

## 📋 Overview
This project provides a Python-based machine learning solution to detect anomalies in multivariate time series data from a simulated industrial process. The goal is to identify data points that deviate significantly from normal operational behavior and to pinpoint the primary features contributing to each anomaly.

This solution was developed for the **Honeywell Hackathon** and implements a sophisticated ensemble model that combines **Principal Component Analysis (PCA)** and a **Bidirectional LSTM Autoencoder** to meet the specific success criteria.

---

## ✨ Features
- **Ensemble Model**: Combines the strengths of a linear model (PCA) for structural changes and a non-linear temporal model (Bidirectional LSTM Autoencoder) for complex pattern deviations.  
- **Z-Score Normalization**: Anomaly scores are calculated as a Z-score relative to the normal training period, providing a statistically robust measure of deviation.  
- **Intelligent Feature Attribution**: Identifies the top 7 contributing features for each anomaly by determining which model in the ensemble was more confident in the detection. All rules for ranking, tie-breaking, and filtering (>1% contribution) are applied.  
- **Output Smoothing**: A rolling average is applied to the final scores to ensure stability and prevent erratic jumps between adjacent time points.  
- **Performance Evaluation**: Includes a separate script to generate key performance graphs for visual validation of the results.  

---

## 🚀 Setup and Installation
Follow these steps to set up the project environment.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-name>
```
### 2. Create a Virtual Environment 

# For Windows
```bash
python -m venv venv
venv\Scripts\activate
```
# For macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies

Install all the required libraries using the requirements.txt file.
```bash
pip install -r requirements.txt
```
## How to Run

### 1. Place the Dataset: Ensure the TEP_Train_Test.csv file is in the main project directory.

### 2. Run the Anomaly Detection Script: Execute the main script from your terminal (assuming it's named anomaly_detector.py).
```bash
python anomaly_detector.py
```

(Note: Model training can take 5–15 minutes depending on your computer's CPU.)

### 3. Generate Evaluation Graphs: After the main script has successfully created the output file (TEP_Train_Test_with_anomalies_FINAL.csv), run the graphing script (assuming it's named generate_graphs.py).
```bash
python generate_graphs.py
```

This will save three PNG files in the project directory for evaluation:

1_scores_over_time.png

2_score_distribution.png

3_anomaly_deep_dive.png

## Output

### TEP_Train_Test_with_anomalies_FINAL.csv: A copy of the original dataset with 8 new columns:

### Abnormality_score: A score from 0 to 100 indicating the severity of the anomaly.

top_feature_1 to top_feature_7: The names of the features that contributed most to the anomaly score.

### Graph Images (.png): Visualizations of the model's performance.
