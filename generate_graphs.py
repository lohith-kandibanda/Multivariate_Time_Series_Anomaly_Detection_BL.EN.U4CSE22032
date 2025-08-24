# generate_graphs.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_evaluation_graphs(file_path='TEP_Train_Test_with_anomalies_FINAL.csv'):
    """
    Loads the anomaly detection results and generates key performance graphs.
    """
    print(f"Loading results from '{file_path}'...")
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
        print("Please run the main anomaly detection script first to generate the output file.")
        return

    # Define the training period for visualization
    train_end_date = pd.to_datetime('2004-01-05 23:59:00')

    # --- Graph 1: Anomaly Scores Over Time ---
    print("📈 Generating Graph 1: Anomaly Scores Over Time...")
    plt.figure(figsize=(18, 6))
    plt.plot(df.index, df['Abnormality_score'], label='Abnormality Score', color='blue', zorder=2)

    # Shade the training region
    plt.axvspan(df.index.min(), train_end_date, color='lightgreen', alpha=0.4, label='Training Period (Normal)')

    # Add a threshold line for high anomalies
    plt.axhline(y=80, color='red', linestyle='--', linewidth=1.5, label='High Anomaly Threshold (80)')

    plt.title('Anomaly Scores Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Abnormality Score (0-100)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('1_scores_over_time.png')
    print("   Saved '1_scores_over_time.png'")

    # --- Graph 2: Distribution of Scores ---
    print("📊 Generating Graph 2: Distribution of Scores...")
    plt.figure(figsize=(12, 6))

    # Separate scores for training and testing periods
    train_scores = df[df.index <= train_end_date]['Abnormality_score']
    test_scores = df[df.index > train_end_date]['Abnormality_score']

    sns.kdeplot(train_scores, label='Training Period Scores', color='green', fill=True)
    sns.kdeplot(test_scores, label='Testing Period Scores', color='blue', fill=True)

    plt.title('Distribution of Anomaly Scores', fontsize=16)
    plt.xlabel('Abnormality Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('2_score_distribution.png')
    print("   Saved '2_score_distribution.png'")

    # --- Graph 3: Investigating the Highest Anomaly ---
    print("🔍 Generating Graph 3: Investigating the Highest Anomaly...")

    # Find the highest anomaly outside of the training period
    test_df = df[df.index > train_end_date].copy()
    if test_df.empty or test_df['Abnormality_score'].max() < 10:
        print("   No significant anomalies found in the test period to plot.")
        return

    highest_anomaly_time = test_df['Abnormality_score'].idxmax()
    highest_anomaly_row = test_df.loc[highest_anomaly_time]

    # Get the top contributing feature (the first one that is not empty)
    top_feature = None
    for i in range(1, 8):
        feature_name = highest_anomaly_row[f'top_feature_{i}']
        if feature_name:
            top_feature = feature_name
            break

    if not top_feature:
        print("   Could not identify a top feature for the highest anomaly.")
        return

    print(
        f"   Highest anomaly found at {highest_anomaly_time} with score {highest_anomaly_row['Abnormality_score']:.2f}. Top feature: {top_feature}")

    # Define a time window around the anomaly
    time_window = pd.Timedelta(minutes=30)
    zoom_df = df.loc[highest_anomaly_time - time_window: highest_anomaly_time + time_window]

    fig, ax1 = plt.subplots(figsize=(18, 6))

    # Plot the abnormality score on the primary y-axis
    color = 'red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Abnormality Score', color=color, fontsize=12)
    ax1.plot(zoom_df.index, zoom_df['Abnormality_score'], color=color, label='Abnormality Score', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a secondary y-axis for the feature value
    ax2 = ax1.twinx()
    color = 'darkblue'
    ax2.set_ylabel(f'Value of {top_feature}', color=color, fontsize=12)
    ax2.plot(zoom_df.index, zoom_df[top_feature], color=color, label=f'Feature: {top_feature}', linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Deep Dive: Highest Anomaly vs. Top Feature ({top_feature})', fontsize=16)
    plt.axvline(highest_anomaly_time, color='black', linestyle='--', linewidth=1.5,
                label=f'Anomaly Peak at {highest_anomaly_time.time()}')
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plt.savefig('3_anomaly_deep_dive.png')
    print("   Saved '3_anomaly_deep_dive.png'")


if __name__ == '__main__':
    generate_evaluation_graphs()