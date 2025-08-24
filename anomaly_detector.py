# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Bidirectional

# Suppress warnings and set seeds
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
tf.random.set_seed(42)
np.random.seed(42)


# ==============================================================================
# 2. ANOMALY DETECTION CLASS (Final Corrected Version with Full Features)
# ==============================================================================
class AnomalyDetectionSystem:
    """
    Final version using a PCA + BiLSTM Autoencoder Ensemble with a
    corrected Z-score, output smoothing, and full feature contribution logic.
    """

    def __init__(self, timesteps=60, pca_variance=0.95, score_smoothing_window=15):
        self.timesteps = timesteps
        self.score_smoothing_window = score_smoothing_window
        self.bilstm_autoencoder = None
        self.pca = PCA(n_components=pca_variance)
        self.lstm_scaler = MinMaxScaler()
        self.pca_scaler = StandardScaler()
        self.feature_names = []
        self.n_features = 0
        print("✅ AnomalyDetectionSystem (Final Corrected Ensemble) initialized.")

    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Loads and prepares data from CSV."""
        print(f"🔄 Loading and preparing data from '{file_path}'...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.feature_names = df.columns.tolist()
        self.n_features = len(self.feature_names)
        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Splits data into train and analysis sets."""
        print("🔪 Splitting data...")
        # FIX: Changed year from 2024 to 2004 ---
        train_start, train_end = '2004-01-01 00:00:00', '2004-01-05 23:59:00'
        train_df = df.loc[train_start:train_end]
        analysis_df = df.copy()
        return train_df, analysis_df

    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Creates sequences for the Autoencoder model."""
        sequences = []
        for i in range(len(data) - self.timesteps + 1):
            sequences.append(data[i:(i + self.timesteps)])
        return np.array(sequences)

    def train_models(self, train_df: pd.DataFrame):
        """Trains both the BiLSTM Autoencoder and the PCA models."""
        print("\n--- Training Models ---")
        # Train PCA
        train_scaled_pca = self.pca_scaler.fit_transform(train_df)
        self.pca.fit(train_scaled_pca)

        # Train BiLSTM Autoencoder
        train_scaled_bilstm = self.lstm_scaler.fit_transform(train_df)
        train_sequences = self.create_sequences(train_scaled_bilstm)

        inputs = Input(shape=(self.timesteps, self.n_features))
        encoded = Bidirectional(LSTM(128, activation='relu', return_sequences=False))(inputs)
        encoded = RepeatVector(self.timesteps)(encoded)
        decoded = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(encoded)
        output = TimeDistributed(Dense(self.n_features))(decoded)

        self.bilstm_autoencoder = Model(inputs, output)
        self.bilstm_autoencoder.compile(optimizer='adam', loss='mae')
        self.bilstm_autoencoder.fit(
            train_sequences, train_sequences,
            epochs=30, batch_size=128, validation_split=0.1, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')]
        )
        print("\n   Both models trained successfully.")

    def get_ensemble_contributors(self, analysis_df: pd.DataFrame, pca_norm: np.ndarray, bilstm_norm: np.ndarray,
                                  analysis_scaled_pca: np.ndarray, analysis_reconstructed_pca: np.ndarray,
                                  analysis_sequences_bilstm: np.ndarray,
                                  reconstructions_bilstm: np.ndarray) -> pd.DataFrame:
        """Gets feature contributions from the model that scored higher for each point, applying all rules."""
        print("📊 Determining intelligent feature contributions...")

        pca_feature_error = np.abs(analysis_scaled_pca - analysis_reconstructed_pca)
        bilstm_feature_error = np.abs(reconstructions_bilstm - analysis_sequences_bilstm)

        top_features_list = []
        for i in range(len(analysis_df)):
            if pca_norm[i] > bilstm_norm[i]:
                # Use PCA contributors
                errors = pca_feature_error[i]
                total_error = np.sum(errors) or 1

                # Filter by >1% contribution rule
                filtered_features = []
                for j, error in enumerate(errors):
                    if (error / total_error) > 0.01:
                        filtered_features.append((self.feature_names[j], error))

                # Sort by magnitude (desc) and name (asc for tie-break)
                sorted_features = sorted(filtered_features, key=lambda x: (-x[1], x[0]))
                top_names = [name for name, error in sorted_features[:7]]
            else:
                # Use BiLSTM contributors
                if i >= self.timesteps - 1:
                    last_step_error = bilstm_feature_error[i - (self.timesteps - 1)][-1]
                    total_error = np.sum(last_step_error) or 1

                    filtered_features = []
                    for j, error in enumerate(last_step_error):
                        if (error / total_error) > 0.01:
                            filtered_features.append((self.feature_names[j], error))

                    sorted_features = sorted(filtered_features, key=lambda x: (-x[1], x[0]))
                    top_names = [name for name, error in sorted_features[:7]]
                else:
                    top_names = []

            top_names.extend([''] * (7 - len(top_names)))
            top_features_list.append(top_names)

        feature_cols = [f'top_feature_{i + 1}' for i in range(7)]
        return pd.DataFrame(top_features_list, columns=feature_cols, index=analysis_df.index)

    def run(self, input_csv_path: str, output_csv_path: str):
        try:
            original_df = self.load_and_prepare_data(input_csv_path)
            train_df, analysis_df = self.split_data(original_df)
            self.train_models(train_df)

            print("\n💯 Calculating final scores...")

            # 1. Calculate raw scores for the ENTIRE analysis period
            analysis_scaled_pca = self.pca_scaler.transform(analysis_df)
            analysis_reconstructed_pca = self.pca.inverse_transform(self.pca.transform(analysis_scaled_pca))
            pca_raw_scores = np.mean(np.square(analysis_scaled_pca - analysis_reconstructed_pca), axis=1)

            analysis_scaled_bilstm = self.lstm_scaler.transform(analysis_df)
            analysis_sequences = self.create_sequences(analysis_scaled_bilstm)
            reconstructions_bilstm = self.bilstm_autoencoder.predict(analysis_sequences)
            mae_loss = np.mean(np.abs(reconstructions_bilstm - analysis_sequences), axis=1)
            bilstm_raw_scores = np.full(len(analysis_df), np.nan)
            bilstm_raw_scores[self.timesteps - 1:] = np.mean(mae_loss, axis=1)
            bilstm_raw_scores = pd.Series(bilstm_raw_scores).fillna(method='bfill').fillna(method='ffill').values

            # 2. Normalize and combine scores
            pca_norm = MinMaxScaler().fit_transform(pca_raw_scores.reshape(-1, 1)).flatten()
            bilstm_norm = MinMaxScaler().fit_transform(bilstm_raw_scores.reshape(-1, 1)).flatten()
            combined_scores = pd.Series((pca_norm + bilstm_norm) / 2.0, index=analysis_df.index)

            # 3. Establish the baseline using ONLY the training period's scores
            train_scores = combined_scores.loc[train_df.index]
            train_mean = train_scores.mean()
            train_std = train_scores.std()
            if train_std == 0: train_std = 1e-6

            # 4. Calculate Z-scores and scale to 0-100
            z_scores = (combined_scores - train_mean) / train_std
            final_scores = np.clip(z_scores.values, 0, 20) * 5

            # 5. Apply smoothing to the final scores
            final_scores_series = pd.Series(final_scores, index=analysis_df.index)
            smoothed_scores = final_scores_series.rolling(window=self.score_smoothing_window, center=True,
                                                          min_periods=1).mean()

            # 6. Get feature contributions
            top_features_df = self.get_ensemble_contributors(analysis_df, pca_norm, bilstm_norm,
                                                             analysis_scaled_pca, analysis_reconstructed_pca,
                                                             analysis_sequences, reconstructions_bilstm)

            # Generate final output file
            print(f"💾 Generating final output file...")
            result_df = original_df.copy()
            result_df['Abnormality_score'] = smoothed_scores
            result_df = result_df.join(top_features_df)

            # --- CORRECTION: THE HARD FILTER HAS BEEN REMOVED ---
            # The logic inside get_ensemble_contributors now correctly handles
            # the >1% rule for all rows, so this extra filter is no longer needed.

            result_df.to_csv(output_csv_path)
            print(f"\n🎉 Success! Final corrected script complete. Output saved to: {output_csv_path}")

        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()


# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main function to define file paths and run the anomaly detection system."""
    input_file = 'TEP_Train_Test.csv'
    output_file = 'TEP_Train_Test_with_anomalies_FINAL.csv'

    print("=====================================================")
    print("      Final Anomaly Detection Script                 ")
    print("=====================================================")

    system = AnomalyDetectionSystem()
    system.run(input_csv_path=input_file, output_csv_path=output_file)


if __name__ == '__main__':
    main()