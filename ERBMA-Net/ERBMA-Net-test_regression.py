import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn

# Fully Connected Model for Regression
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),  # Input size is the concatenated feature vector size
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Final output layer
        )

    def forward(self, x):
        return self.fc(x)

# Function to map test labels based on filenames
def map_test_labels(features_csv, labels_csv):
    """Map filenames in test_features.csv to their corresponding labels in test_labels.csv."""
    feature_data = pd.read_csv(features_csv)
    label_data = pd.read_csv(labels_csv)
    label_mapping = {row['filename']: row['HAMD'] for _, row in label_data.iterrows()}

    features = feature_data.drop(columns=['filename']).values
    filenames = feature_data['filename']

    # Map labels to features
    labels = []
    for filename in filenames:
        base_filename = filename.split('_aug')[0] if '_aug' in filename else filename
        labels.append(label_mapping[base_filename])

    return features, np.array(labels).astype(np.float32), filenames

# Directory and file paths
model_save_dir = r"C:\Users\Turyal\Desktop\MTK\model"
model_weights_path = os.path.join(model_save_dir, "RBC_weights_1.1.pth")  # Updated weight path
save_directory = r"C:\Users\Turyal\Desktop\MTK"
test_features_file = r"C:\Users\Turyal\Desktop\MTK\RBC_LBCNN\test_features.csv"
test_labels_file = r"C:\Users\Turyal\Desktop\MTK\RBC_LBCNN\test_label.csv"

# Prepare the test data
X_test, y_test, test_filenames = map_test_labels(test_features_file, test_labels_file)

# Convert to torch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Instantiate the model
input_size = X_test.shape[1]  # Input size matches the feature vector size
model = RegressionModel(input_size=input_size)

# Load the model weights
model.load_state_dict(torch.load(model_weights_path))
model.eval()  # Set the model to evaluation mode

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_test_tensor = X_test_tensor.to(device)

# Make predictions
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze().cpu().numpy()  # Move predictions back to CPU

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")

# Save RMSE and MAE to CSV
metrics_df = pd.DataFrame({
    'Model': [model_weights_path],
    'RMSE': [rmse],
    'MAE': [mae]
})
metrics_file_path = os.path.join(save_directory, 'test_metrics.csv')
if os.path.exists(metrics_file_path):
    metrics_df.to_csv(metrics_file_path, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(metrics_file_path, index=False)

# Save predictions to a CSV
predictions_df = pd.DataFrame({
    'filename': test_filenames,
    'True HAMD': y_test,
    'Predicted HAMD': y_pred
})
predictions_csv_path = os.path.join(r"C:\Users\Turyal\Desktop\MTK\SECA", "test_predictions_RBC_1.0.csv")
predictions_df.to_csv(predictions_csv_path, index=False)
print(f"Test predictions saved to {predictions_csv_path}")
