import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset Class
class DepressionDataset(Dataset):
    def __init__(self, features_csv, labels_csv):
        # Read feature data
        feature_data = pd.read_csv(features_csv)
        self.features = feature_data.iloc[:, 1:].values  # Feature vectors
        filenames = feature_data.iloc[:, 0].values  # Filenames

        # Read label data
        label_data = pd.read_csv(labels_csv)
        # Create a mapping from filename to label
        label_mapping = {row['filename']: row['HAMD'] for _, row in label_data.iterrows()}

        # Map each filename in features to its label
        self.labels = []
        for filename in filenames:
            # For augmented files, extract the base filename
            base_filename = filename.split('_aug')[0] if '_aug' in filename else filename
            self.labels.append(label_mapping[base_filename])

        # Convert to PyTorch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

# Training Function
def train_model(train_loader, dev_loader, model, criterion, optimizer, scheduler, device, num_epochs):
    train_losses = []  # List to store training losses for each epoch
    val_losses = []  # List to store validation losses for each epoch
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        running_mae = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate RMSE and MAE
            rmse = torch.sqrt(loss)
            mae = torch.abs(outputs - labels).mean()

            running_loss += loss.item()
            running_rmse += rmse.item()
            running_mae += mae.item()

        # Calculate average metrics for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_rmse = running_rmse / len(train_loader)
        epoch_mae = running_mae / len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                # Calculate RMSE and MAE
                rmse = torch.sqrt(loss)
                mae = torch.abs(outputs - labels).mean()

                val_loss += loss.item()
                val_rmse += rmse.item()
                val_mae += mae.item()

        # Calculate average validation metrics
        val_loss_avg = val_loss / len(dev_loader)
        val_rmse_avg = val_rmse / len(dev_loader)
        val_mae_avg = val_mae / len(dev_loader)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}, MAE: {epoch_mae:.4f} | "
              f"Val Loss: {val_loss_avg:.4f}, RMSE: {val_rmse_avg:.4f}, MAE: {val_mae_avg:.4f}")

        # Append the losses to the lists for plotting
        train_losses.append(epoch_loss)
        val_losses.append(val_loss_avg)

        # Update learning rate using ReduceLROnPlateau scheduler
        scheduler.step(val_loss_avg)  # Pass validation loss here
    
    # After training, plot the loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.show()

# Main Script
if __name__ == "__main__":
    # File paths
    data_dir = r"C:\Users\Turyal\Desktop\MTK\RBC_LBCNN"
    train_features_file = os.path.join(data_dir, "train_features.csv")
    train_labels_file = os.path.join(data_dir, "train_label.csv")
    dev_features_file = os.path.join(data_dir, "dev_features.csv")
    dev_labels_file = os.path.join(data_dir, "dev_label.csv")

    # Dataset and Dataloader for Training and Development
    train_dataset = DepressionDataset(train_features_file, train_labels_file)
    dev_dataset = DepressionDataset(dev_features_file, dev_labels_file)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    # Initialize model, criterion, optimizer
    input_size = train_dataset.features.shape[1]  # Number of features (assuming flattened features)
    model = RegressionModel(input_size=input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

    # Learning rate scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    # Train the model
    num_epochs = 100
    train_model(train_loader, dev_loader, model, criterion, optimizer, scheduler, device, num_epochs)
    
    # Save the final trained model and weights
    model_save_dir = r"C:\Users\Turyal\Desktop\MTK\model"
    os.makedirs(model_save_dir, exist_ok=True)

    model_save_path = os.path.join(model_save_dir, "RBC_arch_1.0.pth")
    torch.save(model, model_save_path)
    print(f"Model architecture saved to {model_save_path}")

    model_weights_path = os.path.join(model_save_dir, "RBC_weights_1.1.pth")
    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}")
