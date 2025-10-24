import os

# Base directory (user should update this to their local dataset path)
# By default, assume the dataset is inside the project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input and output directories
input_base_directory = os.path.join(BASE_DIR, "data", "AVEC2014")
output_directory = os.path.join(BASE_DIR, "outputs")

# Make sure output folder exists
os.makedirs(output_directory, exist_ok=True)