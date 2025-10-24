# Facial_Expression_Depression_Recognition

A research-oriented library containing multiple deep learning models for facial depression recognition, developed by Dr. Muhammad Turyalai Khan and other researchers.

This library provides a unified framework for implementing, comparing, and extending various facial depression recognition methods.
Each method can be trained, tested, or used for feature extraction.

Current methods included:

# ERBMA-Net

M. T. Khan, Y. Cao, F. Shafait, and W. Jun "ERBMA-Net: Enhanced Random Binary Multilevel Attention Network for Facial Depression Recognition.” IEEE Transactions on Computational Social Systems, 2025. DOI: https://doi.org/10.1109/TCSS.2025.3596047

Steps to use the code:
First run the feature extraction code that will save train, validation, and test csv files.
Secondly, run train code to train the regression model on the extracted features.
Third, run the test code to verify the performance on test data (test.csv file).

Preprocessing steps:
OpenCV for frame extraction at seconds interval.
MTCNN for facial key points detection (face, eyes, mouth)
Selective cropping of face (224x224), eyes (96x192), and mouth (96x128)

# LMTformer

L. He, J. Zhao, J. Zhang, J. Jiang, S. Qi, Z. Wang, and D. Wu, LMTformer: facial depression recognition with lightweight multi-scale transformer from videos, Appl. Intell. 5, 195, 2025.

# STA-DRN

Pan, Y., Shang, Y., Liu, T., Shao, Z., Guo, G., Ding, H., & Hu, Q. "Spatial–temporal attention network for depression recognition from facial videos." Expert systems with applications, 237, 121410, 2024.

# AVEC2014 Dataset
AVEC2014 dataset is a public dataset, which contains 300 videos of depressed patients. Each video had corresponding ground truth scores and varied in duration from 6 to 248 seconds, recorded at 30 frames per second (fps). 
