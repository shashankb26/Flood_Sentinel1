# Flood_Sentinel1
Flood Detection with Sentinel-1 Imagery
This repository contains a Python script for flood detection using Sentinel-1 satellite imagery and deep learning-based semantic segmentation. The code processes raster data (TIFF files) containing Sentinel-1 bands (e.g., VV, VH, vv_vh, elevation) and corresponding flood masks, trains a U-Net model with a ResNet34 backbone, and logs training metrics using Weights & Biases (WandB).

Flood detection is critical for disaster management and response. This project leverages Sentinel-1 Synthetic Aperture Radar (SAR) imagery to identify flooded areas through semantic segmentation. The model is trained to predict pixel-wise flood masks, distinguishing between flooded (1) and non-flooded (0) regions.

Key features:

Data Preprocessing: Contrast stretching and image sharpening to enhance Sentinel-1 imagery.
Dataset: Custom FloodDataset class for loading and augmenting raster data.
Model: U-Net with a ResNet34 backbone, modified for 4 input channels (VV, VH, vv_vh, elevation).
Training: PyTorch-based training loop with Dice Loss and Adam optimizer.
Logging: Integration with WandB for monitoring training progress.
Requirements
Python 3.11.5

Create a Virtual Environment Using Python 3.11.5 :
python -m venv venv
venv\Scripts\activate

Install Dependencies
With the virtual environment activated, install the required packages:
pip install -r requirements.txt

Datasets Prepaparation 
RUN python datasets_preparation.py

RUN the Model for the flood Mapping
RUN flood_mapping.ipynb
