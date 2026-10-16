# Powder Particle Classification with Scanning Electron Microscopy Images using Machine Learning Techniques

This repository contains the essential files for the classification and analysis of powder particles using SEM images and machine learning techniques, as described in the article *"Powder Particle Classification with Scanning Electron Microscopy Images Using Machine Learning Techniques"*.
It classifies powder materials and alloys from Scanning Electron Microscopy (SEM) images using image processing, particle features, and deep learning.

## Key Files:

- **Cropping.ipynb**: A Jupyter notebook for cropping labeled fragments from SEM images, enabling focused analysis of powder particles.
- **Pipeline.ipynb**: Implements the HCF+PCA+XGBoost method for feature extraction and classification of powder particles.
- **YOLO.ipynb**: A notebook dedicated to training the YOLOv8 model for particle classification in SEM images.
- **contour_coordinates_Al.csv**: Contains the contour coordinates of particles from the AlSiMg powder.
- **contour_coordinates_St.csv**: Contains the contour coordinates of particles from the 316L powder.
- **contour_coordinates_Ti.csv**: Contains the contour coordinates of particles from the TiAlV powder.
- **objects_summary_Al.xlsx**: Excel file summarizing the detected objects (particles) in SEM images for AlSiMg powder.
- **objects_summary_St.xlsx**: Excel file summarizing the detected objects (particles) in SEM images for 316L powder.
- **objects_summary_Ti.xlsx**: Excel file summarizing the detected objects (particles) in SEM images for TiAlV powder.

## Folders:

- **train14**: Contains the best model configuration along with its statistics, providing the optimal setup for particle classification.
- **SEM**: Contains the raw SEM images of powder particles, used for the classification tasks in the repository.

This collection provides an end-to-end solution for powder particle classification, combining data preprocessing, model training, and post-processing analysis.

## PowderScope AI 2.0 web app

The repository includes a modern web application for classifying the alloy material of uploaded Scanning Electron Microscopy (SEM) powder images across three additive manufacturing alloys: **AlSiMg**, **316L Stainless Steel**, and **TiAlV**.

The updated model architecture achieves **95.56% 5-fold cross-validation accuracy** (up from the baseline 86.67%) utilizing a comprehensive 164-dimensional feature pipeline:
- **GLCM (Gray-Level Co-occurrence Matrix)**: Haralick texture descriptors (contrast, dissimilarity, homogeneity, energy, correlation) computed across multiple pixel offsets and angles.
- **2D Fast Fourier Transform (FFT)**: Radial frequency power spectrum energy distributions distinguishing fine versus coarse powder textures.
- **Gabor Filter Banks**: Multi-orientation and multi-wavelength filters capturing microstructural directional grain patterns.
- **Sobel & Canny Gradients**: Gradient magnitude statistics, quantiles, and edge density at multiple thresholds.
- **Shape Moments & Spatial Descriptors**: Hu moments and Local Binary Patterns (LBP) capturing particle boundary curvature.
- **Calibrated Ensemble Classifier**: Standard-scaled soft-voting ensemble combining Calibrated Ridge Classifier, Logistic Regression, and Linear Support Vector Machine (SVM).

### Build and run locally

Use Python 3.10 or newer (tested on Python 3.13). The frontend is served directly from `static/`, so it does not need a separate Node.js build. The Python build process trains the image classifier and saves it to `artifacts/material_classifier.joblib`.

1. Clone the repository and enter its directory:

   ```powershell
   git clone https://github.com/Odin531/sem-powder-particle-classification.git
   cd sem-powder-particle-classification
   ```

2. Create and activate a virtual environment, then install the dependencies:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Build the classifier:

   ```powershell
   python train.py
   ```

   Training reads the labeled images from `SEM/Al/`, `SEM/Steel/`, and `SEM/Ti/`, extracts image features, evaluates the classifier with 5-fold stratified cross-validation, and writes the trained model to `artifacts/material_classifier.joblib`. Each class needs at least five images.

4. Start the web app:

   ```powershell
   python -m uvicorn app:app --reload
   ```

   Open <http://127.0.0.1:8000> in your browser. If the model artifact does not exist, the app trains and saves it when it starts; run `python train.py` first to build it explicitly. Stop the local server with **Ctrl+C**.

### Web Interface Features

- **High-Readability Modern UI**: Built with Plus Jakarta Sans and JetBrains Mono typography, meeting high contrast standards for maximum legibility and visibility.
- **1-Click Quick-Test Gallery**: Directly test 15 built-in micrographs across AlSiMg, 316L Stainless Steel, and TiAlV with a single click.
- **Interactive Region Proposals**: Candidate particle bounding boxes, centroids, and areas mapped dynamically with synchronized row selection.
- **Live Verification Dashboard**: Displays real-time 5-fold cross-validation confusion matrix, precision/recall/F1 breakdowns, and sample counts.
- **Batch Processing**: Supports dragging and dropping up to 20 images (up to 50 MB) simultaneously with distribution breakdown.

The original notebooks, contour tables, object summaries, and training artifacts remain available for further experiments.
