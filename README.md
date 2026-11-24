# AI-Based Brain Tumor Classification Using MRI with Explainable Deep Learning

An explainable deep learning project that classifies brain MRI images into four categories — **Glioma, Meningioma, Pituitary Tumor, and No Tumor** — using Convolutional Neural Networks (CNN) and MobileNetV2 transfer learning, with Grad-CAM visualizations to understand model predictions.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN%20%7C%20MobileNetV2-red" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-purple" alt="Grad-CAM"/>
  <img src="https://img.shields.io/badge/Test%20Accuracy-89.90%25-success" alt="Test Accuracy"/>
</p>

---

## 1. Project Overview

Brain tumors are abnormal growths of cells in the brain. Analyzing MRI scans is an important part of brain tumor assessment, but interpreting medical images requires specialized expertise.

This project explores how deep learning can assist with classifying brain MRI images into different categories.

The system uses:

- A custom Convolutional Neural Network as a baseline model.
- MobileNetV2 with ImageNet pretrained weights for transfer learning.
- Fine-tuning to improve classification performance.
- Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations of model predictions.

The final fine-tuned MobileNetV2 model achieved **89.90% accuracy on an independent test set of 1,000 images**.

### Project objectives

1. Develop an automated brain MRI image classification pipeline.
2. Classify MRI images into four categories.
3. Compare a custom CNN with transfer learning.
4. Improve model performance through fine-tuning.
5. Visualize important image regions using Grad-CAM.
6. Evaluate performance using precision, recall, F1-score and a confusion matrix.

## 2. Dataset

**Dataset:** BRISC 2025 – Brain Tumor MRI Dataset

- Dataset paper: Fateh et al. (2025)
- Research paper: https://arxiv.org/abs/2506.14318
- Kaggle dataset: https://www.kaggle.com/datasets/briscdataset/brisc2025

The dataset contains 6,000 brain MRI images:

| Dataset split | Images |
|---|---:|
| Original training set | 5,000 |
| Original test set | 1,000 |
| **Total** | **6,000** |

### Classification categories

The classification task contains four classes:

| Class | Description |
|---|---|
| Glioma | Brain tumor category |
| Meningioma | Brain tumor category |
| No Tumor | MRI images without a tumor |
| Pituitary | Pituitary tumor category |

### Original dataset distribution

| Class | Training images | Test images |
|---|---:|---:|
| Glioma | 1,147 | 254 |
| Meningioma | 1,329 | 306 |
| No Tumor | 1,067 | 140 |
| Pituitary | 1,457 | 300 |
| **Total** | **5,000** | **1,000** |

The dataset also includes a segmentation task, but this project focuses on the classification task.

## 3. Dataset Quality and Leakage Audit

Before training, the dataset manifest was examined to identify duplicate images and possible data leakage.

SHA-256 hashing was used to identify identical image files.

The audit identified:

- 46 repeated SHA-256 hash groups.
- 7 image hashes appearing across the original training and test splits.
- No conflicting labels among the identified repeated hashes.

To reduce the risk of data leakage:

1. Training images whose hashes appeared in the test set were excluded from training.
2. Duplicate images within the remaining training data were removed.
3. The original test set was retained for independent evaluation.

After cleaning:

| Dataset | Images |
|---|---:|
| Cleaned training pool | 4,955 |
| Independent test set | 1,000 |
| **Total used** | **5,955** |

This step helps ensure that identical images do not appear in both the training and test datasets.

## 4. Train, Validation and Test Split

The cleaned training pool was divided using a stratified split.

- Training: 80%
- Validation: 20%
- Random state: 42
- Stratification: Tumor class labels

The independent test set was not used for training or validation.

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| Glioma | 918 | 229 | 254 |
| Meningioma | 1,048 | 262 | 306 |
| No Tumor | 846 | 212 | 140 |
| Pituitary | 1,152 | 288 | 300 |
| **Total** | **3,964** | **991** | **1,000** |

The total number of images used across all three splits is 5,955.

## 5. Image Preprocessing

All MRI images were prepared for deep learning using a consistent preprocessing pipeline.

| Operation | Configuration |
|---|---|
| Image format | RGB |
| Image dimensions | 224 × 224 |
| Input channels | 3 |
| Pixel normalization | Divide by 255 |
| Input pixel range | 0–1 |
| Batch size | 8 |
| Validation/test shuffling | Disabled |

### Data augmentation

Data augmentation was applied during model training to introduce controlled image variations.

The augmentation operations included:

- Random Rotation: factor 0.03
- Random Zoom: factor 0.08
- Random Contrast: factor 0.1

The model's internal rescaling layer transforms normalized inputs from the 0–1 range to the -1–1 range for MobileNetV2.

The same normalized image scale is maintained during validation and test evaluation.

## 6. Model Development

Two deep learning approaches were implemented and evaluated.

### Model 1: Custom CNN

A custom Convolutional Neural Network was developed as the baseline model.

Architecture:

| Layer | Configuration |
|---|---|
| Convolution | 32 filters |
| Max Pooling | Downsampling |
| Convolution | 64 filters |
| Max Pooling | Downsampling |
| Convolution | 128 filters |
| Max Pooling | Downsampling |
| Convolution | 256 filters |
| Global Average Pooling | Feature aggregation |
| Dense | 128 units |
| Dropout | Regularization |
| Output | 4 classes |

The model was trained for 15 epochs.

**Validation results:**

| Metric | Result |
|---|---:|
| Training accuracy | 81.05% |
| Validation accuracy | 84.16% |
| Validation loss | 0.4349 |
| Macro F1-score | 83.99% |

The custom CNN established a baseline for comparison with transfer learning.

### Model 2: MobileNetV2 Transfer Learning

MobileNetV2 pretrained on ImageNet was used as the feature extraction backbone.

The architecture consists of:

- MobileNetV2 pretrained base.
- Global Average Pooling.
- Dropout.
- Dense layer with 128 units.
- Dropout.
- Four-class output layer.

The pretrained base initially remained frozen while the classification head was trained.

Initial training:

| Metric | Result |
|---|---:|
| Training accuracy | 89.28% |
| Validation accuracy | 88.29% |
| Validation loss | 0.3294 |

### Fine-tuning

After initial transfer learning, selected MobileNetV2 layers were unfrozen for fine-tuning.

Configuration:

- Fine-tuning epochs: 8
- Learning rate: 0.00001 (1e-5)
- Best validation checkpoint selected.

Best fine-tuned validation results:

| Metric | Result |
|---|---:|
| Training accuracy | 95.28% |
| Validation accuracy | 91.93% |
| Validation loss | 0.2189 |
| Macro F1-score | 92.04% |

The best validation checkpoint was selected instead of automatically using the final training epoch.

### Model architecture summary

The fine-tuned MobileNetV2 model contains:

| Parameter | Value |
|---|---:|
| Total parameters | 2,422,468 |
| Trainable parameters | 164,484 |
| Non-trainable parameters | 2,257,984 |

The final fine-tuned model was saved in Keras format for reuse without retraining.

## 7. Explainable AI – Grad-CAM

Deep learning models can produce accurate predictions without clearly explaining which image regions influenced their decisions.

To improve interpretability, Grad-CAM was implemented.

Grad-CAM generates a heatmap that highlights image regions contributing strongly to a selected class prediction.

### How it works

1. Pass an MRI image through the trained model.
2. Extract feature maps from the final convolutional layer.
3. Calculate gradients of the selected class score with respect to those feature maps.
4. Generate a weighted activation heatmap.
5. Overlay the heatmap on the original MRI image.

### What Grad-CAM provides

- A visual representation of model attention.
- Insight into image regions influencing a prediction.
- A way to inspect model behavior beyond the predicted label.
- An additional explainability component for the classification pipeline.

**Important:** Grad-CAM highlights influential regions. It does not provide a medically validated tumor boundary, segmentation mask or clinical explanation.

## 8. Final Independent Test Evaluation

The best fine-tuned MobileNetV2 model was evaluated on the independent test set containing 1,000 images.

### Overall results

| Metric | Result |
|---|---:|
| Test accuracy | **89.90%** |
| Test loss | 0.2776 |
| Macro precision | 91.10% |
| Macro recall | 90.44% |
| Macro F1-score | 90.53% |
| Weighted precision | 90.22% |
| Weighted recall | 89.90% |
| Weighted F1-score | 89.80% |

### Class-wise classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Glioma | 95.91% | 83.07% | 89.03% | 254 |
| Meningioma | 87.07% | 83.66% | 85.33% | 306 |
| No Tumor | 95.04% | 95.71% | 95.37% | 140 |
| Pituitary | 86.38% | 99.33% | 92.40% | 300 |
| **Overall accuracy** | | | **89.90%** | **1,000** |

### Key observations

- No Tumor achieved the highest class-wise F1-score at 95.37%.
- Pituitary achieved the highest recall at 99.33%.
- Glioma achieved 95.91% precision.
- Meningioma was the most challenging category, with an F1-score of 85.33%.
- The model achieved a macro F1-score of 90.53%, indicating good average class-wise performance.

### Confusion matrix

The confusion matrix is available at:

`results/test_confusion_matrix.csv`

It provides the actual-versus-predicted classification counts for all four classes.

The classification report is available at:

`results/test_classification_report.json`

The overall test summary is available at:

`results/test_summary.json`

## 9. Model Comparison

The following table summarizes the main experimental results.

| Model / Stage | Training accuracy | Validation accuracy |
|---|---:|---:|
| Custom CNN | 81.05% | 84.16% |
| MobileNetV2 initial | 89.28% | 88.29% |
| MobileNetV2 fine-tuned | 95.28% | **91.93%** |

The final fine-tuned model achieved **89.90% on the independent test set**.

The test accuracy is the primary result to consider when evaluating performance on unseen data.

## 10. Project Structure

```text
Brain-Tumor-Classification-MRI/
│
├── notebooks/
│   └── 01_Dataset_Inspection.ipynb
│
├── models/
│   ├── baseline_cnn.keras
│   └── brain_tumor_mobilenetv2_tested.keras
│
├── processed_data/
│   ├── train_split.csv
│   ├── validation_split.csv
│   └── test_manifest.csv
│
├── results/
│   ├── test_classification_report.json
│   ├── test_confusion_matrix.csv
│   └── test_summary.json
│
├── .gitignore
├── README.md
└── requirements.txt
```

The original MRI dataset is not included in this GitHub repository because of its size. Download it separately from the dataset source and configure the dataset path in the notebook.

## 11. Installation and Setup

### Prerequisites

- Python 3.13.9
- Jupyter Notebook
- TensorFlow 2.21.0
- Git
- A system capable of running TensorFlow

The project was developed and tested on Windows 11 using an 8 GB RAM system with integrated graphics and CPU-based TensorFlow execution.

### Step 1: Clone the repository

```bash
git clone https://github.com/chaitanyaReddy4/Brain-Tumor-Classification-MRI.git

cd Brain-Tumor-Classification-MRI
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the dataset

Download BRISC 2025 from:

https://www.kaggle.com/datasets/briscdataset/brisc2025

Extract the dataset and configure its location in the notebook.

Expected classification directory:

```text
brain_tumor/
└── classification_task/
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── no_tumor/
    │   └── pituitary/
    │
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── no_tumor/
        └── pituitary/
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`notebooks/01_Dataset_Inspection.ipynb`

Follow the notebook cells to inspect the dataset, reproduce the preprocessing and training workflow, and evaluate the models.

## 12. Using the Saved Model for Prediction

The trained model is saved in:

`models/brain_tumor_mobilenetv2_tested.keras`

It can be loaded directly without retraining.

Example prediction code:

```python
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/brain_tumor_mobilenetv2_tested.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

def predict_brain_tumor(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = CLASSES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    print("Predicted class:", predicted_class)
    print(f"Model score: {confidence:.2f}%")

    return predicted_class, confidence
```

Example:

```python
predict_brain_tumor("sample_mri.jpg")
```

**No retraining is required for ordinary predictions.** The saved model can be loaded whenever needed, provided the required dependencies and compatible input preprocessing are available.

The model expects MRI images compatible with the training data. Its four-class output does not establish that an arbitrary image is a valid brain MRI, and its confidence score should not be interpreted as a clinically calibrated probability.

## 13. Technologies Used

| Category | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow, Keras |
| Model architecture | CNN, MobileNetV2 |
| Transfer learning | ImageNet pretrained weights |
| Explainable AI | Grad-CAM |
| Image processing | Pillow, TensorFlow image utilities |
| Data analysis | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Development | Jupyter Notebook, Anaconda |
| Version control | Git, GitHub |

## 14. Limitations

This project is an academic research prototype and not a clinically validated diagnostic tool.

Important limitations include:

- Performance depends on the quality and distribution of the training dataset.
- Images from different scanners, hospitals and MRI sequences may have different characteristics.
- The model is limited to the four classes used during training.
- It may produce misleading predictions for non-MRI images or images outside the training distribution.
- Grad-CAM is an interpretability technique, not proof of medical reasoning or tumor localization.
- Confidence scores are model outputs and are not necessarily calibrated probabilities.
- Independent clinical validation is required before any real-world medical application.

Predictions should not be used to make medical decisions or replace professional radiological assessment.

## 15. Future Improvements

Potential directions for further development:

1. Build a web interface for uploading MRI images and displaying predictions.
2. Integrate Grad-CAM visualizations into the prediction interface.
3. Experiment with additional pretrained architectures such as EfficientNet.
4. Investigate the misclassification of glioma and meningioma images.
5. Evaluate generalization using external MRI datasets.
6. Explore confidence calibration and out-of-distribution detection.
7. Extend the project toward tumor localization and segmentation.
8. Develop a more comprehensive model comparison and robustness evaluation.

## 16. References

1. Fateh et al. (2025). **BRISC 2025: Brain MRI dataset for classification and segmentation tasks.** Research paper: https://arxiv.org/abs/2506.14318
2. BRISC 2025 Dataset: https://www.kaggle.com/datasets/briscdataset/brisc2025
3. Sandler, M. et al. (2018). **MobileNetV2: Inverted Residuals and Linear Bottlenecks.** IEEE/CVF Conference on Computer Vision and Pattern Recognition.
4. Selvaraju, R. R. et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.** IEEE International Conference on Computer Vision.

---

## 17. Author and Project Information

**Project:** AI-Based Brain Tumor Classification Using MRI with Explainable Deep Learning

**GitHub Repository:**  
https://github.com/chaitanyaReddy4/Brain-Tumor-Classification-MRI

**Primary Model:** Fine-tuned MobileNetV2

**Final Test Accuracy:** 89.90%

**Dataset:** BRISC 2025

**Project Type:** Academic Deep Learning / Medical Image Classification

---

*This project demonstrates the application of deep learning, transfer learning, model evaluation and explainable AI techniques to medical image classification. It is intended for educational and research purposes only.*
