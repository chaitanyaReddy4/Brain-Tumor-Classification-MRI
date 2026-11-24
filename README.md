
# 🧠 AI-Based Brain Tumor Classification Using MRI
### Explainable Deep Learning with MobileNetV2, EfficientNetB0 and Grad-CAM

<p align="center">

**An experimental deep learning system for four-class brain MRI classification with explainable AI and an interactive Streamlit application.**

</p>

---

## 📌 1. Project Overview

Brain tumors are associated with different patterns and appearances in Magnetic Resonance Imaging (MRI) scans. Analyzing these images requires careful examination, and deep learning offers a way to investigate automated image classification.

This project develops an **AI-based brain tumor classification system** using transfer learning and convolutional neural networks.

The system classifies brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

MobileNetV2 is selected as the primary model. EfficientNetB0 is implemented as a comparison model, while a custom CNN serves as a baseline.

To improve interpretability, Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated to visualize image regions that influence model predictions.

An interactive Streamlit application allows users to upload MRI images, view predictions, inspect class scores and visualize Grad-CAM overlays.

**Project type:** Individual Academic Capstone Project  
**Domain:** Artificial Intelligence / Deep Learning / Medical Image Analysis  
**Dataset:** BRISC 2025  
**Primary architecture:** MobileNetV2  
**Explainability:** Grad-CAM  
**Application:** Streamlit

> **Disclaimer:** This is an educational research prototype. It is not a clinically validated diagnostic system and must not be used to make medical decisions.

---

## 🎯 2. Problem Statement

Manual interpretation of brain MRI images requires specialized medical expertise. Exploring automated image classification can help demonstrate how deep learning models identify image patterns associated with different brain tumor categories.

The objective of this project is to develop and evaluate a deep learning pipeline that:

1. Processes brain MRI images.
2. Classifies images into four predefined categories.
3. Compares multiple deep learning architectures.
4. Evaluates classification performance using appropriate metrics.
5. Provides visual explanations through Grad-CAM.
6. Demonstrates inference through an interactive web application.

---

## 💡 3. Proposed Solution

The proposed system follows a complete machine learning workflow:

1. Dataset acquisition and inspection.
2. Duplicate detection and data leakage auditing.
3. Image preprocessing and augmentation.
4. Baseline CNN development.
5. Transfer learning using MobileNetV2.
6. Transfer learning using EfficientNetB0.
7. Fine-tuning and model selection.
8. Test-set evaluation.
9. Grad-CAM explainability.
10. Streamlit application development.

MobileNetV2 is used as the final primary classification model because it achieved better test performance than EfficientNetB0 in the conducted comparison.

---

## 📂 4. Dataset Description

**Dataset:** BRISC 2025

**Dataset source:**  
https://www.kaggle.com/datasets/briscdataset/brisc2025

The dataset provides brain MRI images for classification and segmentation tasks.

This project uses the classification task.

### Original dataset composition

| Dataset | Number of images |
|---|---:|
| Training | 5,000 |
| Testing | 1,000 |
| **Total** | **6,000** |

### Classification categories

| Class | Description |
|---|---|
| Glioma | Glioma category |
| Meningioma | Meningioma category |
| No Tumor | No-tumor category |
| Pituitary | Pituitary category |

### Final experimental data split

A data-quality audit was performed before generating the training and validation sets.

| Split | Images |
|---|---:|
| Training | 3,964 |
| Validation | 991 |
| Testing | 1,000 |
| **Total** | **5,955** |

The original training pool was cleaned to 4,955 images before the stratified split.

The original 1,000-image test set was retained for final evaluation.

### Data leakage prevention

A SHA256-based duplicate audit was performed on the original image files.

Audit observations:

- 46 repeated hash groups were identified.
- 7 hashes were found across the original training and testing sets.
- No conflicting labels were identified among the repeated hashes.
- Training copies of hashes appearing in the test set were excluded.
- Remaining training images were deduplicated.

The cleaned training pool contained 4,955 images.

An 80:20 stratified training-validation split was then created using `random_state=42`.

The same split manifests were used for MobileNetV2 and EfficientNetB0 to maintain a fair model comparison.

---

## ⚙️ 5. Data Preprocessing

All images were prepared for transfer learning.

### Preprocessing operations

- Images decoded and converted to RGB.
- Images resized to 224 × 224.
- Aspect ratio preserved using padding.
- Pixel values converted to floating-point representation.
- Model-specific input preprocessing applied.

### Data augmentation

The MobileNetV2 experiment used:

- Random rotation: 0.03
- Random zoom: 0.08
- Random contrast: 0.1
- Reflect padding for rotation and zoom.

Contrast augmentation was configured with an input range of (0, 1) for the corrected MobileNetV2 pipeline.

EfficientNetB0 used its own compatible input preprocessing and an augmentation contrast range of (0, 255).

These preprocessing differences were maintained to respect the respective model architectures.

---

## 🏗️ 6. Model Architectures

Three model approaches were explored.

### 6.1 Custom CNN — Baseline

A custom convolutional neural network was developed to establish a baseline.

Architecture components:

- Convolutional layers
- Max pooling
- Global average pooling
- Fully connected layer
- Dropout
- Four-class softmax output

The baseline CNN achieved:

| Metric | Validation result |
|---|---:|
| Validation accuracy | 84.16% |
| Validation macro F1-score | 83.99% |

These are validation results, not final test-set results.

### 6.2 MobileNetV2 — Primary Model

MobileNetV2 was selected for transfer learning using ImageNet pretrained weights.

Architecture:

- MobileNetV2 feature extraction backbone
- Global Average Pooling
- Dropout
- Dense layer with 128 units
- Dropout
- Four-unit softmax classification layer

**Input size:** 224 × 224 × 3

#### Training strategy

**Stage 1 — Transfer learning**

- Backbone initially frozen.
- Classification head trained.
- Learning rate: 0.001.
- Training epochs: 12.

**Stage 2 — Fine-tuning**

- Selected backbone layers unfrozen.
- Batch normalization layers kept frozen.
- Learning rate: 0.00001.
- Fine-tuning epochs: 8.

A corrected augmentation experiment was subsequently performed to ensure the contrast transformation used the correct pixel range.

The corrected model was saved as:

`mobilenetv2_corrected_experiment.keras`

#### MobileNetV2 validation performance

| Metric | Result |
|---|---:|
| Best validation accuracy | 92.33% |
| Best validation loss | 0.2105 |

### 6.3 EfficientNetB0 — Comparison Model

EfficientNetB0 was implemented using ImageNet pretrained weights.

Architecture components:

- EfficientNetB0 feature extraction backbone
- Global Average Pooling
- Dropout
- Dense layer with 128 units
- Dropout
- Four-class softmax output

**Input size:** 224 × 224 × 3

#### Training strategy

**Stage 1 — Transfer learning**

- Backbone frozen.
- Classification head trained.
- Learning rate: 0.001.
- Maximum epochs: 8.

**Stage 2 — Fine-tuning**

- Last 30 backbone layers selected for unfreezing.
- Batch normalization layers kept frozen.
- Learning rate: 0.00001.
- Fine-tuning epochs: 8.

#### EfficientNetB0 validation performance

| Metric | Result |
|---|---:|
| Best validation accuracy | 91.02% |
| Best validation loss | 0.21728 |

---

## 📊 7. Final Model Evaluation

Both transfer-learning models were evaluated using the same held-out test set containing 1,000 images.

### 7.1 Overall test performance

| Evaluation Metric | MobileNetV2 | EfficientNetB0 |
|---|---:|---:|
| Test Accuracy | **92.50%** | 90.80% |
| Macro Precision | **93.57%** | 92.09% |
| Macro Recall | **93.04%** | 91.78% |
| Macro F1-score | **93.14%** | 91.77% |
| Weighted F1-score | **92.43%** | 90.66% |
| Correct predictions | **925 / 1000** | 908 / 1000 |

### 7.2 MobileNetV2 — Class-wise test results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Glioma | 96.33% | 82.68% | 88.98% | 254 |
| Meningioma | 86.07% | 90.85% | 88.39% | 306 |
| No Tumor | 97.89% | 99.29% | 98.58% | 140 |
| Pituitary | 94.01% | 99.33% | 96.60% | 300 |
| **Macro Average** | **93.57%** | **93.04%** | **93.14%** | **1,000** |

### 7.3 MobileNetV2 confusion matrix

Rows represent actual classes; columns represent predicted classes.

Class order:

Glioma, Meningioma, No Tumor, Pituitary.

| Actual / Predicted | Glioma | Meningioma | No Tumor | Pituitary |
|---|---:|---:|---:|---:|
| Glioma | 210 | 43 | 0 | 1 |
| Meningioma | 7 | 278 | 3 | 18 |
| No Tumor | 0 | 1 | 139 | 0 |
| Pituitary | 1 | 1 | 0 | 298 |

### 7.4 EfficientNetB0 — Class-wise test results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Glioma | 92.58% | 83.46% | 87.78% | 254 |
| Meningioma | 88.28% | 83.66% | 85.91% | 306 |
| No Tumor | 99.29% | 100.00% | 99.64% | 140 |
| Pituitary | 88.24% | 100.00% | 93.75% | 300 |
| **Macro Average** | **92.09%** | **91.78%** | **91.77%** | **1,000** |

### 7.5 EfficientNetB0 confusion matrix

| Actual / Predicted | Glioma | Meningioma | No Tumor | Pituitary |
|---|---:|---:|---:|---:|
| Glioma | 212 | 34 | 0 | 8 |
| Meningioma | 17 | 256 | 1 | 32 |
| No Tumor | 0 | 0 | 140 | 0 |
| Pituitary | 0 | 0 | 0 | 300 |

### 7.6 Model selection

MobileNetV2 achieved:

- 1.70 percentage points higher test accuracy.
- 1.35 percentage points higher macro F1-score.
- 1.77 percentage points higher weighted F1-score.

Therefore, MobileNetV2 was selected as the primary model for the Streamlit application.

**Important:** The reported metrics describe performance on this particular dataset and split. They do not establish clinical performance or generalization to all hospitals and scanners.

---

## 🔍 8. Explainable AI — Grad-CAM

Grad-CAM was integrated to provide a visual explanation of model predictions.

It uses gradients flowing into the final convolutional feature map to generate a class-specific activation heatmap.

### Implementation

- Primary model: MobileNetV2.
- Target convolutional layer: `out_relu`.
- Final feature map size: 7 × 7 × 1280.
- Heatmap resized to 224 × 224.
- Heatmap combined with the original MRI image.

### Visualization interpretation

| Heatmap color | General interpretation |
|---|---|
| Red / Yellow | Stronger model activation |
| Green | Intermediate activation |
| Blue | Weaker activation |

Grad-CAM highlights regions that influenced the model's prediction.

It does not provide precise tumor segmentation or prove that the model is focusing exclusively on tumor tissue.

Representative Grad-CAM visualizations were generated for all four classes.

Results are stored in:

`results/gradcam/`

---

## 🖥️ 9. Streamlit Web Application

An interactive application was developed to demonstrate the trained MobileNetV2 model.

### Features

- MRI image upload.
- JPG, JPEG and PNG support.
- Image preprocessing.
- Four-class prediction.
- Model confidence score.
- Class-wise prediction scores.
- Grad-CAM heatmap generation.
- Original MRI and heatmap overlay display.
- Medical-use disclaimer.

### Application workflow

1. User uploads an MRI image.
2. Image is converted to RGB.
3. Image is resized and padded to 224 × 224.
4. The trained MobileNetV2 model processes the image.
5. The predicted class and class scores are displayed.
6. Grad-CAM generates a heatmap.
7. The original MRI and heatmap overlay are displayed.

### Run locally

```bash
"D:\anaconda\python.exe" -m streamlit run app.py
```

Alternatively, if the correct Python environment is activated:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

External MRI images have also been used for demonstration of inference. These predictions are qualitative demonstrations and are not included in the reported test accuracy.

---

## 🛠️ 10. Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow / Keras |
| Baseline architecture | Custom CNN |
| Primary architecture | MobileNetV2 |
| Comparison architecture | EfficientNetB0 |
| Explainable AI | Grad-CAM |
| Image processing | Pillow, TensorFlow |
| Numerical computing | NumPy |
| Visualization | Matplotlib |
| Web interface | Streamlit |
| Dataset | BRISC 2025 |
| Development environment | Jupyter Notebook / Anaconda |

---

## 📁 11. Project Structure

```text
Brain_Tumor_AI_Project/
│
├── app.py
├── README.md
├── .gitignore
│
├── 01_Dataset_Inspection.ipynb
│
├── notebooks/
│   ├── 01_Dataset_Inspection.ipynb
│   └── 01_Dataset_Inspection_latest.ipynb
│
├── models/
│   ├── baseline_cnn.keras
│   ├── brain_tumor_mobilenetv2.keras
│   └── brain_tumor_mobilenetv2_tested.keras
│
├── processed_data/
│   ├── train_split.csv
│   ├── validation_split.csv
│   └── test_manifest.csv
│
├── results/
│   ├── confusion_matrix_comparison.png
│   ├── model_comparison.png
│   ├── final_project_results.json
│   ├── efficientnetb0_test_report.json
│   ├── efficientnetb0_test_confusion_matrix.csv
│   ├── mobilenetv2_corrected_summary.json
│   ├── mobilenetv2_corrected_classification_report.json
│   ├── mobilenetv2_corrected_confusion_matrix.csv
│   └── gradcam/
│
├── brain_tumor/
│   ├── classification_task/
│   ├── segmentation_task/
│   ├── manifest.csv
│   └── README.md
│
└── Experimental model checkpoints
    ├── baseline_cnn_model.keras
    ├── best_cnn_model.keras
    ├── best_finetuned_model.keras
    ├── best_mobilenet_model.keras
    ├── efficientnetb0_best.keras
    ├── efficientnetb0_finetuned_best.keras
    └── mobilenetv2_corrected_experiment.keras
```

The dataset and trained model files are not necessarily included in the public Git repository. Refer to the dataset source and project setup instructions.

---

## 🚀 12. Installation and Setup

### Prerequisites

- Python environment compatible with TensorFlow.
- Jupyter Notebook or an IDE.
- Sufficient RAM and storage for model loading and image processing.

### Install dependencies

```bash
pip install tensorflow numpy matplotlib pillow streamlit
```

### Clone the repository

```bash
git clone https://github.com/chaitanyaReddy4/Brain-Tumor-Classification-MRI.git

cd Brain-Tumor-Classification-MRI
```

### Dataset setup

1. Download BRISC 2025 from Kaggle.
2. Extract the dataset.
3. Place the classification dataset in the expected project directory.
4. Ensure the trained MobileNetV2 model is available at the path used by `app.py`.

### Launch the application

```bash
streamlit run app.py
```

---

## 📈 13. Key Contributions and Work Completed

The following are the major technical contributions and completed project activities:

1. Implemented a complete brain MRI classification pipeline for four classes.
2. Performed dataset inspection and class verification.
3. Implemented SHA256-based duplicate detection.
4. Identified and removed training-test duplicate leakage.
5. Created stratified training and validation splits.
6. Developed a custom CNN baseline.
7. Implemented MobileNetV2 transfer learning.
8. Implemented EfficientNetB0 for architecture comparison.
9. Performed fine-tuning of pretrained models.
10. Corrected an augmentation input-range issue in the MobileNetV2 experiment.
11. Compared both transfer-learning models on the same held-out test set.
12. Evaluated performance using accuracy, precision, recall, F1-score and confusion matrices.
13. Integrated Grad-CAM explainability.
14. Developed an interactive Streamlit application.
15. Demonstrated inference using external MRI images.

---

## ⚠️ 14. Limitations

- The model is trained and evaluated on a specific public dataset.
- External images may differ in scanner type, MRI sequence, contrast, resolution and acquisition conditions.
- External-image predictions have not been independently validated against expert annotations.
- Confidence scores are not calibrated probabilities of clinical correctness.
- Grad-CAM provides approximate visual explanations, not exact tumor boundaries.
- The system has not undergone clinical validation.
- The model should not be used for diagnosis, treatment selection or clinical decision-making.

---

## 🔮 15. Future Enhancements

Potential improvements include:

- Evaluation on independent hospital datasets.
- External validation using expert-verified labels.
- MRI tumor segmentation.
- Integration of more advanced explainability techniques.
- Model calibration and uncertainty estimation.
- Robustness testing across MRI modalities and scanners.
- Improved deployment and user-interface design.
- More extensive model comparison.

---

## 📚 16. References

### Dataset

BRISC 2025 Brain MRI Dataset:

https://www.kaggle.com/datasets/briscdataset/brisc2025

### Research paper

Fateh et al. (2025). BRISC 2025.

https://arxiv.org/abs/2506.14318

### Model architectures

- MobileNetV2: https://arxiv.org/abs/1801.04381
- EfficientNet: https://arxiv.org/abs/1905.11946

### Explainable AI

Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.

https://arxiv.org/abs/1610.02391

### Framework documentation

- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- Streamlit: https://docs.streamlit.io/

---

## 👨‍💻 17. Author

**Individual Academic Capstone Project**

**Project Title:** AI-Based Brain Tumor Classification Using MRI with Explainable Deep Learning

**GitHub Repository:**  
https://github.com/chaitanyaReddy4/Brain-Tumor-Classification-MRI

---

<p align="center">

**Built with Python, Deep Learning and Explainable AI.**

</p>
