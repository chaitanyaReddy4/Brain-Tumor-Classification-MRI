
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

PROJECT_PATH = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_PATH / "models" / "brain_tumor_mobilenetv2_final.keras"

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


def preprocess_image(image):
    image_array = np.array(image.convert("RGB"))
    tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    tensor = tf.image.resize_with_pad(tensor, 224, 224)
    tensor = tensor / 255.0
    return tensor


def generate_gradcam(model, image):
    backbone = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = backbone.get_layer("out_relu")

    grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output]
    )

    augmentation = model.get_layer("corrected_augmentation")
    rescaling = model.get_layer("rescaling")
    gap = model.get_layer("global_average_pooling2d_1")
    dropout1 = model.get_layer("dropout_1")
    dense1 = model.get_layer("dense_2")
    dropout2 = model.get_layer("dropout_2")
    classifier = model.get_layer("dense_3")

    image_batch = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        x = augmentation(image_batch, training=False)
        x = rescaling(x)

        conv_outputs, backbone_output = grad_model(
            x, training=False
        )

        x = gap(backbone_output)
        x = dropout1(x, training=False)
        x = dense1(x)
        x = dropout2(x, training=False)
        predictions = classifier(x)

        predicted_class = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(class_score, conv_outputs)

    pooled_gradients = tf.reduce_mean(
        gradients, axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (224, 224)
    )

    return heatmap.numpy().squeeze(), predictions.numpy()[0]


st.title("🧠 AI-Based Brain Tumor Classification")
st.write("MRI Image Analysis using MobileNetV2 and Explainable AI")

st.info(
    "Upload a brain MRI image for experimental four-class "
    "classification and Grad-CAM visualization."
)

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    if st.button("Analyze MRI", type="primary"):

        with st.spinner("Analyzing MRI and generating Grad-CAM..."):

            image_tensor = preprocess_image(image)

            heatmap, predictions = generate_gradcam(
                model, image_tensor
            )

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASSES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            # Prepare image for visualization
            display_tensor = tf.image.resize_with_pad(
                tf.convert_to_tensor(
                    np.array(image), dtype=tf.float32
                ),
                224, 224
            )

            display_image = np.clip(
                display_tensor.numpy() / 255.0, 0, 1
            )

            # Create colored Grad-CAM overlay
            color_map = plt.get_cmap("jet")
            colored_heatmap = color_map(heatmap)[..., :3]

            overlay = (
                0.6 * display_image +
                0.4 * colored_heatmap
            )

            overlay = np.clip(overlay, 0, 1)

        with col2:
            st.subheader("Prediction Result")
            st.success(f"Predicted Class: {predicted_class}")
            st.metric("Model Confidence Score", f"{confidence:.2f}%")

        st.divider()

        st.subheader("Class Probabilities")

        for class_name, probability in zip(CLASSES, predictions):
            st.write(f"{class_name}: {probability * 100:.2f}%")
            st.progress(float(probability))

        st.divider()

        st.header("Explainable AI - Grad-CAM")

        grad_col1, grad_col2 = st.columns(2)

        with grad_col1:
            st.subheader("Original MRI")
            st.image(display_image, use_container_width=True)

        with grad_col2:
            st.subheader("Grad-CAM Heatmap Overlay")
            st.image(overlay, use_container_width=True)

        st.caption(
            "Grad-CAM highlights image regions that influenced "
            "the model's prediction. It does not identify exact "
            "tumor boundaries or establish a medical diagnosis."
        )

st.divider()

st.warning(
    "Research and educational prototype only. "
    "This model is not a medical diagnostic tool. "
    "Do not use its predictions for clinical decisions."
)
