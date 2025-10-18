import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
from streamlit_drawable_canvas import st_canvas

# Load trained model
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Page configuration
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🧠", layout="centered")

# Sidebar
with st.sidebar:
    st.title("🧠 MNIST Classifier")
    st.markdown("""
    Choose one of the options below:
    - ✏️ Draw a digit (0–9)
    - 📁 Upload a 28x28 grayscale image

    The model will predict the digit using a trained CNN.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

    with st.expander("Model Info"):
        st.text("Model: CNN trained on MNIST")
        st.text("Input shape: (28, 28, 1)")
        st.text("Output: 10-class softmax")

# Main title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>MNIST Digit Classifier</h1>", unsafe_allow_html=True)

# Tabs for input methods
tab1, tab2 = st.tabs(["✏️ Draw Digit", "📁 Upload Image"])

def preprocess_image(img):
    img = img.resize((28, 28)).convert('L')
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def predict_digit(img_array):
    if np.sum(img_array) < 0.1:
        st.warning("Please draw a digit or upload a clearer image.")
        return None, None, None
    prediction = model.predict(img_array)
    return np.argmax(prediction), np.max(prediction), prediction

with tab1:
    st.markdown("### Draw a digit below:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img = ImageOps.invert(img).resize((28, 28)).filter(ImageFilter.GaussianBlur(radius=1))
        img_array = preprocess_image(img)

        st.image(img, caption="Preprocessed Digit", width=150)
        st.write("Input shape:", img_array.shape)

        predicted_label, confidence, raw_scores = predict_digit(img_array)

        if predicted_label is not None:
            st.markdown(f"<h3 style='color: #228B22;'>Predicted Digit: {predicted_label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)

            if confidence < 0.5:
                st.warning("Low confidence prediction. Try drawing more clearly.")

            if st.checkbox("Show raw prediction scores"):
                st.write({str(i): float(raw_scores[0][i]) for i in range(10)})

with tab2:
    st.markdown("### Upload a digit image:")
    uploaded_file = st.file_uploader("Upload a PNG/JPG image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        invert = st.checkbox("Invert image colors", value=True)
        if invert:
            image = ImageOps.invert(image)
        image = image.resize((28, 28)).filter(ImageFilter.GaussianBlur(radius=1))
        img_array = preprocess_image(image)

        st.image(image, caption="Preprocessed Image", width=150)
        st.write("Input shape:", img_array.shape)

        predicted_label, confidence, raw_scores = predict_digit(img_array)

        if predicted_label is not None:
            st.markdown(f"<h3 style='color: #228B22;'>Predicted Digit: {predicted_label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)

            if confidence < 0.5:
                st.warning("Low confidence prediction. Try uploading a clearer image.")

            if st.checkbox("Show raw prediction scores"):
                st.write({str(i): float(raw_scores[0][i]) for i in range(10)})

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2025 MNIST Classifier App</p>", unsafe_allow_html=True)
