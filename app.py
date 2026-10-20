
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Car Classifier",
    page_icon="🚗",
    layout="centered"
)

CLASS_NAMES = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_car_model.keras",
        compile=False
    )

st.title("🚗 AI Car Classifier")
st.write("Upload a car image and let AI predict its category.")

uploaded_file = st.file_uploader(
    "Upload a car photo",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded car",
             use_container_width=True)

    if st.button("Identify Car 🚘"):
        try:
            model = load_model()
            image_resized = image.resize((224, 224))
            image_array = np.array(
                image_resized, dtype=np.float32
            )
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(
                image_array, verbose=0
            )[0]

            index = int(np.argmax(predictions))

            st.success(
                f"Predicted Car: {CLASS_NAMES[index]}"
            )
            st.metric(
                "Model confidence",
                f"{predictions[index] * 100:.2f}%"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
