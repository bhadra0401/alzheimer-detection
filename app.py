import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Alzheimer AI",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
}

.subtitle {
    font-size: 18px;
    color: #666666;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #e8f4ff;
    border-left: 6px solid #1f77b4;
}

.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/mobilenet_alzheimer.keras"
    )

model = load_model()

classes = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<div class='title'> Alzheimer Disease Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Transfer Learning + Deep Learning + Ensemble Learning</div>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.header(" Project Information")

    st.success("Model Accuracy: 97.7%")

    st.write("""
    ### Technologies Used
    
    - MobileNetV2
    - Transfer Learning
    - TensorFlow
    - Scikit-Learn
    - Voting Ensemble
    - Streamlit
    """)

    st.write("---")

    st.info("""
    Alzheimer's Classes:
    
    • Mild Demented
    
    • Moderate Demented
    
    • Non Demented
    
    • Very Mild Demented
    """)

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI Scan",
            use_container_width=True
        )

with col2:

    if uploaded_file:

        img = image.resize((224,224))

        img = np.array(img)

        img = img / 255.0

        img = np.expand_dims(img, axis=0)

        with st.spinner("Analyzing MRI Scan..."):

            prediction = model.predict(img, verbose=0)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.markdown(
            f"""
            <div class='result-box'>
            <h2>Prediction Result</h2>
            <h3>{classes[predicted_class]}</h3>
            <h4>Confidence: {confidence:.2f}%</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.subheader("Confidence Scores")

        for i, label in enumerate(classes):

            prob = prediction[0][i]

            st.write(label)

            st.progress(float(prob))

            st.write(f"{prob*100:.2f}%")

# -----------------------------
# DETAILED ANALYSIS
# -----------------------------
if uploaded_file:

    st.divider()

    st.subheader(" Prediction Analysis")

    df = pd.DataFrame({
        "Class": classes,
        "Probability (%)":
        [round(x*100,2) for x in prediction[0]]
    })

    st.dataframe(
        df,
        use_container_width=True
    )

    st.bar_chart(
        df.set_index("Class")
    )

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Developed using Transfer Learning Framework for Early Alzheimer's Prediction"
)