import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Weather Classification Model", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #e8f0fe;
        color: black;
    }
    .stTextArea>div>textarea {
        background-color: #e8f0fe;
        color: black;
    }
    .stFileUploader>div>div>input[type="file"] {
        background-color: #e8f0fe;
        color: black;
    }
    .stDataFrame {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model_cnn.hdf5')
    return model

model = load_model()

st.write("""
# Weather Classification Model
""")
st.text("Paulo Abregunda")
st.text("Nikko Denila")

st.write("This web app classifies weather conditions in uploaded images. Please follow the steps:")
st.markdown("""
1. Upload an image using the 'Browse Files' button.
2. Wait for the model to process the image.
3. View the prediction and confidence score.
""")

image = Image.open('rainorshine.jpg')
st.image(image, caption="Sample Image", use_column_width=True)

with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.header("Model Outputs")
        st.info("Rain and Shine")

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

if 'clear_output' not in st.session_state:
    st.session_state.clear_output = False

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    if file.type not in ['image/jpeg', 'image/png']:
        st.error("Unsupported file type. Please upload a .jpg or .png file.")
    else:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        prediction = import_and_predict(image, model)
        class_names = ['Shine', 'Rain']
        max_prob = np.max(prediction)
        prediction_label = class_names[np.argmax(prediction)]
        
        st.success(f"Prediction: {prediction_label}")
        st.write(f"Confidence Score: {max_prob:.2%}")
        
        # Confusion Matrix
        true_labels = [prediction_label]
        predicted_labels = [prediction_label]
        
        if true_labels and predicted_labels:
            cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
            st.write("Confusion Matrix")
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

st.info("[Github Link](https://github.com/paodgreat/DeploymentEmtech3)")

comment_tab = st.container()
with comment_tab:
    st.header("User Comments and Feedback")
    st.write("Please leave your comments and feedback about the Weather Classification Model.")
    
    user_name = st.text_input("Your Name:")
    user_email = st.text_input("Your Email:")
    user_comment = st.text_area("Comments:")

    if 'feedback_list' not in st.session_state:
        st.session_state.feedback_list = []  # Initialize an empty list to store feedback data

    if st.button("Submit Feedback"):
        # Store feedback data in a list
        feedback_data = {
            "Name": user_name,
            "Email": user_email,
            "Comment": user_comment
        }

        st.session_state.feedback_list.append(feedback_data)

        st.success("Thank you for your feedback! We appreciate your input.")

    if st.session_state.feedback_list:
        st.header("Feedback History")
        feedback_df = pd.DataFrame(st.session_state.feedback_list)  # Create a DataFrame from the feedback list
        st.dataframe(feedback_df)  # Display the feedback data in a DataFrame format
    else:
        st.info("No feedback submitted yet.")
