import streamlit as st
from keras.models import load_model
from PIL import Image

from util import classify
# Load classifier
model = load_model('./final_model.h5')

# Set title
st.title('Bacteria classification')

# Set header
st.header('Please upload an image for classification')

# Upload file
file = st.file_uploader('upload image', type=['jpeg', 'jpg', 'png'])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Create a button for classification
    if st.button("Start Classification"):
        with st.spinner("Classifying..."):
            # Call the classify function to get the class name
            class_name = classify(image, model)
            st.success("Classification complete!")

        # Display the class name
        st.write("## Predicted class: {}".format(class_name))