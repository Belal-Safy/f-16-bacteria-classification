import streamlit as st
# from keras.models import load_model
from PIL import Image

from util import classify

# set title
st.title('Pneumonia classification')

# set header
st.header('Please upload a Bacteria image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
# model = load_model('./model/pneumonia_classifier.h5')
model = "sdf"

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name = classify(image, model)

    # write classification
    st.write("## {}".format(class_name))
