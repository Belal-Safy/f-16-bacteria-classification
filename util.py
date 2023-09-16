import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def get_class_name(prediction):
    label_mapping = {
        'Ascariasis': 0,
        'Babesia': 1,
        'Capillaria p': 2,
        'Enterobius v': 3,
        'Epidermophyton floccosum': 4,
        'Fasciolopsis buski': 5,
        'Hookworm egg': 6,
        'Hymenolepis diminuta': 7,
        'Hymenolepis nana': 8,
        'Leishmania': 9,
        'Opisthorchis viverrine': 10,
        'Paragonimus spp': 11,
        'T. rubrum': 12,
        'Taenia spp': 13,
        'Trichuris trichiura': 14
    }

    prediction = np.array(prediction)
    predicted_class_index = np.argmax(prediction)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_class_name = reverse_label_mapping[predicted_class_index]

    return predicted_class_name


def classify(image, model):
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = np.array(model.predict(data))
    class_name = get_class_name(prediction)
    return class_name
