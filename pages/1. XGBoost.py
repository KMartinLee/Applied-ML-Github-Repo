import streamlit as st
from PIL import Image
import os

# Define the relative path to the image
image_path = os.path.join("images", "chart1.png")

# Open and display the image
image = Image.open(image_path)
st.image(image, caption="My Chart", use_column_width=True)