import streamlit as st
from PIL import Image
import os

st.title("MLP Model Visualisations")

# Accuracy of MLP
accuracy_path = os.path.join("images", "Accuracy_of_MLP.png")
accuracy_img = Image.open(accuracy_path)
st.image(accuracy_img, caption="Accuracy of MLP", use_container_width=True)

st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Performance Heat Map of MLP
heatmap_path = os.path.join("images", "Performance_heat_map_MLP.png")
heatmap_img = Image.open(heatmap_path)
st.image(heatmap_img, caption="Performance Heat Map of MLP", use_container_width=True)

