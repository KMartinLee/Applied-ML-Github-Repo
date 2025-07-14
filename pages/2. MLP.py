import streamlit as st
from PIL import Image
import os

st.title("MLP Model Visualisations")

st.markdown("""
MLP (Multi-Layer Perceptron) is a type of artificial neural network composed of multiple layers of interconnected neurons. It is a foundational deep learning model used for both classification and regression tasks. MLPs consist of an input layer, one or more hidden layers, and an output layer. Each neuron applies a weighted sum followed by a non-linear activation function, allowing the model to learn complex, non-linear relationships in data. MLPs are trained using backpropagation, an algorithm that minimizes the prediction error by adjusting the weights through gradient descent.

**Known for:**
- Capturing complex patterns in data  
- Flexibility with various input types (e.g., text, numerical, images)  
- Foundation for deeper architectures (used in DNNs, CNNs, etc.)
""")
# Accuracy of MLP
accuracy_path = os.path.join("images", "Accuracy_of_MLP.png")
accuracy_img = Image.open(accuracy_path)
st.image(accuracy_img, caption="Accuracy of MLP", use_container_width=True)

st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Performance Heat Map of MLP
heatmap_path = os.path.join("images", "Performance_heat_map_MLP.png")
heatmap_img = Image.open(heatmap_path)
st.image(heatmap_img, caption="Performance Heat Map of MLP", use_container_width=True)

