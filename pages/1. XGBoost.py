import streamlit as st
from PIL import Image
import os

st.title("XGBoost Model Visualisations")

st.markdown("""
    What is XGBoost?
    XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm based on decision trees. It is widely used for classification and regression tasks due to its speed and performance. XGBoost optimises the gradient boosting framework, making it efficient and effective for large datasets.
    It uses a gradient descent algorithm to minimise the loss function, allowing it to learn complex patterns in the data. XGBoost is particularly known for its ability to handle missing values and its robustness against overfitting.
    
    Known for:
    - High accuracy
    - Fast training
    - Robustness to overfitting (via regularisation)

""")            
# Accuracy of XGBoost
accuracy_path = os.path.join("images", "Accuracy_of_XGBoost.png")
accuracy_img = Image.open(accuracy_path)
st.image(accuracy_img, caption="Accuracy of XGBoost", use_container_width=True)

st.markdown("", unsafe_allow_html=True)

st.markdown("""
            

""")


st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Performance Heat Map of XGBoost
heatmap_path = os.path.join("images", "Performance_heat_map_XGB.png")
heatmap_img = Image.open(heatmap_path)
st.image(heatmap_img, caption="Performance Heat Map of XGBoost", use_container_width=True)