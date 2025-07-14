import streamlit as st
from PIL import Image
import os

st.title("Random Forest Model Visualisations")

st.markdown("""Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting. It is widely used for both classification and regression tasks. Each tree in the forest is trained on a random subset of the data and selects a random subset of features at each split. This randomness helps ensure diversity among trees, making the overall model more robust and less prone to noise.

Random Forests aggregate the results of all individual trees (via majority vote for classification or average for regression), leading to more stable and generalisable predictions compared to a single decision tree.

**Known for:**
- Strong performance on structured/tabular data
- Built-in feature importance evaluation
- Robustness to overfitting and noise through ensembling
- Minimal need for feature scaling or preprocessing
""")
# Accuracy of Random Forest
accuracy_path = os.path.join("images", "Accuracy_of_RF.png")
accuracy_img = Image.open(accuracy_path)
st.image(accuracy_img, caption="Accuracy of Random Forest", use_container_width=True)

st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Performance Heat Map of Random Forest
heatmap_path = os.path.join("images", "Performance_heat_map_RF.png")
heatmap_img = Image.open(heatmap_path)
st.image(heatmap_img, caption="Performance Heat Map of Random Forest", use_container_width=True)

