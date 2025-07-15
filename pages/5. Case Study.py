import streamlit as st
from PIL import Image
import os

st.title("Case Study")

st.markdown("""
1. Narrative Noise (Word Diversity) Over Time
This section explores how the diversity (entropy) of words used in tweets has changed over time, indicating shifts in narrative focus and sentiment within the crude oil market.
2. Cumulative Accounts on Oil Topics Over Time
This section examines the total number of tweets related to oil topics over time, providing insights into the evolving conversation around crude oil.
""")
# Accuracy of LSTM

accuracy_path = os.path.join("images", "1.png")
accuracy_img = Image.open(accuracy_path)
st.image(accuracy_img, caption="Cumulative Accounts on Oil topics over time", use_container_width=True)

st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Performance Heat Map of LSTM
heatmap_path = os.path.join("images", "3.png")
heatmap_img = Image.open(heatmap_path)
st.image(heatmap_img, caption="Narrative Noise (Word Diversity) Over Time", use_container_width=True)
