import streamlit as st
from PIL import Image
import os

st.title("LSTM Model Visualisations")

st.markdown("""LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to learn from sequential data by capturing long-range dependencies. Unlike traditional RNNs, LSTMs use gating mechanisms to control the flow of information, allowing them to retain relevant information over long time intervals and forget irrelevant data.

LSTMs are particularly effective for tasks involving time series, natural language, and any data with a temporal structure. They are commonly used for forecasting, text classification, and sentiment analysis, among other applications.

**Known for:**
- Handling sequential and time-dependent data effectively
- Capturing long-term dependencies through gated memory cells
- High performance in NLP, forecasting, and other temporal tasks
""")
# Accuracy of LSTM
accuracy_path = os.path.join("images", "Accuracy_of_LSTM.png")
accuracy_img = Image.open(accuracy_path)
st.image(accuracy_img, caption="Accuracy of LSTM", use_container_width=True)

st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Performance Heat Map of LSTM
heatmap_path = os.path.join("images", "Performance_heat_map_LSTM.png")
heatmap_img = Image.open(heatmap_path)
st.image(heatmap_img, caption="Performance Heat Map of LSTM", use_container_width=True)

