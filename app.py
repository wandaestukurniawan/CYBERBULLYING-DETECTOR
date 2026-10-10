import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_STAGE1 = "wandaestu/stage1-cyberbullying"
MODEL_STAGE2 = "wandaestu/stage2-cyberbullying"

st.set_page_config(page_title="Cyberbullying Detection", layout="centered")

st.title("🛡️ Cyberbullying Detection App")
st.write("Detector App by Using Classification Model 2-Stage RoBERTa.")

# Load Model & Tokenizer
@st.cache_resource
def load_models():
    tok = AutoTokenizer.from_pretrained(MODEL_STAGE1)
    model_s1 = AutoModelForSequenceClassification.from_pretrained(MODEL_STAGE1)
    model_s2 = AutoModelForSequenceClassification.from_pretrained(MODEL_STAGE2)
    return tok, model_s1, model_s2

tokenizer, model_s1, model_s2 = load_models()

# Mapping Stage 1 & 2
stage1_labels = ["not_cyberbullying", "cyberbullying"] # Sesuai target_names stage 1
stage2_labels = ["age", "ethnicity", "gender", "other_cyberbullying", "religion"] # Sesuaikan dengan le_stage2.classes_

# Input Tweets
user_input = st.text_area("Typing tweet text:", "")

if st.button("Text analysis"):
    if user_input.strip() == "":
        st.warning("Input text first!")
    else:
        # Preprocessing & Inference Stage 1
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        
        with torch.no_grad():
            out_s1 = model_s1(**inputs)
            pred_s1_idx = torch.argmax(out_s1.logits, dim=1).item()
            pred_s1_label = stage1_labels[pred_s1_idx]

        st.subheader("Analysis result:")
        
        if pred_s1_label == "not_cyberbullying":
            st.success("✅ **Result of Stage 1:** Not Cyberbullying")
        else:
            st.error("⚠️ **Result of Stage 1:** Cyberbullying!")
            
            # Predict Stage 2 (Bullying Category)
            with torch.no_grad():
                out_s2 = model_s2(**inputs)
                pred_s2_idx = torch.argmax(out_s2.logits, dim=1).item()
                pred_s2_label = stage2_labels[pred_s2_idx]
            
            st.warning(f"🏷️ **Result of Stage 2 (Category):** {pred_s2_label.capitalize()}")
