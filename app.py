import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizerFast
from model_def import NewsClassifier

REPO_ID = "Theara2/cambodian-news-roberta"
LABELS = ["label0", "label1", "label2", "label3", "label4"]  # replace with your real class names, in order

@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="roberta_best.pt")
    model = NewsClassifier(num_labels=len(LABELS))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    return model, tokenizer

model, tokenizer = load_model()

st.title("Cambodian News Classifier")
text = st.text_area("Enter news text:")

if st.button("Classify") and text.strip():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = torch.argmax(probs).item()
    st.subheader(f"Prediction: {LABELS[pred_idx]}")
    st.write({LABELS[i]: f"{probs[i].item():.2%}" for i in range(len(LABELS))})
