# 🛡️ 2-Stage RoBERTa Cyberbullying Detection App

A 2-stage deep learning web application built with Streamlit and PyTorch to detect and categorize cyberbullying text content in real-time.

🔗 **Live Demo Application:** [[Insert Your Streamlit App Link Here](https://cyberbullying-alarm.streamlit.app/)]

---

## 📌 Project Overview
Cyberbullying on social media takes various forms, ranging from subtle toxicity to targeted harassment based on identity. This project implements a **hierarchical 2-stage classification approach** using fine-tuned **RoBERTa** models:

1. **Stage 1 (Binary Classification):** Evaluates whether the input text contains cyberbullying content or not.
2. **Stage 2 (Multi-class Classification):** If cyberbullying is detected, the second model categorizes the nature of the content into specific classes (`age`, `ethnicity`, `gender`, `religion`, or `other_cyberbullying`).

---

## 🚀 Key Features
* **Two-Stage NLP Pipeline:** Efficient model triaging to minimize unnecessary secondary compute.
* **Confidence Percentage:** Real-time probability scores powered by PyTorch Softmax logit transformation.
* **Hugging Face Integration:** Models hosted on Hugging Face Hub to ensure fast deployment without hitting GitHub storage limits.
* **User-Friendly UI:** Interactive layout built with Streamlit for seamless text testing.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 
* **Framework:** Streamlit
* **Deep Learning Framework:** PyTorch
* **Transformers Library:** Hugging Face `transformers`
* **Model Hosting:** Hugging Face Hub (`wandaestu/stage1-cyberbullying` & `wandaestu/stage2-cyberbullying`)

---

## 📂 Repository Structure
```text
CYBERBULLYING-DETECTOR/
├── app.py              # Main Streamlit web application code
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
