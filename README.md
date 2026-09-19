# Bangla Mental Health Classifier

**Mental Health Classification in Bangla Social Media Text with SHAP Explainability**

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://bangla-mental-health-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Encoder-BanglaBERT-yellow)](https://huggingface.co/csebuetnlp/banglabert)

**Live demo:** [https://bangla-mental-health-app.streamlit.app](https://bangla-mental-health-app.streamlit.app)

**Author:** [Saim Hosen Hridoy](https://github.com/Saimhosenhridoy)

---

## Abstract

This project is a web application that classifies **Bengali (Bangla) social-media style text** into three mental-health related categories and explains the decision with **SHAP**.

The classifier is built on **BanglaBERT** (`csebuetnlp/banglabert`) with a hybrid pooling head (`HybridBanglaBERTClassifier`). Users type or paste Bangla text, receive a predicted class with confidence and class probabilities, then see a word-level SHAP visualization for **Depressive**, **Non_depressive**, and **Positive**.

The app is intended for **research and awareness only**. It is **not a clinical diagnosis**, not a screening tool for emergency decisions, and not a substitute for a qualified mental health professional.

---

## Live application

| Item | Link |
|---|---|
| Streamlit app | [https://bangla-mental-health-app.streamlit.app](https://bangla-mental-health-app.streamlit.app) |
| Source code | [https://github.com/Saimhosenhridoy/bangla-mental-health-app](https://github.com/Saimhosenhridoy/bangla-mental-health-app) |
| Author | [https://github.com/Saimhosenhridoy](https://github.com/Saimhosenhridoy) |

If the Cloud app has been idle, the first load can take longer (cold start). After the container is awake, prediction is faster than SHAP.

---

## Problem

Most mental-health NLP tools are English-first. Bangla social-media posts are a common place where people write about mood, stress, and hopelessness, but there are few public, explainable demos for this setting.

This app focuses on:

1. Three-way classification of Bangla text  
2. Transparent, word-level explanation (SHAP)  
3. A simple web UI that non-engineers can try  
4. A basic crisis-language warning  

---

## Output classes

| ID | Label | Display name | Meaning |
|---:|---|---|---|
| 0 | `Depressive` | Depressive | The text shows linguistic indicators associated with sadness, hopelessness, or depressive thoughts. |
| 1 | `Non_depressive` | Non_depressive | The text does not show clear linguistic indicators of depression. |
| 2 | `Positive` | Positive | The text shows linguistic indicators of an optimistic or positive mental state. |

Predictions also include:

- **confidence** of the top class  
- **probability** of all three classes  
- a short support-style message for the predicted class  

---

## Features

- Bangla input with `Hind Siliguri` font support  
- One-click example sentences for each class  
- Hybrid BanglaBERT classifier  
- Probability bar chart (Altair)  
- SHAP text plot for all three classes  
- Text cleaning (invisible characters, extra whitespace)  
- Crisis-pattern warning (self-harm / suicide related phrases)  
- Research disclaimer  
- Streamlit Cloud deployment  

---

## How a request works

```text
User text
    → clean_text()
    → length check
    → contains_crisis_language()   (warning only; prediction still runs)
    → predict_text()
         tokenizer (max_length from settings)
         HybridBanglaBERTClassifier
         softmax over 3 classes
    → probability chart
    → explain_text_interactive()
         shorten text for speed
         SHAP partition explainer
         HTML text plot for 3 classes
         