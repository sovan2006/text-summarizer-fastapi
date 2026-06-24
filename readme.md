# 📝 AI Text Summarizer using FastAPI & Hugging Face Transformers

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)]()

## 🚀 Live Demo

🔗 **Application:** Add Railway Link Here

📚 **API Documentation:** Add Railway Link Here/docs

💻 **GitHub Repository:** https://github.com/sovan2006/text-summarizer-fastapi

---

## 📌 Project Overview

The AI Text Summarizer is a Natural Language Processing (NLP) application that generates concise and meaningful summaries from lengthy text inputs.

The project leverages a fine-tuned T5 Transformer model trained on the SAMSum dataset and provides an interactive web interface powered by FastAPI.

This application demonstrates practical implementation of:

* Natural Language Processing (NLP)
* Transformer-based Deep Learning Models
* Model Deployment
* REST API Development
* Production-ready FastAPI Applications

---

## 🎯 Problem Statement

Reading long documents, articles, and conversations can be time-consuming.

This project aims to automatically generate short and meaningful summaries while preserving the key information from the original content.

---

## ✨ Features

✅ AI-powered Text Summarization

✅ Fine-tuned T5 Transformer Model

✅ FastAPI Backend

✅ Interactive Web Interface

✅ Real-time Summary Generation

✅ REST API Support

✅ Deployable on Railway / Render / Hugging Face Spaces

---

## 🏗️ System Architecture

User Input
↓
FastAPI Backend
↓
Text Preprocessing
↓
Fine-Tuned T5 Model
↓
Summary Generation
↓
Frontend Display

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning

* PyTorch
* Hugging Face Transformers
* T5-Small

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* HTML
* CSS
* JavaScript
* Jinja2 Templates

### Deployment

* Railway
* GitHub

---

## 🤖 Model Information

### Base Model

T5-Small

### Fine-Tuned On

SAMSum Dialogue Summarization Dataset

### Task

Abstractive Text Summarization

### Framework

Transformers + PyTorch

---

## 📂 Project Structure

text-summarizer-fastapi/

├── app.py

├── index.html

├── requirements.txt

├── Procfile

├── README.md

├── screenshots/

└── model/

---

## 📷 Screenshots

### Home Page



### Generated Summary



## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sovan2006/text-summarizer-fastapi.git
cd text-summarizer-fastapi
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn app:app --reload
```

Open:

```bash
http://127.0.0.1:8000
```

---

## 📡 API Endpoint

### POST /summarize

Request:

```json
{
    "dialogue": "Your input text here"
}
```

Response:

```json
{
    "summary": "Generated summary"
}
```

---

## 📈 Future Improvements

* PDF Summarization
* Multi-language Summarization
* Document Upload Support
* User Authentication
* Docker Deployment
* Cloud-based Model Serving
* LLM Integration

---

## 👨‍💻 Author

### Sovan Barik

B.Tech Artificial Intelligence & Machine Learning

Machine Learning | NLP | Generative AI | FastAPI

📧 Email: sovanbarik07@gmail.com

🔗 GitHub: https://github.com/sovan2006

🔗 LinkedIn: www.linkedin.com/in/sovan-barik-711bba326


