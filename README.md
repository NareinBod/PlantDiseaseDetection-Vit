# 🌿 Fine-Tuning a Vision Transformer (ViT) for Plant Disease Detection

## 🧠 Overview
This project focuses on building an **intelligent image classification system** that accurately identifies **plant diseases from leaf images** using the **Vision Transformer (ViT)** architecture — an advanced deep learning model that adapts Transformer concepts for visual tasks.

By **fine-tuning a pre-trained ViT** (such as one trained on ImageNet) on a **custom dataset of healthy and diseased leaves**, the model learns to detect subtle differences in **color, texture, and leaf structure**, achieving high diagnostic accuracy.  

This project showcases how **large-scale models** can be efficiently adapted to specialized **agricultural applications**, contributing to **sustainable farming**, **reduced crop loss**, and **AI-driven disease detection**.

---

## 🚀 Key Features
- Fine-tuning of a **Vision Transformer (ViT)** using **transfer learning**
- **Custom dataset** of segmented leaf images (healthy vs diseased)
- **Data preprocessing** and **augmentation** for better generalization
- **Performance evaluation** with confusion matrices and class-level metrics
- **Explainable AI (XAI)** techniques to interpret model predictions
- End-to-end **AI pipeline** from dataset loading to visualization

---

## 🧩 Project Workflow
1. **Data Preparation**
   - Collected and organized images into labeled categories.
   - Applied **data augmentation** (rotation, flipping, brightness adjustments).

2. **Model Fine-Tuning**
   - Loaded a **pre-trained ViT model** from Hugging Face Transformers.
   - Replaced the classification head to fit the number of plant disease classes.
   - Trained using **PyTorch Lightning / PyTorch** for efficient optimization.

3. **Evaluation**
   - Measured accuracy, precision, recall, and F1-score.
   - Visualized results with **confusion matrices** and **class-wise performance charts**.

4. **Interpretability**
   - Used visualization techniques (e.g., Grad-CAM) to highlight regions influencing model decisions.

---
