**Online Payment Fraud Detection (Gradio + XGBoost)**

A production-style fraud detection demo you can run in the browser. Upload a PaySim-style CSV or generate a synthetic but realistic dataset, train on CPU, view PR-AUC / ROC-AUC, confusion matrix, feature importances, and download predictions.

**Demo**: https://huggingface.co/spaces/aakash-malhan/online-payment-fraud-detection

<img width="572" height="219" alt="Screenshot 2025-10-21 215820" src="https://github.com/user-attachments/assets/71501753-a9e0-4f22-8658-502554ff3c70" />


**Problem & Objective**

Flag likely fraudulent transactions in imbalanced payment data (fraud ≪ 1%) while allowing the user to tune alert volume to business needs.

**Business impact & takeaways**

Imbalanced-aware modeling with tunable alert rate aligns to fraud-ops capacity.

Exportable predictions plug into manual review/queueing with minimal effort.

Feature importance gives quick signals for rule refinement.
