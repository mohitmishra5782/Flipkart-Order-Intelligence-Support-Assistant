# Flipkart Order Intelligence & Support Agent

Hey there! This is my submission for the Flipkart Order Intelligence project. It’s an end-to-end Machine Learning and AI support agent pipeline built to handle return risk evaluation, product image classification, and automated customer support.

What's Inside?

I broke the solution down into three main modules:

1. Part 1: Return Risk Prediction (`part1_return_risk.py`)**  
   Built a Random Forest model on e-commerce order data to figure out which orders are likely to get returned. It checks things like item category, customer history, discounts, and payment methods.

2. Part 2: Product Image Classifier (`part2_image_classifier.py`)**  
   Fine-tuned a pre-trained ResNet-18 model using PyTorch to automatically categorize product images coming through the system.

3. Part 3: Policy-Grounded RAG Support Agent (`part3_agent.py`)**  
   Created a customer support bot using FAISS vector search and `sentence-transformers`. It answers customer queries strictly based on store policies so it doesn't hallucinate random answers.

Repository Setup

flipkart-order-intelligence/
├── data/                  # Sample product images
├── models/                # Saved model weights (.pkl & .pt files)
├── transcripts/           # Saved agent chat logs (.json)
├── generate_orders.py     # Script to build the mock order dataset
├── part1_return_risk.py   # Training script for return risk model
├── part2_image_classifier.py # Fine-tuning script for ResNet-18
├── part3_agent.py         # Support agent execution script
├── orders_dataset.csv     # Synthesized dataset (6,000 orders)
└── requirements.txt       # Necessary Python packages
