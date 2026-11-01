# Flipkart Order Intelligence & Support Assistant

An end-to-end machine-learning and applied-AI project exploring three e-commerce workflows: order-return risk prediction, product-image classification, and policy-grounded customer support.

**Author:** Mohit Mishra

## Project Modules

### 1. Return Risk Prediction

A Random Forest model is used to estimate whether an order may be returned. The project description identifies order and customer attributes such as item category, customer history, discounts, and payment method as model inputs.

Main script: `part1_return_risk.py`

### 2. Product Image Classification

A ResNet-18 model is fine-tuned with PyTorch to classify product images.

Main script: `part2_image_classifier.py`

### 3. Policy-Grounded RAG Support Agent

A retrieval-augmented support agent uses FAISS vector search and sentence-transformers to retrieve relevant store-policy information for customer questions.

Main script: `part3_agent.py`

## Repository Contents

The project includes scripts for generating sample order data, model code, data/assets, saved model files, and support-agent transcripts. The project files are organized under `vs demo/Flipkart-Order-Intelligence-Support-Assistant/`.

- `generate_orders.py` — generates a mock order dataset
- `orders_dataset.csv` — synthesized order data (documented as 6,000 orders)
- `requirements.txt` — Python dependencies
- `data/` — sample product images/data
- `models/` — saved model artifacts
- `transcripts/` — agent conversation logs

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/mohitmishra5782/Flipkart-Order-Intelligence-Support-Assistant.git
   cd Flipkart-Order-Intelligence-Support-Assistant
   ```
2. Move into the project directory:
   ```bash
   cd "vs demo/Flipkart-Order-Intelligence-Support-Assistant"
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```
   Windows:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Run the relevant module after reviewing its configuration and required input files:
```bash
python generate_orders.py
python part1_return_risk.py
python part2_image_classifier.py
python part3_agent.py
```

Some modules may require model artifacts, datasets, or configuration to be generated or placed first. Check each script before running it.

## Reproducibility & Responsible Use

- Treat generated order records as synthetic/demo data unless a script states otherwise.
- Do not use model predictions as the sole basis for real customer decisions.
- Document the dataset split, preprocessing, evaluation metrics, and model versions when reporting results.
- Keep credentials, API keys, and private customer information out of the repository.

## Suggested Next Improvements

- Add screenshots or a short demo video.
- Record actual precision, recall, F1-score, and confusion matrices for the return-risk model.
- Document image classes and per-class metrics for the classifier.
- Explain policy-document ingestion and retrieval evaluation for the RAG agent.
- Add license and data-source notes.

## Author

Mohit Mishra — https://github.com/mohitmishra5782
