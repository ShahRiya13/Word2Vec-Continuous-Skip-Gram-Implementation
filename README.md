# Word2Vec-Continuous-Skip-Gram-Implementation
**Overview**

This project implements the Word2Vec Continuous Skip-Gram model from scratch using PyTorch. The model is trained on WikiText-2 and WikiText-103 (raw-v1) datasets.

All components — including data preprocessing, dataset creation, training, and inference — are custom-built without using torchtext.datasets.

**Components**
**1️ Data Loader**

Downloaded and loaded raw WikiText datasets

Used pyarrow and pandas to read .parquet files:

pip install pyarrow
import pandas as pd
df = pd.read_parquet("file.parquet")


**Tokenized raw text**

Built vocabulary and word-to-index mappings

Generated (center word, context word) pairs using sliding window

Implemented custom torch.utils.data.Dataset

Created training and testing DataLoaders

**2️ Model Architecture**

The Skip-Gram model includes:

Input embedding layer

Output embedding layer

Dot product between embeddings

Softmax for probability distribution

The model learns word representations by predicting surrounding context words given a center word.

**3️ Training**

Mini-batch training strategy

Cross-entropy loss

Optimizer: Adam

Separate training and testing loaders

Periodic loss monitoring

**4️ Inference**

After training, the model:

Extracts learned word embeddings

Computes cosine similarity

Returns Top-K (e.g., K=10) most similar words for a given input word

**Example:**

Input: "king"
Top 10 similar words:
queen, prince, monarch, throne, ...


**Key Learning Objectives**

Implement Skip-Gram from scratch

Build custom PyTorch Dataset and DataLoader

Train embeddings using mini-batch optimization

Perform similarity-based inference

Work with large-scale text corpora
