# LLM Fine-Tuning for Legal Text Summarization

This project demonstrates the fine-tuning of T5 language models for legal text summarization using two different approaches:

1. Traditional fine-tuning
2. LoRA (Low-Rank Adaptation) fine-tuning

## Project Structure

- `Fine_Tuinig_LLM.ipynb`: Traditional fine-tuning approach
- `Fine_Tuning_LLM_using_LORA.ipynb`: Fine-tuning using LoRA technique

## Requirements

```bash
# PyTorch with CUDA support
torch
torchvision
torchaudio

# Core dependencies
transformers
datasets
evaluate
rouge_score
nltk
peft  # for LoRA fine-tuning

# Optional
numpy
```

## Dataset

The project uses the "Legal Text Summarization" dataset from Hugging Face:

- Dataset: `AjayMukundS/Legal_Text_Summarization-llama2`
- Source: [HuggingFace Dataset Link](https://huggingface.co/datasets/AjayMukundS/Legal_Text_Summarization-llama2)

## Models

### Base Model

- Model: `google-t5/t5-small`
- Source: [HuggingFace Model Hub](https://huggingface.co/google-t5/t5-small)

### Fine-tuned Variants

1. Traditional Fine-tuned Model

   - Saved as: `my_awesome_legal_model`
   - Training Parameters:
     - Learning rate: 2e-5
     - Batch size: 16
     - Epochs: 4
     - Weight decay: 0.01

2. LoRA Fine-tuned Model
   - Saved as: `legal-t5-lora`
   - LoRA Configuration:
     - Rank (r): 32
     - Alpha: 64
     - Target modules: query, value, and key matrices
     - Dropout: 0.1
   - Training Parameters:
     - Learning rate: 2e-5
     - Batch size: 16
     - Epochs: 4
     - Weight decay: 0.01

## Features

- Text summarization for legal documents
- ROUGE score evaluation
- Model performance comparison
- Support for both CPU and CUDA devices
- Memory optimization for large models
- Evaluation metrics:
  - ROUGE-1
  - ROUGE-2
  - ROUGE-L

## Hardware Requirements

- CUDA-capable GPU recommended
- Sufficient RAM for model training (16GB+ recommended)

## Usage

The notebooks provide step-by-step implementation of:

1. Data loading and preprocessing
2. Model initialization and configuration
3. Training and evaluation
4. Performance comparison between base and fine-tuned models
5. Example inference with sample texts

## Evaluation

Both models are evaluated using ROUGE metrics on a test subset, with comprehensive performance comparisons between:

- Base T5-small model
- Traditional fine-tuned model
- LoRA fine-tuned model
