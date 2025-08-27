LoRA Fine-Tuning for CNN-Daily Mail with Qwen 2.5-3B
This project fine-tunes the Qwen 2.5-3B large language model (LLM) for the text summarization task using the CNN-Daily Mail dataset. I will employ the Low-Rank Adaptation (LoRA) method, a parameter-efficient fine-tuning (PEFT) technique.

Getting Started
Before you begin, clone the repository and install all required libraries from the requirements.txt file.

git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt

Project Workflow
Follow these steps in order to prepare your data, train the model, and evaluate its performance.

1. Prepare Data with dataset.py
Run this script to load, process, and tokenize the raw CNN-Daily Mail data. It separates articles and summaries into "input_texts" and "labels" columns, which is the required format for training a summarization model.

python dataset.py

2. Train the Model with train.py
Execute this script to begin fine-tuning. It loads the Qwen 2.5-3B model and trains the LoRA adapter on your prepared dataset. The process is memory-efficient and includes early stopping to prevent overfitting.

python train.py

3. Evaluate Performance with eval.py
After training, run this script to assess your model's quality. It generates summaries on a test set and computes ROUGE scores, providing a quantitative measure of performance.

python eval.py

4. Test with a Custom Input using test.py
This script allows you to perform a quick, qualitative check of your fine-tuned model. It loads the trained adapter, merges it with the base model, and prompts you to enter your own text to generate a summary.

python test.py