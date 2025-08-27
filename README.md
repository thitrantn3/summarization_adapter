**LoRA Fine-Tuning for CNN-Daily Mail with Qwen 2.5-3B**

This project fine-tunes the Qwen 2.5-3B large language model (LLM) for the text summarization task using the CNN-Daily Mail dataset. I will employ the Low-Rank Adaptation (LoRA) method, a parameter-efficient fine-tuning (PEFT) technique.

Getting Started
<br>Install all required libraries from the requirements.txt file.

Project Workflow
Follow these steps in order to prepare data, train the model, and evaluate its performance.

1. Prepare Data with dataset.py
<br>Run *python dataset.py* to load, process, and tokenize the raw CNN-Daily Mail data. It separates articles and summaries into "input_texts" and "labels" columns, which is the required format for training a summarization model.

2. Train the Model with train.py
<br>Execute *python train.py* to begin fine-tuning. It loads the Qwen 2.5-3B model and trains the LoRA adapter on your prepared dataset. The process is memory-efficient and includes early stopping to prevent overfitting.

3. Evaluate Performance with eval.py
<br>After training, run *python eval.py* to assess your model's quality. It generates summaries on a test set and computes ROUGE scores, providing a quantitative measure of performance.

4. Test with a Custom Input using test.py
<br>Run *python test.py* allows you to perform a quick, qualitative check of your fine-tuned model. It loads the trained adapter, merges it with the base model, and prompts you to enter your own text to generate a summary.
