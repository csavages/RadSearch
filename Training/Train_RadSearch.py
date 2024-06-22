"""Updated training script for the recent Sentence Transformers 3.0.0 update on May 28th, 2024"""

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "UCSD-VA-health/RadBERT-RoBERTa-4m",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="RadBERT-RoBERTa-4m",
    )
)

# 3. Option 1: Load a dataset to finetune from Hugging Face
dataset = load_dataset("Huggingface_dataset_filepath", "triplet")

# 3. Option 2: Load separate locally saved csv files for training and test datasets
dataset = load_dataset('csv', data_files={'train': your_folder_path + 'train_dataset.csv', 'evaluation': your_folder_path + 'evaluation_dataset.csv', 'test': your_folder_path + 'test_dataset.csv'})

train_dataset = dataset["train"]
eval_dataset = dataset["evaluation"]
test_dataset = dataset["test"]

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="your_output_folder_path/your_model",
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="your_model",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"], # This column of your dataset should contain the "Findings" sections of reports
    positives=eval_dataset["positive"], # This should contain the exact "Impression" section associated with the "Findings" section of a given report
    negatives=eval_dataset["negative"], # This should contain all other "Impression" sections other than the positive example above
    name="your_model_evaluator",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()


# 8. (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"], # This column of your dataset should contain the "Findings" sections of reports
    positives=test_dataset["positive"], # This should contain the exact "Impression" section associated with the "Findings" section of a given report
    negatives=test_dataset["negative"], # This should contain all other "Impression" sections other than the positive example above
    name="your_model_test",
)
test_evaluator(model)

# 9. Save the trained model
model.save_pretrained("your_output_folder_path/RadBERT-RoBERTa-4m/your_model_final")
