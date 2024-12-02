import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable TF onednn to prevent conflicts with transformers
import torch
from transformers import BertForSequenceClassification, BertForQuestionAnswering, BertTokenizerFast, TrainingArguments, Trainer, EarlyStoppingCallback, AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator
from datasets import load_dataset, DatasetDict, Dataset
import random
import json
import matplotlib.pyplot as plt
import matplotlib
import IPython.display as display
if os.environ.get('JUPYTER_RUNNING'):
    matplotlib.use('module://ipykernel.pylab.backend_inline')  # Use Jupyter inline backend
else:
    matplotlib.use('TkAgg')  # Use a non-interactive backend
import numpy as np
from tqdm import tqdm
import warnings
# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sklearn.metrics as metrics

def finetune_squadlike_qa(model_name, output_dir, debug_mode=False, dataset_name="squad2", force_retrain=False, hyperparameters=None, custom_tokenizer=None):
    
    # create the output directory for the selected model
    output_dir = os.path.join(output_dir, model_name)
    
    # Load the appropriate dataset
    if dataset_name == "squad2":
        # Load the SQuAD 2.0 dataset
        data_dir = "./squad2.0"
        squad_dataset = load_dataset("squad_v2")
    elif dataset_name == "eusquad":
        # Load the EUSQUAD dataset
        data_dir = "./eusquad_v1.0"
        eusquad_train = load_dataset('json', data_files='eusquad_v1.0/eusquad-train-v1.0_canine-s.jsonl')
        eusquad_dev = load_dataset('json', data_files='eusquad_v1.0/eusquad-dev-v1.0_canine-s.jsonl')

        # Combine the train and validation datasets into a DatasetDict
        squad_dataset = DatasetDict({
            "train": eusquad_train['train'],
            "validation": eusquad_dev['train']
        })
    else:
        raise ValueError("Invalid dataset name. Choose between 'squad2' or 'eusquad'.")

    # Split 10% of the training data as the test set
    print("Splitting the training data into training and test sets...")
    train_dataset = squad_dataset["train"]
    shuffled_indices = list(range(len(train_dataset)))
    random.shuffle(shuffled_indices)
    
    valid_indices = [i for i in shuffled_indices if len(train_dataset[i]['answers']['text']) > 0]
    invalid_indices = [i for i in shuffled_indices if len(train_dataset[i]['answers']['text']) == 0]
    
    # Determine the number of instances for the test set (approximately 10% of the training data)
    num_test_instances = int(0.1 * len(train_dataset)) // 2
    
    # Select balanced valid and invalid indices for the test set
    test_valid_indices = valid_indices[:num_test_instances]
    test_invalid_indices = invalid_indices[:num_test_instances]
    test_indices = test_valid_indices + test_invalid_indices
    
    # Ensure the remaining indices are used for training
    train_indices = [i for i in shuffled_indices if i not in test_indices]

    # Print the number of valid and invalid instances in the test set
    num_valid_test = len([i for i in test_indices if len(train_dataset[i]['answers']['text']) > 0])
    num_invalid_test = len([i for i in test_indices if len(train_dataset[i]['answers']['text']) == 0])
    print(f"Number of valid answers in the test set: {num_valid_test}")
    print(f"Number of invalid answers in the test set: {num_invalid_test}")
    
    test_dataset = train_dataset.select(test_indices)
    train_dataset = train_dataset.select(train_indices)

    # Save test dataset to a JSON file for later use
    def save_test_data_squad_format(dataset, file_path):
        data = []
        for example in dataset:
            paragraphs = {
                "context": example["context"],
                "qas": [
                    {
                        "id": example["id"],
                        "question": example["question"],
                        "answers": [
                            {
                                "text": answer,
                                "answer_start": start
                            }
                            for answer, start in zip(example["answers"]["text"], example["answers"]["answer_start"])
                        ],
                        "is_impossible": len(example["answers"]["text"]) == 0
                    }
                ]
            }
            data.append({"paragraphs": [paragraphs]})

        squad_format_data = {
            "version": "1.0",
            "data": data
        }

        with open(file_path, 'w') as f:
            json.dump(squad_format_data, f, indent=2)

    # Save the test data in SQuAD format
    test_data_path = os.path.join(data_dir, "test_data.json")
    save_test_data_squad_format(test_dataset, test_data_path)


    # Update dataset to include the newly split training set
    squad_dataset = DatasetDict({
        "train": train_dataset,
        "validation": squad_dataset["validation"],
        "test": test_dataset
    })

    # Truncate datasets if in debug mode
    if debug_mode:
        squad_dataset["train"] = squad_dataset["train"].select(range(5000))
        squad_dataset["validation"] = squad_dataset["validation"].select(range(500))

    # Load the tokenizer and model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to preprocess the dataset for question answering
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # Define a function to preprocess the evaluation examples
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
        
    # Apply preprocessing to the dataset
    print("Tokenizing dataset...")
    tokenized_squad = squad_dataset.map(prepare_train_features, batched=True, remove_columns=squad_dataset["train"].column_names)

    # Set default hyperparameters if not provided
    if hyperparameters is None:
        hyperparameters = {
            'learning_rate': 3e-5,
            'num_train_epochs': 10,
            'batch_size': 4,
            'gradient_accumulation_steps': 16,
            'weight_decay': 0.01,
            'warmup_ratio': 0.2
        }

    learning_rate = hyperparameters['learning_rate']
    num_train_epochs = hyperparameters['num_train_epochs']
    batch_size = hyperparameters['batch_size'] * hyperparameters['gradient_accumulation_steps']
    gradient_accumulation_steps = hyperparameters['gradient_accumulation_steps']
    weight_decay = hyperparameters['weight_decay']
    warmup_ratio = hyperparameters['warmup_ratio']

    num_train_epochs = num_train_epochs

    # Calculate total training steps and warmup steps
    num_training_examples = len(tokenized_squad["train"])
    total_steps = (num_training_examples // batch_size) * num_train_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # logging_dir="./logs",
        logging_dir = os.path.join(output_dir, "logs"),
        logging_steps=10 if debug_mode else 100,
        learning_rate=learning_rate,
        num_train_epochs=1 if debug_mode else hyperparameters['num_train_epochs'],
        per_device_train_batch_size=hyperparameters['batch_size'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,  # Calculated based on warmup_ratio
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )

    # Check for an existing checkpoint
    last_checkpoint = None
    if os.path.isdir(output_dir) and not force_retrain:
        checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith("checkpoint")]
        if checkpoint_files:
            checkpoint_numbers = [int(f.split("-")[-1]) for f in checkpoint_files]
            latest_checkpoint = max(checkpoint_numbers)
            last_checkpoint = os.path.join(output_dir, f"checkpoint-{latest_checkpoint}")

    if last_checkpoint and not force_retrain:
        print(f"Resuming training from checkpoint {last_checkpoint}")
    else:
        print("No checkpoint found or force retrain is set, starting training from scratch.")

    # Add early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["validation"],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )

    # Start or resume training
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save the best model at the end of training
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Extract training and evaluation losses from the logs
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []

    for log in trainer.state.log_history:
        if "loss" in log.keys() and "step" in log.keys():
            train_losses.append(log["loss"])
            train_steps.append(log["step"])
        if "eval_loss" in log.keys() and "epoch" in log.keys():
            eval_losses.append(log["eval_loss"])
            eval_steps.append(log["epoch"])

    # Plot training and evaluation loss
    plt.figure(figsize=(6, 4))
    plt.style.use('dark_background')

    if len(train_losses) > 0:
        plt.plot(train_steps, train_losses, label="Training Loss", linestyle="-", color="blue", alpha=0.7)

    if len(eval_losses) > 0:
        eval_steps_scaled = [step * (train_steps[-1] / max(eval_steps)) for step in eval_steps]
        plt.plot(eval_steps_scaled, eval_losses, label="Evaluation Loss", linestyle="--", color="orange", marker="o", alpha=0.8)

    if len(train_losses) > 0 or len(eval_losses) > 0:
        ax = plt.gca()
        ax.set_facecolor("black")
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss Over Training Steps")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "training_evaluation_loss_plot.png"), facecolor="black")
        if os.environ.get('JUPYTER_RUNNING'):
            display.display(plt.gcf())
        plt.close()
    else:
        print("No data available to plot.")

    # Evaluate the model on the test set
    model = AutoModelForQuestionAnswering.from_pretrained(best_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare validation features for the test set
    validation_data = squad_dataset["test"]
    validation_features = validation_data.map(
        prepare_validation_features,
        batched=True,
        remove_columns=validation_data.column_names,
        )
    

    # Define a function to post-process predictions
    def postprocess_qa_predictions(examples, features, raw_predictions):
        all_start_logits, all_end_logits = raw_predictions

        # Map examples to their corresponding features
        example_to_features = {example["id"]: [] for example in examples}
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        final_predictions = {}

        # Iterate through each example
        for example in tqdm(examples):
            feature_indices = example_to_features[example["id"]]
            context = example["context"]

            # Initialize the minimum null score for deciding "no answer" prediction
            min_null_score = None
            valid_answers = []

            # Iterate over each feature (span) for this example
            for feature_index in feature_indices:
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                offset_mapping = features[feature_index]["offset_mapping"]

                input_ids = features[feature_index]["input_ids"]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                # Calculate null score using CLS token
                null_score = start_logits[cls_index] + end_logits[cls_index]
                min_null_score = min(min_null_score, null_score) if min_null_score is not None else null_score

                # Get the top start and end indices
                start_indexes = np.argsort(start_logits)[-1: -21: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -21: -1].tolist()

                # Collect all valid answers from the feature
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Make sure start and end indices are within the valid range
                        if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                            continue

                        # Ensure that the start and end are within the context tokens (not question or padding)
                        if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                            continue

                        # Ensure that the end index is after the start index and within the maximum answer length
                        if end_index < start_index or end_index - start_index + 1 > 30:
                            continue

                        # Get the character start and end positions from the offset mapping
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append({
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char]
                        })

            # Select the best answer from the valid answers or decide "no answer"
            if len(valid_answers) > 0:
                best_answer = max(valid_answers, key=lambda x: x["score"])
            else:
                best_answer = {"text": ""}

            # Compare the best answer score with the null score to decide the final prediction
            if min_null_score is not None and min_null_score > best_answer.get("score", 0):
                final_predictions[example["id"]] = ""
            else:
                final_predictions[example["id"]] = best_answer["text"]

        return final_predictions

    # Get predictions with DataCollator
    print('Preparing test dataset for evaluation...')
    validation_dataset = validation_features.remove_columns(["example_id", "offset_mapping"])
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=8,
        collate_fn=default_data_collator
        )

    all_start_logits = []
    all_end_logits = []
    print("Generating predictions on the test set...")
    
    for batch in tqdm(validation_dataloader):
        # Move batch to GPU if available
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-process the predictions
    raw_predictions = (all_start_logits, all_end_logits)
    print("Post-processing predictions...")
    final_predictions = postprocess_qa_predictions(squad_dataset["test"], validation_features, raw_predictions)

    # Save predictions to JSON file in the format needed for SQuAD evaluation
    eval_predictions_file = os.path.join(data_dir, "test_predictions.json")
    with open(eval_predictions_file, 'w') as outfile:
        json.dump(final_predictions, outfile)

    print(f"Predictions saved to {eval_predictions_file}")
    
    # Run the evaluation script
    if dataset_name == "squad2":
        # data_file = "dev-v2.0.json"  # Path to the ground truth data
        data_file = os.path.join(data_dir, "test_data.json")
    elif dataset_name == "eusquad":
        # data_file = "eusquad_v1.0/eusquad-dev-v1.0_canine-s.jsonl"
        data_file = os.path.join(data_dir, "test_data.json")    
    
    pred_file = eval_predictions_file  # Path to your predictions file
    out_file = "evaluation_results.json"
    out_file = os.path.join(output_dir, out_file)
    os.system(f"python evaluate-v2.0.py {data_file} {pred_file} --out-file {out_file}")
    print(f"Evaluation results saved to {out_file}")
    
    return final_predictions


## example usage:
# hyperparameters = {
#     'learning_rate': 3e-5,
#     'num_train_epochs': 5,
#     'batch_size': 8,
#     'gradient_accumulation_steps': 8,
#     'weight_decay': 0.02,
#     'warmup_ratio': 0.1
#     }
# final_predictions = finetune_squadlike_qa(model_name="bert-base-uncased", output_dir="./results", debug_mode=False, dataset_name="squad2", force_retrain=False, hyperparameters=hyperparameters)

