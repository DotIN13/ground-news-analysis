import os
import json

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    load_peft_weights,
    set_peft_model_state_dict
)
from peft import PeftModel
from accelerate import Accelerator
import torch.nn.functional as F  # For KL divergence and softmax

MAX_LENGTH = 3072
SEED = 42
HF_TOKEN = os.environ.get("HF_TOKEN")


# ------------------------------
# Data Preparation
# ------------------------------
class TextClassificationDataset(Dataset):
    def __init__(self, data, label_map, tokenizer):
        self.data = data
        self.label_map = label_map
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = f"{self.data[idx]['article_description']} {self.data[idx]['article_text']}"
        label = self.data[idx]["label"]
        logits = self.data[idx]["logit_scores"]
        return {"text": text, "label": label, "logits": logits}


def create_collate_fn(tokenizer):
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        # Convert teacher logits for each sample into a tensor (assumed shape: [num_labels])
        teacher_logits = torch.tensor([item["logits"] for item in batch], dtype=torch.float)
        tokenized = tokenizer(
            texts,
            padding=True,  # Enable dynamic padding
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "teacher_logits": teacher_logits
        }
    return collate_fn


def load_datasets(file_path, label_map, tokenizer):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = TextClassificationDataset(train_data, label_map, tokenizer)
    val_dataset = TextClassificationDataset(val_data, label_map, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, collate_fn=create_collate_fn(tokenizer))
    val_dataloader = DataLoader(
        val_dataset, batch_size=8, collate_fn=create_collate_fn(tokenizer))
    return train_dataloader, val_dataloader


# ------------------------------
# Model & Tokenizer Setup
# ------------------------------
def setup_model_and_tokenizer(model_name, bnb_config, accelerator, lora_weights_path=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    # Load model with quantization config and task-specific number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        num_labels=5,
        device_map={"": accelerator.process_index},
        token=HF_TOKEN
    )

    # Set pad token and resize embeddings
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # Prepare model for efficient training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA configuration for QLoRA training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "down_proj",
            "o_proj",
            "q_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "v_proj"
        ],
        task_type="SEQ_CLS"
    )

    # Load existing LoRA weights if provided, else initialize a new LoRA adapter
    if lora_weights_path is not None:
        model = get_peft_model(model, lora_config)
        state_dict = load_peft_weights(lora_weights_path)
        set_peft_model_state_dict(
            model,
            state_dict,
            adapter_name="default",
            ignore_mismatched_sizes=True,
        )
        print(f"Loaded existing LoRA weights from {lora_weights_path}")
    else:
        model = get_peft_model(model, lora_config)

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameters:", trainable_params)

    # Save tokenizer for later use
    tokenizer.save_pretrained("best_model")

    return model, tokenizer


# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += len(labels)

    accuracy = correct / total
    avg_val_loss = total_val_loss / len(dataloader)
    model.train()
    return avg_val_loss, accuracy


# ------------------------------
# Training Loop
# ------------------------------
def train_loop(model, optimizer, scheduler, train_dataloader, val_dataloader, device, accelerator, writer, num_epochs=3, log_steps=10, eval_steps=100):
    best_val_loss = float("inf")
    global_step = 0

    # Hyperparameters for knowledge distillation
    alpha = 0.5        # Weight for the classification loss (ground-truth labels)
    temperature = 2.0   # Temperature for distillation

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        running_loss = []

        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            # Move batch to the correct device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)

            # Forward pass: the model returns a classification loss using the provided labels.
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            classification_loss = outputs.loss

            # Compute the distillation loss (KL divergence between teacher and student distributions)
            student_logits = outputs.logits
            # Scale logits with temperature
            log_probs = F.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            distill_loss = F.kl_div(log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

            # Combine the losses
            total_loss_batch = alpha * classification_loss + (1 - alpha) * distill_loss
            running_loss.append(total_loss_batch.item())

            accelerator.backward(total_loss_batch)
            optimizer.step()
            scheduler.step()  # Update learning rate scheduler

            total_loss += total_loss_batch.item()
            global_step += 1

            # Log training loss every log_steps steps
            if (step + 1) % log_steps == 0:
                loss_val = sum(running_loss) / len(running_loss)
                print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_dataloader)}, Loss: {loss_val:.4f}")
                writer.add_scalar("Train/Step_Loss", loss_val, global_step)
                running_loss = []

            # Evaluate and save best model every eval_steps steps
            if global_step % eval_steps == 0:
                avg_val_loss, accuracy = evaluate(model, val_dataloader, device)
                print(f"Step {global_step} Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
                writer.add_scalar("Validation/Loss", avg_val_loss, global_step)
                writer.add_scalar("Validation/Accuracy", accuracy, global_step)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained("best_model")
                    torch.cuda.empty_cache()
                    print(f"Best model saved at step {global_step} with Validation Loss: {best_val_loss:.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Train/Average_Loss", avg_train_loss, epoch)

    return best_val_loss


# ------------------------------
# Main Function
# ------------------------------
def main():
    # Setup accelerator and device
    accelerator = Accelerator()
    device = accelerator.device

    # Define label mapping and quantization configuration
    label_map = {"left": 0, "center": 1, "right": 2}  # Adjust if necessary for 5 classes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_name = "meta-llama/Llama-3.2-1B"
    lora_weights_path = None

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name, bnb_config, accelerator,
        lora_weights_path=lora_weights_path)

    # Load data and create dataloaders
    train_dataloader, val_dataloader = load_datasets(
        "../data/topics_10k_deepseek_logits.json", label_map, tokenizer)

    # Initialize optimizer and prepare with accelerator
    optimizer = AdamW(model.parameters(), lr=2e-4)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    # Calculate total training steps and setup the linear scheduler with warmup
    num_epochs = 3
    total_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_training_steps),  # 5% warmup
        num_training_steps=total_training_steps
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="./runs/exp-distill-logits-retrain-e3")

    # Run training loop with evaluation and logging
    train_loop(model,
               optimizer,
               scheduler,
               train_dataloader,
               val_dataloader,
               device,
               accelerator,
               writer,
               num_epochs=num_epochs,
               log_steps=20,
               eval_steps=200)

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
