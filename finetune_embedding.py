import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = self.model.config
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        last_hidden_states = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        loss = None
        if labels is not None:
            # Contrastive learning loss can be implemented here
            # For now, using MSE as a simple reconstruction loss
            reconstructed = self.model.get_input_embeddings()(labels)
            loss = torch.nn.functional.mse_loss(embeddings, reconstructed.mean(dim=1))
        
        return {
            'loss': loss,
            'embeddings': embeddings,
            'last_hidden_state': last_hidden_states
        }

def load_and_preprocess_data(csv_path):
    """Load and preprocess the CSV data"""
    logger.info(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Clean and preprocess text data
    texts = df['text'].dropna().astype(str).tolist()
    
    # Remove very short texts
    texts = [text for text in texts if len(text.strip()) > 10]
    
    logger.info(f"Loaded {len(texts)} text samples")
    return texts

def main():
    # Configuration
    MODEL_NAME = "google/embeddinggemma-300m"
    CSV_PATH = "output_unsupervised.csv"
    OUTPUT_DIR = "./finetune_results"
    MAX_LENGTH = 512
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    texts = load_and_preprocess_data(CSV_PATH)
    
    # Split data
    train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
    logger.info(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = EmbeddingModel(MODEL_NAME)
    model.to(device)
    
    # Create datasets
    train_dataset = EmbeddingDataset(train_texts, tokenizer, MAX_LENGTH)
    val_dataset = EmbeddingDataset(val_texts, tokenizer, MAX_LENGTH)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,
        do_predict=False,
        eval_strategy="steps",
        prediction_loss_only=True,
        per_device_train_batch_size=10000,
        per_device_eval_batch_size=4096,
        per_gpu_train_batch_size=None,
        per_gpu_eval_batch_size=None,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=None,
        torch_empty_cache_steps=None,
        learning_rate=2e-05,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=1.0,
        num_train_epochs=2,
        max_steps=-1,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={'num_decay_steps': 160},
        warmup_ratio=0.05,
        warmup_steps=0,
        log_level="passive",
        log_level_replica="warning",
        log_on_each_node=True,
        logging_nan_inf_filter=True,
        save_safetensors=True,
        save_on_each_node=False,
        save_only_model=False,
        restore_callback_states_from_checkpoint=False,
        no_cuda=False,
        use_cpu=False,
        use_mps_device=False,
        seed=42,
        data_seed=None,
        jit_mode_eval=False,
        use_ipex=False,
        bf16=True,
        fp16=False,
        fp16_opt_level="O1",
        half_precision_backend="auto",
        bf16_full_eval=False,
        fp16_full_eval=False,
        tf32=None,
        local_rank=0,
        ddp_backend=None,
        tpu_num_cores=None,
        tpu_metrics_debug=False,
        debug=[],
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        past_index=-1,
        disable_tqdm=False,
        remove_unused_columns=True,
        label_names=None,
        load_best_model_at_end=False,
        ignore_data_skip=False,
        fsdp=[],
        fsdp_min_num_params=0,
        fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
        fsdp_transformer_layer_cls_to_wrap=None,
        accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None},
        deepspeed=None,
        label_smoothing_factor=0.0,
        optim="adamw_torch",
        optim_args=None,
        adafactor=False,
        group_by_length=False,
        length_column_name="length",
        ddp_find_unused_parameters=None,
        ddp_bucket_cap_mb=None,
        ddp_broadcast_buffers=False,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        skip_memory_metrics=True,
        use_legacy_prediction_loop=False,
        push_to_hub=False,
        resume_from_checkpoint=None,
        hub_model_id=None,
        hub_strategy="every_save",
        hub_private_repo=None,
        hub_always_push=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        include_inputs_for_metrics=False,
        include_for_metrics=[],
        eval_do_concat_batches=True,
        fp16_backend="auto",
        push_to_hub_model_id=None,
        push_to_hub_organization=None,
        mp_parameters="",
        auto_find_batch_size=False,
        full_determinism=False,
        torchdynamo=None,
        ray_scope="last",
        ddp_timeout=1800,
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,
        dispatch_batches=None,
        split_batches=None,
        include_tokens_per_second=False,
        include_num_input_tokens_seen=False,
        neftune_noise_alpha=None,
        optim_target_modules=None,
        batch_eval_metrics=False,
        eval_on_start=False,
        use_liger_kernel=False,
        eval_use_gather_object=False,
        average_tokens_across_devices=False,
        prompts=None,
        batch_sampler="no_duplicates",
        multi_dataset_batch_sampler="proportional",
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_strategy="steps",
        run_name="embeddinggemma-finetune",
        report_to=None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Test embedding generation
    logger.info("Testing embedding generation...")
    test_text = texts[0][:200]  # Use first text sample
    
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        embedding = outputs['embeddings']
        logger.info(f"Generated embedding shape: {embedding.shape}")
        logger.info(f"Sample embedding (first 10 dims): {embedding[0][:10].cpu().numpy()}")
    
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()