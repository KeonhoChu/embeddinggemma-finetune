import pandas as pd
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_examples(texts, num_pairs_per_text=3):
    """
    Create training examples for contrastive learning
    """
    examples = []
    
    for i, text in enumerate(texts):
        # Create positive pairs (same text with slight variations or duplicates)
        examples.append(InputExample(texts=[text, text], label=1.0))
        
        # Create negative pairs (different texts)
        for _ in range(num_pairs_per_text):
            # Random negative example
            neg_idx = random.randint(0, len(texts) - 1)
            while neg_idx == i:
                neg_idx = random.randint(0, len(texts) - 1)
            
            examples.append(InputExample(texts=[text, texts[neg_idx]], label=0.0))
    
    return examples

def load_and_preprocess_data(csv_path):
    """Load and preprocess the CSV data"""
    logger.info(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Clean and preprocess text data
    texts = df['text'].dropna().astype(str).tolist()
    
    # Remove very short texts
    texts = [text for text in texts if len(text.strip()) > 10]
    
    # Limit to reasonable size for training
    if len(texts) > 10000:
        texts = texts[:10000]
        logger.info(f"Limited to first 10000 texts for training efficiency")
    
    logger.info(f"Loaded {len(texts)} text samples")
    return texts

def main():
    # Configuration
    MODEL_NAME = "google/embeddinggemma-300m"
    CSV_PATH = "output_unsupervised.csv"
    OUTPUT_DIR = "./finetune_results_sbert"
    MAX_SEQ_LENGTH = 8192
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    texts = load_and_preprocess_data(CSV_PATH)
    
    # Create training examples
    logger.info("Creating training examples...")
    training_examples = create_training_examples(texts, num_pairs_per_text=2)
    
    # Split examples
    train_examples, val_examples = train_test_split(training_examples, test_size=0.2, random_state=42)
    logger.info(f"Train examples: {len(train_examples)}, Validation examples: {len(val_examples)}")
    
    # Initialize model
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Training arguments with your specified parameters
    args = SentenceTransformerTrainingArguments(
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
        run_name="embeddinggemma-finetune-sbert",
        report_to=None,
    )
    
    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        loss=train_loss,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    model.save(OUTPUT_DIR)
    
    # Test embedding generation
    logger.info("Testing embedding generation...")
    test_texts = texts[:5]  # Use first 5 text samples
    
    embeddings = model.encode(test_texts)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Expected output dimensions: 1024")
    logger.info(f"Actual output dimensions: {embeddings.shape[1]}")
    logger.info(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Test similarity
    if len(test_texts) >= 2:
        similarity = model.similarity(embeddings[0], embeddings[1])
        logger.info(f"Similarity between first two texts: {similarity.item():.4f}")
    
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info(f"Maximum sequence length: {MAX_SEQ_LENGTH} tokens")
    logger.info(f"Output dimensionality: {embeddings.shape[1]} dimensions")

if __name__ == "__main__":
    main()