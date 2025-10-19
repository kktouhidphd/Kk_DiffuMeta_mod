"""
Diffusion Model Training Script for Inverse Design of Shell Metamaterials

This script trains a transformer-based diffusion model for generating mathematical equations.
The model learns to generate implicit equations conditioned on material properties or other conditions.

Usage:
    python main.py --config config.yaml
"""

import argparse
import os
import yaml
import csv  # <-- Import csv for logging

# PyTorch imports
import torch
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

import numpy as np

# ML utilities
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from abc import ABC, abstractmethod
import shutil

# Local imports - project-specific modules
from src.utils import *
from src.model import *
from src.datasets import *
from src.gaussian_diffusion import *
from src.params import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Please create it or check the path.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise

def flatten_config(config, parent_key='', sep='_'):
    """Flatten nested config dictionary for easier access"""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Model Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    return parser.parse_args()

if __name__ == "__main__":
    
    # ========================================================================================
    # CONFIGURATION AND SETUP
    # ========================================================================================
    
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration from YAML file
    yaml_config = load_config(args.config)
    
    # Flatten the config for easier access
    flat_config = flatten_config(yaml_config)
    
    # ========================================================================================
    # LOCAL LOGGING SETUP (Replaces WANDB)
    # ========================================================================================
    
    class Config:
        def __init__(self, d):
            for a, b in d.items():
                setattr(self, a, b)
    config = Config(flat_config)

    sweep_name = 'local_run'

    outdir = f"{run_dir}/{sweep_name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f'{outdir}/check_point', exist_ok=True)

    # --- Setup for local log file ---
    log_file_path = os.path.join(outdir, "training_log.txt")
    with open(log_file_path, "w") as f:
        f.write("epoch,training_loss,test_loss,train_pred_loss,test_pred_loss\n")
    # --------------------------------

    # ========================================================================================
    # DATA PREPARATION
    # ========================================================================================
    
    tokenizer = EquationTokenizer()
    dataset_result = get_dataset(data_path, multi_objective_cond=multi_objective_cond)
    
    if isinstance(dataset_result, tuple):
        dataset = dataset_result[0]
    else:
        dataset = dataset_result

    num_train = len(dataset)
    split = int(np.floor(valid_size * num_train))

    train_dataset, test_dataset = random_split(
        dataset, lengths=[num_train - split, split])

    torch.save(train_dataset, f'{outdir}/train_dataset.pth')

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.training_train_batch_size, shuffle=True, drop_last=True)
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.training_train_batch_size, shuffle=True, drop_last=True)
    
    # ========================================================================================
    # MODEL INITIALIZATION
    # ========================================================================================
    
    model = TransformerNetModel(
        input_dims=config.model_input_dims,
        output_dims=config.model_output_dims,
        hidden_t_dim=config.model_hidden_t_dim,
        transformer_num_hidden_layers=config.model_transformer_num_hidden_layers,
        transformer_num_attention_heads=config.model_transformer_num_attention_heads,
        transformer_hidden_size=config.model_transformer_hidden_size,
        proj_activation_func=config.model_proj_activation_func,
        mlp_ratio=config.model_mlp_ratio,
        depth=config.model_depth,
        cfg_scale=config.cfg_cfg_scale,
        dropout=0,
        config=None,
        config_name=config.model_config_name,
        vocab_size=tokenizer.vocab_size,
        dropout_prob=config.cfg_cfg_dropout_prob,
        cross_attn=config.model_cross_attn,
        latent_model=None,
        embedding_scale=config.model_embedding_scale,
        learn_embedding_scale=config.model_learn_embedding_scale,
    ).to(device)
    
    model.initialize_weights()
    print(model)
    
    # ========================================================================================
    # DIFFUSION PROCESS SETUP
    # ========================================================================================
    
    noise_schedule = config.diffusion_noise_schedule
    diffusion_steps = config.diffusion_diffusion_steps
    predict_xstart = config.diffusion_predict_xstart
    sigma_small = False
    
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    
    rescale_timesteps = True
    learn_sigmas = False
    use_kl = False
    rescale_learned_sigmas = False

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, section_counts=config.diffusion_section_counts),
        betas=betas,
        training_mode=config.diffusion_training_mode,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas=learn_sigmas,
        use_kl=use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas,
        normalize_nll_loss=config.diffusion_normalize_nll_loss
    )

    # ========================================================================================
    # TIMESTEP SAMPLING STRATEGY
    # ========================================================================================

    class ScheduleSampler(ABC):
        @abstractmethod
        def weights(self):
            pass

        def sample(self, batch_size, device):
            w = self.weights()
            if w is None:
                raise ValueError("weights() method returned None")
            p = w / np.sum(w)
            indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
            indices = th.from_numpy(indices_np).long().to(device)
            weights_np = 1 / (len(p) * p[indices_np])
            weights = th.from_numpy(weights_np).float().to(device)
            return indices, weights

    class UniformSampler(ScheduleSampler):
        def __init__(self, diffusion):
            self.diffusion = diffusion
            self._weights = np.ones([diffusion.num_timesteps])

        def weights(self):
            return self._weights

    schedule_sampler = UniformSampler(diffusion)
        
    if config.training_optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training_lr, weight_decay=1e-5)
    elif config.training_optimizer_type == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=config.training_lr, weight_decay=1e-5)
    elif config.training_optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training_lr, weight_decay=0.01)
    elif config.training_optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training_lr, momentum=0.9, weight_decay=0.01)

    warmup_steps = int(0.1 * config.training_num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=config.training_num_epochs
    )

    # ========================================================================================
    # TRAINING PREPARATION
    # ========================================================================================
    
    global_step = 0
    best_test_loss = 1e3
    print("Training model...")

    def evaluate(model, test_dataloader):
        model.eval()
        test_loss_epoch, test_pred_epoch, test_nll_epoch, test_tT_eopch = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                eq = batch[0].to(device)
                label = batch[1].to(device)
                if multi_objective_cond:
                    cond = [label, batch[-1].to(device)]
                else:
                    cond = label
                t, weights = schedule_sampler.sample(eq.shape[0], device)
                losses = diffusion.training_losses(model, eq, t, cond, None)
                if losses is not None:
                    loss = losses["loss"].mean()
                    if "mse" in losses:
                        test_pred_epoch += losses["mse"].mean().detach().item()
                    elif "ce" in losses:
                        test_pred_epoch += losses["ce"].mean().detach().item()
                    test_nll_epoch += losses["decoder_nll"].mean().detach().item()
                    test_tT_eopch += losses["tT_loss"].mean().detach().item()
                    test_loss_epoch += loss.detach().item()
        return test_loss_epoch/(step+1), test_pred_epoch/(step+1), test_nll_epoch, test_tT_eopch

    # ========================================================================================
    # MAIN TRAINING LOOP
    # ========================================================================================
    
    print(f"Starting training for {config.training_num_epochs} epochs...")
    
    for epoch in range(config.training_num_epochs):
        train_loss_epoch, train_pred_loss_epoch, train_decoder_nll_loss_epoch, train_tT_loss_epoch = 0., 0., 0., 0.
        model.train()

        for step, batch in enumerate(train_dataloader):
            eq, label = batch[0].to(device), batch[1].to(device)
            if multi_objective_cond:
                cond = [label, batch[-1].to(device)]
            else:
                cond = label
            optimizer.zero_grad()
            t, weights = schedule_sampler.sample(eq.shape[0], device)
            losses = diffusion.training_losses(model, eq, t, cond, None)
            
            if config.freezing_freeze_embed and epoch == config.freezing_freeze_embed_epoch:
                model.freeze_embedding()
            
            if losses is not None:        
                loss = losses["loss"].mean()
                loss.backward()
                if "mse" in losses:
                    train_pred_loss_epoch += losses["mse"].mean().detach().item()
                elif "ce" in losses:
                    train_pred_loss_epoch += losses["ce"].mean().detach().item()
                train_decoder_nll_loss_epoch += losses["decoder_nll"].mean().detach().item()
                train_tT_loss_epoch += losses["tT_loss"].mean().detach().item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training_max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1
                train_loss_epoch += loss.detach().item()
        
        train_loss_epoch /= (step+1)
        train_pred_loss_epoch /= (step+1)

        # ========================================================================================
        # EVALUATION AND LOGGING
        # ========================================================================================
        
        test_loss_epoch, test_pred_epoch, test_nll_epoch, test_tT_eopch = evaluate(model, test_dataloader)
        
        # --- Print to console ---
        if epoch % 10 == 0 or epoch == 0:
            print(f"--- Epoch {epoch} ---")
            print(f"  Training Loss: {train_loss_epoch:.4f}")
            print(f"  Test Loss:     {test_loss_epoch:.4f}")
            print("--------------------")

        # --- Write to local log file ---
        with open(log_file_path, "a") as f:
            log_line = f"{epoch},{train_loss_epoch:.6f},{test_loss_epoch:.6f},{train_pred_loss_epoch:.6f},{test_pred_epoch:.6f}\n"
            f.write(log_line)
        # --------------------------------

        # ========================================================================================
        # MODEL CHECKPOINTING
        # ========================================================================================
        
        if test_loss_epoch < best_test_loss:
            torch.save(model, f'{outdir}/best_model_checkpoint.pth')
            best_test_loss = test_loss_epoch
        
        if epoch > 0 and epoch % config.training_save_model_step == 0:
            model_checkpoint_path = f'{outdir}/check_point/epoch_{epoch}_model_checkpoint.pth'
            torch.save(model, model_checkpoint_path)
    
    # ========================================================================================
    # TRAINING COMPLETION AND CLEANUP
    # ========================================================================================
    
    print("Training completed!")
    print("Saving final model...")
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    print(f"Model saved to {outdir}/model.pth")
 