import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
import csv
import multiprocessing
from multiprocessing import Pool
import logging
import time
import os
import argparse
from torch.nn.functional import log_softmax
from functools import partial
from src.utils import *
from src.model import *
from dataGeneration.validity_check import *
from src.datasets import *
from src.gaussian_diffusion import *
from src.params import *

# Clear any cached CUDA state at import time.
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser configuration.  Users can override the default
# guidance scale, number of samples to generate, and the checkpoint
# path via the command line.  Defaults mirror the original script.
parser = argparse.ArgumentParser(description='Sample equations using diffusion model')
parser.add_argument('--cfg_scale', type=float, default=0.0,
                    help='Classifier-free guidance scale (default: 0.0 for unconditional sampling)')
parser.add_argument('--num_samples', type=int, default=200,
                    help='Number of samples to generate (default: 200)')
parser.add_argument('--model_checkpoint', type=str, default='model_checkpoints/model_checkpoint.pth',
                    help='Path to model checkpoint (default: model_checkpoints/model_checkpoint.pth)')
args = parser.parse_args()

# Resolve important paths based on the provided arguments and constants
model_checkpoint_path = args.model_checkpoint
inv_target_file = f'data/inv_design_target/inv_target.csv'
train_dataset_path = f'data/dataset/train_dataset.pth'
outdir = f"generation_results"
if os.path.exists(outdir):
    # Remove any previous results directory to ensure clean outputs
    os.system(f'rm -r {outdir}')
os.makedirs(outdir, exist_ok=True)
log_output_file = f'{outdir}/sample_output.txt'

# Number of samples to draw from the diffusion model
num_samples = args.num_samples

# Import parameters from src.params.  seq_len defines the sequence
# length for the transformer model.  num_beams controls how many
# beam-search candidates we keep per sample when decoding from logits.
sequence_length = seq_len
num_beams = 1
cfg_scale = args.cfg_scale

# Diffusion parameters.  These mirror the defaults used during
# training; feel free to adjust them if experimenting with different
# diffusion schedules.
noise_scheduler_type = 'sqrt'
diffusion_steps = 2000
predict_xstart = True
sigma_small = False
training_mode = 'e2e'
rescale_timesteps = True
learn_sigmas = False
use_kl = False
rescale_learned_sigmas = False

# Load training dataset and corresponding normalizer
dataset, label_normalizer = get_dataset(data_path, return_normalizer=True)

def setup_logger(log_file: str) -> None:
    """
    Initialize a simple file-based logger that appends messages to
    ``log_file``.  Logs are written in plain text with no extra
    formatting beyond the message itself.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    """
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        filemode='w',
                        format='%(message)s')


def evaluate_validity_generated_samples(model_checkpoint_path: str, save_idx: int | None = None, num_samples: int = 100) -> None:
    """
    Generate symbolic equations using a diffusion model and evaluate their validity.

    This helper wraps the full sampling and evaluation pipeline:
    1. Loads the target conditioning coefficients and normalises them.
    2. Restores the transformer diffusion model from a checkpoint.
    3. Samples latent representations at multiple diffusion timesteps and decodes
       them into strings using the tokenizer.
    4. Filters out invalid equations based on dependency checks and mesh validity.
    5. Writes valid and invalid equations to separate CSV files and logs
       summary statistics (validity, uniqueness, novelty).

    Parameters
    ----------
    model_checkpoint_path : str
        Path to the pre-trained model checkpoint (.pth file).
    save_idx : int or None, optional
        Optional index to tag output files.  Currently unused.
    num_samples : int, optional
        Number of samples to draw from the diffusion model.  Defaults to 100.
    """
    start = time.time()

    setup_logger(log_output_file)
    logging.info(f"Processing model: {model_checkpoint_path}")

    tokenizer = EquationTokenizer()

    # Read the inverse design target and replicate across the requested number
    # of samples.  Then normalise using the dataset's label normaliser.
    target_c = np.genfromtxt(inv_target_file, delimiter=',').reshape(1, -1)
    target_c = np.tile(target_c, (num_samples, 1))
    target_c = torch.from_numpy(target_c).float().to(device)
    if len(target_c) < num_samples:
        num_samples = len(target_c)
    else:
        target_c = target_c[:num_samples, :]
    target_c = label_normalizer.normalize(target_c)

    # Load the training dataset.  In PyTorch 2.6 and later, torch.load
    # defaults to weights_only=True for security.  Since our dataset is
    # a pickled Subset object, we must explicitly pass weights_only=False.
    if not os.path.exists(train_dataset_path):
        raise FileNotFoundError(f"Training dataset path {train_dataset_path} does not exist")
    train_dataset = torch.load(train_dataset_path, weights_only=False)
    full_dataset = train_dataset.dataset
    subset_indices = train_dataset.indices
    # Get the tokens from the full dataset and index with the subset
    training_tokens = full_dataset.tensors[0][subset_indices]
    training_tokens = training_tokens.cpu().numpy()

    # Verify that the model checkpoint exists before loading.  If the
    # checkpoint comes from older versions of src, patch module names
    # accordingly to allow the pickle to be deserialised.
    if not os.path.exists(model_checkpoint_path):
        raise ValueError(f"Model checkpoint path {model_checkpoint_path} does not exist")
    else:
        import sys
        import src.model as model_module
        import src.transformer_utils as transformer_utils
        import src.params as params
        import src.gaussian_diffusion as gaussian_diffusion
        import src.utils as utils
        import src.datasets as datasets
        # Map old module names to new src locations
        sys.modules['model'] = model_module
        sys.modules['transformer_utils'] = transformer_utils
        sys.modules['params'] = params
        sys.modules['gaussian_diffusion'] = gaussian_diffusion
        sys.modules['utils'] = utils
        sys.modules['datasets'] = datasets
        # Load the model from the checkpoint with weights_only disabled.
        model = torch.load(model_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        model = model.to(device)
        model.eval()
        model.cfg_scale = cfg_scale

        valid_output_filename = f"{outdir}/valid_sample_equations.csv"
        invalid_output_filename = f"{outdir}/invalid_sample_equations.csv"

        # Initialise diffusion process according to configuration.
        betas = get_named_beta_schedule(noise_scheduler_type, diffusion_steps)
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, section_counts='ddim' + str(diffusion_steps)),
            betas=betas,
            rescale_timesteps=rescale_timesteps,
            training_mode=training_mode,
            predict_xstart=predict_xstart,
            learn_sigmas=learn_sigmas,
            use_kl=use_kl,
            rescale_learned_sigmas=rescale_learned_sigmas,
        )
        # Shape for the latent samples: (num_samples, seq_len, embedding_dim)
        sample_shape = (num_samples, seq_len, model.word_embedding.weight.shape[-1])

        # Sample the diffusion process once to capture intermediate timesteps.  The
        # returned tensor has shape (timesteps, num_samples, seq_len, embed_dim).
        all_samples = diffusion.p_sample_loop(model, sample_shape, cond=target_c, langevin_fn=None)

        # Define which timesteps to save.  The final timestep is denoted by -1.
        save_timesteps_percentage = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        save_timesteps = [round(len(all_samples) * p) for p in save_timesteps_percentage]
        save_timesteps[-1] = -1

        intermediate_results = []
        for t in save_timesteps:
            sample_t = all_samples[t]
            logits_t = model.get_logits(sample_t)
            log_probs_t = log_softmax(logits_t, dim=-1)
            cands_t = torch.topk(log_probs_t, k=num_beams, dim=-1)
            decoded_sentences_t: list[str] = []
            for seq_list in cands_t.indices:
                for k in range(num_beams):
                    seq = seq_list[:, k]
                    _, decoded_sentence = tokenizer.decode(seq.tolist())
                    decoded_sentences_t.append(decoded_sentence)
            intermediate_results.append({
                'timestep': t,
                'decoded_sentences': decoded_sentences_t
            })
            # Persist decoded sequences for later inspection
            os.makedirs(f"{outdir}/intermediate_timestep_samples", exist_ok=True)
            with open(f"{outdir}/intermediate_timestep_samples/sample_timestep_{t}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for eq in decoded_sentences_t:
                    writer.writerow([eq])

        # Sample once more to get the final timestep sample
        sample = diffusion.p_sample_loop(model, sample_shape, cond=target_c, langevin_fn=None)
        sample = sample[-1]  # extract the last timestep

        logits = model.get_logits(sample)
        # Use classifier-free guidance if configured
        log_probs = log_softmax(logits, dim=-1)
        cands = torch.topk(log_probs, k=num_beams, dim=-1)

        decoded_sentences: list[str] = []
        decoded_sequences = []
        for seq_list in cands.indices:
            for k in range(num_beams):
                seq = seq_list[:, k]
                decoded_sequence, decoded_sentence = tokenizer.decode(seq.tolist())
                decoded_sentences.append(decoded_sentence)
                decoded_sequences.append(decoded_sequence)

        # Expand the sample and conditioning labels to match the number of beams
        sample = torch.repeat_interleave(sample, num_beams, dim=0)
        target_c = torch.repeat_interleave(target_c, num_beams, dim=0)

        valid_eq: list[str] = []
        invalid_eq: list[str] = []
        valid_eq_target_c: list[np.ndarray] = []

        for i in range(len(decoded_sentences)):
            eq = decoded_sentences[i].lstrip('+')
            try:
                is_dependent = is_equation_dependent_on_xyz(eq)
                is_valid = is_mesh_valid(eq)
            except Exception:
                is_dependent = False
                is_valid = False
            if is_dependent and is_valid:
                valid_eq.append(eq)
                valid_eq_target_c.append(label_normalizer.unnormalize(target_c[i].cpu().detach().numpy()))
            else:
                invalid_eq.append(eq)

        # Write invalid equations to file
        with open(invalid_output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for eq in invalid_eq:
                writer.writerow([eq])

        if len(valid_eq) == 0:
            logging.info("*" * 70)
            logging.info(f"Model loaded from {model_checkpoint_path}")
            logging.info("No valid samples")
        else:
            # Encode the valid equations and compute uniqueness/novelty metrics
            generated_tokens = tokenizer.encode(valid_eq)
            unique_ratio = len(np.unique(generated_tokens, axis=0)) / len(generated_tokens)
            training_set = set(tuple(token) if isinstance(token, (list, np.ndarray)) else token for token in training_tokens)
            generated_set = set(tuple(token) if isinstance(token, list) else token for token in generated_tokens)
            novelty_score = len(generated_set.difference(training_set)) / len(generated_tokens)
            unique_idx = np.unique(generated_tokens, axis=0, return_index=True)[1]
            # Save the unnormalised conditioning values corresponding to unique equations
            np.savetxt(f'{outdir}/valid_target_c.csv', np.array(valid_eq_target_c)[unique_idx, :], delimiter=",")
            unique_valid_eq = np.array(valid_eq)[unique_idx]
            with open(valid_output_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for eq in unique_valid_eq:
                    writer.writerow([eq])
            end = time.time()
            # Log summary statistics
            logging.info("*" * 70)
            logging.info(f"Model loaded from: {model_checkpoint_path}")
            logging.info(f"Number of samples = {num_samples}")
            logging.info(f"Validity score = {len(valid_eq)/num_samples*100: .2f}%")
            logging.info(f"Unique score = {unique_ratio*100: .2f}%")
            logging.info(f"Novelty score = {novelty_score*100: .2f}%")
            logging.info(f"Generation time = {end - start: .2f} s")
            logging.info("*" * 70)


# Invoke the evaluation if executed as a script
if __name__ == '__main__':
    evaluate_validity_generated_samples(model_checkpoint_path, num_samples=num_samples)