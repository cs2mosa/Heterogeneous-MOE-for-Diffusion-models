import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np


class StabilityVAE:
    def __init__(self,
                 vae_name: str = "stabilityai/sd-vae-ft-mse",
                 batch_size=32,
                 # Standard SD Scaling Factor
                 scale_factor=0.18215,
                 # You want latents to have std=0.5 for your EDM config
                 target_std=0.5
                 ):
        self.vae_name = vae_name
        self._vae = None
        self.batch_size = int(batch_size)

        self.scale_factor = scale_factor
        self.target_std = target_std

        # Calculate the multiplier to go from VAE -> N(0, 0.5)
        # Standard VAE output is approx N(0, 1/0.18215).
        # We multiply by 0.18215 to get N(0, 1).
        # We multiply by 0.5 to get N(0, 0.5).
        self.enc_scaler = self.scale_factor * self.target_std

    def init(self, device):
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(self.vae_name).to(device)
            self._vae.requires_grad_(False)
            self._vae.eval()
        else:
            self._vae.to(device)

    def _run_vae_encoder_on_batch(self, pixel_batch):
        latent_dist = self._vae.encode(pixel_batch).latent_dist
        # Return mean and std from the distribution
        return latent_dist.mean, latent_dist.std

    def _run_vae_decoder_on_batch(self, raw_latent_batch):
        decoded = self._vae.decode(raw_latent_batch).sample
        return decoded

    @torch.no_grad()
    def encode(self, x):
        """
        Input: uint8 [0, 255]
        Output: float32 centered latents N(0, 0.5)
        """
        # [0, 255] -> [-1, 1]
        pixels_fp32 = (x.to(torch.float32) / 127.5) - 1.0

        latents_list = []
        # Process in batches to save RAM
        for batch in pixels_fp32.split(self.batch_size):
            mean, std = self._run_vae_encoder_on_batch(batch)
            # Reparameterization trick
            sampled = mean + torch.randn_like(mean) * std
            latents_list.append(sampled)

        raw_latents = torch.cat(latents_list, dim=0)

        # Scale to match EDM config (sigma_data=0.5)
        final_latents = raw_latents * self.enc_scaler

        return final_latents

    @torch.no_grad()
    def decode(self, x):
        """
        Input: float32 centered latents N(0, 0.5)
        Output: uint8 [0, 255]
        """
        self.init(x.device)

        # Inverse scaling: N(0, 0.5) -> VAE Native Space
        raw_latents = x / self.enc_scaler

        decoded_pixels_fp32 = torch.cat(
            [self._run_vae_decoder_on_batch(batch) for batch in raw_latents.split(self.batch_size)]
        )

        # [-1, 1] -> [0, 255]
        final_pixels = ((decoded_pixels_fp32 + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

        return final_pixels

class CLIP_EMBED:
    def __init__(self,
                 embed_name: str = "openai/clip-vit-large-patch14",
                 device: str = 'cuda'
                 ):
        self.embed_name = embed_name  # Store the name for reloading
        self.tokenizer = None
        self.text_encoder = None
        self.device = device

    def init(self):
        """Lazy initialization to be called on first use."""
        if self.tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.embed_name)
            self.text_encoder = CLIPTextModel.from_pretrained(self.embed_name,ignore_mismatched_sizes=True).to(self.device)
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

    def __getstate__(self):
        """Customize serialization to exclude the large models."""
        state = self.__dict__.copy()
        # Do not save the models, only their configuration name.
        state['tokenizer'] = None
        state['text_encoder'] = None
        return state

    def __setstate__(self, state):
        """Restore state during deserialization."""
        self.__dict__.update(state)

    @torch.no_grad()
    def encode_text(self, text_list):
        """
        Input: List of strings ["a dog", "a cat"]
        Output: Embeddings (B, 77, 768)
        """
        inputs = self.tokenizer(
            text_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)

        # Get hidden states (Sequence output)
        encoder_output = self.text_encoder(input_ids)
        return encoder_output.last_hidden_state