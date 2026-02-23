"""
extractor.py — Intercepts and saves latent tensors from Stable Diffusion.
"""

import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import config

def run_extraction():
    # Automatically select GPU / MPS / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {config.MODEL_ID} on {device}...")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(config.MODEL_ID, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)  # Keep logs clean
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # We will accumulate items of form: {"step": int, "label": str, "prompt": str, "latents": ndarray}
    extracted_data = [] 

    print("Starting prompt generation and latent extraction...")
    for class_label, prompts in config.TEXT_PROMPTS.items():
        print(f"\n→ Processing class: [{class_label.upper()}]")
        
        for prompt in prompts:
            print(f"    Generating: '{prompt}'")
            
            # The callback triggered by diffusers at the end of each denoising block
            def callback_on_step_end(pipeline, step_index, timestep, callback_kwargs):
                if step_index in config.TRACKING_INTERVALS:
                    # 'latents' shape is (batch_size, channels, height, width) -> e.g. (1, 4, 64, 64)
                    latents_tensor = callback_kwargs["latents"]
                    
                    # 1. Save the intermediate image visually
                    img_dir = os.path.join(config.OUTPUT_DIR, "images", class_label)
                    os.makedirs(img_dir, exist_ok=True)
                    
                    with torch.no_grad():
                        scaled_latents = latents_tensor / pipeline.vae.config.scaling_factor
                        image_tensor = pipeline.vae.decode(scaled_latents, return_dict=False)[0]
                        image_pil = pipeline.image_processor.postprocess(image_tensor, output_type="pil")[0]
                        
                        # Create a safe filename using the first 20 chars of the prompt
                        safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:20]
                        img_filename = f"{safe_prompt}_step_{step_index:02d}.png"
                        image_pil.save(os.path.join(img_dir, img_filename))
                    
                    # 2. Extract latent data for clustering
                    # Detach from graph, move to CPU, convert to numpy
                    latents_np = latents_tensor.detach().cpu().numpy()
                    
                    extracted_data.append({
                        "step": step_index,
                        "timestep": timestep.item(),
                        "class_label": class_label,
                        "prompt": prompt,
                        "latents": latents_np
                    })
                return callback_kwargs

            # Run inference explicitly in eval mode without calculating gradients
            with torch.no_grad():
                _ = pipe(
                    prompt, 
                    num_inference_steps=config.NUM_INFERENCE_STEPS,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"]
                )
                
    # ─── Compile and Save ────────────────────────────────────────────────────
    embeddings = []
    labels = []
    steps = []
    
    for item in extracted_data:
        # Flatten the spatial volume (1, 4, 64, 64) to a 1D vector of length 16,384
        flat_latent = item["latents"].flatten()
        embeddings.append(flat_latent)
        labels.append(item["class_label"])
        steps.append(item["step"])
        
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    steps = np.array(steps)
    
    out_file = os.path.join(config.OUTPUT_DIR, "extracted_latents.npz")
    np.savez(out_file, embeddings=embeddings, labels=labels, steps=steps)
    
    print(f"\n✓ Extraction complete.")
    print(f"  Total latent snapshots collected: {len(embeddings)}")
    print(f"  Shape of embeddings array: {embeddings.shape} (Flattened vectors)")
    print(f"  Data saved to: {out_file}")

if __name__ == "__main__":
    run_extraction()
