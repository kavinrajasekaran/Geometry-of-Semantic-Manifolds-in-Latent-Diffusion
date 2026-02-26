"""
extractor.py — Intercepts and saves latent tensors from Stable Diffusion.

We extract TWO representations at each tracked denoising step:
  1. The denoising spatial latent (1, 4, 64, 64) — encodes image layout
  2. The U-Net mid-block bottleneck feature — encodes semantic meaning
"""

import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import config

def run_extraction():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {config.MODEL_ID} on {device}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(config.MODEL_ID, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    extracted_data = []

    # Optional: ask user if they want to add a custom prompt for testing
    add_prompt = input("Do you want to add a custom prompt for testing? (y/n, leave blank for no): ").strip().lower()
    if add_prompt == 'y' or add_prompt == 'yes':
        user_prompt = input("What is your custom prompt?: ").strip()
        if user_prompt:
            user_cat = input("What semantic category should this fall under? (e.g. 'custom'): ").strip()
            if not user_cat:
                user_cat = "custom"
                
            if user_cat in config.TEXT_PROMPTS:
                config.TEXT_PROMPTS[user_cat].append(user_prompt)
            else:
                config.TEXT_PROMPTS[user_cat] = [user_prompt]
            print(f"Added custom prompt '{user_prompt}' under category '{user_cat}'.\n")

    # ─── Hook into U-Net mid-block to capture semantic bottleneck features ────
    # The mid_block is the deepest layer of the U-Net. It processes the most
    # abstract, semantically compressed representation of the image.
    # This is where "cat-ness" vs "car-ness" is encoded, NOT in the spatial latent.
    mid_block_output = {}
    
    def mid_block_hook(module, input, output):
        mid_block_output["feat"] = output.detach().cpu().numpy()
    
    hook_handle = pipe.unet.mid_block.register_forward_hook(mid_block_hook)
    
    print("Starting prompt generation and latent extraction...")
    for class_label, prompts in config.TEXT_PROMPTS.items():
        print(f"\n→ Processing class: [{class_label.upper()}]")
        
        for prompt in prompts:
            print(f"    Generating: '{prompt}'")
            
            # NOTE: class_label and prompt are bound via default args to avoid Python closure bug
            def callback_on_step_end(pipeline, step_index, timestep, callback_kwargs,
                                     _label=class_label, _prompt=prompt):
                if step_index in config.TRACKING_INTERVALS:
                    latents_tensor = callback_kwargs["latents"]
                    
                    # 1. Save the intermediate image visually
                    img_dir = os.path.join(config.OUTPUT_DIR, "images", _label)
                    os.makedirs(img_dir, exist_ok=True)
                    
                    with torch.no_grad():
                        scaled_latents = latents_tensor / pipeline.vae.config.scaling_factor
                        image_tensor = pipeline.vae.decode(scaled_latents, return_dict=False)[0]
                        image_pil = pipeline.image_processor.postprocess(image_tensor, output_type="pil")[0]
                        
                        safe_prompt = "".join([c if c.isalnum() else "_" for c in _prompt])[:20]
                        img_filename = f"{safe_prompt}_step_{step_index:02d}.png"
                        image_pil.save(os.path.join(img_dir, img_filename))
                    
                    # 2. Extract spatial latent (image layout)
                    latents_np = latents_tensor.detach().cpu().numpy()
                    
                    # 3. Extract U-Net mid-block semantic bottleneck
                    bottleneck_np = mid_block_output.get("feat", None)
                    
                    # 4. Store flattened image pixels
                    image_np = np.array(image_pil).flatten()
                    
                    extracted_data.append({
                        "step": step_index,
                        "timestep": timestep.item(),
                        "class_label": _label,
                        "prompt": _prompt,
                        "latents": latents_np,
                        "bottleneck": bottleneck_np,
                        "image": image_np
                    })
                return callback_kwargs

            generator = torch.Generator(device).manual_seed(42)
            
            with torch.no_grad():
                _ = pipe(
                    prompt, 
                    num_inference_steps=config.NUM_INFERENCE_STEPS,
                    generator=generator,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"]
                )
    
    # Remove the hook after extraction is done
    hook_handle.remove()
                
    # ─── Compile and Save ────────────────────────────────────────────────────
    embeddings = []
    bottlenecks = []
    images = []
    labels = []
    steps = []
    
    for item in extracted_data:
        flat_latent = item["latents"].flatten()
        embeddings.append(flat_latent)
        
        if item["bottleneck"] is not None:
            bottlenecks.append(item["bottleneck"].flatten())
        
        images.append(item["image"])
        labels.append(item["class_label"])
        steps.append(item["step"])
        
    embeddings = np.array(embeddings)
    images = np.array(images)
    labels = np.array(labels)
    steps = np.array(steps)
    
    save_dict = {
        "embeddings": embeddings,
        "images": images, 
        "labels": labels,
        "steps": steps
    }
    
    if bottlenecks:
        save_dict["bottlenecks"] = np.array(bottlenecks)
        print(f"  U-Net bottleneck shape: {save_dict['bottlenecks'].shape}")
    
    out_file = os.path.join(config.OUTPUT_DIR, "extracted_latents.npz")
    np.savez(out_file, **save_dict)
    
    print(f"\n✓ Extraction complete.")
    print(f"  Total latent snapshots collected: {len(embeddings)}")
    print(f"  Spatial latent shape: {embeddings.shape} (Flattened VAE latents)")
    print(f"  Data saved to: {out_file}")

if __name__ == "__main__":
    run_extraction()

