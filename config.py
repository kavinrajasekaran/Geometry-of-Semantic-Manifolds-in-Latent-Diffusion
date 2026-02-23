"""
config.py — Configuration parameters for the Latent Diffusion Geometry project.
"""

# ─── Model Settings ──────────────────────────────────────────────────────────
MODEL_ID = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 50

# Which inference step indices to intercept (0 = pure noise, 49 = final image)
TRACKING_INTERVALS = [0, 10, 20, 30, 40, 49]

# ─── Prompt Configurations ───────────────────────────────────────────────────
# We group prompts into 5 distinct semantic concepts to track their clustering.
TEXT_PROMPTS = {
    "cat": [
        "A high quality photo of a fluffy cat",
        "A cute kitten playing with yarn",
        "An orange tabby cat sitting on a sunny windowsill"
    ],
    "car": [
        "A red sports car driving rapidly down the street",
        "A vintage classic car parked in a garage",
        "A sleek modern electric vehicle on the highway"
    ],
    "city": [
        "A bustling city street at night with neon lights",
        "A futuristic cyberpunk metropolis in the rain",
        "A wide shot of the New York skyline at dusk"
    ],
    "landscape": [
        "A serene mountain landscape at sunrise",
        "A beautiful sandy beach sunset with palm trees",
        "A wide open green valley with a flowing river"
    ],
    "portrait": [
        "A detailed portrait of a person with dramatic studio lighting",
        "A close-up face shot of a young woman smiling",
        "A black and white professional portrait of a man"
    ]
}

# ─── File Paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
