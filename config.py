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
        "An orange tabby cat sitting on a sunny windowsill",
        "A black cat sitting on a wooden fence",
        "A Siamese cat lounging on a couch",
        "A Persian cat with bright green eyes",
        "Two cats sleeping next to each other",
        "A cat stretching on a rug in a living room"
    ],
    "car": [
        "A red sports car driving rapidly down the street",
        "A vintage classic car parked in a garage",
        "A sleek modern electric vehicle on the highway",
        "A yellow taxi cab in traffic",
        "A white SUV parked in a driveway",
        "A blue convertible on a coastal road",
        "A black luxury sedan at a dealership",
        "A green muscle car at a car show"
    ],
    "city": [
        "A bustling city street at night with neon lights",
        "A futuristic cyberpunk metropolis in the rain",
        "A wide shot of the New York skyline at dusk",
        "An aerial view of downtown Tokyo at night",
        "A busy intersection in London with double decker buses",
        "A panoramic view of Chicago skyscrapers",
        "A crowded city market in Hong Kong",
        "A rainy evening on a city boulevard with street lights"
    ],
    "landscape": [
        "A serene mountain landscape at sunrise",
        "A beautiful sandy beach sunset with palm trees",
        "A wide open green valley with a flowing river",
        "A snowy mountain peak under a clear blue sky",
        "A misty forest with tall pine trees",
        "A vast desert with sand dunes at golden hour",
        "A waterfall in a lush tropical jungle",
        "A calm lake reflecting autumn foliage"
    ],
    "portrait": [
        "A detailed portrait of a person with dramatic studio lighting",
        "A close-up face shot of a young woman smiling",
        "A black and white professional portrait of a man",
        "A headshot of an elderly woman with wrinkles",
        "A portrait of a child with freckles laughing",
        "A moody portrait of a musician holding a guitar",
        "A corporate headshot of a businessman in a suit",
        "A candid portrait of a street artist painting"
    ]
}

# ─── File Paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
