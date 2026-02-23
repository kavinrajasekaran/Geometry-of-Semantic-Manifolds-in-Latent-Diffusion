# The Geometry of Latent Diffusion Models
## Project Plan & Class Deliverables

### 1. 3-Minute Presentation Outline
**Slide 1: What is the Geometry of Latent Diffusion? (Minute 1)**
- *The Problem*: Generative AI (like Stable Diffusion) seems like a "black box" that magically turns text into high-quality images.
- *The Concept*: Instead of pixels, models operate in a "Latent Space." We are investigating the semantic manifold—how abstract concepts (e.g., cats vs. cars) separate and cluster dynamically over time.
- *The Hook*: We visualize the exact moment the neural network decides an image is a "cat" and not a "car."

**Slide 2: How Will We Execute It? (Minute 2)**
- *Framework*: Use `diffusers` (Hugging Face) to run `runwayml/stable-diffusion-v1-5`.
- *Method*: Intercept the U-Net bottleneck tensors at specific denoising timesteps (e.g., t=50, 40, 30, 20, 10, 0).
- *Analysis*: Apply PCA to compress massive 16,384-dimensional tensors, then use t-SNE and UMAP to reduce the data down to 2D.
- *Visuals*: Code a custom visualization pipeline using `matplotlib` and `seaborn`.

**Slide 3: Why Does It Matter? (Minute 3)**
- *Interpretability*: Sheds light on the inner workings of black-box models, satisfying a core goal of unsupervised learning.
- *Geometry of Generation*: Shows *trajectories* (how unstructured noise crystallizes into a structured semantic concept).
- *Impact*: By tracking clustering, we can measure how quickly the model commits to a semantic class, offering insights into latent consistency and generation efficiency.

---

### 2. 1-Page Proposal Outline
**1. Problem Formulation**
- *Objective*: To understand the topological structure of the latent space in diffusion models during the denoising process.
- *Question*: How and when do distinct semantic clusters (e.g., landscapes vs. portraits) diverge from pure isotropic Gaussian noise?
- *Scope*: Limit the study to 5 distinct semantic classes (cats, cars, cities, landscapes, portraits) with multiple prompts per class.

**2. Methodology**
- *Data/Pipeline Setup*: Use PyTorch and the Hugging Face `diffusers` library to orchestrate Stable Diffusion v1.5. Write a custom callback mechanism to intercept and extract latent tensors from the U-Net across identical seeds and prompts.
- *Dimensionality Reduction Pipeline*: 
  1. Flatten latents from $(1, 4, 64, 64)$ to $16,384$-D spatial vectors.
  2. Apply Principal Component Analysis (PCA) to project down to $\approx 50$ dimensions (reducing noise and improving stability).
  3. Apply t-SNE and UMAP to project these 50 dimensions down to 2D.
- *Visualization Strategy*: Plot 2D embeddings, color-coded by semantic class labels. Produce both distinct timestep snapshots and trajectory plots over the 50 inference steps.

**3. Expected Results**
- Early timesteps (high noise) will show indistinguishable overlap among all semantic classes.
- A sudden phase transition/divergence where classes snap into isolated clusters.
- Trajectory plots will reveal clear paths migrating from a central noise distribution towards isolated semantic attractors.

---

### 3. Task Delegation (4-Person Group)
To ensure equal and explicit contributions for the 12-page final report:

- **Person 1: Data & Pipeline Architecture** Create `config.py` and `extractor.py`. Responsible for writing the `diffusers` PyTorch hooks, intercepting the latent variables via step callbacks, flattening the tensors, handling GPU memory, and saving the massive `.npz` arrays.
- **Person 2: Dimensionality Reduction (Unsupervised Learning)** Create `reducer.py`. Responsible for the mathematical pipeline: ensuring proper scaling/normalization, implementing PCA, and tuning perplexity/neighbor hyperparameters for t-SNE and UMAP to accurately preserve local/global geometry.
- **Person 3: Visualization & Analytics** Create `visualizer.py`. Responsible for using `seaborn` and `matplotlib` to convert the 2D arrays into publication-ready figures. Will code both the static scatter plots per timestep and the complex line-trajectory maps connecting points over time.
- **Person 4: Theory, Evaluation, & Writing (Lead Editor)** Responsible for framing the 12-page report, reviewing the visual analytics, synthesizing the findings into the context of Unsupervised Learning, writing the introduction/conclusion, and formatting the references.
