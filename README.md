# 🎓 Adaptive_Tutor_RL

A hybrid deep learning framework integrating Generative Adversarial Networks (GANs) and Reinforcement Learning (PPO) for adaptive text complexity enhancement, aligned with CEFR standards for second language acquisition.

---

## 📌 Overview

**Adaptive_Tutor_RL** is an intelligent language tutoring environment that dynamicallly adapts language learning content to match a student's Proficiency level according to the **CEFR (Common European Framework of Reference for Languages)** standard.

By leveraging a **Qwen-1.8B-Chat** generative language model fine-tuned via **Proximal Policy Optimization (PPO)** alongside a **RoBERTa-based linguistic discriminator**, the system generates tailored target-word explanations while enforcing grammatical accuracy, semantic coherence, and pedagogical suitability.

---

## 🏗️ Core Architecture & System Components

1. **Generative Agent (Policy Model):**
   * **Base Model:** `Qwen/Qwen1.5-1.8B-Chat`
   * **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT/LoRA) optimized using PPO to balance readability and target word insertion.

2. **Linguistic Discriminator:**
   * **Base Model:** `RoBERTa-aligned`
   * **Role:** Acts as a quality filter and reward component by detecting grammatical flaws and calculating linguistic penalties.

3. **Adaptive Environment (`AdaptiveTutorEnv`):**
   * A custom Reinforcement Learning environment implementing a **Finite State Machine (FSM)**.
   * Tracks student mastery via dynamic promotion (`up_streak`) and demotion (`down_streak`) thresholds to prevent reward hacking.

---

## 🎯 Composite Reward Function Breakdown

The PPO agent optimizes output text quality using a multi-objective composite reward function:

$$\text{Reward} = R_{\text{target}} + R_{\text{semantic}} + R_{\text{CEFR}} - P_{\text{linguistic}}$$

* **Target Word Insertion ($R_{\text{target}}$):** Rewards natural inclusion of the intended vocabulary word in generated sentences.
* **Semantic Similarity ($R_{\text{semantic}}$):** Evaluates context preservation using cosine similarity over embeddings.
* **CEFR Readability Shift ($R_{\text{CEFR}}$):** Aligns output readability metrics to the active student proficiency level.
* **Linguistic Penalty ($P_{\text{linguistic}}$):** Deducts points for ungrammatical structures evaluated by RoBERTa.

---

## 📁 Repository Structure

```text
Adaptive_Tutor_RL/
├── app_gradio.py             # Main Gradio FSM web interface
├── requirements.txt          # Python environment dependencies
├── .gitignore                # Protects heavy model weights and local logs
├── README.md                 # Project documentation
├── notebooks/                # Training and exploration notebooks
│   ├── Adaptive_tutor_RL.ipynb
│   └── Dialog5000.ipynb
├── src/                      # Source code modules
│   └── rl/
│       └── env.py            # AdaptiveTutorEnv implementation
└── data/                     # Dataset structures
    └── processed/            # Preprocessed training datasets


## ⚙️ Installation & Usage
1. Clone the Repository
Bash
git clone [https://github.com/Fadi-Toumat/Adaptive_Tutor_RL.git](https://github.com/Fadi-Toumat/Adaptive_Tutor_RL.git)
cd Adaptive_Tutor_RL

---

## 2. Install Dependencies
Bash
pip install -r requirements.txt

---

## 3. Run the Adaptive Web Interface
Bash
python app_gradio.py
---

## 📦 Model Weights & Access Strategy
Due to storage constraints on GitHub, trained heavy model checkpoints (.safetensors / PEFT adapters) are hosted externally.
---

## Academic & Research Access: Model weights are available upon request for research and evaluation purposes.
---
## Contact: To request access to pre-trained checkpoints (Qwen PPO Adapter & RoBERTa Discriminator), please reach out to Fadi Toumat at Fadi.n.toumat@gmail.com.

👤 Author & Contact
Lead Researcher & Developer: Fadi Toumat

Email: Fadi.n.toumat@gmail.com

GitHub: @Fadi-Toumat
