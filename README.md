# 📦 Learning from Demonstration: 3D Bin Packing with LLM Guidance

This project implements a **3D bin packing simulation** powered by **Learning from Demonstrations (LfD)** and **LLM guidance**.  
It allows you to **record demonstrations**, **process them into training-ready datasets**, and **fine-tune a model** using **LoRA / QLoRA** for efficient training on limited hardware.

---

## 🚀 Features
- **3D Bin Packing Environment** – Simulates placing boxes of various sizes into a bin.
- **LLM-Generated Placement Paths** – Use GPT-based models to generate packing strategies.
- **Demonstration Recording** – Save human or LLM-generated runs for training.
- **Dataset Processing** – Convert raw demonstrations into tokenized datasets.
- **LoRA & QLoRA Fine-Tuning** – Efficiently fine-tune large language models on recorded data.
- **Replay Mode** – Visualize saved runs for debugging or presentations.

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kasramojallal1/learning-from-demonstration.git
cd learning-from-demonstration
```

### 2️⃣ Create a Conda Environment
```bash
conda create -n lfd python=3.11
conda activate lfd
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure
```
learning-from-demonstration/
│
├── data/
│   ├── demos/        # Recorded demonstration datasets
│   │   └── binpack_lfd.jsonl   # Main dataset (JSON Lines)
│   ├── runs/         # Individual demonstration run logs
│   └── processed/    # Training-ready datasets
│
├── envs/
│   └── bin_env.py    # 3D bin-packing simulation environment
│
├── lfd/
│   └── logger.py     # Demonstration logger (JSON Lines)
│
├── training/
│   ├── sft_prepare.py               # Dataset preparation for fine-tuning
│   ├── train_lora_llama32_3b.py     # LoRA/QLoRA fine-tuning (Llama 3.2 3B, the paper's planner)
│   ├── train_lora_llama32_3b_nvidia_gpu.py
│   ├── train_lora_llama31_8b.py
│   └── train_lora_gemma2_2b.py
│
├── utils/
│   └── geometry.py   # Rotation and placement geometry
│
├── main.py           # Entry point for recording demonstrations
├── main_datasets.py  # Same, on the paper's three box-sequence datasets
└── README.md
```

---

## 🖥 Usage

### 1️⃣ Record Demonstrations
Run the main script to generate and save demonstrations:
```bash
python main.py
```
This will produce a `binpack_lfd.jsonl` file in `data/demos/`.

---

### 2️⃣ Record on the Paper's Datasets
Same recorder, but boxes are drawn from the three box-sequence datasets used in the paper comparison:
```bash
python main_datasets.py --mode paper:data1 --bin 10 --placements 20 --seed 123   # also paper:data2, paper:data3
```

---

### 3️⃣ Prepare Dataset for Training
Convert raw demonstrations into a processed dataset:
```bash
python training/sft_prepare.py
```
Output will be saved in `data/processed/`.

---

### 4️⃣ Fine-Tune with LoRA / QLoRA
Example for QLoRA fine-tuning:
```bash
python training/train_lora_llama32_3b.py --model meta-llama/Llama-3.2-3B --dataset_dir data/processed --output_dir models/llama3.2-qlora
```

---

## 🔑 Hugging Face Authentication
If your model requires authentication:
```bash
huggingface-cli login
```
Make sure you have a **valid read token** from Hugging Face.

---

## 📊 Example Workflow
1. **Record** – Use `main.py` to create bin packing runs.
2. **Process** – Run `sft_prepare.py` to tokenize and store data.
3. **Fine-Tune** – Train with `training/train_lora_llama32_3b.py` using QLoRA for memory efficiency.
4. **Deploy** – Use the fine-tuned model to generate packing paths for new boxes.

---

## 📊 Results

This is the training half of **Packi**; the simulation, the hosted-model baselines and the dataset runner live in the companion repo, [llm-robotic-packer](https://github.com/kasramojallal1/llm-robotic-packer).

The **Llama 3.2 3B model fine-tuned here with LoRA/QLoRA (4-bit)** on recorded demonstrations reached **87% bin utilization** and **outperformed 11 proprietary API models** (including GPT-4o, GPT-5-mini and Claude 3.7 Sonnet) while running on a **single 16 GB consumer GPU**.

Paper: *Packi: Robotic 3D Bin Packing with LLMs Fine-tuned by Learning from Demonstration* (under review).

---

## 🖼 Example Demonstrations

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf5f7385-06b3-4aac-adef-3e736e99747a" width="30%" />
  <img src="https://github.com/user-attachments/assets/cd2aca7e-7005-400a-8d67-4a993343b421" width="30%" />
  <img src="https://github.com/user-attachments/assets/b4074181-f95d-46f6-a0ad-c62875197fad" width="30%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5dd4a6c4-0f40-4f8d-b0e0-3af9c20a73cf" width="30%" />
  <img src="https://github.com/user-attachments/assets/0dcd514c-4dc3-4641-a1a1-91167dec73f7" width="30%" />
  <img src="https://github.com/user-attachments/assets/2b9e479f-3bff-494f-b352-e8f732b65906" width="30%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d859161b-aa1e-4ae7-a26e-98888e358223" width="30%" />
  <img src="https://github.com/user-attachments/assets/1c754691-cd99-418f-96fb-c781e1aa7c1a" width="30%" />
  <img src="https://github.com/user-attachments/assets/4c2f2b81-606a-4bc0-8b26-0b80941ddd93" width="30%" />
</p>


