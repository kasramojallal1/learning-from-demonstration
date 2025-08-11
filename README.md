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
git clone https://github.com/yourusername/learning-from-demonstration.git
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
├── training/
│   ├── sft_prepare.py   # Dataset preparation for fine-tuning
│   ├── train_lora.py    # LoRA/QLoRA training script
│   └── utils.py
│
├── main.py           # Entry point for recording demonstrations
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

### 2️⃣ Replay Demonstrations
Visualize recorded placements:
```bash
python replay.py --file data/demos/binpack_lfd.jsonl
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
python training/train_lora.py     --model meta-llama/Llama-3.2-1B     --dataset_dir data/processed     --output_dir models/llama3.2-qlora
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
3. **Fine-Tune** – Train with `train_lora.py` using QLoRA for memory efficiency.
4. **Deploy** – Use the fine-tuned model to generate packing paths for new boxes.

---

## 🖼 Example Demonstrations
<img width="960" height="720" alt="placement_0001" src="https://github.com/user-attachments/assets/bf5f7385-06b3-4aac-adef-3e736e99747a" />
<img width="960" height="720" alt="placement_0002" src="https://github.com/user-attachments/assets/cd2aca7e-7005-400a-8d67-4a993343b421" />
<img width="960" height="720" alt="placement_0003" src="https://github.com/user-attachments/assets/b4074181-f95d-46f6-a0ad-c62875197fad" />
<img width="960" height="720" alt="placement_0004" src="https://github.com/user-attachments/assets/5dd4a6c4-0f40-4f8d-b0e0-3af9c20a73cf" />
<img width="960" height="720" alt="placement_0005" src="https://github.com/user-attachments/assets/0dcd514c-4dc3-4641-a1a1-91167dec73f7" />
<img width="960" height="720" alt="placement_0006" src="https://github.com/user-attachments/assets/2b9e479f-3bff-494f-b352-e8f732b65906" />
<img width="960" height="720" alt="placement_0007" src="https://github.com/user-attachments/assets/d859161b-aa1e-4ae7-a26e-98888e358223" />
<img width="960" height="720" alt="placement_0008" src="https://github.com/user-attachments/assets/1c754691-cd99-418f-96fb-c781e1aa7c1a" />
<img width="960" height="720" alt="placement_0009" src="https://github.com/user-attachments/assets/4c2f2b81-606a-4bc0-8b26-0b80941ddd93" />


