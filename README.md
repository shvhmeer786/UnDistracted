# UnDistracted

> Real-time distracted-driving detection powered by deep learning.

UnDistracted is a computer-vision pipeline that recognizes risky in-cabin behaviors (texting, phone use, drinking, etc.) from live camera feeds. The project combines modular data-preparation tooling, configurable training scripts, and a ready-to-run inference demo so you can reproduce results or adapt the system to your own fleet.

---

## Highlights

- **Production-minded workflow** with reproducible scripts for preprocessing, training, evaluation, and deployment.
- **High-accuracy CNN** that reaches 99.5% test accuracy on the held-out benchmark split.
- **Real-time inference demo** that overlays predictions and confidence scores directly on webcam footage.
- **Flexible configuration** via YAML files (augmentation, optimizer, training schedule, and more).

---

## Quick Start

Clone the project and install the Python dependencies inside a virtual environment:

```bash
git clone https://github.com/<username>/UnDistracted.git
cd UnDistracted

python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate  # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset

- 100k+ annotated driver images across multiple distraction classes.
- Default split: `80%` train / `10%` validation / `10%` test.
- Representative classes: `safe_drive`, `texting_left`, `texting_right`, `talking_on_phone`, `drinking`, `adjusting_radio`, and more.
- Customize or plug in your own dataset by matching the folder structure used in `data/raw_images`.

---

## Model & Training

| Component            | Details                                                                          |
| -------------------- | -------------------------------------------------------------------------------- |
| Backbone             | Custom CNN with five convolutional blocks + fully connected classifier head     |
| Frameworks           | Training in PyTorch & Keras (TensorFlow backend); augmentation via OpenCV        |
| Optimizer            | Adam (`lr=1e-4`) with categorical cross-entropy                                 |
| Data augmentation    | Brightness, contrast, rotation, random noise, and horizontal flips               |
| Configuration        | `configs/train_config.yaml` controls batch size, epochs, augmentations, logging  |
| Checkpointing        | Automatically persisted to `experiments/<timestamp>` for reproducibility         |

Run the end-to-end pipeline:

```bash
# 1) Preprocess and augment data
python scripts/preprocess_data.py \
  --input_dir data/raw_images \
  --output_dir data/processed \
  --img_size 224

# 2) Train the CNN (configurable hyperparameters)
python scripts/train.py \
  --config configs/train_config.yaml

# 3) Evaluate and export metrics
python scripts/evaluate.py \
  --model_path experiments/<run_id>/model.pt \
  --test_data data/processed/test
```

---

## Evaluation

- **Top-line metric:** 99.5% test accuracy on the reference split.
- **Artifacts:** confusion matrix (`reports/confusion_matrix.png`) and ROC curves (`reports/roc_curves.png`).
- **Logging:** training curves (loss, accuracy) saved per run inside `experiments/`.

Use the provided notebook `Using_AI_to_Detect_Distracted_Driving.ipynb` for deeper analysis, error inspection, and visualizations.

---

## Real-Time Inference Demo

Stream predictions from any webcam or video feed:

```bash
python scripts/infer_camera.py \
  --model_path experiments/<run_id>/model.pt \
  --camera_id 0
```

The script displays live annotated frames with class labels and confidence scores. Adjust `--camera_id` for multi-camera setups.

---

## Project Structure

```
UnDistracted/
├── configs/           # YAML configs for training & inference
├── data/              # Raw and processed datasets (create via preprocess script)
│   ├── raw_images/
│   └── processed/
├── experiments/       # Saved checkpoints, logs, and metrics
├── reports/           # Visual analytics (confusion matrix, ROC curves, etc.)
├── scripts/           # CLI utilities for preprocessing, training, evaluation, inference
├── Using_AI_to_Detect_Distracted_Driving.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Contributing

Contributions, bug reports, and feature ideas are welcome!

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit with a clear message (`git commit -m "Add your feature"`).
4. Push and open a pull request with context, screenshots, or logs.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.
