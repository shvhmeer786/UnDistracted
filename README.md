Project Overview
UnDistracted is a computer vision project designed to detect distracted driving behaviors in real
time. By leveraging convolutional neural networks (CNNs) and deep learning frameworks, the
system analyzes in-cabin images and flags unsafe driving postures, helping enhance road
safety and prevent accidents.
Features
● Real-time detection: Processes camera frames on-the-fly to identify distracted
behaviors.
● High accuracy: Achieved a 99.5% test accuracy on held-out data.
● Scalable pipeline: Built with modular Python scripts for data preprocessing, model
training, and inference.
Dataset
● Collected and annotated over 100,000+ images of driver behavior
● Class labels include: safe_drive, texting_left, texting_right,
talking_on_phone, drinking, etc.
● Data split: 80% train, 10% validation, 10% test
Model Architecture
● Base model: Custom CNN with 5 convolutional layers
● Frameworks:
○ Training: PyTorch and Keras (TensorFlow backend)
○ Data augmentation: OpenCV for brightness, contrast, rotation, and noise injection
● Optimizer: Adam with learning rate 1e-4
● Loss: Categorical Cross-Entropy
Installation
Clone the repository
git clone https://github.com/<username>/UnDistracted.git
1. cd UnDistracted
Create a virtual environment
python3 -m venv venv
source venv/bin/activate # macOS/Linux
2. venv\Scripts\activate # Windows
3. Install dependencies
pip install -r requirements.txt
Usage
1. Data Preprocessing
python scripts/preprocess_data.py \
--input_dir data/raw_images \
--output_dir data/processed \
--img_size 224
2. Training
python scripts/train.py \
--config configs/train_config.yaml
3. Evaluation
python scripts/evaluate.py \
--model_path experiments/<run_id>/model.pt \
--test_data data/processed/test
4. Real-time Inference Demo
python scripts/infer_camera.py \
--model_path experiments/<run_id>/model.pt \
--camera_id 0
Training
● Configurations stored in configs/train_config.yaml (batch size, epochs, learning
rate)
● Checkpoints saved under experiments/ with timestamped folders
Evaluation & Results
● Test Accuracy: 99.5%
● Confusion Matrix: See reports/confusion_matrix.png
● ROC Curves: reports/roc_curves.png
Real-time Inference
The infer_camera.py script captures frames from a webcam and overlays predicted labels
and confidence scores on the video stream.
Project Structure
UnDistracted/
├── configs/ # YAML config files
├── data/ # Dataset folders
│ ├── raw_images/
│ └── processed/
├── experiments/ # Model checkpoints & logs
├── reports/ # Plots and evaluation artifacts
├── scripts/ # Utility scripts (preprocess, train, evaluate, infer)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # MIT License
Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (git checkout -b feature/YourFeature)
3. Commit your changes (git commit -m 'Add YourFeature')
4. Push to branch (git push origin feature/YourFeature)
5. Open a Pull Request
License
This project is licensed under the MIT License.
