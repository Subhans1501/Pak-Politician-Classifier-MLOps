# Pakistani Politician Image Classification (MLOps Category A)

[![GitHub Repository](https://img.shields.io/badge/GitHub-View_Repository-blue?logo=github)](https://github.com/Subhans1501/Pak-Politician-Classifier-MLOps)

An end-to-end Machine Learning Operations (MLOps) pipeline that classifies facial images of 16 Pakistani public figures (15 politicians and 1 military spokesperson). Features automated dataset versioning (DVC), experiment tracking (MLflow), DAG orchestration (Apache Airflow), and automated CI/CD deployment via Docker and AWS EC2.

This project features fine-tuned deep learning models trained on a custom-scraped dataset, served via a FastAPI backend, and fully containerized for production deployment on an AWS server.

**Institution:** FAST National University of Computer and Emerging Sciences

## Contributors & Team Distribution
* **Subhan** (Lead) - *Data, Infrastructure & Integration*
* **Eeman Khalid** - *Modeling & Containerization*
* **Saad Shafqat** - *MLOps & Automation*

**Official Teaching Staff & TAs:**
* **Sir Asif Ameer** (@asif370)
* **Omer Farooq Khan** (@omerrfarooqq) - ANN-A1 TA
* **Aun Ali** (@Aun-Dev146) - ANN-A2 TA
* **Ahsan Butt** (@ahsan608) - MLOPS TA

---

## Features

* **Deep Learning Engine:** Utilizes fine-tuned ResNet-50 and EfficientNet-B0 architectures (unfreezing the top 30 layers) to capture specific facial features and hit ≥ 90% accuracy.
* **RESTful Backend:** FastAPI handles incoming image predictions, applies architecture-specific preprocessing math, and serves model inferences.
* **Automated Data Pipeline:** DVC tracks the massive image datasets and `.h5` model artifacts, keeping the Git repository lightweight and fast.
* **Robust MLOps Deployment:** Orchestrated by Apache Airflow, tracked by MLflow, containerized using Docker, and deployed via GitHub Actions to an AWS EC2 instance.

## Tech Stack

* **Machine Learning:** TensorFlow, Keras (`tf-keras`), OpenCV, Pillow, Numpy, Pandas
* **MLOps / DevOps:** DVC (Data Version Control), MLflow, Apache Airflow, Docker, GitHub Actions, AWS (Ubuntu EC2)
* **Backend API:** FastAPI, Uvicorn, Python `python-multipart`

## Project Structure

```text
├── app.py                 # FastAPI backend server
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration for AWS deployment
├── dvc.yaml               # DVC pipeline configuration
├── .github/
│   └── workflows/
│       └── deploy.yml     # CI/CD pipeline for GitHub Actions
├── dags/
│   └── mlops_pipeline.py  # Apache Airflow orchestration DAG
├── saved_models/          # DVC-tracked model artifacts (.h5)
├── dataset/               # DVC-tracked raw and split image datasets
├── scripts/
│   ├── model_architectures.py # ResNet50 & EfficientNetB0 configurations
│   ├── train.py           # Training loop and MLflow logging
│   └── evaluate.py        # Confusion matrix and classification reports
└── README.md
```

## Project StructureModel Architecture & Pipeline

* **Input:** Raw facial image (Minimum 80 images per class, sourced from Google, Wikipedia, and Gov pages).

* **Preprocessing:** Image resized to (224, 224). Aggressive data augmentation (rotation, flipping, zooming) applied to the training split.

* **Feature Extraction:** Pre-trained weights from ImageNet are utilized, with a custom Global Average Pooling and Dropout (0.5) layer added to prevent overfitting.

* **Prediction:** A softmax activation layer calculates the probability distribution across the 16 classes.

**Output:** JSON response containing the predicted public figure and the confidence score.

## Local Setup & Development
To run this project locally on your machine:

1. Clone the Repository

```bash
git clone https://github.com/Subhans1501/Pak-Politician-Classifier-MLOps.git
cd Pak-Politician-Classifier-MLOps
```

2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

3. Pull Heavy Data & Models (DVC)
Because models and datasets are not stored in Git, you must pull them from remote storage.

```bash
dvc pull
```

4. Start the FastAPI Backend

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Docker & AWS Deployment
To deploy the production-ready container to an AWS Ubuntu server:

1. Pull Latest Code & Data

```bash
git pull origin main
dvc pull
```

2. Clean Old Docker Cache (Optional)

```bash
sudo docker system prune -a --volumes
```

3. Build Docker Image

```bash
sudo docker build -t pak-politician-classifier .
```

4. Run Docker Container

```bash
sudo docker run -d -p 8000:8000 pak-politician-classifier
```

5. Access the API

Live API Endpoint: http://<your-aws-ip>:8000/predict

FastAPI Swagger UI: http://<your-aws-ip>:8000/docs

Retraining the Pipeline
If you wish to retrain the models with new images or update the architecture, the MLOps pipeline handles everything automatically:

```bash
dvc repro
```

This single command will:

Detect changes in the dataset or scripts.

Trigger train.py (Tracking metrics in MLflow).

Trigger evaluate.py (Generating new confusion matrices).

Save the updated .h5 models to the saved_models/ directory.