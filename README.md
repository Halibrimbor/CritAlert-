# CritAlert

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)
![Healthcare AI](https://img.shields.io/badge/Healthcare-AI%20Prototype-2F6FED?style=for-the-badge)

</div>

CritAlert is an AI-powered medical screening system designed to detect critical abnormalities in X-ray, ECG, and CT scan images using trained deep learning models. The project combines a Flask web app, TensorFlow/Keras inference, and a doctor-facing alert dashboard to support healthcare triage and clinical review workflows.

## Overview

This project is a prototype healthcare AI solution. A user uploads a medical image, selects the correct report type, and the application loads the matching trained model to predict whether the scan is normal or critical. If a critical result is found, an alert is created and shown in the doctor dashboard.

## Why this project matters

Fast triage is essential in healthcare. CritAlert demonstrates how AI can support early detection by helping identify potentially urgent cases more quickly and routing them to clinicians for review.

## Key Features

- X-ray abnormality detection
- ECG anomaly detection
- CT scan analysis
- Critical alert generation
- Secure doctor login and dashboard
- Local image upload and report storage
- Trained model inference using Keras/TensorFlow

## Technology Stack

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- HTML / CSS / JavaScript

## Repository Structure

```text
Critalert_main/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── start_critalert.bat
├── README_START.txt
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── FEATURES.md
│   └── SETUP_GUIDE.md
├── models/
│   ├── image_model.h5
│   ├── ecg_model.h5
│   ├── ct_scan_model.h5
│   ├── ecg_class_indices.json
│   └── ct_class_indices.json
├── data/
│   └── pending_alerts.json
├── training/
│   ├── train_image_model.py
│   ├── train_ecg_model.py
│   └── train_ct_model.py
├── templates/
│   ├── index.html
│   ├── doctor_login.html
│   └── doctor_dashboard.html
├── uploads/
│   └── .gitkeep
├── PROJECT_BANNER.md
├── PROJECT_SUMMARY.md
├── ABOUT_THIS_PROJECT.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Halibrimbor/CritAlert-.git
cd Critalert_main
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the project

```bash
python app.py
```

Then open:

- http://localhost:5000
- http://localhost:5000/doctor/login

## Default doctor password

```text
doctor@1234
```

## Windows quick start

Use the launcher file:

- [start_critalert.bat](start_critalert.bat)

This script creates a virtual environment if missing, installs dependencies, and starts the app.

## How it works

1. Upload a medical report image.
2. Select the type: X-ray, ECG, or CT.
3. The app loads the matching trained model.
4. The image is preprocessed and passed to the model.
5. The prediction result is displayed in the web interface.
6. If the result is critical, the doctor dashboard receives an alert.

## Medical use case

CritAlert is intended as a healthcare AI prototype for early screening and triage support. It can help simulate a workflow where urgent findings are surfaced sooner for human review.

> Important: This project is a research and demo prototype and should not be treated as a clinically validated medical diagnostic system.

## Future improvements

- database-backed alert storage
- secure production authentication
- cloud deployment
- DICOM support
- more advanced notifications
- explainability and confidence reporting

## License

This project is distributed under the MIT License.

## Project status

Prototype / Research Project

---

### Verified status

The project code was validated successfully with:

```bash
python -m py_compile app.py
```

This completed without errors.
