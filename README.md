# CritAlert

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)
![Healthcare%20AI](https://img.shields.io/badge/Healthcare-AI%20Prototype-2F6FED?style=for-the-badge)

</div>

CritAlert is an AI-powered medical screening system that detects critical abnormalities in X-ray, ECG, and CT scan images using trained deep learning models. The project combines a Flask web application, TensorFlow-based inference, and a doctor-facing alert dashboard to support early clinical review and triage workflows.

## Overview

This project is designed for medical image analysis and alert generation. Users upload a report image, select the corresponding report type, and the system loads the matching trained model to predict whether the scan is normal or critical. If a critical result is found, the application creates an alert and presents it in a secure doctor portal for review.

## Why this project matters

In healthcare environments, quick triage is essential. CritAlert demonstrates how AI can assist in early screening by helping staff identify potentially urgent cases faster and provide doctors with a clear, structured alert system for review.

## Features

- X-ray abnormality detection
- ECG abnormality detection
- CT scan image analysis
- Critical alert generation
- Doctor login and dashboard access
- Local upload and alert storage
- Model inference using trained Keras/TensorFlow files

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- HTML / CSS / JavaScript

## Project Structure

```text
Critalert_main/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── start_critalert.bat
├── README_START.txt
├── image_model.h5
├── ecg_model.h5
├── ct_scan_model.h5
├── ecg_class_indices.json
├── ct_class_indices.json
├── pending_alerts.json
├── uploads/
├── templates/
│   ├── index.html
│   ├── doctor_login.html
│   └── doctor_dashboard.html
├── docs/
│   ├── SETUP_GUIDE.md
│   ├── ARCHITECTURE.md
│   ├── FEATURES.md
│   └── DEPLOYMENT.md
├── ABOUT_THIS_PROJECT.md
├── PROJECT_SUMMARY.md
├── PROJECT_BANNER.md
└── train_*.py files
```

## Installation

### 1. Clone or open the project

```bash
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

Start the application:

```bash
python app.py
```

Then open:

- http://localhost:5000
- http://localhost:5000/doctor/login

## Default doctor login

```text
Password: doctor@1234
```

## Quick Start (Windows)

Use the provided startup file:

- [start_critalert.bat](start_critalert.bat)

This script will create a venv if needed, install dependencies, and launch the app.

## Medical Use Case

CritAlert is intended as a medical AI prototype for early screening and clinical review support. It helps in identifying critical findings quickly, signaling doctors to review urgent cases, and demonstrating how deep learning can be embedded into practical healthcare workflows.

> Important: This project is a prototype for research and demonstration. It should not be used as a substitute for professional medical diagnosis or certified clinical decision support.

## Alert Workflow

1. Upload a medical report image.
2. Select the report type.
3. Run analysis through the trained model.
4. View the model result.
5. If critical, a dashboard alert is created.
6. The doctor can review and acknowledge the alert.

## Future Improvements

- Database-backed alert storage
- Better authentication and user management
- Cloud deployment
- DICOM support
- Notification system for critical cases
- Model explainability and confidence analysis

## License

This project is available under the MIT License.

## Contact

This project and the included trained models were developed for a demo and research-oriented healthcare AI application.

---

### Verified status

The project Python code was validated successfully with:

```bash
python -m py_compile app.py
```

This completed without errors.

## Conclusion

CritAlert is a working AI-assisted medical screening prototype that bridges trained deep learning models with a web application for diagnostic review. It demonstrates how medical imaging models can be packaged into a practical hospital-style workflow for detecting critical findings and alerting doctors.

The models in this repository were trained by the project owner and are integrated into the app for local demo and research use.

## License

This project is intended for academic, research, and demo use. Please add your preferred license before public use or distribution.

## Contact / project ownership

This project and its trained models are owned by the developer who trained them. Please keep the repository and model assets within the project context unless proper permissions are granted.

---

If you want, I can also create a more advanced version of the repository with:

- a proper docs folder
- a project architecture diagram
- a sample LICENSE file
- a requirements freeze file
- a deployment guide for Heroku / Render / Azure
