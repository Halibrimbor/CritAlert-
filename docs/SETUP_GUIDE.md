# CritAlert Setup Guide

This guide explains how to install, configure, and run the CritAlert project correctly.

## 1. Prerequisites

Before running the app, make sure you have:

- Python 3.10 or 3.11
- pip
- Git (optional, but recommended)
- A local terminal such as PowerShell, CMD, or Bash

## 2. Clone or open the project

Open the project folder in your machine.

If using Git:

```bash
git clone <your-repository-url>
cd Critalert_main
```

## 3. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux / Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

Run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Start the application

From the project root:

```bash
python app.py
```

Then open:

- http://localhost:5000

Doctor access:

- http://localhost:5000/doctor/login

## 6. Doctor login

Use the default password:

```text
doctor@1234
```

## 7. Startup script for Windows

You can also run the ready-made script:

- [start_critalert.bat](../start_critalert.bat)

This script creates a virtual environment automatically if it does not exist, installs requirements, and starts the app.

## 8. Important notes

- The trained models must remain in the project root directory.
- If the model files are missing, the app will show a model-not-found error.
- Uploaded images are saved in the uploads folder.
- Alerts are stored in the local JSON file named pending_alerts.json.
- This project is a prototype and should be used carefully for medical workflow testing only.

## 9. Troubleshooting

### Module not found

If you get import errors, reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

### TensorFlow not found

Make sure the virtual environment is activated and the package is installed.

### Model not found

Check that these files exist in the project root:

- image_model.h5
- ecg_model.h5
- ct_scan_model.h5

### Port already in use

Stop any other app using port 5000 or change the Flask port in app.py.

## 10. Recommended next steps

- validate the trained models with local test images
- secure the doctor login for production use
- convert alert storage to a database
- add a proper user authentication system
- deploy the app to a real web hosting environment
