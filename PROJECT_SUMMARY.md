# CritAlert — Project Summary

CritAlert is an AI-powered medical screening application designed to detect critical abnormalities in X-Ray, ECG, and CT report images using trained deep learning models. The system allows users to upload medical report images, select the report type, and analyze the result through a Flask-based web application. The backend loads the relevant TensorFlow model, preprocesses the image, and predicts whether the scan is normal or critical.

If the result is flagged as critical, the application creates a medical alert and stores it in a local alert list. A doctor-facing dashboard allows users to log in securely, review incoming alerts, inspect uploaded images, and acknowledge critical findings. The project demonstrates how AI and web technologies can be combined to create a healthcare triage tool for early detection and faster clinical review.

### Key Features

- X-Ray, ECG, and CT scan analysis
- Deep learning inference using trained TensorFlow models
- Critical finding detection and alert generation
- Secure doctor login and dashboard
- Local image storage and alert management
- Research prototype for AI-assisted healthcare support

### Technologies Used

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- HTML / CSS / JavaScript

### Use Case

This project can be used as a healthcare triage prototype where AI helps identify urgent cases and directs them to doctors for review. It is especially useful for learning, academic demonstration, and prototype development in medical AI and healthcare automation.

### Project Status

Prototype / Research Project

### Important Note

This application is designed for demonstration and research purposes and should not be treated as a certified clinical diagnostic system. Real-world medical use requires clinical validation, regulatory approval, and expert oversight.
