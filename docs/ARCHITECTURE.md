# CritAlert Architecture

## Overview

CritAlert is a Flask web application that combines trained deep learning models with a hospital-style alert workflow.

The architecture is divided into four major layers:

1. User Interface Layer
2. Application Server Layer
3. AI Inference Layer
4. Data and Alert Layer

## 1. User Interface Layer

This layer includes:

- the main report upload page
- the doctor login page
- the doctor dashboard

The front-end is built with HTML, CSS, and JavaScript embedded inside the Jinja templates.

Files:

- [templates/index.html](../templates/index.html)
- [templates/doctor_login.html](../templates/doctor_login.html)
- [templates/doctor_dashboard.html](../templates/doctor_dashboard.html)

## 2. Application Server Layer

The Flask app is the central backend server. It handles:

- route management
- request parsing
- file upload validation
- model routing
- alert generation
- session-based doctor authorization

Main file:

- [app.py](../app.py)

## 3. AI Inference Layer

Each medical image is processed using the correct trained Keras model.

The app loads the model once and reuses it across requests:

- X-Ray model for chest image prediction
- ECG model for cardiac abnormality check
- CT model for CT classification

Image preprocessing includes:

- loading the image
- resizing to model input size
- normalization to [0,1]
- batch expansion
- prediction using the loaded model

## 4. Data and Alert Layer

The system stores:

- uploaded files in the uploads folder
- alert metadata in pending_alerts.json

This is a lightweight local prototype workflow rather than a production database system.

## Request Flow

```text
User uploads report
        |
        v
Flask route receives file
        |
        v
Detect report type
        |
        v
Load relevant model
        |
        v
Preprocess image
        |
        v
Run prediction
        |
        v
If critical -> create alert
        |
        v
Doctor reviews in dashboard
```

## Security Considerations

This version includes:

- session-based login protection for the doctor dashboard
- password hashing for the doctor account

For a production environment, you would further improve:

- database-backed authentication
- role-based access
- encrypted file storage
- environment-based secrets
- secure deployment setup

## Scalability Notes

This prototype is suitable for local research and demo use. For larger usage, a production-ready version would need:

- a database instead of JSON
- better file management
- asynchronous processing for heavy model inference
- GPU optimization
- API-based deployment
- monitoring and logging for model health

## Summary

The CritAlert architecture is intentionally simple and effective for a prototype: a Flask app, TensorFlow models, upload handling, and a doctor dashboard. It demonstrates how trained medical models can be integrated into a healthcare workflow for screening and alerting.
