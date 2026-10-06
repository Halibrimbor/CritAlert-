# CritAlert Deployment Guide

## Local deployment

To run this project locally:

1. Create a virtual environment
2. Install dependencies
3. Start Flask app with python app.py
4. Open the browser to localhost:5000

## Requirements

- Python 3.10 or 3.11
- TensorFlow compatible with the installed Python version
- Flask dependencies

## Recommended local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

## Production deployment considerations

For production use, consider:

- a database instead of JSON files
- environment variables for secrets
- proper reverse proxy with Nginx
- WSGI server such as Gunicorn
- secure HTTPS setup
- user authentication and role restrictions
- deployment on Azure, AWS, or Render

## Example gunicorn start command

```bash
gunicorn app:app --bind 0.0.0.0:8000
```

## Recommended security improvements

- move password to environment variable
- do not hardcode secrets in app.py
- set app.secret_key from environment
- add file upload validation and scanning
- restrict file types and size more carefully

## Summary

This project is best seen as a prototype or local research application. It can be deployed locally easily, and with more work it can be adapted for a secure production environment.
