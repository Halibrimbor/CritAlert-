"""
CritAlert — app.py  (FIXED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUGS FIXED:
  1. Duplicate get_ecg_model() — first def returned None always
  2. ECG preprocess used 224×224 but model expects 150×150
  3. Button disabled before form submit cancelled submit in some browsers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, json, random, time, io, base64
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_from_directory)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import tensorflow as tf
    import numpy as np
    TF_OK = True
    print(f"✓ TensorFlow {tf.__version__} loaded")
except ImportError:
    TF_OK = False
    print("✗ TensorFlow not found — image inference disabled")

app = Flask(__name__)
app.secret_key = "critalert-secret-2025-xray-ecg"

# ── Doctor portal password ────────────────────────────────────────
# Password is:  doctor@1234
DOCTOR_PASSWORD_HASH = generate_password_hash("doctor@1234")

# ── Storage ───────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
ALERTS_FILE   = os.path.join("data", "pending_alerts.json")
MODEL_DIR     = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"]        = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"]   = 20 * 1024 * 1024  # 20 MB

# ── Patient names (anonymised) ────────────────────────────────────
PATIENT_NAMES = [
    "Aryan Mehta","Priya Sharma","Rohan Gupta","Sneha Iyer",
    "Vikram Nair","Anjali Verma","Kunal Joshi","Deepa Rao",
    "Manish Patel","Pooja Bhat","Suresh Reddy","Neha Malhotra",
    "Rahul Kapoor","Divya Singh","Amit Tiwari","Kavya Pillai",
    "Nikhil Bansal","Shreya Desai","Tarun Chauhan","Meera Saxena",
]

# ═════════════════════════════════════════════════════════════════
#  ALERT STORE
# ═════════════════════════════════════════════════════════════════
def load_alerts():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE) as f:
            return json.load(f)
    return []

def save_alerts(data):
    with open(ALERTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def push_alert(patient, reason, parameter, source, image_filename=None):
    alerts = load_alerts()
    alert  = {
        "id":             f"CA-{int(time.time()*1000) % 100000:05d}",
        "patient":        patient,
        "pid":            f"PT-{random.randint(1000,9999)}",
        "severity":       "critical",
        "source":         source,
        "parameter":      parameter,
        "reason":         reason,
        "image_filename": image_filename,
        "timestamp":      time.time(),
        "status":         "unacked",
    }
    alerts.append(alert)
    save_alerts(alerts)
    return alert

# ═════════════════════════════════════════════════════════════════
#  MODEL CACHE  — loaded once, reused every request
# ═════════════════════════════════════════════════════════════════
_xray_model = None
_ecg_model  = None
_ct_model   = None
def get_xray_model():
    """Load image_model.h5 once and cache it."""
    global _xray_model
    if _xray_model is not None:
        return _xray_model
    if not TF_OK:
        return None
    path = os.path.join(MODEL_DIR, "image_model.h5")
    if not os.path.exists(path):
        print(f"✗ {path} not found")
        return None
    try:
        print("Loading X-Ray model…")
        _xray_model = tf.keras.models.load_model(path)
        print(f"✓ X-Ray model loaded — input:{_xray_model.input_shape}  output:{_xray_model.output_shape}")
        return _xray_model
    except Exception as e:
        print(f"✗ Failed to load X-Ray model: {e}")
        return None

# ── FIX 1: only ONE get_ecg_model() function — no duplicate ──────
def get_ecg_model():
    """Load ecg_model.h5 once and cache it."""
    global _ecg_model
    if _ecg_model is not None:
        return _ecg_model
    if not TF_OK:
        return None
    path = os.path.join(MODEL_DIR, "ecg_model.h5")
    if not os.path.exists(path):
        print(f"✗ {path} not found")
        return None
    try:
        print("Loading ECG model…")
        _ecg_model = tf.keras.models.load_model(path)
        print(f"✓ ECG model loaded — input:{_ecg_model.input_shape}  output:{_ecg_model.output_shape}")
        return _ecg_model
    except Exception as e:
        print(f"✗ Failed to load ECG model: {e}")
        return None
    
def get_ct_model():
    """Load ct_scan_model.h5 once and cache it."""
    global _ct_model
    if _ct_model is not None:
        return _ct_model
    if not TF_OK:
        return None
    path = os.path.join(MODEL_DIR, "ct_scan_model.h5")
    if not os.path.exists(path):
        print(f"✗ {path} not found")
        return None
    try:
        print("Loading CT Scan model…")
        _ct_model = tf.keras.models.load_model(path)
        print(f"✓ CT model loaded — input:{_ct_model.input_shape} output:{_ct_model.output_shape}")
        return _ct_model
    except Exception as e:
        print(f"✗ Failed to load CT model: {e}")
        return None

# ═════════════════════════════════════════════════════════════════
#  INFERENCE
# ═════════════════════════════════════════════════════════════════

# ── FIX 2: read actual model input size instead of hardcoding ────
def _model_img_size(model) -> int:
    """Extract the spatial input size from a loaded Keras model."""
    try:
        shape = model.input_shape   # e.g. (None, 150, 150, 3)
        return int(shape[1])        # height (assumes square)
    except Exception:
        return 150                  # safe fallback

def _prepare_image(file_path: str, size: int) -> "np.ndarray":
    """Load image, resize to (size, size), normalise to [0,1], add batch dim."""
    img = tf.keras.utils.load_img(file_path, target_size=(size, size))
    arr = tf.keras.utils.img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def run_xray_inference(file_path: str):
    """
    X-Ray model:
      output shape = (None, 2)  — logits, no softmax
      class 0 = Critical (alphabetically first: C < N)
      class 1 = Normal
    Returns (is_critical, reason, parameter)
    """
    m = get_xray_model()
    if m is None:
        return False, "image_model.h5 not found. Ensure it is in the Critalert folder.", "Model Missing"

    try:
        size = _model_img_size(m)
        arr  = _prepare_image(file_path, size)
        pred = m.predict(arr, verbose=0)   # shape (1, 2) — logits

        # softmax to get probabilities, then argmax
        probs = tf.nn.softmax(pred[0]).numpy()
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx]) * 100

        print(f"X-Ray prediction — probs:{probs}  class:{idx}  conf:{conf:.1f}%")

        # Class 0 = Critical (C before N alphabetically)
        if idx == 0:
            return (True,
                    f"X-Ray shows critical findings — severe lung abnormality detected (confidence {conf:.0f}%). "
                    "Immediate respiratory support and cardiology review required.",
                    "CNN: Critical X-Ray finding")
        else:
            return (False,
                    f"X-Ray appears within normal limits. No acute abnormality detected (confidence {conf:.0f}%).",
                    "Normal X-Ray")

    except Exception as e:
        print(f"X-Ray inference error: {e}")
        return False, f"X-Ray analysis error: {str(e)}", "Inference Error"


def run_ecg_inference(file_path: str):
    """
    ECG model:
      output shape = (None, 1)  — sigmoid output (binary)
      ecg_class_indices.json: Abnormal=0, Normal=1
      sigmoid > 0.5 → class 1 (Normal)
      sigmoid < 0.5 → class 0 (Abnormal) → CRITICAL
    Returns (is_critical, reason, parameter)
    """
    m = get_ecg_model()
    if m is None:
        return False, "ecg_model.h5 not found. Ensure it is in the Critalert folder.", "ECG Model Missing"

    try:
        size = _model_img_size(m)          # FIX 2: use actual model size (150)
        arr  = _prepare_image(file_path, size)

        pred  = m.predict(arr, verbose=0)  # shape (1, 1) — sigmoid
        score = float(pred[0][0])

        print(f"ECG prediction — sigmoid score: {score:.4f}")

        # sigmoid > 0.5 → Normal (class index 1)
        # sigmoid < 0.5 → Abnormal (class index 0) → critical
        is_abnormal = score < 0.5
        conf = (1.0 - score) * 100 if is_abnormal else score * 100

        if is_abnormal:
            return (True,
                    f"ECG shows abnormal cardiac rhythm — arrhythmia or ST-segment changes detected "
                    f"(confidence {conf:.0f}%). Urgent cardiology evaluation required.",
                    "CNN: Abnormal ECG")
        else:
            return (False,
                    f"ECG appears normal. No significant arrhythmia or ST changes detected (confidence {conf:.0f}%).",
                    "Normal ECG")

    except Exception as e:
        print(f"ECG inference error: {e}")
        return False, f"ECG analysis error: {str(e)}", "Inference Error"
def run_ct_inference(file_path: str):
    """
    CT model:
      multi-class (normal / pneumonia / tumor etc.)
    Returns (is_critical, reason, parameter)
    """
    m = get_ct_model()
    if m is None:
        return False, "ct_scan_model.h5 not found.", "CT Model Missing"

    try:
        size = _model_img_size(m)
        arr  = _prepare_image(file_path, size)

        pred = m.predict(arr, verbose=0)
        idx  = int(np.argmax(pred))
        conf = float(np.max(pred)) * 100

        # Load class labels
        try:
            class_index_path = os.path.join(MODEL_DIR, "ct_class_indices.json")
            with open(class_index_path, "r") as f:
                class_indices = json.load(f)
            labels = {v: k for k, v in class_indices.items()}
            label = labels[idx]
        except:
            label = f"class_{idx}"

        print(f"CT prediction — {label} ({conf:.1f}%)")

        # 🔥 Critical logic
        if label.lower() in ["abnormal"]:
            return (True,
                    f"CT scan shows {label} — critical abnormality detected (confidence {conf:.0f}%). Immediate medical attention required.",
                    f"CNN: {label} detected")
        else:
            return (False,
                    f"CT scan appears normal (confidence {conf:.0f}%). No critical abnormality detected.",
                    "Normal CT Scan")

    except Exception as e:
        print(f"CT inference error: {e}")
        return False, f"CT analysis error: {str(e)}", "Inference Error"

# ═════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════
def image_to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ═════════════════════════════════════════════════════════════════
#  PATH LAB — MAIN ROUTE
# ═════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def index():
    result      = None
    is_critical = False
    alert_fired = None
    report_type = None
    image_b64   = None
    image_mime  = "image/jpeg"
    error_msg   = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            error_msg = "No file selected. Please choose a JPG or PNG image."
        else:
            ext      = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            saved    = os.path.join(UPLOAD_FOLDER, filename)
            file.save(saved)

            if ext in ("jpg", "jpeg", "png"):
                rtype = request.form.get("report_type", "xray").lower()

                if rtype == "ecg":
                    report_type = "ECG"
                    crit, reason, param = run_ecg_inference(saved)
                elif rtype == "ct":
                    report_type = "CT Scan"
                    crit, reason, param = run_ct_inference(saved)           
                else:
                    report_type = "X-Ray"
                    crit, reason, param = run_xray_inference(saved)

                result      = reason
                is_critical = crit
                image_b64   = image_to_b64(saved)
                image_mime  = "image/png" if ext == "png" else "image/jpeg"

                if crit:
                    alert_fired = push_alert(
                        patient        = random.choice(PATIENT_NAMES),
                        reason         = reason,
                        parameter      = param,
                        source         = report_type,
                        image_filename = filename,
                    )
            else:
                error_msg = f"Unsupported format '.{ext}'. Please upload JPG, JPEG, or PNG."
                if os.path.exists(saved):
                    os.remove(saved)

    return render_template("index.html",
                           result=result,
                           is_critical=is_critical,
                           alert_fired=alert_fired,
                           report_type=report_type,
                           image_b64=image_b64,
                           image_mime=image_mime,
                           error_msg=error_msg)

# ═════════════════════════════════════════════════════════════════
#  DOCTOR PORTAL
# ═════════════════════════════════════════════════════════════════
@app.route("/doctor/login", methods=["GET", "POST"])
def doctor_login():
    error = None
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if check_password_hash(DOCTOR_PASSWORD_HASH, pwd):
            session["doctor_logged_in"] = True
            return redirect(url_for("doctor"))
        error = "Incorrect password. Please try again."
    return render_template("doctor_login.html", error=error)

@app.route("/doctor/logout")
def doctor_logout():
    session.pop("doctor_logged_in", None)
    return redirect(url_for("doctor_login"))

@app.route("/doctor")
def doctor():
    if not session.get("doctor_logged_in"):
        return redirect(url_for("doctor_login"))
    return render_template("doctor_dashboard.html")

# ── Doctor API ────────────────────────────────────────────────────
def require_doctor(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("doctor_logged_in"):
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

@app.route("/get_alerts")
@require_doctor
def get_alerts():
    return jsonify(load_alerts())

@app.route("/ack_alert/<aid>", methods=["POST"])
@require_doctor
def ack_alert(aid):
    alerts = load_alerts()
    for a in alerts:
        if a["id"] == aid:
            a["status"] = "acked"
            break
    save_alerts(alerts)
    return jsonify({"ok": True})

@app.route("/clear_alerts", methods=["POST"])
@require_doctor
def clear_alerts():
    save_alerts([])
    return jsonify({"ok": True})

@app.route("/view_image/<filename>")
@require_doctor
def view_image(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))

@app.route("/download_image/<filename>")
@require_doctor
def download_image(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename),
                               as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
