import os
import sys
import json
import pandas as pd
import webview
from flask import Flask, render_template, request, redirect, url_for
import threading
import re

# ✅ Setup BASE_DIR for .exe and .py
BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# ✅ File Paths
CONFIG_PATH = os.path.join(BASE_DIR, 'data', 'config.json')
EXCEL_FILE = os.path.join(BASE_DIR, 'data', 'qc_data.xlsx')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

# ✅ Load Config with error handling
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"Error loading config: {e}")
    CONFIG = {"RECORD_ID": "DEFAULT_"}

RECORD_ID = CONFIG.get("RECORD_ID", "DEFAULT_")

# ✅ Load Excel with error handling
try:
    df = pd.read_excel(EXCEL_FILE, dtype=str).fillna('')
except Exception as e:
    print(f"Error loading Excel file: {e}")
    # Create empty DataFrame with expected columns
    default_columns = [
        "TYPE OF ACCOUNT", "REGD ADDRESS", "DATE OF BIRTH", "GENDER", "NATIONALITY",
        "ADDRESS", "CITY/DISTRICT/TOWN", "STATE", "ADDRESS VERIFICATION",
        "ID VERIFICATION", "CENTER NAME", "CENTER CODE", "REMARKS",
        "MOBILE NUMBER", "CONTACT 1", "CONTACT 2", "FULL NAME", "FATHER/ HUSBAND NAME",
        "PINCODE", "EMAIL", "RECORD NUMBER", "IMAGE NAME"
    ]
    df = pd.DataFrame(columns=default_columns)
    print("Created empty DataFrame as fallback")

# ✅ Flask App
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.secret_key = 'your_secret'

# ✅ Toggle Fields
toggle_fields = [
    "TYPE OF ACCOUNT", "REGD ADDRESS", "DATE OF BIRTH", "GENDER", "NATIONALITY",
    "ADDRESS", "CITY/DISTRICT/TOWN", "STATE", "ADDRESS VERIFICATION",
    "ID VERIFICATION", "CENTER NAME", "CENTER CODE", "REMARKS"
]
toggle_fields_lower = [f.lower() for f in toggle_fields]


# ✅ Helper Functions
def format_contact(value):
    if value == "N.A":
        return value
    if re.search(r'[a-zA-Z]', value):
        return value
    digits = ''.join(filter(str.isdigit, str(value)))
    if len(digits) <= 3:
        return f"({digits})"
    elif len(digits) <= 6:
        return f"({digits[:3]}) {digits[3:]}"
    else:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def format_recordname(name):
    digits = ''.join(filter(str.isdigit, str(name)))
    return RECORD_ID + digits


def proper_case_name(name):
    if name == "N.A":
        return name
    name = re.sub(r"[^A-Za-z0-9&/()'\-.\s]", '', str(name))  # allow period
    name = re.sub(r'\s+', ' ', name.strip())
    return re.sub(r'(\b\w)', lambda m: m.group(1).upper(), name.lower())


def date_of_birth(value):
    if value == "N.A":
        return value
    value = str(value).strip().lower()
    value = value.replace("'", "")
    symbol = ". "
    alpha = ""
    digits = ""
    for i in value:
        if i.isalpha():
            alpha = alpha + i
        if i.isdigit():
            digits = digits + i
    a = digits
    if ("." not in value) or ("," in value):
        symbol = ", "
    if alpha == "" or a == "":
        symbol = ""
    if alpha != "":
        alpha = alpha[0].upper() + alpha[1:]
    part1 = digits[:2]
    part2 = digits[2:4]
    part3 = digits[4:]
    if a != "":
        a = f"{part1}/{part2}/{part3}"
    value = alpha + symbol + a
    return value


def get_image_files():
    image_dir = os.path.join(STATIC_FOLDER, 'images')
    if not os.path.exists(image_dir):
        return []
    return sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])


# ✅ Routes
@app.route("/", methods=["GET"])
def index():
    record_param = request.args.get("record", "0")
    try:
        record_no = int(record_param)
    except ValueError:
        record_no = 0
    record_no = max(0, min(record_no, len(df) - 1))

    record = df.iloc[record_no].to_dict()

    record = {k: (str(v).replace(".0", "") if isinstance(v, float) and v.is_integer() else str(v)) for k, v in
              record.items()}
    for key in ["RECORD NUMBER","TYPE OF ACCOUNT","REGD ADDRESS","FULL NAME", "FATHER/ HUSBAND NAME","DATE OF BIRTH","GENDER","NATIONALITY","ADDRESS","CITY/DISTRICT/TOWN","STATE","PINCODE","MOBILE NUMBER","EMAIL","CONTACT 1","CONTACT 2","ADDRESS VERIFICATION","ID VERIFICATION","CENTER NAME","CENTER CODE","REMARKS"]:
        if key in record:
            record[key] = record[key].strip()
            if record[key] in ["","N.A.","NA","na","n.a","N.a"]:
                record[key] = "N.A"
    for key in ["MOBILE NUMBER", "CONTACT 1", "CONTACT 2"]:
        if key in record:
            record[key] = format_contact(record[key])

    for key in ["FULL NAME", "FATHER/ HUSBAND NAME"]:
        if key in record:
            record[key] = proper_case_name(record[key])
    for key in ["GENDER","EMAIL","STATE","ADDRESS VERIFICATION","ID VERIFICATION","CENTER NAME","CENTER CODE"]:
        if key in record:
            record[key] = record[key].replace(" ","")

    if "PINCODE" in record:
        if record["PINCODE"] != "N.A":
            record["PINCODE"] = str(record["PINCODE"]).zfill(5)

    if "DATE OF BIRTH" in record:
        if record["DATE OF BIRTH"] != "N.A":
            record["DATE OF BIRTH"] = date_of_birth(record["DATE OF BIRTH"])

    if "RECORD NUMBER" in record:
        record["RECORD NUMBER"] = format_recordname(record["RECORD NUMBER"])
    if "EMAIL" in record:
        if record["EMAIL"] != "N.A":
            record["EMAIL"] = record["EMAIL"].lower()

    image_files = get_image_files()
    image_name = record.get("IMAGE NAME", "").strip()

    current_image = image_name if image_name in image_files else (image_files[0] if image_files else "not_found.jpg")

    prev_record = max(0, record_no - 1)
    next_record = min(len(df) - 1, record_no + 1)

    return render_template(
        "index.html",
        record=record,
        record_no=record_no,
        total=len(df),
        toggle_fields_lower=toggle_fields_lower,
        current_image=current_image,
        prev_record=prev_record,
        next_record=next_record
    )


@app.route("/save", methods=["POST"])
def save():
    record_no = int(request.form["record_no"])

    for key in df.columns:
        if key in request.form:
            val = request.form[key]
            val = val.strip()
            if key.upper() in ["MOBILE NUMBER", "CONTACT 1", "CONTACT 2"]:
                df.at[record_no, key] = format_contact(val)
            elif key.upper() in ["FULL NAME", "FATHER/ HUSBAND NAME"]:
                df.at[record_no, key] = proper_case_name(val)
            elif key.upper() == "EMAIL":
                if val == "N.A":
                    df.at[record_no, key] = val
                else:
                    df.at[record_no, key] = val.lower()
            elif key.upper() == "PINCODE":
                if val == "N.A":
                    df.at[record_no, key] = val
                else:
                    df.at[record_no, key] = str(val).zfill(5)
            elif key.upper() == "DATE OF BIRTH":
                df.at[record_no, key] = date_of_birth(val)
            elif key.upper() == "RECORD NUMBER":
                df.at[record_no, key] = format_recordname(val)
            else:
                df.at[record_no, key] = val

    try:
        df.to_excel(EXCEL_FILE, index=False)
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    return redirect(url_for("index", record=record_no, saved='true'))


# ✅ Run as Desktop App with Clean Close
if __name__ == "__main__":
    def start_flask():
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)


    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True  # Mark as daemon thread
    flask_thread.start()

    window = webview.create_window("KYC Record Viewer", "http://127.0.0.1:5000", width = 1366, height = 768, resizable=True)
    webview.start()  # Start without any arguments
