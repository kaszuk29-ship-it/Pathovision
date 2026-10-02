import io
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ocr import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from rapidfuzz import fuzz

st.set_page_config(
    page_title="PathoVision | Medical OCR",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(53, 211, 153, 0.12), transparent 28%),
        radial-gradient(circle at 90% 0%, rgba(99, 102, 241, 0.13), transparent 30%),
        #f6f9fc;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #0f766e 0%, #2563eb 55%, #7c3aed 100%);
    padding: 28px 32px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 16px 40px rgba(37, 99, 235, 0.18);
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 42px;
    margin: 0 0 7px 0;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero p {
    margin: 0;
    font-size: 16px;
    opacity: 0.94;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.28);
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 13px;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #102a43;
    margin: 12px 0 10px 0;
}

.card {
    background: white;
    border: 1px solid #e5edf5;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 7px 22px rgba(15, 23, 42, 0.05);
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    border: 1px solid #e5edf5;
    border-radius: 17px;
    padding: 16px 18px;
    min-height: 108px;
    box-shadow: 0 7px 22px rgba(15, 23, 42, 0.05);
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #0f172a;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

.info-strip {
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    padding: 13px 15px;
    border-radius: 12px;
    color: #1e3a8a;
    margin: 10px 0 18px 0;
}

.warning-strip {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 13px 15px;
    border-radius: 12px;
    color: #9a3412;
    margin: 10px 0 18px 0;
}

.success-strip {
    background: #ecfdf5;
    border-left: 5px solid #10b981;
    padding: 13px 15px;
    border-radius: 12px;
    color: #065f46;
    margin: 10px 0 18px 0;
}

.small-muted {
    color: #64748b;
    font-size: 12px;
}

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #93c5fd;
    border-radius: 16px;
    padding: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Test dictionary / aliases
# -----------------------------
TEST_ALIASES = {
    "Hemoglobin": ["hemoglobin", "hgb", "hb"],
    "White Blood Cell Count": ["wbc", "white blood cell", "total leukocyte count", "tlc"],
    "Red Blood Cell Count": ["rbc", "red blood cell"],
    "Platelet Count": ["platelet", "platelets", "plt"],
    "Hematocrit": ["hematocrit", "hct", "pcv"],
    "MCV": ["mcv", "mean corpuscular volume"],
    "MCH": ["mch", "mean corpuscular hemoglobin"],
    "MCHC": ["mchc", "mean corpuscular hb concentration"],
    "Neutrophils": ["neutrophils", "neutrophil"],
    "Lymphocytes": ["lymphocytes", "lymphocyte"],
    "Eosinophils": ["eosinophils", "eosinophil"],
    "Monocytes": ["monocytes", "monocyte"],
    "Basophils": ["basophils", "basophil"],
    "Glucose": ["glucose", "blood glucose", "fasting glucose", "fbs", "rbs"],
    "Creatinine": ["creatinine", "serum creatinine"],
    "Urea": ["urea", "blood urea"],
    "Total Cholesterol": ["total cholesterol", "cholesterol"],
    "Triglycerides": ["triglycerides", "triglyceride", "tg"],
    "HDL": ["hdl", "hdl cholesterol"],
    "LDL": ["ldl", "ldl cholesterol"],
    "TSH": ["tsh", "thyroid stimulating hormone"],
    "T3": ["t3", "triiodothyronine"],
    "T4": ["t4", "thyroxine"],
    "ALT": ["alt", "sgpt", "alanine aminotransferase"],
    "AST": ["ast", "sgot", "aspartate aminotransferase"],
    "Bilirubin": ["bilirubin", "total bilirubin"],
    "Vitamin B12": ["vitamin b12", "b12", "cobalamin"],
    "Vitamin D": ["vitamin d", "25-oh vitamin d", "25 oh vitamin d"],
}


def canonical_test_name(raw_name: str):
    """Map a raw OCR label to a canonical test name."""
    clean = re.sub(r"[^a-z0-9\s]", " ", raw_name.lower())
    clean = re.sub(r"\s+", " ", clean).strip()

    best_name = None
    best_score = 0

    for canonical, aliases in TEST_ALIASES.items():
        for alias in aliases:
            score = fuzz.token_set_ratio(clean, alias)
            if alias in clean:
                score = max(score, 96)
            if score > best_score:
                best_score = score
                best_name = canonical

    return (best_name, best_score) if best_score >= 70 else (raw_name.strip(), best_score)


# -----------------------------
# OCR
# -----------------------------
def preprocess_image(pil_image: Image.Image):
    image = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Upscaling improves OCR for small report text.
    gray = cv2.resize(gray, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return Image.fromarray(processed)


def ocr_image(pil_image: Image.Image):
    processed = preprocess_image(pil_image)
    text = pytesseract.image_to_string(processed, config="--psm 6")
    return text, processed


def extract_text_from_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix == ".pdf":
        pages = convert_from_bytes(data, dpi=220)
        all_text = []
        previews = []

        for page in pages[:8]:
            text, processed = ocr_image(page)
            all_text.append(text)
            previews.append(processed)

        return "\n\n".join(all_text), previews

    image = Image.open(io.BytesIO(data))
    text, processed = ocr_image(image)
    return text, [processed]


# -----------------------------
# Structured extraction
# -----------------------------
NUMBER_RE = r"(-?\d+(?:\.\d+)?)"
UNIT_RE = r"([a-zA-Zµ/%][a-zA-Z0-9µ/%\^\-]*)"


def parse_reference_range(text):
    patterns = [
        rf"({NUMBER_RE}\s*[-–]\s*{NUMBER_RE})",
        rf"({NUMBER_RE}\s*to\s*{NUMBER_RE})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(1).replace("–", "-").strip()
    return ""


def classify_value(value, reference):
    if not reference:
        return "Needs review"

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)", reference)
    if not m:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*to\s*(-?\d+(?:\.\d+)?)", reference, re.I)

    if not m:
        return "Needs review"

    low, high = float(m.group(1)), float(m.group(2))
    if low <= value <= high:
        return "Within reference"
    return "Outside reference"


def extract_tests(text):
    rows = []
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    for line in lines:
        # Typical: Hemoglobin 13.2 g/dL 12-16
        match = re.search(
            rf"^(.{{2,60}}?)\s+{NUMBER_RE}\s*({UNIT_RE})?\s*(.*)$",
            line,
        )
        if not match:
            continue

        raw_name = match.group(1).strip(" :-|")
        value = float(match.group(2))
        unit = (match.group(3) or "").strip()
        tail = match.group(4) or ""

        # Ignore obvious dates, phone-like values, ages, and report metadata.
        if re.fullmatch(r"\d{1,4}", raw_name):
            continue
        if len(raw_name.split()) > 10:
            continue

        canonical, similarity = canonical_test_name(raw_name)

        # The regex can catch lots of unrelated numbers. Keep known medical
        # test names or high-confidence fuzzy matches.
        known = canonical in TEST_ALIASES
        if not known and similarity < 86:
            continue

        reference = parse_reference_range(tail)
        status = classify_value(value, reference)

        confidence = min(99, max(60, int(similarity if similarity else 78)))
        if reference:
            confidence = min(99, confidence + 3)

        rows.append(
            {
                "Test": canonical,
                "Result": value,
                "Unit": unit,
                "Reference Range": reference,
                "Status": status,
                "Confidence": f"{confidence}%",
                "OCR Label": raw_name,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Test", "Result", "Unit", "Reference Range",
                "Status", "Confidence", "OCR Label"
            ]
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Test"], keep="first")
    return df


def compare_reports(old_df, new_df):
    if old_df.empty or new_df.empty:
        return pd.DataFrame()

    left = old_df[["Test", "Result", "Unit"]].rename(
        columns={"Result": "Previous Result", "Unit": "Previous Unit"}
    )
    right = new_df[["Test", "Result", "Unit"]].rename(
        columns={"Result": "Current Result", "Unit": "Current Unit"}
    )

    merged = pd.merge(left, right, on="Test", how="inner")
    if merged.empty:
        return merged

    merged["Change"] = merged["Current Result"] - merged["Previous Result"]
    merged["Change %"] = np.where(
        merged["Previous Result"] != 0,
        (merged["Change"] / merged["Previous Result"]) * 100,
        np.nan,
    )

    def direction(x):
        if pd.isna(x):
            return "—"
        if abs(x) < 0.000001:
            return "No change"
        return "Increased" if x > 0 else "Decreased"

    merged["Trend"] = merged["Change"].apply(direction)
    return merged


# -----------------------------
# Session state
# -----------------------------
if "reports" not in st.session_state:
    st.session_state.reports = []

if "active_report" not in st.session_state:
    st.session_state.active_report = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🩺 PathoVision")
    st.caption("Medical OCR & pathology report analysis")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyze Report", "Compare Reports", "Report History", "About"],
        index=0,
    )

    st.divider()

    st.markdown("### Project pipeline")
    st.markdown(
        "1. Upload report\n"
        "2. Pre-process image\n"
        "3. OCR extraction\n"
        "4. Structure test values\n"
        "5. Match test names\n"
        "6. Compare reports\n"
        "7. Review flagged values"
    )

    st.divider()
    st.caption("PathoVision is an academic prototype. It does not diagnose disease or replace a healthcare professional.")


# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="badge">AI + OCR + Medical NLP</div>
    <h1>🩺 PathoVision</h1>
    <p>Turn difficult pathology reports into structured, readable and comparable information.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    st.markdown('<div class="section-title">Medical report intelligence at a glance</div>', unsafe_allow_html=True)

    total_reports = len(st.session_state.reports)
    total_tests = sum(len(r["data"]) for r in st.session_state.reports)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Reports analyzed</div><div class="metric-value">{total_reports}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tests extracted</div><div class="metric-value">{total_tests}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">OCR engine</div><div class="metric-value" style="font-size:20px;">Tesseract</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">Purpose</div><div class="metric-value" style="font-size:20px;">Compare</div></div>', unsafe_allow_html=True)

    st.markdown("### What PathoVision does")

    cols = st.columns(3)
    cards = [
        ("📄", "OCR Extraction", "Reads scanned pathology reports and converts them into machine-readable text."),
        ("🧠", "Smart Matching", "Maps variations such as Hb, HGB and Hemoglobin to one standardized test name."),
        ("📊", "Report Comparison", "Shows previous vs current values and highlights measurable changes."),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="card"><div style="font-size:30px;">{icon}</div>'
                f'<h3 style="margin:8px 0;">{title}</h3><p style="color:#64748b;">{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="info-strip"><b>Academic innovation:</b> PathoVision focuses on the difficult part of pathology-report workflows: extracting, standardizing and comparing information across differently formatted reports.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Recommended demo flow")
    st.write("Upload one report → inspect extracted values → upload a second report → compare matching tests → review changes.")

# -----------------------------
# Analyze
# -----------------------------
elif page == "Analyze Report":
    st.markdown('<div class="section-title">Analyze a pathology report</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-strip">For the best OCR result, upload a clear JPG/PNG image or a readable PDF with visible test names and values.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload pathology report",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Supported: PNG, JPG, JPEG and PDF",
    )

    report_label = st.text_input(
        "Report name",
        value=f"Report {len(st.session_state.reports) + 1}",
        placeholder="Example: CBC - September 2026",
    )

    if uploaded:
        with st.spinner("Reading and structuring the report..."):
            try:
                text, previews = extract_text_from_upload(uploaded)
                data = extract_tests(text)
            except Exception as e:
                st.error(f"OCR could not process this file: {e}")
                st.stop()

        st.markdown("### Extraction result")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Characters read", len(text))
        with m2:
            st.metric("Tests detected", len(data))
        with m3:
            confident = sum(int(x.replace("%", "")) >= 85 for x in data["Confidence"]) if not data.empty else 0
            st.metric("High-confidence matches", confident)

        if not data.empty:
            st.markdown("#### Structured values")
            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.TextColumn("OCR confidence"),
                    "Status": st.column_config.TextColumn("Reference status"),
                },
            )

            outside = data[data["Status"] == "Outside reference"]
            if not outside.empty:
                st.markdown(
                    f'<div class="warning-strip"><b>{len(outside)} value(s)</b> are outside the reference range shown in the extracted report. This is not a diagnosis; verify the original report and consult a qualified healthcare professional.</div>',
                    unsafe_allow_html=True,
                )

            if st.button("Save this report to PathoVision history", type="primary"):
                st.session_state.reports.append(
                    {
                        "name": report_label.strip() or f"Report {len(st.session_state.reports) + 1}",
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "data": data.copy(),
                        "raw_text": text,
                    }
                )
                st.session_state.active_report = st.session_state.reports[-1]
                st.success("Report saved successfully. Open Compare Reports to compare it with another saved report.")

        else:
            st.warning("No structured test values were confidently detected. Check the OCR text and try a clearer image.")

        with st.expander("View OCR text"):
            st.text_area("Extracted text", text, height=280)

        with st.expander("View processed image"):
            if previews:
                st.image(previews[0], caption="Pre-processed image used for OCR", use_container_width=True)


# -----------------------------
# Compare
# -----------------------------
elif page == "Compare Reports":
    st.markdown('<div class="section-title">Compare two pathology reports</div>', unsafe_allow_html=True)

    if len(st.session_state.reports) < 2:
        st.markdown(
            '<div class="warning-strip">Save at least two reports from the Analyze Report page before starting a comparison.</div>',
            unsafe_allow_html=True,
        )
    else:
        names = [r["name"] for r in st.session_state.reports]
        a, b = st.columns(2)

        with a:
            old_idx = st.selectbox("Previous report", range(len(names)), format_func=lambda i: names[i], index=0)
        with b:
            new_default = 1 if len(names) > 1 else 0
            new_idx = st.selectbox("Current report", range(len(names)), format_func=lambda i: names[i], index=new_default)

        old_report = st.session_state.reports[old_idx]
        new_report = st.session_state.reports[new_idx]

        result = compare_reports(old_report["data"], new_report["data"])

        if result.empty:
            st.warning("No common test names were found between the two reports.")
        else:
            st.markdown("### Change overview")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Common tests", len(result))
            with c2:
                st.metric("Increased", int((result["Trend"] == "Increased").sum()))
            with c3:
                st.metric("Decreased", int((result["Trend"] == "Decreased").sum()))

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                    "Change": st.column_config.NumberColumn("Absolute change", format="%.2f"),
                },
            )

            st.markdown(
                '<div class="info-strip"><b>How to read this:</b> “Increased” and “Decreased” are mathematical changes between two extracted results. They do not indicate improvement, worsening, or a diagnosis.</div>',
                unsafe_allow_html=True,
            )


# -----------------------------
# History
# -----------------------------
elif page == "Report History":
    st.markdown('<div class="section-title">Report history</div>', unsafe_allow_html=True)

    if not st.session_state.reports:
        st.info("No reports saved in this session yet.")
    else:
        for i, report in enumerate(reversed(st.session_state.reports), start=1):
            with st.expander(f"🩺 {report['name']}  •  {report['date']}"):
                st.write(f"**Tests extracted:** {len(report['data'])}")
                st.dataframe(report["data"], use_container_width=True, hide_index=True)


# -----------------------------
# About
# -----------------------------
elif page == "About":
    st.markdown('<div class="section-title">About PathoVision</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<h3>Problem</h3>
<p>Pathology reports can contain many abbreviations, tables, units and reference ranges.
When a person has multiple reports from different dates or laboratories, manually finding
and comparing the same test can be difficult.</p>

<h3>Solution</h3>
<p>PathoVision uses OCR to read report images, extracts test-result pairs, standardizes common
test names, attaches reference information when available, and compares repeated reports.</p>

<h3>Core pipeline</h3>
<p><b>Report image/PDF → Image preprocessing → OCR → Test extraction → Semantic name matching → Structured table → Comparison</b></p>

<h3>Innovation</h3>
<p>The prototype is designed around <b>longitudinal pathology-report understanding</b> rather than
simple OCR alone. It attempts to recognize the same medical test even when laboratories use
different labels or abbreviations.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="warning-strip"><b>Important:</b> PathoVision is an academic software prototype. OCR can make mistakes. Reference ranges differ between laboratories and individuals. Never use this prototype to diagnose, start, stop, or change treatment.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="small-muted" style="text-align:center;margin-top:35px;">PathoVision • Academic AI/OCR prototype • Verify all extracted values against the original laboratory report.</div>',
    unsafe_allow_html=True,
)
