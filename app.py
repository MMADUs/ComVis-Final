import html
import time

import streamlit as st

from module import predict_retinal, predict_retinal_proba

DISEASE_CLASSES = {
    "normal": {
        "name": "Normal",
        "desc": "No signs of retinal disease were detected in the uploaded image.",
        "status": "Low-risk screen",
    },
    "glaucoma": {
        "name": "Glaucoma",
        "desc": "Possible optic nerve damage pattern associated with glaucoma.",
        "status": "Review recommended",
    },
    "cataract": {
        "name": "Cataract",
        "desc": "Possible lens clouding pattern that can reduce visual clarity.",
        "status": "Review recommended",
    },
    "diabetic_retinopathy": {
        "name": "Diabetic Retinopathy",
        "desc": "Possible diabetes-related retinal vessel damage pattern.",
        "status": "Review recommended",
    },
}

MODEL_OPTIONS = {
    "Logistic Regression": "lr",
    "Support Vector Machine": "svm",
    "Random Forest": "rf",
}

CLASS_COLORS = {
    "normal": {"bg": "#ecfdf5", "border": "#34d399", "fg": "#065f46", "bar": "#10b981"},
    "glaucoma": {
        "bg": "#fff7ed",
        "border": "#fb923c",
        "fg": "#9a3412",
        "bar": "#f97316",
    },
    "cataract": {
        "bg": "#eff6ff",
        "border": "#60a5fa",
        "fg": "#1d4ed8",
        "bar": "#3b82f6",
    },
    "diabetic_retinopathy": {
        "bg": "#fef2f2",
        "border": "#f87171",
        "fg": "#991b1b",
        "bar": "#ef4444",
    },
}


def normalize_class_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def extract_proba_dict(proba_result):
    if isinstance(proba_result, tuple) and len(proba_result) == 2:
        return proba_result[0] or {}
    return proba_result or {}


def class_info_for(value: str) -> dict:
    return DISEASE_CLASSES.get(
        normalize_class_name(value),
        {
            "name": str(value).replace("_", " ").title(),
            "desc": "",
            "status": "Prediction result",
        },
    )


def color_for(value: str) -> dict:
    return CLASS_COLORS.get(
        normalize_class_name(value),
        {"bg": "#f8fafc", "border": "#94a3b8", "fg": "#334155", "bar": "#64748b"},
    )


def render_probability_bars(proba_dict: dict, predicted_key: str) -> None:
    sorted_proba = sorted(
        proba_dict.items(), key=lambda item: float(item[1]), reverse=True
    )

    rows = []
    for class_key, probability in sorted_proba:
        normalized = normalize_class_name(class_key)
        info = class_info_for(normalized)
        colors = color_for(normalized)
        pct = max(0.0, min(float(probability) * 100, 100.0))
        active_class = " is-active" if normalized == predicted_key else ""

        rows.append(f"""<div class="prob-row{active_class}">
    <div class="prob-name">
        <span>{html.escape(info["name"])}</span>
        <small>{html.escape(normalized.replace("_", " ").title())}</small>
    </div>
    <div class="prob-track" aria-hidden="true">
        <div class="prob-fill" style="width:{pct:.2f}%; background:{colors["bar"]};"></div>
    </div>
    <div class="prob-value">{pct:.1f}%</div>
</div>""")

    st.markdown(
        f"""<section class="surface result-surface">
    <div class="surface-header">
        <span>Confidence Scores</span>
        <small>All classes</small>
    </div>
    {"".join(rows)}
</section>""",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="RetinaScan AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
    --page-bg: #f5f7fb;
    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #d9e2ef;
    --accent: #2563eb;
    --accent-dark: #1d4ed8;
}

html,
body,
.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--page-bg) !important;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] > .main {
    background:
        linear-gradient(180deg, rgba(219, 234, 254, 0.75), rgba(245, 247, 251, 0) 280px),
        var(--page-bg) !important;
}

#MainMenu,
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton,
[data-testid="stStatusWidget"],
[data-testid="InputInstructions"] {
    display: none !important;
}

.block-container {
    max-width: 1120px;
    padding: 2rem 2rem 3.5rem;
}

body,
button,
input,
textarea,
label,
.stMarkdown,
.stText,
.stSelectbox,
.stFileUploader {
    font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.hero {
    border-bottom: 1px solid rgba(148, 163, 184, 0.28);
    margin-bottom: 1.25rem;
    padding: 0.35rem 0 1.45rem;
}

.eyebrow {
    color: var(--accent-dark);
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    margin: 0 0 0.55rem;
    text-transform: uppercase;
}

.hero h1 {
    color: var(--ink);
    font-size: clamp(2.15rem, 5vw, 4.15rem);
    font-weight: 800;
    letter-spacing: 0;
    line-height: 0.98;
    margin: 0;
}

.hero p {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.65;
    margin: 1rem 0 0;
    max-width: 680px;
}

.surface {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
    padding: 1.25rem;
}

.surface-title {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 800;
    margin: 0 0 0.15rem;
}

.surface-copy {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.55;
    margin: 0 0 1.1rem;
}

.tiny-label {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    margin: 0.95rem 0 0.5rem;
    text-transform: uppercase;
}

.requirements {
    display: grid;
    gap: 0.75rem;
    margin-top: 1rem;
}

.requirement-item {
    background: var(--surface-soft);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.85rem 0.95rem;
}

.requirement-item strong {
    color: var(--ink);
    display: block;
    font-size: 0.84rem;
    margin-bottom: 0.25rem;
}

.requirement-item span {
    color: var(--muted);
    display: block;
    font-size: 0.82rem;
    line-height: 1.45;
}

[data-testid="stSelectbox"] label,
[data-testid="stFileUploader"] label {
    color: var(--ink) !important;
    font-size: 0.85rem !important;
    font-weight: 750 !important;
}

[data-baseweb="select"] > div,
[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] * {
    color: var(--ink) !important;
}

[data-testid="stFileUploader"] {
    margin-top: 0.25rem;
}

[data-testid="stFileUploader"] section {
    border-style: dashed !important;
    min-height: 104px;
    padding: 0.8rem !important;
}

[data-testid="stFileUploader"] section * {
    color: #334155 !important;
}

[data-testid="stFileUploader"] button {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    font-weight: 750 !important;
}

[data-testid="stFileUploader"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent-dark) !important;
}

[data-testid="stFileUploaderFile"] {
    align-items: center !important;
    background: #f8fafc !important;
    border: 1px solid #dbe4ef !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    color: var(--ink) !important;
    min-height: 44px !important;
    padding: 0.45rem 0.55rem !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #334155 !important;
}

[data-testid="stFileUploaderFile"] svg {
    color: var(--accent-dark) !important;
}

[data-testid="stFileUploaderFileName"] {
    color: #1e293b !important;
    font-size: 0.82rem !important;
    font-weight: 750 !important;
    line-height: 1.15 !important;
}

[data-testid="stFileUploaderFileData"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
    line-height: 1.15 !important;
}

[data-testid="stFileUploaderFile"] button {
    align-items: center !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 999px !important;
    display: inline-flex !important;
    height: 24px !important;
    justify-content: center !important;
    min-height: 24px !important;
    padding: 0 !important;
    width: 24px !important;
}

[data-testid="stFileUploaderFile"] button:hover {
    background: #eff6ff !important;
    border-color: var(--accent) !important;
}

.stButton > button {
    background: var(--accent-dark) !important;
    border: 1px solid var(--accent-dark) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 0.96rem !important;
    font-weight: 800 !important;
    min-height: 3rem;
    width: 100%;
}

.stButton > button:hover {
    background: #1e40af !important;
    border-color: #1e40af !important;
}

.stButton > button * {
    color: #ffffff !important;
}

.preview-note {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.45;
    margin: 0.6rem 0 0;
}

.result-surface {
    margin-top: 1.2rem;
}

.diagnosis-card {
    border-radius: 8px;
    border: 1.5px solid;
    margin-top: 1.2rem;
    padding: 1.35rem;
}

.diagnosis-topline {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}

.diagnosis-pill {
    border: 1px solid currentColor;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.65rem;
    text-transform: uppercase;
}

.diagnosis-model {
    font-size: 0.78rem;
    font-weight: 700;
    opacity: 0.72;
}

.diagnosis-card h2 {
    font-size: clamp(1.9rem, 4vw, 3rem);
    font-weight: 850;
    letter-spacing: 0;
    line-height: 1.03;
    margin: 0;
}

.diagnosis-card p {
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0.8rem 0 0;
    max-width: 680px;
}

.surface-header {
    align-items: baseline;
    display: flex;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.surface-header span {
    color: var(--ink);
    font-size: 0.98rem;
    font-weight: 800;
}

.surface-header small {
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 700;
}

.prob-row {
    align-items: center;
    display: grid;
    gap: 0.8rem;
    grid-template-columns: minmax(132px, 190px) 1fr 56px;
    padding: 0.55rem 0;
}

.prob-row + .prob-row {
    border-top: 1px solid #edf2f7;
}

.prob-row.is-active .prob-name span,
.prob-row.is-active .prob-value {
    color: var(--ink);
    font-weight: 850;
}

.prob-name span,
.prob-name small {
    display: block;
}

.prob-name span {
    color: #1e293b;
    font-size: 0.86rem;
    font-weight: 750;
    line-height: 1.25;
}

.prob-name small {
    color: #94a3b8;
    font-size: 0.72rem;
    margin-top: 0.1rem;
}

.prob-track {
    background: #edf2f7;
    border-radius: 999px;
    height: 0.55rem;
    overflow: hidden;
}

.prob-fill {
    border-radius: 999px;
    height: 100%;
}

.prob-value {
    color: #475569;
    font-size: 0.82rem;
    font-weight: 800;
    text-align: right;
}

.disclaimer {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 8px;
    color: #9a3412;
    font-size: 0.82rem;
    line-height: 1.55;
    margin-top: 1.2rem;
    padding: 0.95rem 1rem;
}

.footer {
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    text-align: center;
}

@media (max-width: 760px) {
    .block-container {
        padding: 1.25rem 1rem 2.5rem;
    }

    .hero {
        padding-top: 0;
    }

    .surface {
        padding: 1rem;
    }

    .prob-row {
        grid-template-columns: 1fr 52px;
        gap: 0.45rem 0.75rem;
    }

    .prob-name {
        grid-column: 1 / -1;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="hero">
    <p class="eyebrow">RetinaScan AI</p>
    <h1>Retinal disease screening from fundus images</h1>
    <p>
        Upload one retinal image, choose a traditional machine learning model, and review
        the predicted class with confidence scores for Normal, Glaucoma, Cataract, and
        Diabetic Retinopathy.
    </p>
</section>
""",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.92, 1.08], gap="large")

with left_col:
    st.markdown(
        """
        <section class="surface">
            <p class="surface-title">Image checklist</p>
            <p class="surface-copy">
                Better input images make the model output easier to trust. Use a clear,
                centered fundus photograph before running inference.
            </p>
            <div class="requirements">
                <div class="requirement-item">
                    <strong>File type</strong>
                    <span>JPG, JPEG, or PNG fundus camera output.</span>
                </div>
                <div class="requirement-item">
                    <strong>Image quality</strong>
                    <span>Keep the retina in focus with enough light and minimal blur.</span>
                </div>
                <div class="requirement-item">
                    <strong>Framing</strong>
                    <span>Include the optic disc and macula near the center where possible.</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    with st.container(border=True):
        st.markdown(
            '<p class="surface-title">Run classification</p>', unsafe_allow_html=True
        )
        st.markdown(
            '<p class="surface-copy">Select a model and upload a retinal fundus image.</p>',
            unsafe_allow_html=True,
        )

        selected_model_label = st.selectbox(
            "Classification model",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
        )

        uploaded_file = st.file_uploader(
            "Retinal fundus image",
            type=["jpg", "jpeg", "png"],
            help="Accepted formats: JPG, JPEG, PNG.",
        )

        image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

        if image_bytes is not None:
            st.markdown('<p class="tiny-label">Preview</p>', unsafe_allow_html=True)
            st.image(image_bytes, use_container_width=True)
            st.markdown(
                '<p class="preview-note">Confirm the image is clear before analysing.</p>',
                unsafe_allow_html=True,
            )

        submitted = st.button("Analyze image", type="primary")

if submitted:
    if image_bytes is None:
        st.error("Please upload a retinal image before running classification.")
    else:
        model_key = MODEL_OPTIONS[selected_model_label]

        with st.spinner("Analyzing retinal texture and intensity features..."):
            time.sleep(0.4)
            predicted_class = predict_retinal(image_bytes, model=model_key)
            proba_result = predict_retinal_proba(image_bytes, model=model_key)

        proba_dict = extract_proba_dict(proba_result)
        pred_key = normalize_class_name(predicted_class)
        class_info = class_info_for(pred_key)
        colors = color_for(pred_key)

        st.markdown(
            f"""
            <section class="diagnosis-card" style="
                background: {colors["bg"]};
                border-color: {colors["border"]};
                color: {colors["fg"]};">
                <div class="diagnosis-topline">
                    <span class="diagnosis-pill">{html.escape(class_info["status"])}</span>
                    <span class="diagnosis-model">{html.escape(selected_model_label)}</span>
                </div>
                <h2>{html.escape(class_info["name"])}</h2>
                <p>{html.escape(class_info["desc"])}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        if proba_dict:
            render_probability_bars(proba_dict, pred_key)

        st.markdown(
            """
            <div class="disclaimer">
                <strong>Medical disclaimer:</strong> This app is not a substitute for a professional ophthalmological
                examination or clinical decision-making.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<p class="footer">2026 Kelompok 10 - RetinaScan AI</p>',
    unsafe_allow_html=True,
)
