# app.py
# Advanced Local ATS Validation Platform
# Fully Offline + Apple Silicon Optimized
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import re
import pandas as pd
import streamlit as st

from ats_engine import AdvancedHybridATS
from analyzer import generate_html_report
from file_parser import parse_uploaded_file
from database import save_report, get_all_reports

# ==========================================================
# ENVIRONMENT DETECTION
# ==========================================================

IS_CLOUD = os.environ.get("STREAMLIT_CLOUD", "false").lower() == "true"
INFERENCE_TYPE = "Linux CPU (Cloud Optimized)" if IS_CLOUD else "Apple Metal (MPS Optimized)"

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="ATS Hybrid Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* ======================================================
   GLOBAL
====================================================== */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ======================================================
   METRIC CARDS
====================================================== */

div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1px solid #e5e7eb;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* ======================================================
   BUTTONS
====================================================== */

div.stButton > button:first-child {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-weight: 700;
    border: none;
    transition: 0.2s ease-in-out;
}

div.stButton > button:first-child:hover {
    transform: translateY(-1px);
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
}

/* ======================================================
   TEXT AREAS
====================================================== */

textarea {
    border-radius: 10px !important;
}

/* ======================================================
   EXPANDERS
====================================================== */

.streamlit-expanderHeader {
    font-size: 1rem;
    font-weight: 600;
}

/* ======================================================
   SECTION SPACING
====================================================== */

.block-container {
    padding-top: 2rem;
}

/* ======================================================
   HIDE FOOTER
====================================================== */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def extract_keywords_from_text(text: str) -> list:
    """Extract technical keywords from text (simple heuristic)."""
    # Common tech keywords pattern
    keywords = re.findall(
        r'\b(?:Python|Java|C\+\+|SQL|JavaScript|TypeScript|React|Vue|Angular|'
        r'Node\.js|Django|Flask|FastAPI|AWS|GCP|Azure|Docker|Kubernetes|'
        r'TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|Spark|Hadoop|'
        r'Git|Linux|Windows|MacOS|REST|GraphQL|PostgreSQL|MongoDB|'
        r'Machine Learning|Deep Learning|NLP|Computer Vision|Data Science|'
        r'Analytics|BI|ETL|DevOps|CI\/CD|Agile|Scrum)\b',
        text,
        re.IGNORECASE
    )
    return list(set(keywords))[:5]  # Top 5 unique keywords

def get_color_coded_score(score: float) -> str:
    """Return emoji and color for ATS score."""
    if score >= 70:
        return "🟢"
    elif score >= 40:
        return "🟠"
    else:
        return "🔴"

# ==========================================================
# RESOURCE CACHING
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_ats_engine():

    return AdvancedHybridATS()

# ==========================================================
# TITLE & INTRO
# ==========================================================

st.title("🚀 ATS Validation Platform")
st.caption("Local AI-powered resume screening and recruiter-style evaluation")

st.info(
    "🚀 AI-powered ATS validation platform combining semantic embeddings, "
    "recruiter-style analysis, keyword matching, and local LLM evaluation "
    "to assess resume-job fit completely offline."
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("🚀 System Info")

    st.success("🧠 Embedding Model: MiniLM-L6-v2")
    st.success("⚡ Local LLM: Qwen2.5-1.5B Q4")
    st.success(f"⚡ Inference: {INFERENCE_TYPE}")

    st.markdown("---")

    st.markdown("""
    ### ✨ Features
    
    - 📊 Hybrid ATS scoring
    - 🧠 Semantic similarity matching
    - 🔍 Lexical keyword analysis
    - 📝 Local recruiter-style analysis
    - 📄 PDF/DOCX/TXT parsing
    - 💾 Persistent report history
    """)

# ==========================================================
# LOAD ENGINE
# ==========================================================

engine = load_ats_engine()

# ==========================================================
# TABS
# ==========================================================

tab_analysis, tab_history = st.tabs([
    "📄 New Analysis",
    "🗂️ Saved Reports"
])

# ==========================================================
# ANALYSIS TAB
# ==========================================================

with tab_analysis:

    # ------------------------------------------------------
    # META INPUTS
    # ------------------------------------------------------

    meta_col1, meta_col2 = st.columns(2)

    candidate_name = meta_col1.text_input(
        "Candidate Name",
        placeholder="e.g. Jane Doe"
    )

    job_title = meta_col2.text_input(
        "Target Job Title",
        placeholder="e.g. Senior Data Scientist"
    )

    # ------------------------------------------------------
    # FILE INPUT SECTION
    # ------------------------------------------------------

    with st.expander(
        "📥 Upload & Review Documents",
        expanded=True
    ):

        col1, col2 = st.columns(2)

        # ==================================================
        # RESUME
        # ==================================================

        with col1:

            st.subheader("📄 Candidate Resume")

            resume_file = st.file_uploader(
                "Upload Resume",
                type=["pdf", "docx", "txt"],
                key="resume_upload"
            )

            parsed_resume = ""

            if resume_file:

                with st.spinner("Parsing resume..."):

                    parsed_resume = parse_uploaded_file(
                        resume_file
                    )

            resume_input = st.text_area(
                "Resume Text",
                value=parsed_resume,
                height=300,
                placeholder="Paste resume content or upload a file..."
            )

        # ==================================================
        # JOB DESCRIPTION
        # ==================================================

        with col2:

            st.subheader("💼 Job Description")

            jd_file = st.file_uploader(
                "Upload Job Description",
                type=["pdf", "docx", "txt"],
                key="jd_upload"
            )

            parsed_jd = ""

            if jd_file:

                with st.spinner("Parsing job description..."):

                    parsed_jd = parse_uploaded_file(
                        jd_file
                    )

            jd_input = st.text_area(
                "Job Description Text",
                value=parsed_jd,
                height=300,
                placeholder="Paste job description or upload a file..."
            )

    # ------------------------------------------------------
    # VALIDATION ACTION
    # ------------------------------------------------------

    st.markdown("")
    
    col_spacer1, col_button, col_spacer2 = st.columns([1, 2, 1])
    
    with col_button:
        run_analysis = st.button(
            "⚡ Run Validation Pipeline",
            use_container_width=True,
            type="primary"
        )
    
    st.markdown("")

    # ======================================================
    # VALIDATION
    # ======================================================

    if run_analysis:

        if not candidate_name.strip():

            st.warning(
                "Please enter candidate name."
            )

            st.stop()

        if not job_title.strip():

            st.warning(
                "Please enter target job title."
            )

            st.stop()

        if not resume_input.strip():

            st.warning(
                "Resume content is empty."
            )

            st.stop()

        if not jd_input.strip():

            st.warning(
                "Job description content is empty."
            )

            st.stop()

        # ==================================================
        # PROCESSING WITH PROGRESS BAR
        # ==================================================

        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Stage 1: Load Models
        status_text.write("0% — Loading semantic embedding engine...")
        progress_bar.progress(10)
        time.sleep(0.2)

        # Stage 2: Parse Resume
        status_text.write("25% — Parsing resume and job description...")
        progress_bar.progress(25)
        time.sleep(0.2)

        # Stage 3: Semantic Analysis
        status_text.write("50% — Computing semantic similarity...")
        progress_bar.progress(50)
        score_data = engine.get_evaluation(
            resume_input,
            jd_input
        )

        # Stage 4: LLM Evaluation
        status_text.write("75% — Generating recruiter-style analysis...")
        progress_bar.progress(75)
        html_report = generate_html_report(
            resume_input,
            jd_input
        )

        # Stage 5: Save Report
        status_text.write("100% — Saving report to database...")
        progress_bar.progress(100)
        save_report(
            candidate_name=candidate_name,
            job_title=job_title,
            ats_score=score_data["ats_score"],
            fit_category=score_data["good_fit"],
            semantic_score=score_data["breakdown"]["semantic_score"],
            lexical_score=score_data["breakdown"]["lexical_score"],
            html_report=html_report
        )

        elapsed_time = time.time() - start_time
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show success confirmation
        st.success(f"✅ Report saved successfully — Processed in {elapsed_time:.2f}s")

        # ==================================================
        # SCROLL ANCHOR
        # ==================================================
        
        st.markdown('<a id="dashboard"></a>', unsafe_allow_html=True)
        st.markdown("")

        # ==================================================
        # SCORE DASHBOARD - KPI CARDS
        # ==================================================

        st.markdown("---")
        st.subheader("📊 ATS Evaluation Dashboard")
        
        # Execution timer
        st.caption(f"⚡ Pipeline executed in {elapsed_time:.2f} seconds")

        col1, col2, col3 = st.columns(3)
        
        score_emoji = get_color_coded_score(score_data["ats_score"])

        col1.metric(
            f"{score_emoji} Overall ATS Score",
            f"{score_data['ats_score']}/100"
        )

        col2.metric(
            "📋 Fit Classification",
            score_data["good_fit"]
        )

        semantic_score = score_data["breakdown"]["semantic_score"]
        lexical_score = score_data["breakdown"]["lexical_score"]

        col3.metric(
            "🔍 Semantic / Lexical",
            f"{semantic_score} / {lexical_score}"
        )

        # ==================================================
        # MISSING KEYWORDS SECTION
        # ==================================================

        st.markdown("---")
        st.subheader("🎯 Missing Keywords & Top Skills")
        
        missing_keywords = extract_keywords_from_text(jd_input)
        resume_keywords = extract_keywords_from_text(resume_input)
        
        col_missing, col_present = st.columns(2)
        
        with col_missing:
            st.write("**Keywords in JD:**")
            if missing_keywords:
                for kw in missing_keywords[:5]:
                    st.write(f"• {kw}")
            else:
                st.caption("No common keywords detected")
        
        with col_present:
            st.write("**Keywords in Resume:**")
            if resume_keywords:
                for kw in resume_keywords[:5]:
                    st.write(f"✓ {kw}")
            else:
                st.caption("No common keywords detected")

        # ==================================================
        # SCORE INTERPRETATION
        # ==================================================

        st.markdown("---")

        if score_data["ats_score"] >= 70:
            st.success("✅ Excellent ATS alignment detected.")
        elif score_data["ats_score"] >= 40:
            st.warning("⚠️ Moderate ATS alignment detected.")
        else:
            st.error("❌ Low ATS alignment detected.")

        # ==================================================
        # COLLAPSE UPLOAD SECTION AFTER ANALYSIS
        # ==================================================

        st.markdown("---")

        # ==================================================
        # HTML REPORT
        # ==================================================

        st.subheader("📑 Detailed Recruiter Analysis")

        st.html(
            html_report
        )

# ==========================================================
# HISTORY TAB
# ==========================================================

with tab_history:

    st.subheader("🗂️ Saved Candidate Reports")

    reports = get_all_reports()

    # ------------------------------------------------------
    # NO REPORTS
    # ------------------------------------------------------

    if not reports:

        st.info(
            "No reports available yet."
        )

    # ------------------------------------------------------
    # REPORT TABLE
    # ------------------------------------------------------

    else:

        overview_data = []

        for report in reports:

            overview_data.append({

                "Date":
                report.created_at.strftime(
                    "%Y-%m-%d %H:%M"
                ),

                "Candidate":
                report.candidate_name,

                "Role":
                report.job_title,

                "ATS Score":
                report.ats_score,

                "Fit":
                report.fit_category
            })

        df = pd.DataFrame(overview_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # ==================================================
        # REPORT SELECTOR
        # ==================================================

        report_options = {

            f"{r.candidate_name} | "
            f"{r.job_title} | "
            f"{r.ats_score}/100":
            r

            for r in reports
        }

        selected_key = st.selectbox(
            "Select Saved Report",
            list(report_options.keys())
        )

        # ==================================================
        # DISPLAY REPORT
        # ==================================================

        if selected_key:

            selected_report = report_options[
                selected_key
            ]

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(
                "ATS Score",
                selected_report.ats_score
            )

            metric_col2.metric(
                "Fit",
                selected_report.fit_category
            )

            metric_col3.metric(
                "Semantic / Lexical",
                f"{selected_report.semantic_score} / "
                f"{selected_report.lexical_score}"
            )

            st.markdown("---")

            st.html(
                selected_report.html_report
            )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#64748b;font-size:0.85rem;margin-top:40px;">
        Deblina Roy | 2026 | NLP ATS Project
    </div>
    """,
    unsafe_allow_html=True
)
