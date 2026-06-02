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
# SESSION STATE INITIALIZATION
# ==========================================================

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ==========================================================
# DEVICE DETECTION (Torch-based)
# ==========================================================

try:
    import torch
    if torch.backends.mps.is_available():
        INFERENCE_TYPE = "🔧 Inference: Apple Metal (MPS Optimized)"
    else:
        INFERENCE_TYPE = "⚡ Inference: Linux CPU"
except ImportError:
    INFERENCE_TYPE = "⚡ Inference: CPU (PyTorch Unavailable)"

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
    # Convert everything to lowercase to fix duplicate/missing bugs
    return list(set([k.lower() for k in keywords]))[:5]

def get_color_coded_score(score: float) -> str:
    """Return emoji and color for ATS score."""
    if score >= 80:
        return "🟢"
    elif score >= 60:            # Changed from 40 to 60 to match the engine
        return "🟡"
    else:
        return "🔴"

def get_score_interpretation(score: float) -> tuple:
    """Return interpretation and color for ATS score."""
    if score >= 80:              # Changed from 70 to 80
        return "✅ Strong ATS compatibility. Resume is well-optimized.", "success"
    elif score >= 60:            # Changed from 40 to 60
        return "⚠️ Moderate ATS compatibility. Several improvements possible.", "warning"
    else:
        return "❌ Low ATS compatibility. Significant resume optimization recommended.", "error"

def compare_keywords(jd_keywords: list, resume_keywords: list) -> tuple:
    """Return missing and matched keywords."""
    jd_set = set(jd_keywords)
    resume_set = set(resume_keywords)
    missing = list(jd_set - resume_set)
    matched = list(jd_set & resume_set)
    return missing, matched

# ==========================================================
# RESOURCE CACHING
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_ats_engine():
    return AdvancedHybridATS(semantic_weight=0.60)

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

    st.header("🚀 System Architecture")

    # Technical Specs
    st.success("🧠 Embedding: MiniLM-L6-v2")
    st.success("⚡ LLM: Qwen2.5-1.5B Q4_K_M")
    st.success(INFERENCE_TYPE)

    st.markdown("---")

    # Quick Start Guide
    st.markdown("""
    ### 📖 Quick Start
    1. Input candidate name & role
    2. Upload/Paste Resume & JD
    3. Click **Run Validation Pipeline**
    """)

    st.markdown("---")

    # Trimmed Features List
    st.markdown("""
    ### ✨ Core Features
    - 📊 Hybrid ATS scoring engine
    - 🧠 Semantic similarity matching
    - 🔍 Lexical keyword extraction
    - 📝 Offline recruiter-style analysis
    - 💾 Persistent SQLite history
    """)
    
    st.markdown("---")

    # Portfolio Links
    st.markdown("""
    ### 👩‍💻 Developer
    **Deblina Roy**  
    [View Source on GitHub ↗](https://github.com/roy-deblina/nlp-ats-project)
    """)

    # Privacy Disclaimer (Massive green flag for tech interviews)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("🔒 **Privacy Note:** All NLP processing is executed securely in-memory. Uploaded resumes are never stored permanently or used to train public models.")

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
    # FILE INPUT SECTION (COLLAPSES AFTER ANALYSIS)
    # ------------------------------------------------------

    with st.expander(
        "📥 Upload & Review Documents",
        expanded=not st.session_state.analysis_complete
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
            
            # ADD THIS WARNING
            st.caption("⚠️ **Note:** To prevent memory limits on the cloud, resume text is automatically truncated to the first 3,500 characters.")

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
            
            # ADD THIS WARNING
            st.caption("⚠️ **Note:** To prevent memory limits on the cloud, job description text is automatically truncated to the first 1,500 characters.")

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
        status_text.write("75% — Generating analysis...")
        progress_bar.progress(75)
        # Add the spinning circle context manager here
        jd_keywords_list = extract_keywords_from_text(jd_input)
        resume_keywords_list = extract_keywords_from_text(resume_input)
        missing_kws, matched_kws = compare_keywords(jd_keywords_list, resume_keywords_list)
        with st.spinner("🔄 Complete report analysis... (This may take a moment)"):
            html_report = generate_html_report(
                resume_text=resume_input,
                jd_text=jd_input,
                ats_score=score_data["ats_score"],
                missing_keywords=missing_kws
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

        # Mark analysis as complete for session state
        st.session_state.analysis_complete = True

        # Show success confirmation (compact)
        st.toast(f"✅ Report generated in {elapsed_time:.2f}s")

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
        # MISSING VS MATCHED KEYWORDS
        # ==================================================

        st.markdown("---")
        st.subheader("🎯 Keyword Analysis")
        
        jd_keywords = extract_keywords_from_text(jd_input)
        resume_keywords = extract_keywords_from_text(resume_input)
        missing, matched = compare_keywords(jd_keywords, resume_keywords)
        
        col_missing, col_matched = st.columns(2)
        
        with col_missing:
            st.write("**❌ Missing Keywords:**")
            if missing:
                for kw in missing[:5]:
                    st.write(f"   ❌ {kw}")
            elif not jd_keywords:
                st.caption("No technical keywords found in Job Description.")
            else:
                st.caption("✓ All detected keywords are present!")
        
        with col_matched:
            st.write("**✅ Matched Keywords:**")
            if matched:
                for kw in matched[:5]:
                    st.write(f"   ✓ {kw}")
            elif not jd_keywords:
                st.caption("No technical keywords found in Job Description.")
            else:
                st.caption("No keyword overlap detected")

        # ==================================================
        # RESUME STRENGTHS CARD
        # ==================================================

        st.markdown("---")
        st.subheader("💪 Resume Strengths")
        
        strengths_text = f"""
        ✓ Resume parsed successfully
        ✓ {len(resume_keywords)} technical keywords detected
        ✓ Semantic similarity: {semantic_score}%
        ✓ Ready for {job_title} role
        """
        st.success(strengths_text)

        # ==================================================
        # SCORE BREAKDOWN EXPLANATION
        # ==================================================

        st.markdown("---")
        st.subheader("📈 Score Breakdown")
        
        breakdown_cols = st.columns(3)
        
        # Added 'help' tooltips to explain the metrics on hover
        breakdown_cols[0].metric(
            label="Semantic Match", 
            value=f"{semantic_score}%",
            help="Measures contextual similarity and meaning between resume and JD using AI embeddings."
        )
        breakdown_cols[1].metric(
            label="Keyword Match", 
            value=f"{lexical_score}%",
            help="Measures exact and standardized technical keyword overlap."
        )
        breakdown_cols[2].metric(
            label="Overall ATS", 
            value=f"{score_data['ats_score']}%",
            help="Weighted combination: 60% Semantic + 40% Keyword"
        )
        
        # Added expander to explain the calculations clearly to the user
        with st.expander("ℹ️ How are these scores calculated?"):
            st.markdown("""
            * **Semantic Match (60%):** Uses an AI embedding model to understand the *meaning* and *context* of your resume compared to the job description. It recognizes related concepts even if they aren't exact word matches.
            * **Keyword/Lexical Match (40%):** Scans for exact technical skills, tools, and industry keywords. It applies a standard ATS taxonomy and calculates the overlap percentage.
            * **Overall Score:** A weighted hybrid of both metrics simulating how a modern ATS will rank your profile.
            * **Fit Classification:** 
                * **Strong Fit:** 80-100
                * **Potential Fit:** 60-79
                * **Not a Fit:** Below 60
            """)
        
        st.caption("ATS Score = (Semantic Match × 0.6) + (Keyword Match × 0.4)")

        # ==================================================
        # SCORE INTERPRETATION
        # ==================================================

        st.markdown("---")
        interpretation, msg_type = get_score_interpretation(score_data["ats_score"])
        
        if msg_type == "success":
            st.success(interpretation)
        elif msg_type == "warning":
            st.warning(interpretation)
        else:
            st.error(interpretation)

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
        Deblina Roy | 2026 | NLP ATS Platform
    </div>
    """,
    unsafe_allow_html=True
)
