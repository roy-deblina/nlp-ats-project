# analyzer.py
# Fully Local ATS HTML Analyzer using Qwen2.5 GGUF + llama.cpp
# Optimized for Apple Silicon (M1/M2/M3) + Cloud Deployment

import os
import warnings
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import textwrap
import streamlit as st

# Suppress harmless llama_context size mismatch warning
warnings.filterwarnings(
    "ignore",
    message=".*n_ctx_seq.*n_ctx_train.*"
)

# ==========================================================
# LOAD LOCAL QWEN MODEL - CLOUD & STORAGE OPTIMIZED
# ==========================================================
@st.cache_resource
def load_llm():
    # 1. Resolve model path dynamically inside the cached resource execution
    raw_path = os.environ.get(
        "QWEN_MODEL_PATH", 
        "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    
    # 2. Check if this is a HuggingFace Hub URL and pull down via huggingface_hub safely
    if "huggingface.co" in raw_path:
        with st.spinner("📥 Downloading Qwen model from Hugging Face hub (first run only)..."):
            resolved_model_path = hf_hub_download(
                repo_id="111deblina/qwen-ats-model",
                filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                local_dir_use_symlinks=False  # Avoids deprecated resume_download warnings completely
            )
    else:
        resolved_model_path = raw_path
    
    # 3. Detect if running on cloud vs local
    is_cloud = os.environ.get("STREAMLIT_CLOUD", "false").lower() == "true"
    
    # CONFIGURATION: Adjust based on cloud memory thresholds
    if is_cloud:
        n_ctx = int(os.environ.get("LLAMA_N_CTX", "512"))       # Drastically reduced to prevent OOM
        n_batch = int(os.environ.get("LLAMA_N_BATCH", "32"))    # Lower batch size
        n_threads = int(os.environ.get("LLAMA_N_THREADS", "1")) # Single thread for lower overhead
    else:
        n_ctx = int(os.environ.get("LLAMA_N_CTX", "2048"))
        n_batch = int(os.environ.get("LLAMA_N_BATCH", "128"))
        n_threads = int(os.environ.get("LLAMA_N_THREADS", "4"))

    # 4. Initialize Llama engine with your resolved local cache directory pointer
    return Llama(
        model_path=resolved_model_path,
        n_ctx=n_ctx,
        n_gpu_layers=0,
        n_threads=n_threads,
        use_mmap=True,
        use_mlock=False,
        n_batch=n_batch,
        logits_all=False,
        verbose=False
    )

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are a senior technical recruiter and ATS evaluator.

Your job is to analyze a candidate resume against
a job description and generate a professional ATS report.

You must think like a real recruiter.

FOCUS ON:

- exact skill matches
- missing technologies
- years of experience alignment
- role relevance
- education alignment
- keyword optimization
- recruiter hiring confidence
- ATS compatibility

STRICT RULES:

1. Return ONLY valid HTML.
2. No markdown.
3. No explanations outside HTML.
4. Use clean professional styling.
5. Use short recruiter-style bullet points.
6. Be highly analytical and specific.
7. Mention exact missing skills from JD.
8. Mention exact strengths from resume.
9. Add recruiter recommendation section.
10. Add ATS optimization suggestions.

The report should look like a professional recruiter dashboard.
"""

# ==========================================================
# MAIN REPORT GENERATION
# ==========================================================

def generate_html_report(
    resume_text: str,
    jd_text: str
) -> str:

    try:
        llm = load_llm()
        # ==================================================
        # INPUT CLEANING
        # ==================================================

        if not resume_text.strip():
            return """
            <div style='padding:20px;color:red;'>
                Resume text is empty.
            </div>
            """

        if not jd_text.strip():
            return """
            <div style='padding:20px;color:red;'>
                Job description text is empty.
            </div>
            """

        # ==================================================
        # NORMALIZE WHITESPACES
        # VERY IMPORTANT FOR LOCAL LLM QUALITY
        # ==================================================

        resume_text = " ".join(resume_text.split())
        jd_text = " ".join(jd_text.split())

        # ==================================================
        # TRUNCATION SAFETY
        # ==================================================

        resume_text = resume_text[:3500]
        jd_text = jd_text[:1500]

        # ==================================================
        # USER PROMPT
        # ==================================================

        user_prompt = f"""
        Analyze this candidate professionally.

        ==================================================
        JOB DESCRIPTION
        ==================================================

        {jd_text}

        ==================================================
        CANDIDATE RESUME
        ==================================================

        {resume_text}

        ==================================================

        Generate a professional ATS recruiter report.

        Required Sections:

        1. Executive Summary
        2. ATS Match Analysis
        3. Technical Skills Evaluation
        4. Experience Evaluation
        5. Education & Certifications
        6. Missing Keywords & Skill Gaps
        7. Recruiter Concerns
        8. ATS Optimization Suggestions
        9. Final Hiring Recommendation

        Instructions:

        - Mention important matching skills
        - Mention exact missing keywords
        - Mention experience gaps
        - Mention ATS strengths
        - Mention recruiter concerns
        - Mention resume weaknesses
        - Use recruiter-style analysis
        - Keep analysis detailed but concise
        - Return ONLY HTML
        """

        # ==================================================
        # QWEN CHAT TEMPLATE
        # ==================================================

        prompt = textwrap.dedent(f"""
        <|im_start|>system
        {SYSTEM_PROMPT}<|im_end|>
        <|im_start|>user
        {user_prompt}<|im_end|>
        <|im_start|>assistant
        """)

        # ==================================================
        # LOCAL INFERENCE
        # ==================================================

        response = llm(

            prompt,

            max_tokens=600,

            temperature=0.2,

            top_p=0.92,

            repeat_penalty=1.12,

            stop=[
                "<|im_end|>",
                "<|im_start|>"
            ]
        )

        generated_text = response["choices"][0]["text"].strip()

        # ==================================================
        # SAFETY CLEANING
        # ==================================================

        generated_text = generated_text.replace("```html", "")
        generated_text = generated_text.replace("```", "")
        
        # Strip out conversational filler if it exists before the first HTML tag
        if "<div" in generated_text:
            generated_text = generated_text[generated_text.find("<div"):]
        elif "<html" in generated_text:
            generated_text = generated_text[generated_text.find("<html"):]  

        # ==================================================
        # BASIC HTML VALIDATION
        # ==================================================

        if "<div" not in generated_text and "<html" not in generated_text:

            generated_text = f"""
            <div style="
                font-family: Arial, sans-serif;
                padding: 24px;
                border-radius: 12px;
                background: #ffffff;
                border: 1px solid #dee2e6;
                line-height: 1.6;
            ">

                <h2 style="color:#0d6efd;">
                    ATS Recruiter Analysis
                </h2>

                <div style="
                    background:#f8f9fa;
                    padding:18px;
                    border-radius:8px;
                    margin-top:15px;
                ">

            {generated_text}

        </div>

    </div>
    """
        # st.code(generated_text, language="html")

        return generated_text

    # ======================================================
    # ERROR HANDLING
    # ======================================================

    except Exception as e:

        return f"""
        <div style="
            padding:20px;
            background-color:#f8d7da;
            color:#721c24;
            border-radius:10px;
            font-family:Arial;
        ">
            <h3>Local LLM Error</h3>
            <p>{str(e)}</p>
        </div>
        """