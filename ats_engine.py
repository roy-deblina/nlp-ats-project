# ats_engine.py
# Lightweight Hybrid ATS Scoring Engine
# Optimized for Apple Silicon (M1/M2/M3) + Cloud Deployment

import re
import os
import torch
from typing import Dict, Any
from sentence_transformers import SentenceTransformer


class AdvancedHybridATS:

    def __init__(
        self,
        model_name: str = None,
        semantic_weight: float = 0.40
    ):
        # Default to HuggingFace Hub for cloud compatibility
        if model_name is None:
            # Check if local model exists, otherwise use HF Hub
            local_path = "models/embeddings/all-MiniLM-L6-v2"
            if os.path.exists(local_path):
                model_name = local_path
            else:
                # Cloud: auto-download from HuggingFace
                model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # ---------------------------------------------------------
        # DEVICE SELECTION - CLOUD COMPATIBLE
        # ---------------------------------------------------------

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---------------------------------------------------------
        # LOAD LIGHTWEIGHT EMBEDDING MODEL
        # ---------------------------------------------------------

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        # ---------------------------------------------------------
        # WEIGHT CONFIGURATION
        # ---------------------------------------------------------

        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight

        # ---------------------------------------------------------
        # TECHNICAL & BUSINESS TAXONOMY
        # ---------------------------------------------------------

        self.taxonomy = {

            # =====================================================
            # CLOUD / DEVOPS
            # =====================================================

            "amazon web services": "aws",
            "google cloud platform": "gcp",
            "google cloud": "gcp",
            "microsoft azure": "azure",
            "kubernetes": "k8s",
            "docker": "containerization",
            "continuous integration": "cicd",
            "continuous deployment": "cicd",
            "continuous delivery": "cicd",
            "infrastructure as code": "iac",
            "github actions": "cicd",

            # =====================================================
            # AI / ML / DATA SCIENCE
            # =====================================================

            "machine learning": "ml",
            "artificial intelligence": "ai",
            "natural language processing": "nlp",
            "deep learning": "dl",
            "computer vision": "cv",
            "large language models": "llm",
            "generative ai": "genai",
            "pytorch": "dl",
            "tensorflow": "dl",
            "scikit-learn": "sklearn",

            # =====================================================
            # FRONTEND / BACKEND
            # =====================================================

            "react.js": "react",
            "node.js": "node",
            "vue.js": "vue",
            "angular.js": "angular",
            "javascript": "js",
            "typescript": "ts",
            "golang": "go",
            "c plus plus": "c++",
            "c sharp": "c#",
            "ruby on rails": "ruby",

            # =====================================================
            # DATABASES / DATA ENGINEERING
            # =====================================================

            "sql server": "sql",
            "microsoft sql server": "sql",
            "postgresql": "sql",
            "postgres": "sql",
            "mongodb": "nosql",
            "apache spark": "spark",
            "snowflake": "data warehousing",

            # =====================================================
            # HR / RECRUITING
            # =====================================================

            "human resources": "hr",
            "talent acquisition": "recruiting",
            "applicant tracking system": "ats",
            "applicant tracking systems": "ats",
            "employee relations": "hr",
            "human resource information system": "hris",
            "performance management": "hr",
            "diversity, equity, and inclusion": "dei",
            "diversity and inclusion": "dei",

            # =====================================================
            # FINANCE / ACCOUNTING
            # =====================================================

            "financial planning and analysis": "fp&a",
            "certified public accountant": "cpa",
            "generally accepted accounting principles": "gaap",
            "profit and loss": "p&l",
            "accounts payable": "ap",
            "accounts receivable": "ar",
            "mergers and acquisitions": "m&a",
            "return on investment": "roi",

            # =====================================================
            # MARKETING / SEO
            # =====================================================

            "search engine optimization": "seo",
            "search engine marketing": "sem",
            "social media marketing": "smm",
            "pay per click": "ppc",
            "content management system": "cms",
            "customer relationship management": "crm",
            "public relations": "pr",
            "key performance indicators": "kpi",
            "go to market": "gtm",

            # =====================================================
            # SALES
            # =====================================================

            "business to business": "b2b",
            "business to consumer": "b2c",
            "business development": "sales",
            "salesforce": "crm",
            "hubspot": "crm",
            "account executive": "sales",
            "customer success": "sales",

            # =====================================================
            # OPERATIONS / PM
            # =====================================================

            "project management professional": "pmp",
            "supply chain management": "scm",
            "enterprise resource planning": "erp",
            "lean six sigma": "six sigma",
            "standard operating procedures": "sop",
            "quality assurance": "qa",

            # =====================================================
            # DESIGN
            # =====================================================

            "user experience": "ux",
            "user interface": "ui",
            "adobe creative suite": "adobe",
            "adobe creative cloud": "adobe",
            "interaction design": "ui/ux",

            # =====================================================
            # METHODOLOGIES
            # =====================================================

            "agile methodology": "agile",
            "scrum master": "scrum",
            "project management": "management"
        }

        # ---------------------------------------------------------
        # STOP WORDS
        # ---------------------------------------------------------

        self.stop_words = {
            "this", "that", "with", "from", "your",
            "have", "will", "team", "work", "client",
            "must", "using", "experience", "skills",
            "knowledge", "ability", "strong", "good",
            "excellent", "candidate", "preferred",
            "required", "responsible"
        }

    # =============================================================
    # TEXT CLEANING
    # =============================================================

    def _clean_text(self, text: str) -> str:

        text = text.lower()

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # =============================================================
    # TAXONOMY NORMALIZATION
    # =============================================================

    def _apply_taxonomy(self, text: str) -> str:

        text = self._clean_text(text)

        for variant, standard in self.taxonomy.items():

            text = re.sub(
                rf"\b{re.escape(variant)}\b",
                standard,
                text
            )

        return text

    # =============================================================
    # KEYWORD EXTRACTION
    # =============================================================

    def _extract_keywords(self, text: str) -> set:

        words = set(
            re.findall(r"(?<!\S)[a-zA-Z0-9\+#\.]{2,}(?!\S)", text)
        )

        return {
            word
            for word in words
            if word not in self.stop_words
        }

    # =============================================================
    # LEXICAL MATCHING
    # =============================================================

    def _calculate_lexical_overlap(
        self,
        resume: str,
        jd: str
    ) -> float:

        resume_clean = self._apply_taxonomy(resume)
        jd_clean = self._apply_taxonomy(jd)

        jd_keywords = self._extract_keywords(jd_clean)
        resume_keywords = self._extract_keywords(resume_clean)

        if not jd_keywords:
            return 0.0

        matched_keywords = jd_keywords.intersection(resume_keywords)

        # ---------------------------------------------------------
        # TECH BONUS
        # ---------------------------------------------------------

        taxonomy_terms = set(self.taxonomy.values())

        tech_matches = matched_keywords.intersection(
            taxonomy_terms
        )

        base_score = (
            len(matched_keywords) / len(jd_keywords)
        ) * 100

        tech_bonus = (
            len(tech_matches) / len(jd_keywords)
        ) * 20

        final_lexical_score = min(
            100.0,
            base_score + tech_bonus
        )

        return round(final_lexical_score, 2)

    # =============================================================
    # SEMANTIC EMBEDDINGS
    # =============================================================

    def _get_embedding(self, text: str):

        text = self._apply_taxonomy(text)

        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        return embedding

    # =============================================================
    # SEMANTIC SCORING
    # =============================================================

    def _calculate_semantic_similarity(
        self,
        resume: str,
        jd: str
    ) -> float:

        resume_embedding = self._get_embedding(resume)
        jd_embedding = self._get_embedding(jd)

        similarity = torch.nn.functional.cosine_similarity(
            resume_embedding,
            jd_embedding,
            dim=0
        ).item()

        # ---------------------------------------------------------
        # SCORE NORMALIZATION
        # ---------------------------------------------------------

        semantic_score = ((similarity + 1) / 2) * 100

        semantic_score = min(
            100.0,
            max(0.0, semantic_score)
        )

        return round(semantic_score, 2)

    # =============================================================
    # FINAL EVALUATION
    # =============================================================

    def get_evaluation(
        self,
        resume: str,
        jd: str
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------

        if (
            not isinstance(resume, str)
            or not isinstance(jd, str)
            or not resume.strip()
            or not jd.strip()
        ):

            return {
                "ats_score": 0.0,
                "good_fit": "No Fit",
                "breakdown": {
                    "semantic_score": 0.0,
                    "lexical_score": 0.0
                }
            }

        # ---------------------------------------------------------
        # CALCULATE SCORES
        # ---------------------------------------------------------

        semantic_score = self._calculate_semantic_similarity(
            resume,
            jd
        )

        lexical_score = self._calculate_lexical_overlap(
            resume,
            jd
        )

        # ---------------------------------------------------------
        # WEIGHTED HYBRID SCORE
        # ---------------------------------------------------------

        final_score = (
            (semantic_score * self.semantic_weight)
            +
            (lexical_score * self.lexical_weight)
        )

        final_score = round(final_score, 2)

        # ---------------------------------------------------------
        # FIT CLASSIFICATION
        # ---------------------------------------------------------

        if final_score >= 80:
            fit = "Strong Fit"

        elif final_score >= 60:
            fit = "Potential Fit"

        else:
            fit = "Not a Fit"

        # ---------------------------------------------------------
        # RESPONSE
        # ---------------------------------------------------------

        return {
            "ats_score": final_score,
            "good_fit": fit,
            "breakdown": {
                "semantic_score": semantic_score,
                "lexical_score": lexical_score
            }
        }