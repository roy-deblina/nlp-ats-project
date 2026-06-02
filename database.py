# database.py
# Advanced SQLite Persistence Layer
# Optimized for Local ATS Platform

from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

from sqlalchemy.exc import SQLAlchemyError

# ==========================================================
# DATABASE CONFIG
# ==========================================================

DATABASE_URL = "sqlite:///ats_database.db"

# ==========================================================
# ENGINE
# ==========================================================

engine = create_engine(

    DATABASE_URL,

    # SQLite multithread safety
    connect_args={
        "check_same_thread": False
    },

    # Better production behavior
    pool_pre_ping=True,

    # Disable verbose logs
    echo=False
)

# ==========================================================
# SESSION
# ==========================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# ==========================================================
# BASE MODEL
# ==========================================================

Base = declarative_base()

# ==========================================================
# ATS REPORT TABLE
# ==========================================================

class ATSReport(Base):

    __tablename__ = "ats_reports"

    # ------------------------------------------------------
    # PRIMARY KEY
    # ------------------------------------------------------

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # ------------------------------------------------------
    # CANDIDATE INFO
    # ------------------------------------------------------

    candidate_name = Column(
        String(255),
        nullable=False,
        default="Unknown Candidate"
    )

    job_title = Column(
        String(255),
        nullable=False,
        default="Unknown Role"
    )

    # ------------------------------------------------------
    # ATS METRICS
    # ------------------------------------------------------

    ats_score = Column(
        Float,
        nullable=False,
        default=0.0
    )

    fit_category = Column(
        String(100),
        nullable=False,
        default="Not Evaluated"
    )

    semantic_score = Column(
        Float,
        nullable=False,
        default=0.0
    )

    lexical_score = Column(
        Float,
        nullable=False,
        default=0.0
    )

    # ------------------------------------------------------
    # GENERATED REPORT
    # ------------------------------------------------------

    html_report = Column(
        Text,
        nullable=False
    )

    # ------------------------------------------------------
    # OPTIONAL RAW INPUT STORAGE
    # ------------------------------------------------------

    resume_text = Column(
        Text,
        nullable=True
    )

    jd_text = Column(
        Text,
        nullable=True
    )

    # ------------------------------------------------------
    # SYSTEM INFO
    # ------------------------------------------------------

    embedding_model = Column(
        String(255),
        nullable=False,
        default="all-MiniLM-L6-v2"
    )

    llm_model = Column(
        String(255),
        nullable=False,
        default="Qwen2.5-1.5B-Instruct-Q4_K_M"
    )

    # ------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # ------------------------------------------------------
    # STRING REPRESENTATION
    # ------------------------------------------------------

    def __repr__(self):

        return (
            f"<ATSReport("
            f"id={self.id}, "
            f"candidate='{self.candidate_name}', "
            f"score={self.ats_score}"
            f")>"
        )

# ==========================================================
# CREATE TABLES
# ==========================================================

Base.metadata.create_all(bind=engine)

# ==========================================================
# DATABASE SESSION CONTEXT
# ==========================================================

@contextmanager
def get_db_session():

    session = SessionLocal()

    try:

        yield session

        session.commit()

    except Exception:

        session.rollback()

        raise

    finally:

        session.close()

# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    candidate_name: str,
    job_title: str,

    ats_score: float,
    fit_category: str,

    semantic_score: float,
    lexical_score: float,

    html_report: str,

    resume_text: str = None,
    jd_text: str = None,

    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str = "Qwen2.5-1.5B-Instruct-Q4_K_M"
) -> bool:

    try:

        with get_db_session() as session:

            report = ATSReport(

                # ----------------------------------------------
                # META
                # ----------------------------------------------

                candidate_name=candidate_name.strip(),

                job_title=job_title.strip(),

                # ----------------------------------------------
                # SCORES
                # ----------------------------------------------

                ats_score=round(
                    float(ats_score),
                    2
                ),

                fit_category=fit_category,

                semantic_score=round(
                    float(semantic_score),
                    2
                ),

                lexical_score=round(
                    float(lexical_score),
                    2
                ),

                # ----------------------------------------------
                # REPORT
                # ----------------------------------------------

                html_report=html_report,

                # ----------------------------------------------
                # OPTIONAL RAW TEXT
                # ----------------------------------------------

                resume_text=resume_text,

                jd_text=jd_text,

                # ----------------------------------------------
                # MODEL INFO
                # ----------------------------------------------

                embedding_model=embedding_model,

                llm_model=llm_model
            )

            session.add(report)

            return True

    except SQLAlchemyError as e:

        print(
            f"[DATABASE ERROR] Failed to save report: {e}"
        )

        return False

    except Exception as e:

        print(
            f"[SYSTEM ERROR] Unexpected DB failure: {e}"
        )

        return False

# ==========================================================
# GET ALL REPORTS
# ==========================================================

def get_all_reports():

    try:

        with get_db_session() as session:

            reports = (

                session.query(ATSReport)

                .order_by(
                    ATSReport.created_at.desc()
                )

                .all()
            )

            return reports

    except Exception as e:

        print(
            f"[DATABASE ERROR] Failed fetching reports: {e}"
        )

        return []

# ==========================================================
# GET SINGLE REPORT
# ==========================================================

def get_report_by_id(report_id: int):

    try:

        with get_db_session() as session:

            report = (

                session.query(ATSReport)

                .filter(
                    ATSReport.id == report_id
                )

                .first()
            )

            return report

    except Exception as e:

        print(
            f"[DATABASE ERROR] Failed fetching report: {e}"
        )

        return None

# ==========================================================
# DELETE REPORT
# ==========================================================

def delete_report(report_id: int) -> bool:

    try:

        with get_db_session() as session:

            report = (

                session.query(ATSReport)

                .filter(
                    ATSReport.id == report_id
                )

                .first()
            )

            if not report:

                return False

            session.delete(report)

            return True

    except Exception as e:

        print(
            f"[DATABASE ERROR] Failed deleting report: {e}"
        )

        return False

# ==========================================================
# DATABASE HEALTH CHECK
# ==========================================================

def database_health_check() -> bool:

    try:

        with get_db_session() as session:

            session.execute("SELECT 1")

            return True

    except Exception as e:

        print(
            f"[DATABASE ERROR] Health check failed: {e}"
        )

        return False