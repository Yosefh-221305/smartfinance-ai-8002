"""
SmartFinance AI — Classification API (v3-fixed, Port 8002)

PERUBAHAN v3 (perbaikan):
  - HAPUS is_sheet3 dari build_features() — tidak relevan di production
  - build_features() sekarang murni menghitung dari data keuangan aktual
  - Feature list di-load dari feature_columns.json yang sudah diperbarui
  - Validasi: jika is_sheet3 ada di _feature_cols, otomatis dihapus

Jalankan:
  uvicorn step4_predict_api:app --reload --port 8002

Dokumentasi API: http://localhost:8002/docs
"""

import os
import sys
import json
import pickle
import time
import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smartfinance.classify")

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
FEATURE_PATH= os.path.join(BASE_DIR, "models", "feature_columns.json")

RECOMMENDATION_API_URL = os.getenv("RECOMMENDATION_API_URL", "http://localhost:8001")
API_KEY        = os.getenv("SMARTFINANCE_API_KEY", "")
RATE_LIMIT     = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
ALLOWED_ORIGINS= os.getenv("ALLOWED_ORIGINS", "*").split(",")

LABEL_NAMES = {0: "Hemat", 1: "Normal", 2: "Boros"}
VALID_MONTHS = {
    "Januari", "Februari", "Maret",    "April",
    "Mei",     "Juni",     "Juli",     "Agustus",
    "September","Oktober", "November", "Desember",
}
VALID_CATEGORIES = {
    "makanan_minuman", "transportasi", "hiburan", "belanja_online",
    "tagihan_utilitas", "kesehatan", "pendidikan", "tabungan_investasi", "lainnya",
}

# Kolom yang TIDAK boleh ada di fitur model — meski ada di JSON lama
FORBIDDEN_FEATURES = {"is_sheet3"}

# ── Rate Limiter ───────────────────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(request: Request) -> None:
    ip  = request.client.host if request.client else "unknown"
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 60.0]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit: maks {RATE_LIMIT} request/menit.",
            headers={"Retry-After": "60"},
        )
    _rate_store[ip].append(now)


# ── Auth ───────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> None:
    if not API_KEY:
        return
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key tidak valid.")


# ── Load Model ─────────────────────────────────────────────────────────────
_model = None
_feature_cols: list[str] = []


def load_model():
    global _model, _feature_cols
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            "Jalankan step3_train_model.py untuk melatih model baru."
        )
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(FEATURE_PATH, "r") as f:
        meta = json.load(f)

    raw_features = meta["features"]

    # Pastikan kolom terlarang (is_sheet3) tidak ada di feature list
    removed = [f for f in raw_features if f in FORBIDDEN_FEATURES]
    _feature_cols = [f for f in raw_features if f not in FORBIDDEN_FEATURES]

    if removed:
        logger.warning(
            f"Kolom berikut dihapus dari feature list (tidak valid di production): {removed}"
        )
    logger.info(f"Model loaded: {len(_feature_cols)} fitur aktif")


try:
    load_model()
except FileNotFoundError as e:
    logger.warning(f"Model belum ada: {e}")


# ── Feature Engineering ────────────────────────────────────────────────────
def build_features(data: dict) -> pd.DataFrame:
    """
    Hitung semua fitur dari input mentah.
    TIDAK ADA is_sheet3 — di production nilai ini selalu 0 dan tidak informatif.
    Konsisten dengan step2_preprocessing.py v3.
    """
    income  = data["total_income"]
    expense = data["total_expense"]
    savings = income - expense

    def ratio(val):
        return val / income if income > 0 else 0.0

    cat_expenses = {
        "makanan_minuman":    data.get("expense_makanan_minuman",    0),
        "transportasi":       data.get("expense_transportasi",       0),
        "hiburan":            data.get("expense_hiburan",            0),
        "belanja_online":     data.get("expense_belanja_online",     0),
        "tagihan_utilitas":   data.get("expense_tagihan_utilitas",   0),
        "kesehatan":          data.get("expense_kesehatan",          0),
        "pendidikan":         data.get("expense_pendidikan",         0),
        "tabungan_investasi": data.get("expense_tabungan_investasi", 0),
    }

    IDEAL_LIMITS = {
        "makanan_minuman":    0.30,
        "transportasi":       0.15,
        "hiburan":            0.10,
        "belanja_online":     0.10,
        "tagihan_utilitas":   0.10,
        "kesehatan":          0.10,
        "pendidikan":         0.05,
        "tabungan_investasi": 0.20,
    }
    n_overspent = sum(
        1 for cat, val in cat_expenses.items()
        if ratio(val) > IDEAL_LIMITS.get(cat, 0.10)
    )

    konsumtif = cat_expenses["hiburan"] + cat_expenses["belanja_online"]
    produktif = cat_expenses["pendidikan"] + cat_expenses["tabungan_investasi"]

    row = {
        "total_income":                income,
        "total_expense":               expense,
        "savings_pct":                 ratio(savings) * 100,
        "total_expense_pct":           ratio(expense) * 100,
        "expense_ratio_makanan":       ratio(cat_expenses["makanan_minuman"]),
        "expense_ratio_transportasi":  ratio(cat_expenses["transportasi"]),
        "expense_ratio_hiburan":       ratio(cat_expenses["hiburan"]),
        "expense_ratio_belanja_online":ratio(cat_expenses["belanja_online"]),
        "expense_ratio_tabungan":      ratio(cat_expenses["tabungan_investasi"]),
        "n_overspent_categories":      n_overspent,
        "expense_makanan_minuman":     cat_expenses["makanan_minuman"],
        "expense_transportasi":        cat_expenses["transportasi"],
        "expense_hiburan":             cat_expenses["hiburan"],
        "expense_belanja_online":      cat_expenses["belanja_online"],
        "expense_tagihan_utilitas":    cat_expenses["tagihan_utilitas"],
        "expense_kesehatan":           cat_expenses["kesehatan"],
        "expense_pendidikan":          cat_expenses["pendidikan"],
        "expense_tabungan_investasi":  cat_expenses["tabungan_investasi"],
        # is_sheet3 → TIDAK ADA (dihapus)
        "savings_ratio":               max(-1, min(1, ratio(savings))),
        "expense_konsumtif":           konsumtif,
        "expense_ratio_konsumtif":     ratio(konsumtif),
        "expense_produktif":           produktif,
        "expense_ratio_produktif":     ratio(produktif),
        "is_deficit":                  int(expense > income),
    }

    df = pd.DataFrame([row])

    # Pastikan kolom sesuai dengan training (isi 0 jika ada kolom baru)
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Pastikan is_sheet3 tidak masuk meski ada di data dict
    return df[_feature_cols]


# ── Pydantic Models ─────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id":                   "USR001",
                "month":                     "Mei",
                "year":                      2026,
                "total_income":              5000000,
                "total_expense":             4300000,
                "expense_makanan_minuman":   2000000,
                "expense_transportasi":       800000,
                "expense_hiburan":            600000,
                "expense_belanja_online":     500000,
                "expense_tagihan_utilitas":   300000,
                "expense_kesehatan":          100000,
                "expense_pendidikan":               0,
                "expense_tabungan_investasi":       0,
            }
        }
    }

    user_id:                   str   = Field(..., min_length=1, max_length=50)
    month:                     str
    year:                      int   = Field(..., ge=2020, le=2100)
    total_income:              float = Field(..., gt=0)
    total_expense:             float = Field(..., ge=0)
    expense_makanan_minuman:   float = Field(default=0, ge=0)
    expense_transportasi:      float = Field(default=0, ge=0)
    expense_hiburan:           float = Field(default=0, ge=0)
    expense_belanja_online:    float = Field(default=0, ge=0)
    expense_tagihan_utilitas:  float = Field(default=0, ge=0)
    expense_kesehatan:         float = Field(default=0, ge=0)
    expense_pendidikan:        float = Field(default=0, ge=0)
    expense_tabungan_investasi:float = Field(default=0, ge=0)

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: str) -> str:
        if v not in VALID_MONTHS:
            raise ValueError(f"Bulan '{v}' tidak valid. Gunakan nama bulan Indonesia.")
        return v

    @model_validator(mode="after")
    def validate_expense_range(self) -> "ClassifyRequest":
        if self.total_expense > self.total_income * 2:
            raise ValueError("total_expense melebihi 2x total_income. Periksa data input.")
        return self


class CategorySummaryForRec(BaseModel):
    category:             str
    total_amount:         float = Field(..., ge=0)
    transaction_count:    int   = Field(..., ge=0)
    top_transactions:     str
    percentage_of_income: float = Field(..., ge=0, le=200)


class ClassifyAndRecommendRequest(ClassifyRequest):
    categories: list[CategorySummaryForRec] = Field(
        default_factory=list,
        description="Detail per kategori. Jika kosong, dibuat otomatis dari expense_ fields.",
    )


class ClassifyResponse(BaseModel):
    user_id:      str
    month:        str
    year:         int
    label:        str
    confidence:   float
    probabilities:dict[str, float]
    savings_pct:  float
    generated_at: str


class ClassifyAndRecommendResponse(BaseModel):
    user_id:                  str
    month:                    str
    year:                     int
    label:                    str
    confidence:               float
    savings_pct:              float
    financial_health:         str
    recommendation_summary:   Optional[str] = None
    category_recommendations: Optional[dict[str, str]] = None
    recommendation_error:     Optional[str] = None
    generated_at:             str


class HealthResponse(BaseModel):
    status:                 str
    model_loaded:           bool
    feature_count:          int
    is_sheet3_excluded:     bool
    auth_enabled:           bool
    recommendation_api_url: str
    timestamp:              str
    version:                str


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartFinance AI - Classification API v3",
    description=(
        "Klasifikasi pengguna ke Boros/Normal/Hemat + integrasi rekomendasi.\n\n"
        "**v3**: is_sheet3 dihapus dari fitur (data leakage fix).\n\n"
        "**Auth**: Set header `X-API-Key` jika `SMARTFINANCE_API_KEY` di-set di `.env`."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def determine_health(savings_pct: float) -> str:
    if savings_pct >= 20:
        return "Sehat — Pengguna menabung dengan baik (≥20%)"
    elif savings_pct >= 10:
        return "Cukup baik — Masih ada ruang perbaikan (10–20%)"
    elif savings_pct > 0:
        return "Perlu perhatian — Tabungan sangat minim (<10%)"
    else:
        return "Kritis — Pengeluaran melebihi pemasukan (defisit)"


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        feature_count=len(_feature_cols),
        is_sheet3_excluded="is_sheet3" not in _feature_cols,
        auth_enabled=bool(API_KEY),
        recommendation_api_url=RECOMMENDATION_API_URL,
        timestamp=datetime.now().isoformat(),
        version="3.0.0",
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    tags=["Klasifikasi"],
    summary="Klasifikasi tipe keuangan pengguna",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def classify(request: ClassifyRequest):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model ML belum di-load. Jalankan step3_train_model.py.",
        )

    savings_pct = (
        (request.total_income - request.total_expense) / request.total_income * 100
    )

    try:
        X         = build_features(request.model_dump())
        proba     = _model.predict_proba(X)[0]
        pred_idx  = int(np.argmax(proba))
        label     = LABEL_NAMES[pred_idx]
        confidence= round(float(proba[pred_idx]), 4)
        probs     = {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)}
        logger.info(f"Classified {request.user_id}: {label} ({confidence:.2%})")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error prediksi: {e}")

    return ClassifyResponse(
        user_id=request.user_id,
        month=request.month,
        year=request.year,
        label=label,
        confidence=confidence,
        probabilities=probs,
        savings_pct=round(savings_pct, 1),
        generated_at=datetime.now().isoformat(),
    )


@app.post(
    "/classify-and-recommend",
    response_model=ClassifyAndRecommendResponse,
    tags=["Klasifikasi"],
    summary="Klasifikasi + rekomendasi dalam satu endpoint",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def classify_and_recommend(request: ClassifyAndRecommendRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model ML belum di-load.")

    income      = request.total_income
    expense     = request.total_expense
    savings_pct = ((income - expense) / income * 100) if income > 0 else 0

    # Step 1: Klasifikasi
    try:
        X        = build_features(request.model_dump())
        proba    = _model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        label    = LABEL_NAMES[pred_idx]
        confidence = round(float(proba[pred_idx]), 4)
        logger.info(f"Classified {request.user_id}: {label} ({confidence:.2%})")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {e}")

    # Step 2: Build kategori
    categories = request.categories
    if not categories:
        cat_map = {
            "makanan_minuman":    request.expense_makanan_minuman,
            "transportasi":       request.expense_transportasi,
            "hiburan":            request.expense_hiburan,
            "belanja_online":     request.expense_belanja_online,
            "tagihan_utilitas":   request.expense_tagihan_utilitas,
            "kesehatan":          request.expense_kesehatan,
            "pendidikan":         request.expense_pendidikan,
            "tabungan_investasi": request.expense_tabungan_investasi,
        }
        categories = [
            CategorySummaryForRec(
                category=cat,
                total_amount=amt,
                transaction_count=max(1, int(amt / 50_000)),
                top_transactions=f"Pengeluaran {cat.replace('_', ' ')} Rp {amt:,.0f}",
                percentage_of_income=round((amt / income * 100) if income > 0 else 0, 1),
            )
            for cat, amt in cat_map.items()
            if amt > 0
        ]

    # Step 3: Panggil Recommendation API
    rec_summary = None
    cat_recs    = None
    rec_error   = None

    try:
        headers = {}
        if API_KEY:
            headers["X-API-Key"] = API_KEY

        payload = {
            "user_id":        request.user_id,
            "month":          request.month,
            "year":           request.year,
            "total_income":   income,
            "total_expense":  expense,
            "financial_label": label,
            "categories":     [cat.model_dump() for cat in categories],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{RECOMMENDATION_API_URL}/recommendations",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            rec_data    = resp.json()
            rec_summary = rec_data.get("summary_recommendation")
            cat_recs    = rec_data.get("category_recommendations")
            logger.info(f"Rekomendasi berhasil untuk {request.user_id}")

    except httpx.ConnectError:
        rec_error = (
            f"Recommendation API tidak dapat dihubungi di {RECOMMENDATION_API_URL}. "
            "Pastikan port 8001 sudah berjalan."
        )
        logger.warning(rec_error)
    except httpx.HTTPStatusError as e:
        rec_error = f"Recommendation API error: {e.response.status_code}"
        logger.error(rec_error)
    except Exception as e:
        rec_error = f"Unexpected error: {str(e)}"
        logger.error(rec_error)

    return ClassifyAndRecommendResponse(
        user_id=request.user_id,
        month=request.month,
        year=request.year,
        label=label,
        confidence=confidence,
        savings_pct=round(savings_pct, 1),
        financial_health=determine_health(savings_pct),
        recommendation_summary=rec_summary,
        category_recommendations=cat_recs,
        recommendation_error=rec_error,
        generated_at=datetime.now().isoformat(),
    )
