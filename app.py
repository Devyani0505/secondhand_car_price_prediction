from flask import Flask, render_template, request, jsonify
import json
import os
import re
import time
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODEL_DIR, "car_price_pipeline.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data2.csv")
MARKET_TRENDS_PATH = os.path.join(BASE_DIR, "market_trends.json")
CURRENT_YEAR = 2026

pipeline_model = joblib.load(PIPELINE_PATH) if os.path.exists(PIPELINE_PATH) else None

_MARKET_TRENDS_CACHE = {"loaded_at": 0.0, "payload": None}
_MARKET_TRENDS_TTL_SECONDS = 300
MIN_MAKE_MODEL_GUARDRAIL_SAMPLES = 8
MIN_MAKE_GUARDRAIL_SAMPLES = 20

FEATURE_COLUMNS = [
    "Make",
    "Model",
    "Variant",
    "City",
    "Color",
    "Fuel",
    "Transmission Type",
    "RC Status",
    "Loan Clearance",
    "Accident History",
    "Insurance Claims",
    "Service Records",
    "Body Panels",
    "Rust",
    "Glass Condition",
    "Lights Condition",
    "Tyres",
    "Seats Condition",
    "Electronics Condition",
    "Smell",
    "Engine Condition",
    "Transmission Condition",
    "Brakes Condition",
    "Steering Condition",
    "Suspension Condition",
    "Battery Condition",
    "AC Cooling",
    "Rattling Noises",
    "Engine Start",
    "Idle Vibration",
    "Acceleration Condition",
    "Brakes Drive",
    "Airbags Present",
    "Safety Systems",
    "Manufacturing Year",
    "Kilometers Driven",
    "Mileage",
    "Owners",
    "Airbags Count",
    "Car_Age",
    "kms_per_year",
    "condition_score",
    "has_airbags",
    "is_engine_start_ok",
    "rattling_present",
]

FIELD_MAP = {
    "Make": "make",
    "Model": "model",
    "Variant": "variant",
    "City": "city",
    "Color": "color",
    "Fuel": "fuel",
    "Transmission Type": "transmission",
    "RC Status": "rc_status",
    "Loan Clearance": "loan_clearance",
    "Accident History": "accident_history",
    "Insurance Claims": "insurance_claims",
    "Service Records": "service_records",
    "Body Panels": "body_panels",
    "Rust": "rust",
    "Glass Condition": "glass_condition",
    "Lights Condition": "lights_condition",
    "Tyres": "tyres",
    "Seats Condition": "seats_condition",
    "Electronics Condition": "electronics_condition",
    "Smell": "smell",
    "Engine Condition": "engine_condition",
    "Transmission Condition": "transmission_condition",
    "Brakes Condition": "brakes_condition",
    "Steering Condition": "steering_condition",
    "Suspension Condition": "suspension_condition",
    "Battery Condition": "battery_condition",
    "AC Cooling": "ac_cooling",
    "Rattling Noises": "rattling_noises",
    "Engine Start": "engine_start",
    "Idle Vibration": "idle_vibration",
    "Acceleration Condition": "acceleration_condition",
    "Brakes Drive": "brakes_drive",
    "Airbags Present": "airbags_present",
    "Safety Systems": "safety_systems",
    "Manufacturing Year": "year",
    "Kilometers Driven": "kms",
    "Mileage": "mileage",
    "Owners": "owners",
    "Airbags Count": "airbags_count",
}

DEFAULTS = {
    "Make": "Maruti",
    "Model": "Alto",
    "Variant": "VXI",
    "City": "Hyderabad",
    "Color": "White",
    "Fuel": "Petrol",
    "Transmission Type": "Manual",
    "RC Status": "Active",
    "Loan Clearance": "Cleared",
    "Accident History": "Not Reported",
    "Insurance Claims": "Single",
    "Service Records": "Partial",
    "Body Panels": "Average",
    "Rust": "Average",
    "Glass Condition": "Average",
    "Lights Condition": "Average",
    "Tyres": "Average",
    "Seats Condition": "Average",
    "Electronics Condition": "Average",
    "Smell": "Average",
    "Engine Condition": "Average",
    "Transmission Condition": "Average",
    "Brakes Condition": "Average",
    "Steering Condition": "Average",
    "Suspension Condition": "Average",
    "Battery Condition": "Average",
    "AC Cooling": "Average",
    "Rattling Noises": "Occasional",
    "Engine Start": "Yes",
    "Idle Vibration": "Average",
    "Acceleration Condition": "Average",
    "Brakes Drive": "Average",
    "Airbags Present": "Yes",
    "Safety Systems": "Seat belts, ABS, EBD",
    "Manufacturing Year": 2018,
    "Kilometers Driven": 40000,
    "Mileage": 18.0,
    "Owners": 1,
    "Airbags Count": 2,
}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_str(value, default=""):
    if value in (None, ""):
        return default
    return str(value).strip()


def _normalized_guardrail_key(value):
    return _safe_str(value, "").lower()


def _bounded_multiplier(value, lo=0.7, hi=1.35):
    return float(min(max(_safe_float(value, 1.0), lo), hi))


def _default_market_trends():
    return {
        "last_updated_utc": None,
        "overall_index": 1.0,
        "city_adjustments": {},
        "make_adjustments": {},
        "fuel_adjustments": {},
    }


def _normalize_market_trends(payload):
    out = _default_market_trends()
    if not isinstance(payload, dict):
        return out

    out["last_updated_utc"] = payload.get("last_updated_utc")
    out["overall_index"] = _bounded_multiplier(payload.get("overall_index", 1.0))

    for key in ("city_adjustments", "make_adjustments", "fuel_adjustments"):
        raw_map = payload.get(key, {})
        if isinstance(raw_map, dict):
            normalized = {}
            for raw_k, raw_v in raw_map.items():
                clean_k = _safe_str(raw_k, "")
                if clean_k:
                    normalized[clean_k.lower()] = _bounded_multiplier(raw_v)
            out[key] = normalized
    return out


def _load_market_trends():
    now = time.time()
    cached = _MARKET_TRENDS_CACHE.get("payload")
    if cached is not None and now - float(_MARKET_TRENDS_CACHE.get("loaded_at", 0.0)) < _MARKET_TRENDS_TTL_SECONDS:
        return cached

    payload = None
    remote_url = os.environ.get("MARKET_TRENDS_URL", "").strip()

    if remote_url:
        try:
            with urlopen(remote_url, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
            payload = None

    if payload is None and os.path.exists(MARKET_TRENDS_PATH):
        try:
            with open(MARKET_TRENDS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            payload = None

    normalized = _normalize_market_trends(payload)
    _MARKET_TRENDS_CACHE["loaded_at"] = now
    _MARKET_TRENDS_CACHE["payload"] = normalized
    return normalized


def _market_multiplier_from_row(row, request_data):
    manual = request_data.get("market_trend_index")
    if manual not in (None, ""):
        return _bounded_multiplier(manual), "manual"

    market = _load_market_trends()
    city = _safe_str(row.get("City"), "").lower()
    make = _safe_str(row.get("Make"), "").lower()
    fuel = _safe_str(row.get("Fuel"), "").lower()

    overall = _bounded_multiplier(market.get("overall_index", 1.0), 0.8, 1.25)
    city_mult = _bounded_multiplier(market["city_adjustments"].get(city, 1.0), 0.85, 1.20)
    make_mult = _bounded_multiplier(market["make_adjustments"].get(make, 1.0), 0.85, 1.20)
    fuel_mult = _bounded_multiplier(market["fuel_adjustments"].get(fuel, 1.0), 0.85, 1.20)

    combined = _bounded_multiplier(overall * city_mult * make_mult * fuel_mult)
    return combined, "auto"


def _build_condition_score(row):
    quality_map = {"Poor": 0, "Average": 1, "Good": 2}
    condition_cols = [
        "Body Panels",
        "Rust",
        "Glass Condition",
        "Lights Condition",
        "Tyres",
        "Seats Condition",
        "Electronics Condition",
        "Smell",
        "Engine Condition",
        "Transmission Condition",
        "Brakes Condition",
        "Steering Condition",
        "Suspension Condition",
        "Battery Condition",
        "AC Cooling",
        "Idle Vibration",
        "Acceleration Condition",
        "Brakes Drive",
    ]
    vals = []
    for col in condition_cols:
        vals.append(quality_map.get(_safe_str(row.get(col)), 1))
    return float(np.mean(vals))


def _load_price_guardrails():
    guardrails = {
        "global": (10000, 12000000),
        "by_make_model": {},
        "by_make": {},
        "by_age_bucket": {},
    }
    if not os.path.exists(DATA_PATH):
        return guardrails

    try:
        hist = pd.read_csv(DATA_PATH, low_memory=False)
    except Exception:
        return guardrails

    hist["Selling Price"] = pd.to_numeric(hist.get("Selling Price"), errors="coerce")
    hist["Manufacturing Year"] = pd.to_numeric(hist.get("Manufacturing Year"), errors="coerce")
    hist = hist.dropna(subset=["Selling Price", "Manufacturing Year"]).copy()
    if hist.empty:
        return guardrails

    hist["Make"] = hist["Make"].astype(str).str.strip()
    hist["Model"] = hist["Model"].astype(str).str.strip()
    hist["Make_key"] = hist["Make"].map(_normalized_guardrail_key)
    hist["Model_key"] = hist["Model"].map(_normalized_guardrail_key)
    hist["Car_Age"] = (CURRENT_YEAR - hist["Manufacturing Year"]).clip(lower=0, upper=35)

    global_low = int(hist["Selling Price"].quantile(0.02))
    global_high = int(hist["Selling Price"].quantile(0.98))
    guardrails["global"] = (max(10000, global_low), max(global_low + 5000, global_high))

    mm = (
        hist.groupby(["Make_key", "Model_key"])["Selling Price"]
        .agg(
            n="count",
            low=lambda s: s.quantile(0.05),
            high=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )
    for _, r in mm.iterrows():
        if int(r["n"]) >= MIN_MAKE_MODEL_GUARDRAIL_SAMPLES:
            key = (_normalized_guardrail_key(r["Make_key"]), _normalized_guardrail_key(r["Model_key"]))
            lo = int(max(10000, r["low"]))
            hi = int(max(lo + 5000, r["high"]))
            guardrails["by_make_model"][key] = (lo, hi)

    mk = (
        hist.groupby("Make_key")["Selling Price"]
        .agg(
            n="count",
            low=lambda s: s.quantile(0.05),
            high=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )
    for _, r in mk.iterrows():
        if int(r["n"]) >= MIN_MAKE_GUARDRAIL_SAMPLES:
            lo = int(max(10000, r["low"]))
            hi = int(max(lo + 5000, r["high"]))
            guardrails["by_make"][_normalized_guardrail_key(r["Make_key"])] = (lo, hi)

    age_buckets = [(0, 3), (4, 7), (8, 15), (16, 35)]
    for lo_age, hi_age in age_buckets:
        seg = hist[(hist["Car_Age"] >= lo_age) & (hist["Car_Age"] <= hi_age)]
        if len(seg) < 50:
            continue
        lo = int(max(10000, seg["Selling Price"].quantile(0.05)))
        hi = int(max(lo + 5000, seg["Selling Price"].quantile(0.95)))
        guardrails["by_age_bucket"][(lo_age, hi_age)] = (lo, hi)

    return guardrails


PRICE_GUARDRAILS = _load_price_guardrails()


def _bucket_for_age(age):
    if age <= 3:
        return (0, 3)
    if age <= 7:
        return (4, 7)
    if age <= 15:
        return (8, 15)
    return (16, 35)


def _apply_price_guardrails(raw_price, row):
    make = _normalized_guardrail_key(row.get("Make"))
    model = _normalized_guardrail_key(row.get("Model"))
    age = int(_safe_float(row.get("Car_Age"), 0))

    lo, hi = PRICE_GUARDRAILS["global"]
    mm_key = (make, model)
    if mm_key in PRICE_GUARDRAILS["by_make_model"]:
        lo, hi = PRICE_GUARDRAILS["by_make_model"][mm_key]
    elif make in PRICE_GUARDRAILS["by_make"]:
        lo, hi = PRICE_GUARDRAILS["by_make"][make]

    age_key = _bucket_for_age(age)
    if age_key in PRICE_GUARDRAILS["by_age_bucket"]:
        age_lo, age_hi = PRICE_GUARDRAILS["by_age_bucket"][age_key]
        lo = max(lo, age_lo)
        hi = min(hi, age_hi)
        if hi <= lo:
            hi = lo + 5000

    bounded = int(min(max(int(raw_price), lo), hi))
    return bounded, lo, hi


def _apply_depreciation_penalty(raw_price, row):
    age = int(_safe_float(row.get("Car_Age"), 0))
    kms = float(_safe_float(row.get("Kilometers Driven"), 0))
    owners = int(_safe_float(row.get("Owners"), 1))
    condition_score = float(_safe_float(row.get("condition_score"), 1.0))
    airbags_present = _safe_str(row.get("Airbags Present"), "Yes")
    safety = _safe_str(row.get("Safety Systems"), "Seat belts")
    claims = _safe_str(row.get("Insurance Claims"), "Single")

    price = float(raw_price)

    # Multiplicative penalty for age.
    if age >= 20:
        price *= 0.45
    elif age >= 15:
        price *= 0.60
    elif age >= 10:
        price *= 0.75

    # Multiplicative penalty for high running.
    if kms >= 500000:
        price *= 0.35
    elif kms >= 300000:
        price *= 0.50
    elif kms >= 200000:
        price *= 0.70

    # Minor owner-based wear penalty.
    if owners >= 4:
        price *= 0.85
    elif owners == 3:
        price *= 0.92

    # Strong condition impact: average/past wear should materially reduce price.
    # condition_score is roughly in [0, 2] where 1 means "Average".
    condition_factor = min(max(0.45 + 0.30 * condition_score, 0.45), 1.05)
    price *= condition_factor

    # Additional practical market penalties.
    if airbags_present == "No":
        price *= 0.90
    if safety == "Seat belts":
        price *= 0.92
    if claims == "Multiple":
        price *= 0.90

    # Hard caps for old/high-km combinations.
    hard_cap = None
    if age >= 20:
        hard_cap = 130000
    elif age >= 15:
        hard_cap = 180000

    if kms >= 500000:
        hard_cap = min(hard_cap, 120000) if hard_cap is not None else 120000
    elif kms >= 300000:
        hard_cap = min(hard_cap, 180000) if hard_cap is not None else 180000
    elif kms >= 200000:
        hard_cap = min(hard_cap, 250000) if hard_cap is not None else 250000

    final_price = int(max(price, 10000))
    if hard_cap is not None:
        final_price = min(final_price, hard_cap)
    return final_price


def _profile_depreciation_cap(row, hi_guard):
    age = int(_safe_float(row.get("Car_Age"), 0))
    kms = float(_safe_float(row.get("Kilometers Driven"), 0))

    if age <= 3:
        age_factor = 0.95
    elif age <= 7:
        age_factor = 0.75
    elif age <= 10:
        age_factor = 0.60
    elif age <= 13:
        age_factor = 0.45
    elif age <= 16:
        age_factor = 0.35
    else:
        age_factor = 0.25

    if kms <= 50000:
        km_factor = 1.00
    elif kms <= 100000:
        km_factor = 0.90
    elif kms <= 150000:
        km_factor = 0.80
    elif kms <= 200000:
        km_factor = 0.70
    elif kms <= 300000:
        km_factor = 0.55
    elif kms <= 500000:
        km_factor = 0.40
    else:
        km_factor = 0.30

    factor = max(0.15, age_factor * km_factor)
    return int(max(50000, hi_guard * factor))


DEALER_SEARCH_TEMPLATES = {
    "CARS24": "https://www.cars24.com/buy-used-car?search={query}",
    "Spinny": "https://www.spinny.com/used-cars/?search={query}",
    "CarDekho / Gaadi": "https://www.cardekho.com/used-cars+in+{query}",
    "CarWale": "https://www.carwale.com/used/cars-for-sale/#q={query}",
    "Mahindra First Choice": "https://www.mahindrafirstchoice.com/used-cars/{query}",
    "Maruti True Value": "https://www.marutisuzukitruevalue.com/?s={query}",
    "OLX Autos / Cars & Bikes": "https://www.olx.in/items/q-{query}",
    "Big Boy Toyz": "https://www.bigboytoyz.com/used-cars-search?q={query}",
}


def _normalize_price(n):
    try:
        value = int(float(n))
    except (TypeError, ValueError):
        return None
    if value < 10000 or value > 50000000:
        return None
    return value


def _extract_inr_prices(text):
    if not text:
        return []

    candidates = []
    patterns = [
        r"(?:₹|&#8377;|Rs\.?|INR)\s*([0-9][0-9,]{3,})",
        r'"price"\s*:\s*"?([0-9]{4,8})"?',
        r'"priceAmount"\s*:\s*"?([0-9]{4,8})"?',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            if isinstance(match, tuple):
                match = match[0]
            clean = str(match).replace(",", "").strip()
            value = _normalize_price(clean)
            if value is not None:
                candidates.append(value)

    return sorted(set(candidates))


def _fetch_text(url, timeout=3.5):
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        content_type = str(resp.headers.get("Content-Type", "")).lower()
        if "text/html" not in content_type and "application/json" not in content_type:
            return ""
        return resp.read().decode("utf-8", errors="ignore")


def _build_compare_query(payload):
    parts = [
        _safe_str(payload.get("make"), ""),
        _safe_str(payload.get("model"), ""),
        _safe_str(payload.get("variant"), ""),
        _safe_str(payload.get("year"), ""),
        _safe_str(payload.get("fuel"), ""),
        _safe_str(payload.get("transmission"), ""),
        _safe_str(payload.get("city"), ""),
        "India",
    ]
    return " ".join(p for p in parts if p).strip()


def _fetch_dealer_prices(payload):
    query = _build_compare_query(payload)
    encoded_query = quote_plus(query)
    out = {}

    for dealer_name, template in DEALER_SEARCH_TEMPLATES.items():
        url = template.replace("{query}", encoded_query)
        dealer_out = {"url": url, "price": None, "sample_count": 0, "error": None}
        try:
            html = _fetch_text(url)
            prices = _extract_inr_prices(html)
            if prices:
                dealer_out["price"] = int(min(prices))
                dealer_out["sample_count"] = len(prices)
        except Exception as exc:
            dealer_out["error"] = str(exc)
        out[dealer_name] = dealer_out

    return query, out


# -----------------------
# Page routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/valuation")
def valuation():
    return render_template("valuation.html")


@app.route("/emi")
def emi():
    return render_template("emi.html")


@app.route("/dealers")
def dealers():
    return render_template("dealers.html")


# -----------------------
# Prediction API
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if pipeline_model is None:
            return jsonify(
                {"success": False, "error": "Model pipeline not found. Run: python train.py"}
            )

        data = request.get_json() or {}
        required_fields = [
            "make",
            "model",
            "variant",
            "city",
            "year",
            "kms",
            "mileage",
            "owners",
            "fuel",
            "transmission",
            "color",
        ]
        missing = [f for f in required_fields if data.get(f) in (None, "")]
        if missing:
            return jsonify(
                {
                    "success": False,
                    "error": f"Missing required fields: {', '.join(missing)}",
                }
            )

        year_in = int(_safe_float(data.get("year"), 0))
        kms_in = _safe_float(data.get("kms"), -1)
        mileage_in = _safe_float(data.get("mileage"), -1)
        owners_in = int(_safe_float(data.get("owners"), 0))

        if year_in < 1990 or year_in > CURRENT_YEAR:
            return jsonify({"success": False, "error": "Enter a valid manufacturing year."})
        if kms_in <= 0:
            return jsonify({"success": False, "error": "Kilometers driven must be greater than 0."})
        if mileage_in <= 0:
            return jsonify({"success": False, "error": "Mileage must be greater than 0."})
        if owners_in <= 0:
            return jsonify({"success": False, "error": "Owners must be at least 1."})

        row = {}

        for feature in FIELD_MAP:
            payload_key = FIELD_MAP[feature]
            raw = data.get(payload_key)
            if feature in {"Manufacturing Year", "Kilometers Driven", "Mileage", "Owners", "Airbags Count"}:
                row[feature] = _safe_float(raw, DEFAULTS[feature])
            else:
                row[feature] = _safe_str(raw, DEFAULTS[feature])

        row["Manufacturing Year"] = int(max(1990, min(row["Manufacturing Year"], CURRENT_YEAR)))
        row["Car_Age"] = float(max(0, CURRENT_YEAR - row["Manufacturing Year"]))
        row["kms_per_year"] = float(row["Kilometers Driven"] / max(row["Car_Age"], 1))
        row["condition_score"] = _build_condition_score(row)
        row["has_airbags"] = 1.0 if row["Airbags Present"] == "Yes" else 0.0
        row["is_engine_start_ok"] = 1.0 if row["Engine Start"] == "Yes" else 0.0
        row["rattling_present"] = 1.0 if row["Rattling Noises"] == "Present" else 0.0

        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        predicted_log = float(pipeline_model.predict(df)[0])
        raw_price = int(max(np.expm1(predicted_log), 10000))
        penalized_price = _apply_depreciation_penalty(raw_price, row)
        predicted_price, lo_guard, hi_guard = _apply_price_guardrails(penalized_price, row)

        profile_cap = _profile_depreciation_cap(row, hi_guard)
        predicted_price = min(predicted_price, profile_cap, penalized_price)
        trend_multiplier, trend_source = _market_multiplier_from_row(row, data)
        predicted_price = int(max(10000, round(predicted_price * trend_multiplier)))
        predicted_price, lo_guard, hi_guard = _apply_price_guardrails(predicted_price, row)
        predicted_price = min(predicted_price, _profile_depreciation_cap(row, hi_guard))

        if predicted_price < lo_guard:
            low = max(10000, int(predicted_price * 0.92))
            high = int(predicted_price * 1.08)
        else:
            low = max(lo_guard, int(predicted_price * 0.92))
            high = min(hi_guard, int(predicted_price * 1.08))

        if high <= low:
            high = low + 5000
        return jsonify(
            {
                "success": True,
                "predicted_price": predicted_price,
                "price_low": low,
                "price_high": high,
                "raw_model_price": raw_price,
                "penalized_model_price": penalized_price,
                "profile_cap": profile_cap,
                "market_trend_multiplier": round(float(trend_multiplier), 4),
                "market_trend_source": trend_source,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/dealer-prices", methods=["POST"])
def dealer_prices():
    try:
        data = request.get_json() or {}
        make = _safe_str(data.get("make"), "")
        model = _safe_str(data.get("model"), "")
        if not make and not model:
            return jsonify(
                {
                    "success": False,
                    "error": "Enter at least Make or Model to fetch dealer prices.",
                }
            )

        query, results = _fetch_dealer_prices(data)
        found_count = sum(1 for item in results.values() if item.get("price") is not None)
        return jsonify(
            {
                "success": True,
                "query": query,
                "found_count": found_count,
                "results": results,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
