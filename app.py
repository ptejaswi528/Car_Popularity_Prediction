import json
import os
import sys
import hashlib
import time
import urllib.error
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "car_market_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "vehicle_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")

app = Flask(__name__)

pipeline = None
meta = None
df_all = None

IMAGE_CACHE_PATH = os.path.join(BASE_DIR, "image_cache.json")
IMAGE_CACHE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

_image_cache = {}


def _load_image_cache() -> dict:
    if not os.path.exists(IMAGE_CACHE_PATH):
        return {}
    try:
        with open(IMAGE_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_image_cache(cache: dict) -> None:
    try:
        with open(IMAGE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _cache_get(key: str) -> str | None:
    item = _image_cache.get(key)
    if not item:
        return None
    ts = item.get("ts", 0)
    if ts and (time.time() - float(ts)) > IMAGE_CACHE_TTL_SECONDS:
        return None
    return item.get("url")


def _cache_set(key: str, url: str) -> None:
    _image_cache[key] = {"url": url, "ts": time.time()}
    _save_image_cache(_image_cache)


def _http_get_json(url: str, timeout_s: float = 6.0) -> dict:
    req = Request(url, headers={"User-Agent": "CarPopularityAnalytics/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
        return json.loads(payload)


def _wikimedia_search_file_url(query: str) -> str | None:
    """
    Best-effort lookup of a car-related image from Wikimedia Commons.
    Returns a thumbnail URL (smaller, faster) if possible.
    """
    # 1) Search only in File namespace (6)
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,
        "srlimit": 3,
        "format": "json",
    }
    search_url = "https://commons.wikimedia.org/w/api.php?" + urlencode(search_params)
    data = _http_get_json(search_url)
    hits = (data.get("query") or {}).get("search") or []
    if not hits:
        return None

    file_title = hits[0].get("title")
    if not file_title:
        return None
    file_title = str(file_title)
    if not file_title.lower().startswith("file:"):
        file_title = "File:" + file_title.split(":", 1)[-1]

    # 2) Fetch the thumbnail url for the file
    info_params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 480,
        "format": "json",
    }
    info_url = "https://commons.wikimedia.org/w/api.php?" + urlencode(info_params)
    info = _http_get_json(info_url)
    pages = (info.get("query") or {}).get("pages") or {}
    for _page_id, page in pages.items():
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue
        ii = imageinfo[0]
        thumb_url = ii.get("thumburl")
        if thumb_url:
            return str(thumb_url)
        url = ii.get("url")
        if url:
            return str(url)
    return None


def get_car_image_url(brand: str, model: str, year: int | str | None) -> str | None:
    brand = (brand or "").strip()
    model = (model or "").strip()
    if not brand or not model:
        return None

    year_str = str(year).strip() if year is not None else ""
    base_query = f"{brand} {model} {year_str}".strip()
    cache_key = hashlib.sha256(base_query.encode("utf-8")).hexdigest()

    cached = _cache_get(cache_key)
    if cached:
        return cached

    candidates = [base_query]
    candidates.append(f"{base_query} car".strip())
    candidates.append(f"{base_query} vehicle".strip())
    candidates.append(f"{brand} {model}".strip())
    candidates.append(f"{brand} {model} car".strip())

    for q in candidates:
        if not q:
            continue
        try:
            url = _wikimedia_search_file_url(q)
            if url:
                _cache_set(cache_key, url)
                return url
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue
        except Exception:
            continue
    return None


def get_image_url(car: str, model: str, year: int | str | None = None) -> str | None:
    """
    Return a best-effort real car image URL.
    Uses Wikimedia Commons search and may return None if no confident match.
    """
    return get_car_image_url(
        brand=(car or "").strip(),
        model=(model or "").strip(),
        year=year,
    )


_image_cache = _load_image_cache()


def _load_data():
    global pipeline, meta, df_all
    if not os.path.exists(CSV_PATH):
        return False
    df_all = pd.read_csv(CSV_PATH)
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        pipeline = joblib.load(MODEL_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        pipeline = None
        meta = None
    return True


_load_data()


@app.before_request
def _lazy_load_csv_if_missing():
    """If the server started before the CSV existed, load it on the next request."""
    global df_all
    if df_all is None and os.path.isfile(CSV_PATH):
        _load_data()


def effective_meta():
    if meta is not None:
        return meta
    if df_all is None:
        return {}
    cat_cols = [
        "brand",
        "model",
        "fuel_type",
        "transmission",
        "driven_wheels",
        "market_category",
    ]
    return {
        "categories": {c: sorted(df_all[c].astype(str).unique().tolist()) for c in cat_cols},
        "numeric_ranges": {
            "year": [int(df_all["year"].min()), int(df_all["year"].max())],
            "engine_hp": [int(df_all["engine_hp"].min()), int(df_all["engine_hp"].max())],
            "engine_cylinders": sorted(df_all["engine_cylinders"].astype(int).unique().tolist()),
            "doors": sorted(df_all["doors"].astype(int).unique().tolist()),
            "price": [int(df_all["price"].min()), int(df_all["price"].max())],
        },
        "r2_score": None,
        "mae": None,
        "feature_importance_top": {},
        "level_thresholds": {"q1": 400.0, "q2": 700.0},
        "popularity_score_min": float(df_all["popularity_score"].min()),
        "popularity_score_max": float(df_all["popularity_score"].max()),
    }


def score_to_level(score: float) -> str:
    th = (meta or {}).get("level_thresholds") or {}
    q1, q2 = th.get("q1"), th.get("q2")
    if q1 is not None and q2 is not None:
        if score <= q1:
            return "Low"
        if score <= q2:
            return "Medium"
        return "High"
    if score < 400:
        return "Low"
    if score < 700:
        return "Medium"
    return "High"


def score_to_percent(score: float) -> float:
    """
    Convert raw popularity_score into a 0-100 "relative popularity index"
    based on min/max seen in the dataset used to train the dashboard.
    """
    minv = (meta or {}).get("popularity_score_min")
    maxv = (meta or {}).get("popularity_score_max")
    if (minv is None or maxv is None) and df_all is not None:
        minv = float(df_all["popularity_score"].min())
        maxv = float(df_all["popularity_score"].max())
    if minv is None or maxv is None or maxv <= minv:
        return 0.0
    pct = (float(score) - float(minv)) / (float(maxv) - float(minv)) * 100.0
    return float(np.clip(pct, 0.0, 100.0))


def apply_filters(dframe: pd.DataFrame, args) -> pd.DataFrame:
    out = dframe.copy()
    brand = args.get("brand", "").strip()
    if brand and brand != "__all__":
        out = out[out["brand"] == brand]
    fuel = args.get("fuel_type", "").strip()
    if fuel and fuel != "__all__":
        out = out[out["fuel_type"] == fuel]
    trans = args.get("transmission", "").strip()
    if trans and trans != "__all__":
        out = out[out["transmission"] == trans]
    year = args.get("year", "").strip()
    if year and year != "__all__":
        out = out[out["year"] == int(year)]
    try:
        pmin = int(args.get("min_price", "") or 0)
    except ValueError:
        pmin = 0
    try:
        pmax = int(args.get("max_price", "") or 0)
    except ValueError:
        pmax = 0
    if pmin > 0:
        out = out[out["price"] >= pmin]
    if pmax > 0:
        out = out[out["price"] <= pmax]
    return out


def overview_cards(dframe: pd.DataFrame) -> dict:
    if dframe.empty:
        return {
            "total_cars": 0,
            "avg_popularity": "—",
            "top_brand": "—",
            "least_car": "—",
        }
    total = len(dframe)
    avg_score = float(dframe["popularity_score"].mean())
    avg = round(score_to_percent(avg_score), 1)
    by_brand = dframe.groupby("brand")["popularity_score"].mean()
    top_brand = str(by_brand.idxmax()) if len(by_brand) else "—"
    worst = dframe.loc[dframe["popularity_score"].idxmin()]
    least_car = f"{worst['brand']} {worst['model']} ({int(worst['year'])})"
    return {
        "total_cars": total,
        "avg_popularity": avg,
        "top_brand": top_brand,
        "least_car": least_car,
    }


def chart_payload(dframe: pd.DataFrame) -> dict:
    empty = {
        "brand_bar": {"labels": [], "values": []},
        "brand_share": {"labels": [], "values": []},
        "scatter": {"points": []},
        "fuel": {"labels": [], "values": []},
        "fuel_share_counts": {"labels": [], "values": []},
        "year_trend": {"labels": [], "values": []},
        "top10": [],
        "insights": {
            "avg_hp_by_brand": {"labels": [], "values": []},
            "common_transmission": "—",
            "common_fuel": "—",
        },
        "hp_hist": {"bins": [], "counts": []},
    }
    if dframe.empty:
        return empty

    by_brand_mean = (
        dframe.groupby("brand")["popularity_score"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
    )
    brand_bar = {
        "labels": by_brand_mean.index.tolist(),
        "values": [round(float(v), 1) for v in by_brand_mean.values],
    }

    share = dframe["brand"].value_counts()
    top_n = 8
    if len(share) > top_n:
        head = share.head(top_n)
        other = int(share.iloc[top_n:].sum())
        labels = head.index.tolist() + (["Other"] if other else [])
        values = head.tolist() + ([other] if other else [])
    else:
        labels = share.index.tolist()
        values = share.tolist()

    sample = dframe
    if len(sample) > 450:
        sample = sample.sample(n=450, random_state=42)
    scatter_points = [
        {"x": int(r["price"]), "y": float(r["popularity_score"])}
        for _, r in sample.iterrows()
    ]

    fuel_mean = dframe.groupby("fuel_type")["popularity_score"].mean().sort_values(ascending=False)
    fuel_chart = {
        "labels": fuel_mean.index.tolist(),
        "values": [round(float(v), 1) for v in fuel_mean.values],
    }
    fuel_counts = dframe["fuel_type"].value_counts()
    fuel_share_counts = {
        "labels": fuel_counts.index.tolist(),
        "values": [int(v) for v in fuel_counts.values],
    }

    year_mean = dframe.groupby("year")["popularity_score"].mean().sort_index()
    year_trend = {
        "labels": [str(int(y)) for y in year_mean.index],
        "values": [round(float(v), 1) for v in year_mean.values],
    }

    # Remove duplicate car names (same brand+model) by keeping the most popular variant.
    top10_rows = (
        dframe.sort_values("popularity_score", ascending=False)
        .drop_duplicates(subset=["brand", "model"], keep="first")
        .head(10)
    )
    top10 = [
        {
            "car_id": int(r["car_id"]),
            "brand": str(r["brand"]),
            "model": str(r["model"]),
            "year": int(r["year"]),
            "popularity_score": int(r["popularity_score"]),
            "popularity_percent": round(score_to_percent(float(r["popularity_score"])), 1),
            "popularity_level": str(r["popularity_level"]),
        }
        for _, r in top10_rows.iterrows()
    ]

    hp_by_brand = dframe.groupby("brand")["engine_hp"].mean().sort_values(ascending=False).head(10)
    insights = {
        "avg_hp_by_brand": {
            "labels": hp_by_brand.index.tolist(),
            "values": [round(float(v), 1) for v in hp_by_brand.values],
        },
        "common_transmission": str(dframe["transmission"].mode().iloc[0])
        if len(dframe["transmission"].mode())
        else "—",
        "common_fuel": str(dframe["fuel_type"].mode().iloc[0])
        if len(dframe["fuel_type"].mode())
        else "—",
    }

    hp = dframe["engine_hp"].astype(float)
    counts, bins = np.histogram(hp, bins=12)
    hp_hist = {
        "bins": [f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)],
        "counts": [int(c) for c in counts],
    }

    return {
        "brand_bar": brand_bar,
        "brand_share": {"labels": labels, "values": values},
        "scatter": {"points": scatter_points},
        "fuel": fuel_chart,
        "fuel_share_counts": fuel_share_counts,
        "year_trend": year_trend,
        "top10": top10,
        "insights": insights,
        "hp_hist": hp_hist,
    }


def car_options_list(dframe: pd.DataFrame, limit=1200):
    if dframe.empty:
        return []
    # Keep only one entry per brand+model (avoid duplicates across years).
    # Choose the most popular variant to represent the model in the selector.
    sub = dframe.sort_values(["brand", "model", "popularity_score"], ascending=[True, True, False])
    dedup = sub.drop_duplicates(subset=["brand", "model"], keep="first")
    rows = []
    for _, r in dedup.head(limit).iterrows():
        rows.append({
            "id": int(r["car_id"]),
            "label": f"{r['brand']} {r['model']}",
        })
    return rows


@app.route("/")
def index():
    if df_all is None:
        return render_template(
            "index.html",
            ready=False,
            error="Dataset not found.",
            csv_expected_path=CSV_PATH,
        )
    filt = apply_filters(df_all, request.args)
    charts = chart_payload(filt)
    overview = overview_cards(filt)
    cars_sel = car_options_list(filt)
    eff_meta = effective_meta()

    # For the prediction dropdown we want models constrained by brand.
    # This also improves UX (prevents selecting an "impossible" brand+model combo).
    models_by_brand: dict[str, list[str]] = {}
    try:
        brands = (eff_meta or {}).get("categories", {}).get("brand", [])
        for b in brands:
            sub = df_all[df_all["brand"].astype(str) == str(b)]
            models_by_brand[str(b)] = sorted(sub["model"].astype(str).unique().tolist())
    except Exception:
        models_by_brand = {}

    prediction_default_brand = request.args.get("pred_brand", "").strip()
    if not prediction_default_brand or prediction_default_brand == "__all__":
        # Fall back to interactive filter brand (if present), otherwise first brand.
        prediction_default_brand = request.args.get("brand", "").strip()
        if not prediction_default_brand or prediction_default_brand == "__all__":
            brand_list = ((eff_meta or {}).get("categories", {}).get("brand") or [])
            prediction_default_brand = brand_list[0] if brand_list else ""

    prediction_defaults = {
        "brand": prediction_default_brand,
        "model": request.args.get("pred_model", "").strip(),
        "year": request.args.get("pred_year", "").strip(),
        "engine_hp": request.args.get("pred_engine_hp", "").strip(),
        "engine_cylinders": request.args.get("pred_engine_cylinders", "").strip(),
        "fuel_type": request.args.get("pred_fuel_type", "").strip(),
        "transmission": request.args.get("pred_transmission", "").strip(),
        "driven_wheels": request.args.get("pred_driven_wheels", "").strip(),
        "doors": request.args.get("pred_doors", "").strip(),
        "market_category": request.args.get("pred_market_category", "").strip(),
        "price": request.args.get("pred_price", "").strip(),
    }
    pred_result = None
    if request.args.get("predicted"):
        try:
            score = float(request.args.get("score", 0))
            pred_result = {
                "score": round(score, 2),
                "level": request.args.get("level", score_to_level(score)),
                "percent": round(score_to_percent(score), 2),
            }
        except ValueError:
            pred_result = None

    filter_state = {
        "brand": request.args.get("brand", "__all__"),
        "fuel_type": request.args.get("fuel_type", "__all__"),
        "transmission": request.args.get("transmission", "__all__"),
        "year": request.args.get("year", "__all__"),
        "min_price": request.args.get("min_price", ""),
        "max_price": request.args.get("max_price", ""),
    }
    fq = {}
    for key in ("brand", "fuel_type", "transmission", "year", "min_price", "max_price"):
        v = request.args.get(key, "").strip()
        if v and v != "__all__":
            fq[key] = v
    return_query = urlencode(fq)
    years_options = (
        sorted(df_all["year"].unique().tolist(), reverse=True) if df_all is not None else []
    )

    return render_template(
        "index.html",
        ready=True,
        model_ready=pipeline is not None and meta is not None,
        overview=overview,
        charts=charts,
        car_options=cars_sel,
        meta=eff_meta,
        models_by_brand=models_by_brand,
        prediction_defaults=prediction_defaults,
        filter_state=filter_state,
        pred_result=pred_result,
        return_query=return_query,
        years_options=years_options,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None or meta is None or df_all is None:
        return redirect(url_for("index"))

    try:
        submitted = {
            "brand": request.form.get("brand", ""),
            "model": request.form.get("model", ""),
            "year": request.form.get("year", ""),
            "engine_hp": request.form.get("engine_hp", ""),
            "engine_cylinders": request.form.get("engine_cylinders", ""),
            "fuel_type": request.form.get("fuel_type", ""),
            "transmission": request.form.get("transmission", ""),
            "driven_wheels": request.form.get("driven_wheels", ""),
            "doors": request.form.get("doors", ""),
            "market_category": request.form.get("market_category", ""),
            "price": request.form.get("price", ""),
        }
        row = pd.DataFrame(
            [
                {
                    "brand": request.form["brand"],
                    "model": request.form["model"],
                    "year": int(request.form["year"]),
                    "engine_hp": int(request.form["engine_hp"]),
                    "engine_cylinders": int(request.form["engine_cylinders"]),
                    "fuel_type": request.form["fuel_type"],
                    "transmission": request.form["transmission"],
                    "driven_wheels": request.form["driven_wheels"],
                    "doors": int(request.form["doors"]),
                    "market_category": request.form["market_category"],
                    "price": int(request.form["price"]),
                }
            ]
        )
        score = float(pipeline.predict(row)[0])
        score = float(np.clip(score, 0, 1000))
        level = score_to_level(score)
    except (KeyError, ValueError):
        return redirect(url_for("index"))

    qs = request.form.get("return_query", "").strip()
    base = url_for("index")

    # Preserve what the user selected, so the prediction form doesn't reset
    # to the first brand/model after redirect.
    tail = urlencode(
        {
            "predicted": "1",
            "score": f"{score:.2f}",
            "level": level,
            "pred_brand": submitted["brand"],
            "pred_model": submitted["model"],
            "pred_year": submitted["year"],
            "pred_engine_hp": submitted["engine_hp"],
            "pred_engine_cylinders": submitted["engine_cylinders"],
            "pred_fuel_type": submitted["fuel_type"],
            "pred_transmission": submitted["transmission"],
            "pred_driven_wheels": submitted["driven_wheels"],
            "pred_doors": submitted["doors"],
            "pred_market_category": submitted["market_category"],
            "pred_price": submitted["price"],
        }
    )
    if qs:
        return redirect(f"{base}?{qs}&{tail}")
    return redirect(f"{base}?{tail}")


@app.route("/api/car/<int:car_id>")
def api_car(car_id):
    if df_all is None:
        return jsonify({"error": "no data"}), 404
    hit = df_all[df_all["car_id"] == car_id]
    if hit.empty:
        return jsonify({"error": "not found"}), 404
    r = hit.iloc[0]
    pred_score = None
    pred_level = None
    pred_percent = None
    image_url = None
    with_image = request.args.get("with_image", "0") == "1"
    if pipeline is not None and meta is not None:
        feat = pd.DataFrame(
            [
                {
                    "brand": r["brand"],
                    "model": r["model"],
                    "year": int(r["year"]),
                    "engine_hp": int(r["engine_hp"]),
                    "engine_cylinders": int(r["engine_cylinders"]),
                    "fuel_type": r["fuel_type"],
                    "transmission": r["transmission"],
                    "driven_wheels": r["driven_wheels"],
                    "doors": int(r["doors"]),
                    "market_category": r["market_category"],
                    "price": int(r["price"]),
                }
            ]
        )
        pred_score = round(float(pipeline.predict(feat)[0]), 1)
        pred_score = float(np.clip(pred_score, 0, 1000))
        pred_level = score_to_level(pred_score)
        pred_percent = round(score_to_percent(pred_score), 1)

    if with_image:
        # Real-image lookup (best effort). Falls back to frontend SVG placeholder when None.
        try:
            image_url = get_image_url(
                car=str(r["brand"]),
                model=str(r["model"]),
                year=r["year"],
            )
        except Exception:
            image_url = None

    return jsonify(
        {
            "car_id": int(r["car_id"]),
            "brand": str(r["brand"]),
            "model": str(r["model"]),
            "year": int(r["year"]),
            "engine_hp": int(r["engine_hp"]),
            "engine_cylinders": int(r["engine_cylinders"]),
            "fuel_type": str(r["fuel_type"]),
            "transmission": str(r["transmission"]),
            "driven_wheels": str(r["driven_wheels"]),
            "doors": int(r["doors"]),
            "market_category": str(r["market_category"]),
            "price": int(r["price"]),
            "popularity_score": int(r["popularity_score"]),
            "popularity_percent": round(score_to_percent(float(r["popularity_score"])), 1),
            "popularity_level": str(r["popularity_level"]),
            "predicted_score": pred_score,
            "predicted_level": pred_level,
            "predicted_percent": pred_percent,
            "image_seed": int(r["car_id"]),
            "image_url": image_url,
        }
    )


if __name__ == "__main__":
    if not _load_data():
        print("Missing car_market_data.csv — run generate_car_market_data.py", file=sys.stderr)
    app.run(debug=True)
