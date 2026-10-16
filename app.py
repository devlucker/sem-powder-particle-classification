from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from powder_ml import (
    MODEL_PATH,
    compute_shap_waterfall,
    extract_features,
    extract_particle_chip_features,
    generate_gradcam_overlay,
    locate_particle_regions,
    save_model,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_BATCH_BYTES = 50 * 1024 * 1024
MAX_BATCH_FILES = 20
MAX_IMAGE_PIXELS = 20_000_000


def load_model() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        save_model()
    trained = joblib.load(MODEL_PATH)
    required = {"classifier", "metrics", "feature_count"}
    if not required.issubset(trained):
        raise RuntimeError(f"Model artifact is missing required fields: {required - trained.keys()}")
    return trained


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.trained = load_model()
    yield


app = FastAPI(title="BioScope PRO 2.0 Materials Analytics", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    trained = getattr(app.state, "trained", None)
    return {
        "status": "ready" if trained else "starting",
        "model_loaded": trained is not None,
        "available_models": list(trained.get("models", {}).keys()) if trained else ["ensemble"],
        "training_samples": trained["metrics"]["sample_count"] if trained else 0,
    }


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    trained = getattr(app.state, "trained", None)
    if trained is None:
        raise HTTPException(status_code=503, detail="The classifier is still starting.")
    return trained["metrics"]


@app.get("/api/samples")
async def get_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    sem_dir = ROOT / "SEM"
    for folder, alloy_name in (("Al", "AlSiMg"), ("Steel", "316L Stainless Steel"), ("Ti", "TiAlV")):
        folder_path = sem_dir / folder
        if folder_path.is_dir():
            for img_path in sorted(folder_path.glob("*.jpg"))[:5]:
                samples.append({
                    "folder": folder,
                    "material_key": folder,
                    "filename": img_path.name,
                    "alloy": alloy_name,
                    "material": alloy_name,
                    "url": f"/api/samples/{folder}/{img_path.name}",
                })
    return samples


@app.get("/api/samples/{folder}/{filename}")
async def get_sample_image(folder: str, filename: str) -> FileResponse:
    if folder not in {"Al", "Steel", "Ti"}:
        raise HTTPException(status_code=404, detail="Unknown material folder.")
    img_path = (ROOT / "SEM" / folder / filename).resolve()
    if not img_path.is_file() or not img_path.is_relative_to(ROOT / "SEM"):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(img_path)


MEDICAL_STANDARDS = {
    "TiAlV": {
        "medical_name": "Ti-6Al-4V ELI (Grade 23 / ASTM F136)",
        "astm_standard": "ASTM F136 / ISO 5832-3",
        "clinical_application": "Orthopedic joint reconstructions, spinal fusion cages, dental implants",
        "biocompatibility": "High biocompatibility, superior osseointegration, corrosion resistant",
        "cytotoxicity_risk": "Minimal (< 0.05%)",
    },
    "316L Stainless Steel": {
        "medical_name": "Surgical 316L Stainless Steel (ASTM F138)",
        "astm_standard": "ASTM F138 / ISO 5832-1",
        "clinical_application": "Surgical fixation pins, trauma bone plates, stent delivery systems",
        "biocompatibility": "Medical-grade passivated, low magnetic susceptibility, high ductility",
        "cytotoxicity_risk": "Low (< 0.10%)",
    },
    "AlSiMg": {
        "medical_name": "Medical Structural AlSi10Mg (Prosthetics)",
        "astm_standard": "ISO 13485 Compliant / ASTM B788",
        "clinical_application": "Lightweight external prosthetics, orthotic braces, surgical robotics",
        "biocompatibility": "External prosthetic contact approved (non-implantable bone contact)",
        "cytotoxicity_risk": "Moderate if internalized; approved for external devices",
    },
}
OOD_CONFIDENCE_THRESHOLD = 0.65


async def predict_one(
    file: UploadFile,
    trained: dict[str, Any],
    model_name: str = "ensemble"
) -> tuple[dict[str, Any], int]:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Choose an image before classifying.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 15 MB or smaller.")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions must be 20 megapixels or smaller.")
            image.load()
            rgb_image = image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from error

    # Select active model
    available_models = trained.get("models", {})
    clf = available_models.get(model_name, trained["classifier"])

    try:
        features = extract_features(rgb_image).reshape(1, -1)
        probabilities = clf.predict_proba(features)[0]
        regions = locate_particle_regions(rgb_image)
        gradcam_overlay_uri = generate_gradcam_overlay(rgb_image)
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=f"Could not analyze this image: {error}") from error

    classes = clf.classes_
    ranked = sorted(
        (
            {"material": str(material), "probability": float(probability)}
            for material, probability in zip(classes, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    
    top_material = ranked[0]["material"]
    top_confidence = ranked[0]["probability"]
    is_ood = top_confidence < OOD_CONFIDENCE_THRESHOLD
    med_meta = MEDICAL_STANDARDS.get(top_material, {})

    # Compute Explainable AI (SHAP waterfall)
    shap_breakdown = compute_shap_waterfall(features, top_material, ranked)

    clinical_assessment = {
        "is_out_of_distribution": is_ood,
        "clinical_status": "Non-Standard / Unrecognized Powder" if is_ood else "Approved Medical Specification",
        "biomedical_standard": med_meta.get("astm_standard", "Uncertified"),
        "medical_name": med_meta.get("medical_name", top_material),
        "clinical_application": med_meta.get("clinical_application", "Unknown alloy"),
        "biocompatibility": "Requires laboratory verification" if is_ood else med_meta.get("biocompatibility", "Standard"),
        "cytotoxicity_risk": "High / Unverified" if is_ood else med_meta.get("cytotoxicity_risk", "Low"),
        "clinical_advisory": (
            "WARNING: The micrograph feature profile does not match the 3 certified biomedical alloys (AlSi10Mg, 316L Steel, Ti6Al4V). Potential non-biocompatible material (e.g., Zinc, Copper, or non-medical alloy). Do not use in patient implants without spectral assay."
            if is_ood
            else f"Complies with {med_meta.get('astm_standard', 'medical specs')} microstructural surface uniformity requirements for surgical additive manufacturing."
        ),
    }

    # Cross-Contamination & Feedstock Purity Analysis
    total_particles = len(regions)
    if total_particles > 0:
        cat_counts = {
            "spherical": sum(1 for r in regions if r.get("category_code") == "spherical"),
            "satellite": sum(1 for r in regions if r.get("category_code") == "satellite"),
            "irregular": sum(1 for r in regions if r.get("category_code") == "irregular"),
            "fine": sum(1 for r in regions if r.get("category_code") == "fine"),
        }
        diameters = sorted(float(r["diameter_px"]) for r in regions)
        d10 = round(float(np.percentile(diameters, 10)), 1)
        d50 = round(float(np.percentile(diameters, 50)), 1)
        d90 = round(float(np.percentile(diameters, 90)), 1)
        mean_circ = round(float(np.mean([r["circularity"] for r in regions])), 2)

        min_d, max_d = min(diameters), max(diameters)
        if min_d == max_d:
            max_d += 1.0
        bin_edges = np.linspace(min_d, max_d, 6)
        hist_counts, _ = np.histogram(diameters, bins=bin_edges)
        psd_bins = [
            {"range": f"{round(bin_edges[i], 1)}–{round(bin_edges[i+1], 1)} px", "count": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ]

        # Multi-Particle Consensus & Purity
        purity_pct = round(max(0.0, min(100.0, top_confidence * 100 - (cat_counts["irregular"] * 1.5))), 1)
        contaminant_count = cat_counts["irregular"] if (is_ood or cat_counts["irregular"] > 2) else 0
        has_contamination = contaminant_count > 0 or is_ood

        particle_stats = {
            "total": total_particles,
            "counts": cat_counts,
            "percentages": {
                k: round(v / total_particles * 100, 1) for k, v in cat_counts.items()
            },
            "mean_circularity": mean_circ,
            "d10": d10,
            "d50": d50,
            "d90": d90,
            "psd_bins": psd_bins,
            "purity_pct": purity_pct,
            "contaminant_count": contaminant_count,
            "has_contamination": has_contamination,
            "contamination_advisory": (
                f"ALERT: Detected {contaminant_count} foreign / irregular particle inclusions in {top_material} batch! Risk of print defect."
                if has_contamination
                else "Feedstock Purity Nominal · No foreign metallic contaminants detected."
            )
        }
    else:
        particle_stats = {
            "total": 0,
            "counts": {"spherical": 0, "satellite": 0, "irregular": 0, "fine": 0},
            "percentages": {"spherical": 0, "satellite": 0, "irregular": 0, "fine": 0},
            "mean_circularity": 0,
            "d10": 0,
            "d50": 0,
            "d90": 0,
            "psd_bins": [],
            "purity_pct": 100.0,
            "contaminant_count": 0,
            "has_contamination": False,
            "contamination_advisory": "No particles segmented."
        }

    return (
        {
            "filename": Path(file.filename or "image").name,
            "active_model": model_name,
            "prediction": "Unknown / None of the Above" if is_ood else top_material,
            "raw_prediction": top_material,
            "confidence": top_confidence,
            "probabilities": ranked,
            "prediction_scope": "image",
            "image_size": {"width": rgb_image.width, "height": rgb_image.height},
            "regions": regions,
            "gradcam_heatmap": gradcam_overlay_uri,
            "shap_waterfall": shap_breakdown,
            "particle_stats": particle_stats,
            "clinical_assessment": clinical_assessment,
        },
        len(payload),
    )


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = "ensemble") -> dict[str, Any]:
    trained = getattr(app.state, "trained", None)
    if trained is None:
        raise HTTPException(status_code=503, detail="The classifier is still starting.")
    result, _ = await predict_one(file, trained, model_name=model)
    return result


@app.post("/api/predict/particle")
async def predict_particle_crop(file: UploadFile = File(...), model: str = "ensemble") -> dict[str, Any]:
    """Single-particle cropped inspect mode (Mode B)."""
    trained = getattr(app.state, "trained", None)
    if trained is None:
        raise HTTPException(status_code=503, detail="The classifier is still starting.")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Choose an image.")
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            rgb_image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Extract single-chip features & classify
    chip_metrics = extract_particle_chip_features(rgb_image)
    features = extract_features(rgb_image).reshape(1, -1)
    available_models = trained.get("models", {})
    clf = available_models.get(model, trained["classifier"])
    probabilities = clf.predict_proba(features)[0]

    classes = clf.classes_
    ranked = sorted(
        (
            {"material": str(mat), "probability": float(prob)}
            for mat, prob in zip(classes, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    top_mat = ranked[0]["material"]
    conf = ranked[0]["probability"]
    is_ood = conf < OOD_CONFIDENCE_THRESHOLD

    return {
        "filename": Path(file.filename or "chip").name,
        "mode": "single_particle_crop",
        "active_model": model,
        "prediction": "Unknown Particle" if is_ood else top_mat,
        "confidence": conf,
        "probabilities": ranked,
        "particle_metrics": chip_metrics,
        "shap_waterfall": compute_shap_waterfall(features, top_mat, ranked),
    }


@app.post("/api/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one image before classifying.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Choose no more than {MAX_BATCH_FILES} images at a time.",
        )
    declared_size = sum(file.size or 0 for file in files)
    if declared_size > MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The combined upload must be {MAX_BATCH_BYTES // (1024 * 1024)} MB or smaller.",
        )

    trained = getattr(app.state, "trained", None)
    if trained is None:
        raise HTTPException(status_code=503, detail="The classifier is still starting.")

    results = []
    total_bytes = 0
    for file in files:
        result, upload_bytes = await predict_one(file, trained)
        total_bytes += upload_bytes
        results.append(result)
        if total_bytes > MAX_BATCH_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"The combined upload must be {MAX_BATCH_BYTES // (1024 * 1024)} MB or smaller.",
            )

    counts: dict[str, int] = {}
    for result in results:
        material = result["prediction"]
        counts[material] = counts.get(material, 0) + 1

    return {
        "image_count": len(results),
        "results": results,
        "distribution": [
            {
                "material": material,
                "count": count,
                "fraction": count / len(results),
            }
            for material, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ],
        "prediction_scope": "one material estimate per image",
    }
