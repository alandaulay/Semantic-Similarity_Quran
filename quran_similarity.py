"""Reproducible pipeline for Qur'anic semantic similarity and theme discovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embeddings"
SIMILARITY_DIR = PROJECT_DIR / "results" / "similarity"
CLUSTERING_DIR = PROJECT_DIR / "results" / "clustering"
THEME_DIR = PROJECT_DIR / "results" / "theme_analysis"
VISUALIZATION_DIR = PROJECT_DIR / "visualizations"
LOG_DIR = PROJECT_DIR / "logs"

API_BASE = "https://api.alquran.cloud/v1"
ARABIC_EDITION = "quran-uthmani"
TRANSLATION_EDITION = "id.indonesian"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_SURAH_COUNT = 114
EXPECTED_AYAH_COUNT = 6236
RANDOM_STATE = 42

for directory in (
    RAW_DIR,
    PROCESSED_DIR,
    EMBEDDING_DIR,
    SIMILARITY_DIR,
    CLUSTERING_DIR,
    THEME_DIR,
    VISUALIZATION_DIR,
    LOG_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def _load_gemini_environment() -> None:
    """Load a user-created key file from the workspace .env folder, if present."""
    candidates = (
        WORKSPACE_DIR / ".env" / "gemini.env",
        PROJECT_DIR / ".env",
        WORKSPACE_DIR / ".env",
    )
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _read_or_download_edition(identifier: str, force_download: bool = False) -> dict[str, Any]:
    path = RAW_DIR / f"{identifier}.json"
    if path.exists() and not force_download:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        endpoint = f"{API_BASE}/quran/{identifier}"
        response = requests.get(endpoint, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "OK" or not payload.get("data"):
            raise RuntimeError(f"Al Quran Cloud returned an invalid response for {identifier}.")
        # Preserve the downloaded response bytes without normalizing or rewriting the raw data.
        path.write_bytes(response.content)
    return payload


def acquire_dataset(force_download: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch or reuse both complete API editions and record source metadata."""
    metadata_path = RAW_DIR / "source_metadata.json"
    use_cached_raw = (
        not force_download
        and (RAW_DIR / f"{ARABIC_EDITION}.json").exists()
        and (RAW_DIR / f"{TRANSLATION_EDITION}.json").exists()
    )
    previous_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if use_cached_raw and metadata_path.exists()
        else {}
    )
    arabic = _read_or_download_edition(ARABIC_EDITION, force_download)
    translation = _read_or_download_edition(TRANSLATION_EDITION, force_download)
    _write_json(
        metadata_path,
        {
            "api_name": "Al Quran Cloud API",
            "api_base": API_BASE,
            "arabic_endpoint": f"{API_BASE}/quran/{ARABIC_EDITION}",
            "arabic_edition": arabic["data"]["edition"],
            "translation_endpoint": f"{API_BASE}/quran/{TRANSLATION_EDITION}",
            "translation_edition": translation["data"]["edition"],
            "downloaded_at_utc": previous_metadata.get(
                "downloaded_at_utc",
                None if use_cached_raw else datetime.now(timezone.utc).isoformat(),
            ),
            "metadata_updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_files": [
                f"{ARABIC_EDITION}.json",
                f"{TRANSLATION_EDITION}.json",
            ],
        },
    )
    print(
        f"Raw API responses are available in {RAW_DIR}; "
        f"{len(translation['data']['surahs'])} surahs returned."
    )
    return arabic, translation


def _flatten_edition(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    surahs = payload["data"]["surahs"]
    verses: list[dict[str, Any]] = []
    surah_metadata: list[dict[str, Any]] = []
    for surah in surahs:
        surah_metadata.append(
            {
                "surah_number": surah["number"],
                "surah_arabic_name": surah["name"],
                "surah_english_name": surah["englishName"],
                "surah_english_translation": surah["englishNameTranslation"],
                "revelation_type": surah["revelationType"],
                "api_ayah_count": len(surah["ayahs"]),
            }
        )
        for ayah in surah["ayahs"]:
            verses.append(
                {
                    "global_number": ayah["number"],
                    "ayah_number": ayah["numberInSurah"],
                    "juz": ayah.get("juz"),
                    "text": ayah["text"],
                    "surah_number": surah["number"],
                }
            )
    return verses, surah_metadata


def validate_and_prepare(
    arabic_payload: dict[str, Any],
    translation_payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align the two editions by global verse number and report data-integrity checks."""
    arabic_verses, arabic_surahs = _flatten_edition(arabic_payload)
    translated_verses, translation_surahs = _flatten_edition(translation_payload)
    arabic_by_number = {verse["global_number"]: verse for verse in arabic_verses}
    translation_by_number = {verse["global_number"]: verse for verse in translated_verses}

    checks: dict[str, Any] = {
        "arabic_surah_count": len(arabic_surahs),
        "translation_surah_count": len(translation_surahs),
        "arabic_surah_coverage_valid": [
            row["surah_number"] for row in arabic_surahs
        ] == list(range(1, EXPECTED_SURAH_COUNT + 1)),
        "translation_surah_coverage_valid": [
            row["surah_number"] for row in translation_surahs
        ] == list(range(1, EXPECTED_SURAH_COUNT + 1)),
        "arabic_ayah_count": len(arabic_verses),
        "translation_ayah_count": len(translated_verses),
        "expected_surah_count": EXPECTED_SURAH_COUNT,
        "expected_ayah_count": EXPECTED_AYAH_COUNT,
        "global_ayah_numbers_match": set(arabic_by_number) == set(translation_by_number),
        "surah_metadata_match": [
            row["surah_number"] for row in arabic_surahs
        ] == [row["surah_number"] for row in translation_surahs],
        "duplicate_arabic_global_numbers": len(arabic_by_number) != len(arabic_verses),
        "duplicate_translation_global_numbers": len(translation_by_number) != len(translated_verses),
        "verse_location_mismatches": 0,
        "juz_mismatches": 0,
    }
    if checks["global_ayah_numbers_match"]:
        rows: list[dict[str, Any]] = []
        for global_number in sorted(translation_by_number):
            arabic = arabic_by_number[global_number]
            translated = translation_by_number[global_number]
            if (
                arabic["surah_number"] != translated["surah_number"]
                or arabic["ayah_number"] != translated["ayah_number"]
            ):
                checks["verse_location_mismatches"] += 1
            if arabic["juz"] != translated["juz"]:
                checks["juz_mismatches"] += 1
            rows.append(
                {
                    "global_number": global_number,
                    "surah_number": translated["surah_number"],
                    "surah_name_ar": arabic_surahs[translated["surah_number"] - 1]["surah_arabic_name"],
                    "surah_name_en": translation_surahs[translated["surah_number"] - 1]["surah_english_name"],
                    "ayah_number": translated["ayah_number"],
                    "juz": translated["juz"],
                    "text_arabic": arabic["text"],
                    "text_indonesian": translated["text"],
                }
            )
        dataset = pd.DataFrame(rows)
    else:
        dataset = pd.DataFrame()

    if not dataset.empty:
        checks.update(
            {
                "rows": len(dataset),
                "unique_verse_identities": int(
                    dataset[["surah_number", "ayah_number"]].drop_duplicates().shape[0]
                ),
                "missing_values_by_column": {
                    key: int(value)
                    for key, value in dataset.isna().sum().to_dict().items()
                },
                "empty_arabic_texts": int(
                    dataset["text_arabic"].astype(str).str.strip().eq("").sum()
                ),
                "empty_translation_texts": int(
                    dataset["text_indonesian"].astype(str).str.strip().eq("").sum()
                ),
                "duplicate_verse_identities": int(
                    dataset.duplicated(["surah_number", "ayah_number"]).sum()
                ),
                "surah_ayah_sequence_valid": all(
                    group["ayah_number"].tolist()
                    == list(range(1, len(group) + 1))
                    for _, group in dataset.groupby("surah_number", sort=True)
                ),
                "juz_coverage_valid": set(dataset["juz"].dropna().astype(int))
                == set(range(1, 31)),
                "no_missing_values": not bool(dataset.isna().any().any()),
                "arabic_translation_alignment_valid": bool(
                    dataset["global_number"].is_unique
                    and dataset[["surah_number", "ayah_number"]].notna().all().all()
                    and checks["verse_location_mismatches"] == 0
                    and checks["juz_mismatches"] == 0
                ),
            }
        )
    else:
        checks["rows"] = 0

    checks["expected_counts_match"] = (
        checks["arabic_surah_count"] == EXPECTED_SURAH_COUNT
        and checks["translation_surah_count"] == EXPECTED_SURAH_COUNT
        and checks["arabic_ayah_count"] == EXPECTED_AYAH_COUNT
        and checks["translation_ayah_count"] == EXPECTED_AYAH_COUNT
    )
    checks["validation_passed"] = bool(
        checks["expected_counts_match"]
        and checks["arabic_surah_coverage_valid"]
        and checks["translation_surah_coverage_valid"]
        and checks["global_ayah_numbers_match"]
        and checks["surah_metadata_match"]
        and not checks["duplicate_arabic_global_numbers"]
        and not checks["duplicate_translation_global_numbers"]
        and checks["verse_location_mismatches"] == 0
        and checks["juz_mismatches"] == 0
        and not dataset.empty
        and checks.get("empty_arabic_texts", 1) == 0
        and checks.get("empty_translation_texts", 1) == 0
        and checks.get("duplicate_verse_identities", 1) == 0
        and checks.get("surah_ayah_sequence_valid", False)
        and checks.get("juz_coverage_valid", False)
        and checks.get("no_missing_values", False)
        and checks.get("arabic_translation_alignment_valid", False)
    )
    checks["validation_note"] = (
        "Counts are compared with the expected 114 surahs and 6,236 ayahs. "
        "Raw API files remain unchanged if a check fails."
    )
    _write_json(RAW_DIR / "validation_report.json", checks)
    metadata_path = RAW_DIR / "source_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "surah_count": checks["translation_surah_count"],
            "ayah_count": checks["translation_ayah_count"],
            "validation_status": "passed" if checks["validation_passed"] else "failed",
        }
    )
    _write_json(metadata_path, metadata)

    if not checks["validation_passed"]:
        raise ValueError(
            "Dataset validation failed. Raw downloads were preserved; review "
            f"{RAW_DIR / 'validation_report.json'} before continuing."
        )
    processed_path = PROCESSED_DIR / "quran_id_indonesian.csv"
    dataset.to_csv(processed_path, index=False, encoding="utf-8")
    print(
        f"Validation passed: {checks['translation_surah_count']} surahs, "
        f"{checks['translation_ayah_count']:,} aligned ayahs."
    )
    return dataset, checks


def preprocess_text(text: str) -> str:
    """Apply conservative Unicode and whitespace normalization without stemming."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = "".join(char for char in normalized if char.isprintable() or char.isspace())
    return re.sub(r"\s+", " ", normalized).strip()


def prepare_embedding_texts(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    result["text_indonesian_preprocessed"] = result["text_indonesian"].map(preprocess_text)
    empty_count = int(result["text_indonesian_preprocessed"].eq("").sum())
    if empty_count:
        raise ValueError(f"{empty_count} verses have empty preprocessed translations.")
    result.to_csv(PROCESSED_DIR / "quran_id_indonesian_preprocessed.csv", index=False, encoding="utf-8")
    print(f"Preprocessing complete; {empty_count} empty translations.")
    return result


def run_eda(dataset: pd.DataFrame) -> dict[str, Path]:
    """Create reproducible EDA charts from the validated dataset."""
    output_paths: dict[str, Path] = {}
    surah_counts = dataset.groupby(["surah_number", "surah_name_en"]).size().reset_index(name="ayah_count")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(surah_counts["surah_number"], surah_counts["ayah_count"], color="#168AAD")
    ax.set(title="Jumlah Ayat per Surah", xlabel="Nomor surah", ylabel="Jumlah ayat")
    fig.tight_layout()
    output_paths["surah_counts"] = VISUALIZATION_DIR / "eda_ayah_per_surah.png"
    fig.savefig(output_paths["surah_counts"], dpi=160)
    plt.show()
    plt.close(fig)

    text_lengths = dataset["text_indonesian"].str.len()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(text_lengths, bins=40, ax=ax, color="#52B788")
    ax.set(title="Distribusi Panjang Terjemahan Ayat", xlabel="Panjang karakter", ylabel="Jumlah ayat")
    fig.tight_layout()
    output_paths["translation_lengths"] = VISUALIZATION_DIR / "eda_translation_length.png"
    fig.savefig(output_paths["translation_lengths"], dpi=160)
    plt.show()
    plt.close(fig)

    missing = pd.DataFrame(
        {"column": dataset.columns, "missing_count": dataset.isna().sum().to_numpy()}
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=missing, x="column", y="missing_count", color="#E76F51", ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set(title="Pemeriksaan Missing Values", xlabel="Kolom", ylabel="Jumlah nilai kosong")
    fig.tight_layout()
    output_paths["missing_values"] = VISUALIZATION_DIR / "eda_missing_values.png"
    fig.savefig(output_paths["missing_values"], dpi=160)
    plt.show()
    plt.close(fig)

    juz_counts = dataset.groupby("juz").size().reindex(range(1, 31), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(juz_counts.index, juz_counts.values, color="#457B9D")
    ax.set(title="Distribusi Jumlah Ayat per Juz", xlabel="Juz", ylabel="Jumlah ayat")
    fig.tight_layout()
    output_paths["juz_counts"] = VISUALIZATION_DIR / "eda_ayah_per_juz.png"
    fig.savefig(output_paths["juz_counts"], dpi=160)
    plt.show()
    plt.close(fig)
    print(
        f"EDA: {len(dataset):,} verses; translation length median "
        f"{text_lengths.median():.0f} characters; {int(dataset.isna().sum().sum())} missing values."
    )
    return output_paths


def load_or_create_embeddings(
    dataset: pd.DataFrame,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
    force_recompute: bool = False,
) -> tuple[np.ndarray, SentenceTransformer]:
    """Load cached embeddings or encode verse translations with the selected model."""
    embeddings_path = EMBEDDING_DIR / "quran_id_indonesian_minilm_l12.npy"
    metadata_path = EMBEDDING_DIR / "embedding_metadata.json"
    input_hash = hashlib.sha256(
        "\n".join(dataset["text_indonesian_preprocessed"].tolist()).encode("utf-8")
    ).hexdigest()
    cached_metadata = {}
    if metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        embeddings_path.exists()
        and not force_recompute
        and cached_metadata.get("model_name") == model_name
        and cached_metadata.get("verse_count") == len(dataset)
        and cached_metadata.get("input_sha256") == input_hash
    ):
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] == len(dataset):
            print(f"Loaded cached embeddings: shape={embeddings.shape}.")
            return embeddings, SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        dataset["text_indonesian_preprocessed"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(embeddings_path, embeddings)
    _write_json(
        metadata_path,
        {
            "model_name": model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "verse_count": len(dataset),
            "input_sha256": input_hash,
            "normalized": True,
            "device": device,
            "batch_size": batch_size,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Created embeddings: shape={embeddings.shape}; device={device}.")
    return embeddings, model


def calculate_similarity(
    dataset: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
    global_pair_count: int = 100,
) -> dict[str, pd.DataFrame]:
    """Save each verse's nearest neighbors and exact top unique global pairs."""
    if embeddings.shape[0] != len(dataset):
        raise ValueError("Embedding count must match the validated dataset row count.")
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=4)
    neighbors.fit(embeddings)
    distances, indices = neighbors.kneighbors(embeddings, n_neighbors=top_k + 1)

    verses = dataset.reset_index(drop=True)
    nearest_rows: list[dict[str, Any]] = []
    for source_index in range(len(verses)):
        rank = 0
        for distance, candidate_index in zip(distances[source_index], indices[source_index]):
            if candidate_index == source_index:
                continue
            rank += 1
            if rank > top_k:
                break
            source = verses.iloc[source_index]
            candidate = verses.iloc[candidate_index]
            nearest_rows.append(
                {
                    "source_global_number": int(source["global_number"]),
                    "source_surah": int(source["surah_number"]),
                    "source_surah_name": source["surah_name_en"],
                    "source_ayah": int(source["ayah_number"]),
                    "source_text": source["text_indonesian"],
                    "rank": rank,
                    "similarity": float(1 - distance),
                    "match_global_number": int(candidate["global_number"]),
                    "match_surah": int(candidate["surah_number"]),
                    "match_surah_name": candidate["surah_name_en"],
                    "match_ayah": int(candidate["ayah_number"]),
                    "match_text": candidate["text_indonesian"],
                    "translation_exact_match": bool(
                        source["text_indonesian"] == candidate["text_indonesian"]
                    ),
                }
            )
    nearest_df = pd.DataFrame(nearest_rows)
    nearest_df.to_csv(SIMILARITY_DIR / "top_5_similar_verses_per_source.csv", index=False, encoding="utf-8")

    # A global top-K unordered pair must occur in the top K list of each endpoint.
    candidate_count = min(global_pair_count, len(verses) - 1)
    global_distances, global_indices = neighbors.kneighbors(
        embeddings, n_neighbors=candidate_count + 1
    )
    pair_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for source_index in range(len(verses)):
        for distance, candidate_index in zip(
            global_distances[source_index], global_indices[source_index]
        ):
            if source_index == candidate_index:
                continue
            left, right = sorted((source_index, int(candidate_index)))
            if (left, right) in pair_rows:
                continue
            first, second = verses.iloc[left], verses.iloc[right]
            pair_rows[(left, right)] = {
                "global_number_1": int(first["global_number"]),
                "surah_name_1": first["surah_name_en"],
                "ayah_number_1": int(first["ayah_number"]),
                "text_1": first["text_indonesian"],
                "global_number_2": int(second["global_number"]),
                "surah_name_2": second["surah_name_en"],
                "ayah_number_2": int(second["ayah_number"]),
                "text_2": second["text_indonesian"],
                "translation_exact_match": bool(
                    first["text_indonesian"] == second["text_indonesian"]
                ),
                "similarity": float(1 - distance),
            }
    global_df = pd.DataFrame(pair_rows.values()).sort_values(
        "similarity", ascending=False
    ).head(global_pair_count).reset_index(drop=True)
    global_df.insert(0, "rank", np.arange(1, len(global_df) + 1))
    global_df.to_csv(SIMILARITY_DIR / "global_most_similar_pairs.csv", index=False, encoding="utf-8")

    top_verses = (
        nearest_df.groupby(
            ["source_global_number", "source_surah", "source_surah_name", "source_ayah", "source_text"],
            as_index=False,
        )
        .agg(mean_top5_similarity=("similarity", "mean"))
        .sort_values("mean_top5_similarity", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_verses.insert(0, "rank", np.arange(1, len(top_verses) + 1))
    top_verses.to_csv(SIMILARITY_DIR / "top_5_source_verses_by_mean_neighbor_similarity.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(nearest_df["similarity"], bins=40, color="#168AAD", edgecolor="white")
    ax.set(title="Distribusi Similarity Top-5 per Ayat", xlabel="Cosine similarity embedding", ylabel="Jumlah pasangan")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "similarity_top5_distribution.png", dpi=160)
    plt.show()
    plt.close(fig)

    sample_indices = np.linspace(0, len(verses) - 1, min(40, len(verses)), dtype=int)
    sample_similarity = embeddings[sample_indices] @ embeddings[sample_indices].T
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sample_similarity, cmap="viridis", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Cosine similarity"})
    ax.set(title="Heatmap Similarity Sampel Sistematis (maks. 40 ayat)", xlabel="Ayat sampel", ylabel="Ayat sampel")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "similarity_sample_heatmap.png", dpi=160)
    plt.show()
    plt.close(fig)

    top_global = global_df.head(20).sort_values("similarity")
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [
        f"{row.surah_name_1} {row.ayah_number_1} ↔ {row.surah_name_2} {row.ayah_number_2}"
        for row in top_global.itertuples()
    ]
    ax.barh(labels, top_global["similarity"], color="#52B788")
    ax.set(title="20 Pasangan Ayat dengan Similarity Tertinggi", xlabel="Cosine similarity", ylabel="Pasangan ayat")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "similarity_global_top20.png", dpi=160)
    plt.show()
    plt.close(fig)
    print(
        f"Similarity done: {len(nearest_df):,} directed top-{top_k} matches; "
        f"{len(global_df)} unique global pairs; source verse excludes itself."
    )
    return {"nearest_verses": nearest_df, "global_pairs": global_df, "top_verses": top_verses}


def _mean_pairwise_cosine(cluster_embeddings: np.ndarray) -> float:
    count = len(cluster_embeddings)
    if count < 2:
        return float("nan")
    resultant_norm_squared = float(np.sum(np.sum(cluster_embeddings, axis=0) ** 2))
    return (resultant_norm_squared - count) / (count * (count - 1))


def run_clustering(
    dataset: pd.DataFrame,
    embeddings: np.ndarray,
    k_values: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate K-Means configurations and retain the assignment with K fixed at five."""
    evaluation_rows: list[dict[str, Any]] = []
    final_labels: np.ndarray | None = None
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(embeddings)
        evaluation_rows.append(
            {
                "k": k,
                "silhouette_sampled": float(
                    silhouette_score(
                        embeddings,
                        labels,
                        sample_size=min(2000, len(dataset)),
                        random_state=RANDOM_STATE,
                    )
                ),
                "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(embeddings, labels)),
                "inertia": float(model.inertia_),
                "cluster_count": int(len(np.unique(labels))),
            }
        )
        if k == 5:
            final_labels = labels
    if final_labels is None:
        raise ValueError("The clustering experiment must include K=5.")

    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(CLUSTERING_DIR / "kmeans_evaluation.csv", index=False, encoding="utf-8")
    clustered = dataset.reset_index(drop=True).copy()
    clustered["cluster"] = final_labels + 1

    summary_rows: list[dict[str, Any]] = []
    for cluster_id, group in clustered.groupby("cluster", sort=True):
        member_indices = np.flatnonzero(final_labels == cluster_id - 1)
        member_embeddings = embeddings[member_indices]
        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "member_count": len(group),
                "mean_pairwise_cosine": _mean_pairwise_cosine(member_embeddings),
            }
        )
    summary = pd.DataFrame(summary_rows)
    clustered.to_csv(CLUSTERING_DIR / "quran_ayah_clusters_k5.csv", index=False, encoding="utf-8")
    summary.to_csv(CLUSTERING_DIR / "cluster_summary_k5.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(evaluation["k"], evaluation["silhouette_sampled"], marker="o")
    axes[0].set(title="Silhouette (sampel <= 2.000)", xlabel="Jumlah cluster K", ylabel="Silhouette")
    axes[1].plot(evaluation["k"], evaluation["davies_bouldin"], marker="o", color="#E76F51")
    axes[1].set(title="Davies-Bouldin", xlabel="Jumlah cluster K", ylabel="DBI (lebih rendah lebih baik)")
    axes[2].plot(evaluation["k"], evaluation["calinski_harabasz"], marker="o", color="#52B788")
    axes[2].set(title="Calinski-Harabasz", xlabel="Jumlah cluster K", ylabel="CH (lebih tinggi lebih baik)")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "clustering_metric_comparison.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="cluster", y="member_count", color="#457B9D", ax=ax)
    ax.set(title="Jumlah Ayat dalam Masing-Masing Cluster K=5", xlabel="Cluster", ylabel="Jumlah ayat")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "clustering_k5_sizes.png", dpi=160)
    plt.show()
    plt.close(fig)

    projected = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings)
    projection = pd.DataFrame(
        {"PC1": projected[:, 0], "PC2": projected[:, 1], "cluster": final_labels + 1}
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=projection,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        s=18,
        alpha=0.65,
        linewidth=0,
        ax=ax,
    )
    ax.set(title="Proyeksi PCA Embedding Ayat (K-Means K=5)", xlabel="Komponen utama 1", ylabel="Komponen utama 2")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "clustering_pca_k5.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=summary, x="cluster", y="mean_pairwise_cosine", color="#52B788", ax=ax)
    ax.set(title="Rata-rata Similarity Internal Exact per Cluster", xlabel="Cluster K=5", ylabel="Rata-rata cosine pasangan internal")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "cluster_internal_similarity_k5.png", dpi=160)
    plt.show()
    plt.close(fig)

    sample_indices_by_cluster: dict[int, np.ndarray] = {}
    representative_rows: list[dict[str, Any]] = []
    for cluster_id in sorted(clustered["cluster"].unique()):
        indices = np.flatnonzero(clustered["cluster"].to_numpy() == cluster_id)
        centroid = embeddings[indices].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroid_similarities = embeddings[indices] @ centroid
        selected = indices[np.argsort(-centroid_similarities)[:15]]
        sample_indices_by_cluster[int(cluster_id)] = selected
        for sample_index in selected:
            row = clustered.iloc[sample_index]
            representative_rows.append(
                {
                    "cluster": int(cluster_id),
                    "global_number": int(row["global_number"]),
                    "surah_name": row["surah_name_en"],
                    "ayah_number": int(row["ayah_number"]),
                    "text_indonesian": row["text_indonesian"],
                }
            )
    representatives = pd.DataFrame(representative_rows)
    representatives.to_csv(
        CLUSTERING_DIR / "cluster_centroid_representatives.csv",
        index=False,
        encoding="utf-8",
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for cluster_id, ax in zip(sorted(sample_indices_by_cluster), axes.flat):
        selected = sample_indices_by_cluster[cluster_id]
        similarity_block = embeddings[selected] @ embeddings[selected].T
        labels = [
            f"{clustered.iloc[index]['surah_name_en'][:8]}:{int(clustered.iloc[index]['ayah_number'])}"
            for index in selected
        ]
        sns.heatmap(
            similarity_block,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=cluster_id == 5,
        )
        ax.tick_params(axis="x", labelrotation=70, labelsize=6)
        ax.tick_params(axis="y", labelrotation=0, labelsize=6)
        ax.set(
            title=f"Cluster {cluster_id}: 15 ayat terdekat ke centroid",
            xlabel="Referensi ayat",
            ylabel="Referensi ayat",
        )
    axes.flat[-1].set_visible(False)
    fig.suptitle("Kedekatan Internal Cluster — Sampel Paling Representatif", y=1.01)
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "cluster_within_group_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(
        "Clustering complete. K=5 is used to meet the assignment's five-group requirement; "
        "metric scores, cluster sizes, exact internal similarities, and within-cluster "
        "representative heatmaps are reported separately."
    )
    return evaluation, clustered


THEME_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "main_theme": {"type": "STRING"},
        "explanation": {"type": "STRING"},
        "connecting_concepts": {"type": "ARRAY", "items": {"type": "STRING"}},
        "meaning_similarities": {"type": "ARRAY", "items": {"type": "STRING"}},
        "context_differences": {"type": "ARRAY", "items": {"type": "STRING"}},
        "representative_verses": {"type": "ARRAY", "items": {"type": "STRING"}},
        "group_conclusion": {"type": "STRING"},
        "limitations": {"type": "STRING"},
    },
    "required": [
        "main_theme",
        "explanation",
        "connecting_concepts",
        "meaning_similarities",
        "context_differences",
        "representative_verses",
        "group_conclusion",
        "limitations",
    ],
}


def _extract_gemini_json(response: dict[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidate response.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Gemini response was not valid JSON despite the JSON response setting.") from error
    missing = set(THEME_SCHEMA["required"]) - set(parsed)
    if missing:
        raise RuntimeError(f"Gemini JSON response is missing fields: {sorted(missing)}")
    return parsed


def discover_themes(clustered: pd.DataFrame, api_key: str | None = None) -> dict[str, Any]:
    """Analyze each K=5 cluster independently with Gemini and cache raw prompts/results."""
    _load_gemini_environment()
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        status = {
            "status": "blocked_configuration_required",
            "message": (
                "Gemini analysis was not run: set GEMINI_API_KEY in "
                ".env/gemini.env and rerun this section."
            ),
            "completed_clusters": 0,
            "required_clusters": 5,
        }
        _write_json(THEME_DIR / "theme_analysis_status.json", status)
        print(status["message"])
        return status

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    all_results: list[dict[str, Any]] = []
    for cluster_id, group in clustered.groupby("cluster", sort=True):
        sample_count = min(30, len(group))
        ordered_group = group.sort_values(
            "cosine_similarity_to_cluster_centroid", kind="stable"
        )
        sampled_group = ordered_group.iloc[
            np.linspace(0, len(ordered_group) - 1, sample_count, dtype=int)
        ]
        verses = [
            {
                "surah": row.surah_name_en,
                "surah_number": int(row.surah_number),
                "ayah_number": int(row.ayah_number),
                "global_number": int(row.global_number),
                "translation": row.text_indonesian,
                "cosine_similarity_to_cluster_centroid": round(
                    float(row.cosine_similarity_to_cluster_centroid), 6
                ),
            }
            for row in sampled_group.itertuples()
        ]
        prompt = (
            "Anda membantu penelitian komputasional ayat Al-Qur'an. Analisis hanya berdasarkan "
            "teks terjemahan yang diberikan. Ini bukan tafsir resmi. Similarity embedding "
            "menunjukkan kedekatan representasi teks, bukan bukti ayat bermakna identik. "
            "Jangan mengarang kandungan; jelaskan keterbatasan bila konteks tidak cukup. "
            "Identifikasi tema utama, penjelasan, konsep penghubung, persamaan makna, "
            "perbedaan konteks, ayat representatif (tulis referensi surah:ayat), kesimpulan, "
            "dan keterbatasan interpretasi. Data yang diberikan adalah sampel, bukan seluruh "
            "cluster; jangan menganggap semua ayat cluster memiliki satu tema.\n\n"
            f"Nomor cluster: {cluster_id}\n"
            f"Jumlah ayat dalam cluster: {len(group)}\n"
            f"Jumlah ayat dalam sampel: {len(verses)}\n"
            "Metode sampel: maksimal 30 ayat dipilih sistematis di sepanjang urutan similarity "
            "ke centroid cluster (terendah hingga tertinggi) agar tingkat keterwakilan berbeda "
            "tercakup. Sampel tidak mewakili pemeriksaan semua anggota.\n"
            "Similarity terlampir adalah cosine embedding terhadap centroid cluster, "
            "bukan skor tafsir atau kesamaan pasangan langsung.\n"
            "Daftar ayat:\n"
            + json.dumps(verses, ensure_ascii=False)
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = THEME_DIR / f"cluster_{cluster_id}_{prompt_hash[:16]}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            all_results.append(cached)
            print(f"Cluster {cluster_id}: loaded cached LLM response.")
            continue

        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": THEME_SCHEMA,
                "temperature": 0.1,
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        response = None
        for attempt in range(5):
            try:
                response = requests.post(
                    url,
                    headers={"x-goog-api-key": resolved_key},
                    json=request_body,
                    timeout=180,
                )
            except (requests.Timeout, requests.ConnectionError) as error:
                if attempt == 4:
                    raise RuntimeError(
                        f"Gemini request for cluster {cluster_id} failed after "
                        f"{attempt + 1} attempts due to a network timeout or connection error."
                    ) from error
                time.sleep(2**attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = min(float(retry_after), 60) if retry_after else 2**attempt
                time.sleep(wait_seconds)
                continue
            break
        if response is None:
            raise RuntimeError(f"No response received for Gemini cluster {cluster_id}.")
        if not response.ok:
            raise RuntimeError(
                f"Gemini API failed for cluster {cluster_id} with HTTP "
                f"{response.status_code}: {response.text[:1000]}"
            )
        response_payload = response.json()
        result = {
            "cluster": int(cluster_id),
            "model": model_name,
            "cluster_verse_count": len(group),
            "sampled_verse_count": len(verses),
            "sampling_method": "Systematic sample of up to 30 verses ordered by centroid cosine similarity.",
            "prompt_sha256": prompt_hash,
            "prompt": prompt,
            "response": _extract_gemini_json(response_payload),
            "raw_response": response_payload,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(cache_path, result)
        all_results.append(result)
        print(f"Cluster {cluster_id}: Gemini theme analysis saved.")

    status = {
        "status": "completed",
        "model": model_name,
        "completed_clusters": len(all_results),
        "required_clusters": 5,
    }
    _write_json(THEME_DIR / "theme_analysis.json", all_results)
    _write_json(THEME_DIR / "theme_analysis_status.json", status)
    cluster_summary = pd.read_csv(CLUSTERING_DIR / "cluster_summary_k5.csv")
    _plot_theme_results(all_results, cluster_summary)
    return status


def _plot_theme_results(
    theme_results: list[dict[str, Any]],
    cluster_summary: pd.DataFrame,
) -> None:
    if len(theme_results) != 5:
        print("Theme comparison chart not generated: five actual LLM responses are required.")
        return
    cluster_lookup = cluster_summary.set_index("cluster")
    table_rows = []
    for result in theme_results:
        cluster_id = result["cluster"]
        if cluster_id not in cluster_lookup.index:
            raise ValueError(f"Missing computed summary for cluster {cluster_id}.")
        summary = cluster_lookup.loc[cluster_id]
        table_rows.append(
            [
                cluster_id,
                result["response"]["main_theme"],
                f"{int(summary['member_count']):,}",
                f"{summary['mean_pairwise_cosine']:.3f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_title("Ringkasan Tema Aktual dan Ukuran Kelompok", pad=18)
    ax.table(
        cellText=table_rows,
        colLabels=["Cluster", "Tema dari Gemini", "Jumlah ayat", "Rata-rata cosine internal"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.1, 0.5, 0.18, 0.22],
    )
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "theme_summary.png", dpi=160)
    plt.show()
    plt.close(fig)


def attach_centroid_similarity(clustered: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Add per-verse cosine similarity to its assigned cluster centroid for LLM context."""
    result = clustered.reset_index(drop=True).copy()
    similarities = np.zeros(len(result), dtype=np.float32)
    for cluster_id in sorted(result["cluster"].unique()):
        indices = np.flatnonzero(result["cluster"].to_numpy() == cluster_id)
        centroid = embeddings[indices].mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            raise ValueError(f"Cluster {cluster_id} has a zero-norm centroid.")
        centroid /= centroid_norm
        similarities[indices] = embeddings[indices] @ centroid
    result["cosine_similarity_to_cluster_centroid"] = similarities
    return result
