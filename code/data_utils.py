from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

FOCUS_ISO = "ARG"
FOCUS_COUNTRY_NAME = "Argentina"
V2_METRICS_FILE = "opportunity_metrics_hs4_arg.csv"
CORDOBA_EXPORTS_FILE = "cordoba_exports.csv"
CORDOBA_RUBRO_CROSSWALK_FILE = "cordoba_rubro_to_hs.csv"

HS_SECTION_RULES = [
    (1, range(1, 6), "1. Live animals; animal products"),
    (2, range(6, 15), "2. Vegetable products"),
    (3, range(15, 16), "3. Animal or vegetable fats and oils"),
    (4, range(16, 25), "4. Prepared foodstuffs; beverages, spirits and tobacco"),
    (5, range(25, 28), "5. Mineral products"),
    (6, range(28, 39), "6. Products of the chemical or allied industries"),
    (7, range(39, 41), "7. Plastics and articles thereof; rubber and articles thereof"),
    (8, range(41, 44), "8. Raw hides and skins, leather, furskins and articles thereof"),
    (9, range(44, 47), "9. Wood and articles of wood"),
    (10, range(47, 50), "10. Pulp, paper and paperboard"),
    (11, range(50, 64), "11. Textiles and textile articles"),
    (12, range(64, 68), "12. Footwear, headgear, umbrellas and related articles"),
    (13, range(68, 71), "13. Articles of stone, plaster, cement, ceramics and glass"),
    (14, range(71, 72), "14. Pearls, precious stones and metals"),
    (15, range(72, 84), "15. Base metals and articles of base metal"),
    (16, range(84, 86), "16. Machinery, mechanical appliances and electrical equipment"),
    (17, range(86, 90), "17. Vehicles, aircraft, vessels and associated transport equipment"),
    (18, range(90, 93), "18. Optical, photographic, medical and musical instruments"),
    (19, range(93, 94), "19. Arms and ammunition"),
    (20, range(94, 97), "20. Miscellaneous manufactured articles"),
    (21, range(97, 98), "21. Works of art, collectors' pieces and antiques"),
]


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "input").exists() and (parent / "data" / "intermediate").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def input_dir() -> Path:
    return project_root() / "data" / "input"


def intermediate_dir() -> Path:
    return project_root() / "data" / "intermediate"


def output_dir() -> Path:
    return project_root() / "data" / "output"


def _file_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


def _resolve_intermediate_csv(primary_name: str, fallback_name: str | None = None) -> Path:
    primary = intermediate_dir() / primary_name
    if primary.exists():
        return primary
    if fallback_name:
        fallback = intermediate_dir() / fallback_name
        if fallback.exists():
            return fallback
    raise FileNotFoundError(
        f"Missing intermediate file. Tried: {primary_name}"
        + (f", {fallback_name}" if fallback_name else "")
    )


def _parse_percent_share(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace("%", "", regex=False)
    return pd.to_numeric(text, errors="coerce") / 100.0


def _expand_hs_prefix_token(token: str) -> list[str]:
    token = str(token).strip()
    if not token:
        return []
    if "-" not in token:
        return [token.zfill(2) if token.isdigit() and len(token) <= 2 else token]

    start, end = [part.strip() for part in token.split("-", 1)]
    if not start.isdigit() or not end.isdigit():
        return []
    width = max(len(start), len(end), 2)
    return [str(value).zfill(width) for value in range(int(start), int(end) + 1)]


def normalize_0_1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    lo = s.min()
    hi = s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def normalize_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma


def hs4_to_section_name(code: str) -> str:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) < 2:
        return "Other"
    chapter = int(digits[:2])
    for _, chapters, label in HS_SECTION_RULES:
        if chapter in chapters:
            return label
    return "Other"


def ensure_required_opportunity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required_defaults = {
        "dai_index": 0.0,
        "dai_percentile": 0.0,
        "dai_lead": 0.0,
        "accessible_market_size": 0.0,
        "accessible_market_growth_5y": 0.0,
        "accessible_market_size_share": 0.0,
        "accessible_market_to_market_ratio": 0.0,
    }
    for col, default in required_defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


@st.cache_data(show_spinner=False)
def load_rankings_countries(year: int = 2024) -> set[str]:
    path = input_dir() / "rankings.csv"
    df = pd.read_csv(path)
    if "year" not in df.columns:
        return set()
    df = df[pd.to_numeric(df["year"], errors="coerce") == int(year)].copy()
    if df.empty:
        return set()

    iso_col = "country_iso3_code" if "country_iso3_code" in df.columns else None
    if iso_col is None:
        candidates = [c for c in df.columns if "iso" in c.lower() and "3" in c.lower()]
        if candidates:
            iso_col = candidates[0]
    if iso_col is None:
        return set()

    s = df[iso_col].astype(str).str.upper().str.strip().str[:3]
    s = s[s.str.len() == 3]
    return set(s.unique().tolist())


@st.cache_data(show_spinner=False)
def load_hs92_reference() -> pd.DataFrame:
    path = input_dir() / "hs92_4digits.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["hs4"] = df["product_hs92_code"].astype(str).str.zfill(4)
    return df[["hs4", "product_name_short", "product_name", "sector", "green_product"]].drop_duplicates("hs4")


@st.cache_data(show_spinner=False)
def _load_anchor_proximity_dataset_cached(_mtime_ns: int) -> pd.DataFrame:
    path = output_dir() / "anchors_proximity_percentile.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for col in ["anchor_hs4", "candidate_hs4"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(4)

    df = ensure_required_opportunity_metrics(df)
    numeric_cols = [
        "proximity",
        "proximity_above_country_median",
        "proximity_rank",
        "eligible_candidate_count",
        "pci",
        "cog",
        "distance_travelled",
        "accessible_market_size",
        "accessible_market_size_share",
        "accessible_market_growth_5y",
        "dai_index",
        "dai_percentile",
        "dai_lead",
        "attractiveness_score",
        "feasibility_score",
        "combined_score",
        "anchor_density",
        "anchor_density_percentile",
        "anchor_embeddedness",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["anchor_sector"] = df.get("anchor_sector", "").fillna("Other").astype(str)
    df["anchor_hs_section_name"] = df.get(
        "anchor_hs_section_name",
        df.get("anchor_hs4", "").astype(str).map(hs4_to_section_name),
    )
    df["anchor_hs_section_name"] = df["anchor_hs_section_name"].fillna("Other").astype(str)
    df["candidate_sector"] = df.get("candidate_sector", "").fillna("Other").astype(str)
    df["candidate_hs_section_name"] = df.get(
        "candidate_hs_section_name",
        df.get("candidate_hs4", "").astype(str).map(hs4_to_section_name),
    )
    df["candidate_hs_section_name"] = df["candidate_hs_section_name"].fillna("Other").astype(str)
    df["accessible_market_size_b"] = (
        pd.to_numeric(df.get("accessible_market_size", 0), errors="coerce").fillna(0.0) / 1_000_000_000
    )

    try:
        metrics = pd.read_csv(intermediate_dir() / V2_METRICS_FILE, usecols=["hs4", "raw_rca_trade"])
        metrics["hs4"] = metrics["hs4"].astype(str).str.zfill(4)
        metrics["raw_rca_trade"] = pd.to_numeric(metrics["raw_rca_trade"], errors="coerce").fillna(0.0)
        df = df.merge(
            metrics.rename(columns={"hs4": "candidate_hs4", "raw_rca_trade": "candidate_raw_rca"}),
            on="candidate_hs4",
            how="left",
        )
    except Exception:
        df["candidate_raw_rca"] = 0.0
    df["candidate_raw_rca"] = pd.to_numeric(df.get("candidate_raw_rca", 0), errors="coerce").fillna(0.0)
    return df


def load_anchor_proximity_dataset() -> pd.DataFrame:
    path = output_dir() / "anchors_proximity_percentile.csv"
    return _load_anchor_proximity_dataset_cached(_file_mtime_ns(path))


@st.cache_data(show_spinner=False)
def load_eff_num_exp() -> pd.DataFrame:
    path = _resolve_intermediate_csv("hs92_attributes.csv")
    df = pd.read_csv(path, usecols=["hs92", "eff_num_exp"])
    df["hs4"] = df["hs92"].astype(str).str.zfill(4)
    return df[["hs4", "eff_num_exp"]].drop_duplicates("hs4")


@st.cache_data(show_spinner=False)
def load_hs4_opportunity_metrics(valid_hs4: Iterable[str], year: int = 2024) -> pd.DataFrame:
    valid_hs4_set = set(str(x).zfill(4) for x in valid_hs4)
    primary_path = intermediate_dir() / V2_METRICS_FILE
    required_cols = {
        "hs4",
        "accessible_market_size",
        "accessible_market_growth_5y",
        "dai_percentile",
    }

    if primary_path.exists():
        m = pd.read_csv(primary_path)
        if "hs4" in m.columns:
            m = m.loc[:, ~m.columns.duplicated()].copy()
            m["hs4"] = m["hs4"].astype(str).str.zfill(4)
            m = m[m["hs4"].isin(valid_hs4_set)].copy()
            m = ensure_required_opportunity_metrics(m)
            if not m.empty and required_cols.issubset(set(m.columns)):
                growth_signal = pd.to_numeric(m["accessible_market_growth_5y"], errors="coerce").fillna(0.0).abs().sum()
                if growth_signal > 0:
                    return m

    raise FileNotFoundError(
        f"Missing generated opportunity metrics for {FOCUS_ISO} {year}. "
        f"Run code/data_processing.ipynb to create {primary_path.name}."
    )


@st.cache_data(show_spinner=False)
def load_cordoba_context_annotations(valid_hs4: Iterable[str]) -> pd.DataFrame:
    """Expand Cordoba export RCA annotations to HS4 without additive allocation."""
    exports_path = input_dir() / CORDOBA_EXPORTS_FILE
    crosswalk_path = input_dir() / CORDOBA_RUBRO_CROSSWALK_FILE
    valid = (
        pd.Series(list(valid_hs4), dtype="string")
        .dropna()
        .astype(str)
        .str.zfill(4)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not valid:
        return pd.DataFrame(columns=["hs4", "rca_cordoba_exports"])
    if not exports_path.exists() or not crosswalk_path.exists():
        return pd.DataFrame({"hs4": valid, "rca_cordoba_exports": np.nan})

    exports = pd.read_csv(exports_path, dtype=str, encoding="utf-8-sig")
    exports.columns = exports.columns.str.strip()
    required_export_cols = {"CCOD_RUBRO", "rca_cordoba_exports"}
    if not required_export_cols.issubset(exports.columns):
        return pd.DataFrame({"hs4": valid, "rca_cordoba_exports": np.nan})
    exports["CCOD_RUBRO"] = exports["CCOD_RUBRO"].astype(str).str.strip()
    exports["rca_cordoba_exports"] = pd.to_numeric(exports["rca_cordoba_exports"], errors="coerce")
    source_cols = ["CCOD_RUBRO", "rca_cordoba_exports"]
    if "DESCRIP_RUBRO" in exports.columns:
        source_cols.append("DESCRIP_RUBRO")
    exports = exports[source_cols].drop_duplicates("CCOD_RUBRO")

    crosswalk = pd.read_csv(crosswalk_path, dtype=str)
    crosswalk.columns = crosswalk.columns.str.strip()
    if not {"CCOD_RUBRO", "hs_prefixes", "mapping_priority"}.issubset(crosswalk.columns):
        return pd.DataFrame({"hs4": valid, "rca_cordoba_exports": np.nan})
    crosswalk["CCOD_RUBRO"] = crosswalk["CCOD_RUBRO"].astype(str).str.strip()
    crosswalk["mapping_priority"] = pd.to_numeric(crosswalk["mapping_priority"], errors="coerce").fillna(0)
    valid_df = pd.DataFrame({"hs4": valid})

    expanded_rows: list[dict[str, object]] = []
    for row in crosswalk.itertuples(index=False):
        prefixes_raw = getattr(row, "hs_prefixes", "")
        if pd.isna(prefixes_raw) or not str(prefixes_raw).strip():
            continue
        prefixes: list[str] = []
        for token in str(prefixes_raw).split(";"):
            prefixes.extend(_expand_hs_prefix_token(token))
        for prefix in prefixes:
            matches = valid_df[valid_df["hs4"].str.startswith(prefix)]["hs4"]
            for hs4 in matches:
                expanded_rows.append(
                    {
                        "hs4": hs4,
                        "CCOD_RUBRO": getattr(row, "CCOD_RUBRO"),
                        "cordoba_mapping_priority": float(getattr(row, "mapping_priority")),
                        "cordoba_mapping_level": len(prefix),
                        "cordoba_mapping_note": getattr(row, "mapping_note", ""),
                    }
                )

    if not expanded_rows:
        return pd.DataFrame({"hs4": valid, "rca_cordoba_exports": np.nan})

    expanded = pd.DataFrame(expanded_rows).merge(exports, on="CCOD_RUBRO", how="left")
    expanded["rca_cordoba_exports_sort"] = pd.to_numeric(expanded["rca_cordoba_exports"], errors="coerce").fillna(-1)
    expanded = expanded.sort_values(
        ["hs4", "cordoba_mapping_priority", "cordoba_mapping_level", "rca_cordoba_exports_sort"],
        ascending=[True, False, False, False],
    ).drop_duplicates("hs4", keep="first")
    return expanded[
        [
            "hs4",
            "rca_cordoba_exports",
            "CCOD_RUBRO",
            "DESCRIP_RUBRO" if "DESCRIP_RUBRO" in expanded.columns else "CCOD_RUBRO",
            "cordoba_mapping_priority",
            "cordoba_mapping_level",
            "cordoba_mapping_note",
        ]
    ].rename(
        columns={
            "CCOD_RUBRO": "cordoba_rubro_code",
            "DESCRIP_RUBRO": "cordoba_rubro_description",
        }
    )


@st.cache_data(show_spinner=True)
def load_opportunity_dataset() -> pd.DataFrame:
    hs_ref = load_hs92_reference()
    valid_hs4 = hs_ref["hs4"].tolist()

    complexity_path = _resolve_intermediate_csv("complexity_arg_2024.csv", "complexity_calculations.csv")
    c = pd.read_csv(complexity_path)
    c["hs4"] = c["product"].astype(str).str.zfill(4)
    if "location" in c.columns:
        c = c[c["location"] == FOCUS_ISO]
    c = c[c["time"] == 2024]
    c["raw_rca"] = pd.to_numeric(c.get("rca", 0), errors="coerce").fillna(0.0)
    c["density_percentile"] = pd.to_numeric(c.get("density_percentile", 0), errors="coerce").fillna(0.0)
    if "rca_transformation" in c.columns:
        c["rca_transformed"] = pd.to_numeric(c["rca_transformation"], errors="coerce").fillna(0.0)
    elif "rca_transformed" in c.columns:
        c["rca_transformed"] = pd.to_numeric(c["rca_transformed"], errors="coerce").fillna(0.0)
    else:
        c["rca_transformed"] = c["raw_rca"]
    c = c[["hs4", "raw_rca", "rca_transformed", "pci", "cog", "density", "density_percentile"]].drop_duplicates("hs4")
    metrics = load_hs4_opportunity_metrics(valid_hs4, year=2024)

    df = c.merge(hs_ref, on="hs4", how="left")
    df = df.merge(metrics, on="hs4", how="left")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = ensure_required_opportunity_metrics(df)
    # Prefer RCA recomputed from trade shares for filtering/display as "raw RCA".
    if "raw_rca_trade" in df.columns:
        df["raw_rca"] = pd.to_numeric(df["raw_rca_trade"], errors="coerce").fillna(0.0)
    df = df.fillna(0)
    if "country_exporter_rank" in df.columns:
        df["country_exporter_rank"] = pd.to_numeric(df["country_exporter_rank"], errors="coerce")
        df.loc[df["country_exporter_rank"] <= 0, "country_exporter_rank"] = np.nan

    numeric_cols = [
        "rca_transformed",
        "raw_rca",
        "pci",
        "cog",
        "density",
        "density_percentile",
        "eff_num_exp",
        "distance_travelled",
        "dai_index",
        "dai_percentile",
        "dai_lead",
        "market_growth_5y",
        "market_size_share",
        "accessible_market_size_share",
        "accessible_market_size",
        "accessible_market_growth_5y",
        "accessible_market_to_market_ratio",
        "market_size",
        "total_trade",
        "country_current_exports",
        "country_export_growth_5y",
        "country_exporter_rank",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Keep min-max normalized values for views that need bounded scales (e.g., sizes/ranking views).
    for col in ["raw_rca", "rca_transformed", "density", "eff_num_exp", "distance_travelled", "dai_percentile", "pci", "cog", "market_growth_5y", "accessible_market_growth_5y", "market_size_share", "accessible_market_size_share"]:
        if col not in df.columns:
            df[col] = 0.0
        df[f"{col}_norm"] = normalize_0_1(df[col])

    # Use z-score normalization for feasibility/attractiveness index construction.
    for col in ["raw_rca", "rca_transformed", "density", "eff_num_exp", "distance_travelled", "dai_percentile", "pci", "cog", "market_growth_5y", "accessible_market_growth_5y", "market_size_share", "accessible_market_size_share"]:
        if col not in df.columns:
            df[col] = 0.0
        df[f"{col}_z"] = normalize_zscore(df[col])

    df["feasibility_index"] = df[
        ["rca_transformed_z", "density_z", "eff_num_exp_z", "dai_percentile_z"]
    ].mean(axis=1)
    df["attractiveness_index"] = df[
        ["pci_z", "cog_z", "accessible_market_growth_5y_z", "accessible_market_size_share_z"]
    ].mean(axis=1)
    df["combined_score"] = (df["feasibility_index"] + df["attractiveness_index"]) / 2
    df["accessible_market_size_b"] = pd.to_numeric(df["accessible_market_size"], errors="coerce").fillna(0.0) / 1_000_000_000
    df["market_size_b"] = pd.to_numeric(df["market_size"], errors="coerce").fillna(0.0) / 1_000_000_000
    df["total_trade_b"] = df["total_trade"] / 1_000_000_000
    df["country_current_exports_b"] = df["country_current_exports"] / 1_000_000_000
    cordoba = load_cordoba_context_annotations(valid_hs4)
    if not cordoba.empty:
        df = df.merge(cordoba, on="hs4", how="left")
    return df
