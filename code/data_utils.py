from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

FOCUS_ISO = "ARG"
V1_METRICS_FILE = "opportunity_metrics_hs4_arg.csv"
V1_METRICS_FALLBACK = "v1_metrics_arg.csv"

COUNTRY_METRIC_RENAME = {
    "ecu_total_trade": "country_current_exports",
    "ecu_export_growth_5y": "country_export_growth_5y",
    "ecu_exporter_rank": "country_exporter_rank",
    "ecu_market_share_2020": "country_market_share_2020",
    "ecu_market_share_2024": "country_market_share_2024",
}


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
def load_hs92_level6_reference() -> pd.DataFrame:
    path = input_dir() / "product_hs92.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["product_level"] = pd.to_numeric(df["product_level"], errors="coerce")
    df = df[df["product_level"] == 6].copy()
    if df.empty:
        return pd.DataFrame(columns=["hs6", "product_name"])

    code = (
        df["product_hs92_code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
        .str[:6]
    )
    df["hs6"] = code
    return df[["hs6", "product_name"]].drop_duplicates("hs6")


@st.cache_data(show_spinner=False)
def _load_gdp_ppp_weights(year: int = 2024) -> pd.Series:
    path = input_dir() / "weights_gdp_ppp.csv"
    ycol = str(int(year))
    df = pd.read_csv(path)
    if ycol not in df.columns:
        return pd.Series(dtype="float64")
    df["importer"] = df["COUNTRY.ID"].astype(str).str.upper().str.strip().str[:3]
    df = df[df["importer"].str.len() == 3].copy()
    df["gdp_raw"] = pd.to_numeric(df[ycol], errors="coerce").fillna(0.0)
    out = df.groupby("importer")["gdp_raw"].sum()
    total = float(out.sum())
    if total <= 0:
        return pd.Series(dtype="float64")
    return (out / total).rename("gdp_weight")


@st.cache_data(show_spinner=True)
def compute_network_alignment_indices_hs4(valid_hs4: Iterable[str], year: int = 2024) -> pd.DataFrame:
    valid_hs4 = set(valid_hs4)
    allowed_countries = load_rankings_countries(2024)
    trade_path = input_dir() / "hs92_country_country_product_year_6_2020_2024.csv"
    usecols = [
        "country_iso3_code",
        "partner_iso3_code",
        "product_hs92_code",
        "year",
        "export_value",
    ]

    # X_{z->y} (bilateral exports), X_{z,i} (exports by product), and M_{i,y} (partner imports by product)
    xzy_acc = pd.Series(dtype="float64")  # index: (exporter, importer)
    xzi_acc = pd.Series(dtype="float64")  # index: (exporter, product_code)
    miy_acc = pd.Series(dtype="float64")  # index: (importer, product_code)

    for chunk in pd.read_csv(trade_path, usecols=usecols, chunksize=1_000_000, low_memory=False):
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[chunk["year"] == int(year)]
        if chunk.empty:
            continue

        chunk["product_code"] = chunk["product_hs92_code"].astype(str).str[:4].str.zfill(4)
        chunk = chunk[chunk["product_code"].isin(valid_hs4)]
        if chunk.empty:
            continue

        chunk["value"] = pd.to_numeric(chunk["export_value"], errors="coerce").fillna(0.0)
        chunk["exporter_iso"] = chunk["country_iso3_code"].astype(str).str.upper().str.strip()
        chunk["importer"] = chunk["partner_iso3_code"].astype(str).str.upper().str.strip()
        if allowed_countries:
            chunk = chunk[
                chunk["exporter_iso"].isin(allowed_countries) & chunk["importer"].isin(allowed_countries)
            ]
            if chunk.empty:
                continue

        xzy_grp = chunk.groupby(["exporter_iso", "importer"])["value"].sum()
        xzy_acc = xzy_grp.copy() if xzy_acc.empty else xzy_acc.add(xzy_grp, fill_value=0)

        xzi_grp = chunk.groupby(["exporter_iso", "product_code"])["value"].sum()
        xzi_acc = xzi_grp.copy() if xzi_acc.empty else xzi_acc.add(xzi_grp, fill_value=0)

        miy_grp = chunk.groupby(["importer", "product_code"])["value"].sum()
        miy_acc = miy_grp.copy() if miy_acc.empty else miy_acc.add(miy_grp, fill_value=0)

    exporters = sorted(allowed_countries) if allowed_countries else []
    if "ARG" not in exporters:
        exporters = sorted(set(exporters).union({"ARG"}))
    products = sorted(valid_hs4)
    if xzy_acc.empty or miy_acc.empty or not exporters or not products:
        base = pd.MultiIndex.from_product(
            [[int(year)], products, exporters],
            names=["year", "product_code", "exporter_iso"],
        ).to_frame(index=False)
        base["unweighted_index"] = 0.0
        base["weighted_index"] = 0.0
        base["unweighted_percentile"] = 0.0
        base["weighted_percentile"] = 0.0
        return base[
            [
                "year",
                "product_code",
                "exporter_iso",
                "unweighted_index",
                "unweighted_percentile",
                "weighted_index",
                "weighted_percentile",
            ]
        ]

    xzy = xzy_acc.rename("X_zy").reset_index()
    xzy.columns = ["exporter_iso", "importer", "X_zy"]
    M_y = xzy.groupby("importer")["X_zy"].sum().rename("M_y")
    X_z = xzy.groupby("exporter_iso")["X_zy"].sum().rename("X_z")
    world_total = float(M_y.sum())

    pipe = xzy.merge(M_y.reset_index(), on="importer", how="left").merge(
        X_z.reset_index(), on="exporter_iso", how="left"
    )
    pipe["local_ms"] = np.where(pipe["M_y"] > 0, pipe["X_zy"] / pipe["M_y"], 0.0)
    pipe["world_ms"] = np.where(world_total > 0, pipe["X_z"] / world_total, 0.0)
    pipe["tii"] = np.where(pipe["world_ms"] > 0, pipe["local_ms"] / pipe["world_ms"], 0.0)
    pipe["tii"] = pipe["tii"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    miy = miy_acc.rename("M_iy").reset_index()
    miy.columns = ["importer", "product_code", "M_iy"]
    pbr = miy.merge(M_y.reset_index(), on="importer", how="left")
    pbr["PBR"] = np.where(pbr["M_y"] > 0, pbr["M_iy"] / pbr["M_y"], 0.0)

    partners = sorted(set(pipe["importer"]).union(set(pbr["importer"])))
    tii_mat = (
        pipe.pivot(index="exporter_iso", columns="importer", values="tii")
        .reindex(index=exporters, columns=partners, fill_value=0.0)
        .fillna(0.0)
    )
    pbr_mat = (
        pbr.pivot(index="importer", columns="product_code", values="PBR")
        .reindex(index=partners, columns=products, fill_value=0.0)
        .fillna(0.0)
    )

    unweighted_arr = tii_mat.to_numpy(dtype=float) @ pbr_mat.to_numpy(dtype=float)

    gdp_w = _load_gdp_ppp_weights(year).reindex(partners).fillna(0.0)
    pbr_weighted_mat = pbr_mat.mul(gdp_w, axis=0)
    weighted_arr = tii_mat.to_numpy(dtype=float) @ pbr_weighted_mat.to_numpy(dtype=float)

    unweighted_df = (
        pd.DataFrame(unweighted_arr, index=exporters, columns=products)
        .stack(future_stack=True)
        .rename("unweighted_index")
        .reset_index()
        .rename(columns={"level_0": "exporter_iso", "level_1": "product_code"})
    )
    weighted_df = (
        pd.DataFrame(weighted_arr, index=exporters, columns=products)
        .stack(future_stack=True)
        .rename("weighted_index")
        .reset_index()
        .rename(columns={"level_0": "exporter_iso", "level_1": "product_code"})
    )

    out = unweighted_df.merge(weighted_df, on=["exporter_iso", "product_code"], how="left")
    out["year"] = int(year)

    if not xzi_acc.empty:
        xzi = xzi_acc.rename("X_zi").reset_index()
        xzi.columns = ["exporter_iso", "product_code", "X_zi"]
        top30 = (
            xzi.sort_values(["product_code", "X_zi", "exporter_iso"], ascending=[True, False, True])
            .groupby("product_code", sort=False)
            .head(30)[["product_code", "exporter_iso"]]
        )
    else:
        top30 = pd.DataFrame(columns=["product_code", "exporter_iso"])

    ecu_rows = pd.DataFrame({"product_code": products, "exporter_iso": "ARG"})
    comparison_set = pd.concat([top30, ecu_rows], ignore_index=True).drop_duplicates()
    out = out.merge(comparison_set, on=["product_code", "exporter_iso"], how="inner")

    out["unweighted_percentile"] = (
        out.groupby("product_code")["unweighted_index"].rank(method="average", pct=True) * 100
    )
    out["weighted_percentile"] = (
        out.groupby("product_code")["weighted_index"].rank(method="average", pct=True) * 100
    )
    return out[
        [
            "year",
            "product_code",
            "exporter_iso",
            "unweighted_index",
            "unweighted_percentile",
            "weighted_index",
            "weighted_percentile",
        ]
    ]


@st.cache_data(show_spinner=False)
def compute_alignment_leads_hs4(valid_hs4: Iterable[str], year: int = 2024) -> pd.DataFrame:
    valid_hs4 = set(valid_hs4)
    allowed_countries = load_rankings_countries(2024)
    trade_path = input_dir() / "hs92_country_country_product_year_6_2020_2024.csv"
    usecols = [
        "country_iso3_code",
        "partner_iso3_code",
        "product_hs92_code",
        "year",
        "export_value",
    ]

    xzi_acc = pd.Series(dtype="float64")  # index: (exporter_iso, hs4)
    for chunk in pd.read_csv(trade_path, usecols=usecols, chunksize=1_000_000, low_memory=False):
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[chunk["year"] == int(year)]
        if chunk.empty:
            continue
        chunk["hs4"] = chunk["product_hs92_code"].astype(str).str[:4].str.zfill(4)
        chunk = chunk[chunk["hs4"].isin(valid_hs4)]
        if chunk.empty:
            continue
        chunk["exporter_iso"] = chunk["country_iso3_code"].astype(str).str.upper().str.strip()
        chunk["importer_iso"] = chunk["partner_iso3_code"].astype(str).str.upper().str.strip()
        if allowed_countries:
            chunk = chunk[
                chunk["exporter_iso"].isin(allowed_countries) & chunk["importer_iso"].isin(allowed_countries)
            ]
            if chunk.empty:
                continue
        chunk["value"] = pd.to_numeric(chunk["export_value"], errors="coerce").fillna(0.0)
        grp = chunk.groupby(["exporter_iso", "hs4"])["value"].sum()
        xzi_acc = grp.copy() if xzi_acc.empty else xzi_acc.add(grp, fill_value=0.0)

    base = pd.DataFrame({"hs4": sorted(valid_hs4)})
    if xzi_acc.empty:
        base["alignment_lead_unweighted"] = 0.0
        base["alignment_lead_weighted"] = 0.0
        return base

    exports = xzi_acc.rename("exports_2024").reset_index()
    exports.columns = ["exporter_iso", "hs4", "exports_2024"]

    top5 = (
        exports[exports["exporter_iso"] != "ARG"]
        .sort_values(["hs4", "exports_2024", "exporter_iso"], ascending=[True, False, True])
        .groupby("hs4", sort=False)
        .head(5)[["hs4", "exporter_iso"]]
    )

    align = compute_network_alignment_indices_hs4(valid_hs4, year=year).rename(columns={"product_code": "hs4"})[
        ["hs4", "exporter_iso", "unweighted_percentile", "weighted_percentile"]
    ]

    comp = top5.merge(align, on=["hs4", "exporter_iso"], how="left")
    comp_med = comp.groupby("hs4", as_index=False).agg(
        competitor_median_unweighted=("unweighted_percentile", "median"),
        competitor_median_weighted=("weighted_percentile", "median"),
    )

    ecu = align[align["exporter_iso"] == "ARG"][
        ["hs4", "unweighted_percentile", "weighted_percentile"]
    ].rename(
        columns={
            "unweighted_percentile": "ecu_unweighted_percentile",
            "weighted_percentile": "ecu_weighted_percentile",
        }
    )

    out = base.merge(ecu, on="hs4", how="left").merge(comp_med, on="hs4", how="left")
    out["alignment_lead_unweighted"] = (
        pd.to_numeric(out["ecu_unweighted_percentile"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["competitor_median_unweighted"], errors="coerce").fillna(0.0)
    )
    out["alignment_lead_weighted"] = (
        pd.to_numeric(out["ecu_weighted_percentile"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["competitor_median_weighted"], errors="coerce").fillna(0.0)
    )
    return out[["hs4", "alignment_lead_unweighted", "alignment_lead_weighted"]]


@st.cache_data(show_spinner=True)
def compute_trade_metrics(valid_hs4: Iterable[str]) -> pd.DataFrame:
    valid_hs4 = set(valid_hs4)
    allowed_countries = load_rankings_countries(2024)
    trade_path = input_dir() / "hs92_country_country_product_year_6_2020_2024.csv"
    distance_path = intermediate_dir() / "argentina_distance.csv"
    potential_path = intermediate_dir() / "potential_market_by_product.csv"
    potential_growth_path = intermediate_dir() / "potential_market_growth_by_product.csv"
    potential_growth_yearly_path = intermediate_dir() / "potential_market_by_product_year.csv"

    world_acc = pd.Series(dtype="float64")  # index: (year, hs4)
    ecu_year_acc = pd.Series(dtype="float64")  # index: (year, hs4)
    imports_2024_acc = pd.Series(dtype="float64")  # index: (importer, hs4)
    exp_hs4_2024_acc = pd.Series(dtype="float64")  # index: (exporter, hs4)

    # Potential market size and its 2020-2024 CAGR by product for Argentina (HS4).
    # Accept both pre-aggregated growth file and yearly file for robustness across deployments.
    if potential_growth_path.exists():
        pm = pd.read_csv(potential_growth_path)
        pm["iso3_d"] = pm["iso3_d"].astype(str).str.upper().str.strip()
        pm = pm[pm["iso3_d"] == FOCUS_ISO].copy()
        pm["hs4"] = (
            pm["hs92"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(4)
            .str[-4:]
        )
        pm["potential_market_size_2020"] = pd.to_numeric(pm.get("potential_market_size_2020", 0), errors="coerce").fillna(0.0)
        pm["potential_market_size_2024"] = pd.to_numeric(pm.get("potential_market_size_2024", 0), errors="coerce").fillna(0.0)
        pm = pm.groupby("hs4", as_index=False)[["potential_market_size_2020", "potential_market_size_2024"]].sum()
        pm["potential_market_size"] = pm["potential_market_size_2024"]
        pm["potential_market_growth_5y"] = np.where(
            (pm["potential_market_size_2020"] > 0) & (pm["potential_market_size_2024"] > 0),
            (pm["potential_market_size_2024"] / pm["potential_market_size_2020"]) ** (1 / 5) - 1,
            0.0,
        )
        pm = pm[["hs4", "potential_market_size", "potential_market_growth_5y"]]
    elif potential_growth_yearly_path.exists():
        pm = pd.read_csv(potential_growth_yearly_path)
        pm["iso3_d"] = pm["iso3_d"].astype(str).str.upper().str.strip()
        pm = pm[pm["iso3_d"] == FOCUS_ISO].copy()
        pm["hs4"] = (
            pm["hs92"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(4)
            .str[-4:]
        )
        pm["year"] = pd.to_numeric(pm["year"], errors="coerce")
        pm["potential_market_size"] = pd.to_numeric(pm["potential_market_size"], errors="coerce").fillna(0.0)
        pm = pm[pm["year"].isin([2020, 2024])].copy()
        pm = pm.groupby(["hs4", "year"], as_index=False)["potential_market_size"].sum()
        pm = pm.pivot(index="hs4", columns="year", values="potential_market_size").reset_index()
        for y in [2020, 2024]:
            if y not in pm.columns:
                pm[y] = 0.0
        pm["potential_market_size_2020"] = pd.to_numeric(pm[2020], errors="coerce").fillna(0.0)
        pm["potential_market_size_2024"] = pd.to_numeric(pm[2024], errors="coerce").fillna(0.0)
        pm["potential_market_size"] = pm["potential_market_size_2024"]
        pm["potential_market_growth_5y"] = np.where(
            (pm["potential_market_size_2020"] > 0) & (pm["potential_market_size_2024"] > 0),
            (pm["potential_market_size_2024"] / pm["potential_market_size_2020"]) ** (1 / 5) - 1,
            0.0,
        )
        pm = pm[["hs4", "potential_market_size", "potential_market_growth_5y"]]
    elif potential_path.exists():
        pm = pd.read_csv(potential_path)
        pm["iso3_d"] = pm["iso3_d"].astype(str).str.upper().str.strip()
        pm = pm[pm["iso3_d"] == FOCUS_ISO].copy()
        pm["hs4"] = (
            pm["hs92"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(4)
            .str[-4:]
        )
        pm["potential_market_size"] = pd.to_numeric(pm["potential_market_imports_sum"], errors="coerce").fillna(0.0)
        pm = pm.groupby("hs4", as_index=False)["potential_market_size"].sum()
        pm["potential_market_growth_5y"] = 0.0
    else:
        pm = pd.DataFrame({"hs4": sorted(valid_hs4), "potential_market_size": 0.0, "potential_market_growth_5y": 0.0})

    usecols = [
        "country_iso3_code",
        "partner_iso3_code",
        "product_hs92_code",
        "year",
        "export_value",
    ]

    for chunk in pd.read_csv(trade_path, usecols=usecols, chunksize=1_000_000, low_memory=False):
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[chunk["year"].isin([2020, 2024])]
        if chunk.empty:
            continue

        chunk["year"] = chunk["year"].astype(int)
        chunk["hs4"] = chunk["product_hs92_code"].astype(str).str[:4].str.zfill(4)
        chunk = chunk[chunk["hs4"].isin(valid_hs4)]
        if chunk.empty:
            continue

        chunk["value"] = pd.to_numeric(chunk["export_value"], errors="coerce").fillna(0)
        chunk["exporter"] = chunk["country_iso3_code"].astype(str).str.upper().str.strip()
        chunk["importer"] = chunk["partner_iso3_code"].astype(str).str.upper().str.strip()
        if allowed_countries:
            chunk = chunk[
                chunk["exporter"].isin(allowed_countries) & chunk["importer"].isin(allowed_countries)
            ]
            if chunk.empty:
                continue

        world_grp = chunk.groupby(["year", "hs4"])["value"].sum()
        if world_acc.empty:
            world_acc = world_grp.copy()
        else:
            world_acc = world_acc.add(world_grp, fill_value=0)

        ecu_year_grp = chunk[chunk["exporter"] == "ARG"].groupby(["year", "hs4"])["value"].sum()
        if ecu_year_acc.empty:
            ecu_year_acc = ecu_year_grp.copy()
        else:
            ecu_year_acc = ecu_year_acc.add(ecu_year_grp, fill_value=0)

        c2024 = chunk[chunk["year"] == 2024]
        if c2024.empty:
            continue

        imp_grp = c2024.groupby(["importer", "hs4"])["value"].sum()
        if imports_2024_acc.empty:
            imports_2024_acc = imp_grp.copy()
        else:
            imports_2024_acc = imports_2024_acc.add(imp_grp, fill_value=0)

        exp_grp = c2024.groupby(["exporter", "hs4"])["value"].sum()
        if exp_hs4_2024_acc.empty:
            exp_hs4_2024_acc = exp_grp.copy()
        else:
            exp_hs4_2024_acc = exp_hs4_2024_acc.add(exp_grp, fill_value=0)

    world = world_acc.rename("value").reset_index()
    world.columns = ["year", "hs4", "world_value"]

    world_pivot = world.pivot(index="hs4", columns="year", values="world_value").reset_index()
    if 2020 not in world_pivot.columns:
        world_pivot[2020] = 0
    if 2024 not in world_pivot.columns:
        world_pivot[2024] = 0
    world_pivot["market_growth_5y"] = np.where(
        (world_pivot[2020] > 0) & (world_pivot[2024] > 0),
        (world_pivot[2024] / world_pivot[2020]) ** (1 / 5) - 1,
        0,
    )

    world_2024 = world[world["year"] == 2024][["hs4", "world_value"]].rename(columns={"world_value": "total_trade"})
    world_total_2024 = world_2024["total_trade"].sum()
    world_2024["market_size_share"] = np.where(
        world_total_2024 > 0,
        world_2024["total_trade"] / world_total_2024,
        0,
    )

    if ecu_year_acc.empty:
        ecu_pivot = pd.DataFrame({"hs4": sorted(valid_hs4), 2020: 0.0, 2024: 0.0})
    else:
        ecu_year = ecu_year_acc.rename("ecu_value").reset_index()
        ecu_year.columns = ["year", "hs4", "ecu_value"]
        ecu_pivot = ecu_year.pivot(index="hs4", columns="year", values="ecu_value").reset_index()
        if 2020 not in ecu_pivot.columns:
            ecu_pivot[2020] = 0.0
        if 2024 not in ecu_pivot.columns:
            ecu_pivot[2024] = 0.0

    ecu_pivot["ecu_export_growth_5y"] = np.where(
        (ecu_pivot[2020] > 0) & (ecu_pivot[2024] > 0),
        (ecu_pivot[2024] / ecu_pivot[2020]) ** (1 / 5) - 1,
        0,
    )
    ecu_2024 = ecu_pivot[["hs4", 2024, "ecu_export_growth_5y"]].rename(columns={2024: "ecu_total_trade"})

    # Argentina market share by product and absolute change (2024 - 2020).
    share_df = world_pivot[["hs4", 2020, 2024]].rename(columns={2020: "world_2020", 2024: "world_2024"})
    share_df = share_df.merge(
        ecu_pivot[["hs4", 2020, 2024]].rename(columns={2020: "ecu_2020", 2024: "ecu_2024"}),
        on="hs4",
        how="left",
    ).fillna(0)
    share_df["ecu_market_share_2020"] = np.where(share_df["world_2020"] > 0, share_df["ecu_2020"] / share_df["world_2020"], 0)
    share_df["ecu_market_share_2024"] = np.where(share_df["world_2024"] > 0, share_df["ecu_2024"] / share_df["world_2024"], 0)
    share_df["market_share_change_abs"] = share_df["ecu_market_share_2024"] - share_df["ecu_market_share_2020"]
    share_df = share_df[["hs4", "ecu_market_share_2020", "ecu_market_share_2024", "market_share_change_abs"]]

    if distance_path.exists():
        distance = pd.read_csv(distance_path)
    else:
        bilateral = intermediate_dir() / "bilateral_distances.csv"
        if bilateral.exists():
            distance = pd.read_csv(bilateral, usecols=["iso3_o", "iso3_d", "dist"])
            distance = distance[distance["iso3_o"].astype(str).str.upper().str.strip() == "ARG"][
                ["iso3_d", "dist"]
            ]
        else:
            distance = pd.DataFrame(columns=["iso3", "distance"])

    if "iso3" not in distance.columns and "iso3_d" in distance.columns:
        distance = distance.rename(columns={"iso3_d": "iso3"})
    if "distance" not in distance.columns and "dist" in distance.columns:
        distance = distance.rename(columns={"dist": "distance"})
    distance["iso3"] = distance["iso3"].astype(str).str.upper().str.strip()

    imports_2024 = imports_2024_acc.rename("imports_dest").reset_index()
    imports_2024.columns = ["importer", "hs4", "imports_dest"]
    world_trade_2024 = world[world["year"] == 2024][["hs4", "world_value"]].rename(columns={"world_value": "world_trade_hs4"})

    dist_terms = imports_2024.merge(world_trade_2024, on="hs4", how="left")
    dist_terms = dist_terms.merge(distance, left_on="importer", right_on="iso3", how="left")
    dist_terms["distance"] = pd.to_numeric(dist_terms["distance"], errors="coerce").fillna(0)
    dist_terms["distance_term"] = np.where(
        dist_terms["world_trade_hs4"] > 0,
        dist_terms["distance"] * (dist_terms["imports_dest"] / dist_terms["world_trade_hs4"]),
        0,
    )
    distance_travelled = (
        dist_terms.groupby("hs4", as_index=False)["distance_term"]
        .sum()
        .rename(columns={"distance_term": "distance_travelled"})
    )

    # Effective number of exporters (Hill number of order 2 / inverse HHI) under the same 145-country filter.
    if exp_hs4_2024_acc.empty:
        eff_df = pd.DataFrame({"hs4": sorted(valid_hs4), "eff_num_exp": 0.0})
        rank_df = pd.DataFrame({"hs4": sorted(valid_hs4), "ecu_exporter_rank": np.nan})
    else:
        exp_hs4 = exp_hs4_2024_acc.rename("value").reset_index()
        exp_hs4.columns = ["exporter", "hs4", "value"]
        totals = exp_hs4.groupby("hs4")["value"].sum().rename("total")
        exp_hs4 = exp_hs4.merge(totals, on="hs4", how="left")
        exp_hs4["share"] = np.where(exp_hs4["total"] > 0, exp_hs4["value"] / exp_hs4["total"], 0)
        eff_df = (
            exp_hs4.groupby("hs4", as_index=False)["share"]
            .apply(lambda s: float(1 / np.sum(np.square(s))) if np.sum(np.square(s)) > 0 else 0.0)
            .rename(columns={"share": "eff_num_exp"})
        )
        # Argentina's rank among exporters by product in 2024 (1 = largest exporter).
        exp_hs4["exporter_rank"] = exp_hs4.groupby("hs4")["value"].rank(method="min", ascending=False)
        ecu_rank = exp_hs4[exp_hs4["exporter"] == "ARG"][["hs4", "exporter_rank"]].rename(
            columns={"exporter_rank": "ecu_exporter_rank"}
        )
        rank_df = pd.DataFrame({"hs4": sorted(valid_hs4)}).merge(ecu_rank, on="hs4", how="left")

    out = world_pivot[["hs4", "market_growth_5y"]].merge(world_2024, on="hs4", how="outer")
    out = out.merge(ecu_2024, on="hs4", how="left")
    out = out.merge(share_df, on="hs4", how="left")
    out = out.merge(distance_travelled, on="hs4", how="left")
    out = out.merge(eff_df, on="hs4", how="left")
    out = out.merge(rank_df, on="hs4", how="left")
    out = out.merge(pm, on="hs4", how="left")
    out = out.fillna(0)
    # Raw RCA for ARG in 2024 from trade shares:
    # RCA_i = (X_ARG_i / X_ARG_total) / (X_world_i / X_world_total)
    arg_total_2024 = float(pd.to_numeric(out["ecu_total_trade"], errors="coerce").fillna(0.0).sum())
    world_total_2024 = float(pd.to_numeric(out["total_trade"], errors="coerce").fillna(0.0).sum())
    out["raw_rca_trade"] = np.where(
        (arg_total_2024 > 0) & (world_total_2024 > 0) & (out["total_trade"] > 0),
        (pd.to_numeric(out["ecu_total_trade"], errors="coerce").fillna(0.0) / arg_total_2024)
        / (pd.to_numeric(out["total_trade"], errors="coerce").fillna(0.0) / world_total_2024),
        0.0,
    )
    out["raw_rca_trade"] = (
        pd.to_numeric(out["raw_rca_trade"], errors="coerce")
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    out["market_size"] = pd.to_numeric(out["total_trade"], errors="coerce").fillna(0.0)
    total_potential_size = float(pd.to_numeric(out["potential_market_size"], errors="coerce").fillna(0.0).sum())
    out["potential_market_size_share"] = np.where(
        total_potential_size > 0,
        pd.to_numeric(out["potential_market_size"], errors="coerce").fillna(0.0) / total_potential_size,
        0.0,
    )
    out["potential_market_to_market_ratio"] = np.where(
        out["market_size"] > 0,
        pd.to_numeric(out["potential_market_size"], errors="coerce").fillna(0.0) / out["market_size"],
        0.0,
    )
    out["ecu_exporter_rank"] = pd.to_numeric(out["ecu_exporter_rank"], errors="coerce")
    out.loc[out["ecu_exporter_rank"] <= 0, "ecu_exporter_rank"] = np.nan
    median_cagr = out["market_growth_5y"].median()
    median_potential_market_growth = out["potential_market_growth_5y"].median()
    median_ecu_export_cagr = out["ecu_export_growth_5y"].median()
    out["above_median_cagr"] = out["market_growth_5y"] > median_cagr
    out["above_median_potential_market_growth"] = out["potential_market_growth_5y"] > median_potential_market_growth
    out["above_median_export_cagr"] = out["ecu_export_growth_5y"] > median_ecu_export_cagr
    return out


@st.cache_data(show_spinner=False)
def load_product_market_deep_dive(hs4: str, focus_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allowed_countries = load_rankings_countries(2024)
    trade_path = input_dir() / "hs92_country_country_product_year_6_2020_2024.csv"
    usecols = [
        "country_iso3_code",
        "partner_iso3_code",
        "product_hs92_code",
        "year",
        "export_value",
    ]

    hs4 = str(hs4).zfill(4)
    world_by_year = pd.Series(dtype="float64")  # index: year
    ecu_by_year = pd.Series(dtype="float64")  # index: year
    dest_focus = pd.Series(dtype="float64")  # index: importer
    hs6_year_acc = pd.Series(dtype="float64")  # index: (hs6, year), ARG exports

    for chunk in pd.read_csv(trade_path, usecols=usecols, chunksize=1_000_000, low_memory=False):
        chunk["hs4"] = chunk["product_hs92_code"].astype(str).str[:4].str.zfill(4)
        chunk = chunk[chunk["hs4"] == hs4]
        if chunk.empty:
            continue

        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[chunk["year"].notna()]
        if chunk.empty:
            continue
        chunk["year"] = chunk["year"].astype(int)
        chunk = chunk[chunk["year"].between(2020, 2024)]
        if chunk.empty:
            continue
        chunk["value"] = pd.to_numeric(chunk["export_value"], errors="coerce").fillna(0)
        chunk["exporter"] = chunk["country_iso3_code"].astype(str).str.upper().str.strip()
        chunk["importer"] = chunk["partner_iso3_code"].astype(str).str.upper().str.strip()
        chunk["hs6"] = chunk["product_hs92_code"].astype(str).str.zfill(6).str[:6]
        if allowed_countries:
            chunk = chunk[
                chunk["exporter"].isin(allowed_countries) & chunk["importer"].isin(allowed_countries)
            ]
            if chunk.empty:
                continue

        w = chunk.groupby("year")["value"].sum()
        world_by_year = w.copy() if world_by_year.empty else world_by_year.add(w, fill_value=0)

        ecu = chunk[chunk["exporter"] == "ARG"].groupby("year")["value"].sum()
        ecu_by_year = ecu.copy() if ecu_by_year.empty else ecu_by_year.add(ecu, fill_value=0)

        ecu_hs6 = chunk[chunk["exporter"] == "ARG"].groupby(["hs6", "year"])["value"].sum()
        hs6_year_acc = ecu_hs6.copy() if hs6_year_acc.empty else hs6_year_acc.add(ecu_hs6, fill_value=0)

        focus = chunk[(chunk["year"] == focus_year) & (chunk["exporter"] == "ARG")].groupby("importer")["value"].sum()
        dest_focus = focus.copy() if dest_focus.empty else dest_focus.add(focus, fill_value=0)

    if world_by_year.empty:
        share_df = pd.DataFrame(columns=["year", "world_value", "ecu_value", "ecu_market_share"])
    else:
        share_df = world_by_year.rename("world_value").reset_index()
        share_df.columns = ["year", "world_value"]
        ecu_df = ecu_by_year.rename("ecu_value").reset_index()
        ecu_df.columns = ["year", "ecu_value"]
        share_df = share_df.merge(ecu_df, on="year", how="left").fillna(0)
        share_df["ecu_market_share"] = np.where(
            share_df["world_value"] > 0,
            share_df["ecu_value"] / share_df["world_value"],
            0,
        )
        share_df = share_df.sort_values("year").reset_index(drop=True)

    if dest_focus.empty:
        dest_df = pd.DataFrame(columns=["importer", "value", "share"])
    else:
        dest_df = dest_focus.rename("value").reset_index()
        dest_df.columns = ["importer", "value"]
        total = dest_df["value"].sum()
        dest_df["share"] = np.where(total > 0, dest_df["value"] / total, 0)
        dest_df = dest_df.sort_values("value", ascending=False).reset_index(drop=True)

    if hs6_year_acc.empty:
        hs6_table = pd.DataFrame(
            columns=["hs6", "product_name", "2020", "2021", "2022", "2023", "2024", "total_2020_2024", "cagr_5y"]
        )
    else:
        hs6_long = hs6_year_acc.rename("value").reset_index()
        hs6_long.columns = ["hs6", "year", "value"]
        hs6_table = hs6_long.pivot(index="hs6", columns="year", values="value").reset_index()
        for y in [2020, 2021, 2022, 2023, 2024]:
            if y not in hs6_table.columns:
                hs6_table[y] = 0.0
        hs6_table = hs6_table[["hs6", 2020, 2021, 2022, 2023, 2024]].fillna(0.0)
        hs6_table["total_2020_2024"] = hs6_table[[2020, 2021, 2022, 2023, 2024]].sum(axis=1)
        hs6_table["cagr_5y"] = np.where(
            (hs6_table[2020] > 0) & (hs6_table[2024] > 0),
            (hs6_table[2024] / hs6_table[2020]) ** (1 / 5) - 1,
            0.0,
        )
        hs6_table = hs6_table.sort_values("total_2020_2024", ascending=False).reset_index(drop=True)
        hs6_table = hs6_table.rename(columns={2020: "2020", 2021: "2021", 2022: "2022", 2023: "2023", 2024: "2024"})
        hs6_ref = load_hs92_level6_reference()
        hs6_table = hs6_table.merge(hs6_ref, on="hs6", how="left")
        hs6_table["product_name"] = hs6_table["product_name"].fillna("")
        hs6_table = hs6_table[
            ["hs6", "product_name", "2020", "2021", "2022", "2023", "2024", "total_2020_2024", "cagr_5y"]
        ]

    return dest_df, share_df, hs6_table


@st.cache_data(show_spinner=False)
def load_eff_num_exp() -> pd.DataFrame:
    path = _resolve_intermediate_csv("hs92_attributes.csv")
    df = pd.read_csv(path, usecols=["hs92", "eff_num_exp"])
    df["hs4"] = df["hs92"].astype(str).str.zfill(4)
    return df[["hs4", "eff_num_exp"]].drop_duplicates("hs4")


@st.cache_data(show_spinner=False)
def load_or_build_v1_hs4_metrics(valid_hs4: Iterable[str], year: int = 2024) -> pd.DataFrame:
    valid_hs4_set = set(str(x).zfill(4) for x in valid_hs4)
    primary_path = intermediate_dir() / V1_METRICS_FILE
    fallback_path = intermediate_dir() / V1_METRICS_FALLBACK
    required_cols = {"potential_market_growth_5y"}

    for p in [primary_path, fallback_path]:
        if not p.exists():
            continue
        m = pd.read_csv(p)
        if "hs4" not in m.columns:
            continue
        safe_rename = {
            k: v for k, v in COUNTRY_METRIC_RENAME.items()
            if (k in m.columns) and (v not in m.columns)
        }
        m = m.rename(columns=safe_rename)
        m = m.loc[:, ~m.columns.duplicated()].copy()
        m["hs4"] = m["hs4"].astype(str).str.zfill(4)
        m = m[m["hs4"].isin(valid_hs4_set)].copy()
        if not m.empty and required_cols.issubset(set(m.columns)):
            growth_signal = pd.to_numeric(m["potential_market_growth_5y"], errors="coerce").fillna(0.0).abs().sum()
            if growth_signal > 0:
                return m

    align = compute_network_alignment_indices_hs4(valid_hs4_set, year=year)
    align = align[align["exporter_iso"] == FOCUS_ISO][
        ["product_code", "unweighted_percentile", "weighted_percentile"]
    ].rename(
        columns={
            "product_code": "hs4",
            "unweighted_percentile": "alignment_unweighted_percentile",
            "weighted_percentile": "alignment_weighted_percentile",
        }
    )
    lead = compute_alignment_leads_hs4(valid_hs4_set, year=year)
    trade_metrics = compute_trade_metrics(valid_hs4_set)
    metrics = trade_metrics.merge(align, on="hs4", how="left").merge(lead, on="hs4", how="left")
    safe_rename = {
        k: v for k, v in COUNTRY_METRIC_RENAME.items()
        if (k in metrics.columns) and (v not in metrics.columns)
    }
    metrics = metrics.rename(columns=safe_rename)
    metrics["hs4"] = metrics["hs4"].astype(str).str.zfill(4)
    metrics = metrics[metrics["hs4"].isin(valid_hs4_set)].copy()
    metrics.to_csv(primary_path, index=False)
    return metrics


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
    metrics = load_or_build_v1_hs4_metrics(valid_hs4, year=2024)

    df = c.merge(hs_ref, on="hs4", how="left")
    df = df.merge(metrics, on="hs4", how="left")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # Prefer RCA recomputed from trade shares for filtering/display as "raw RCA".
    if "raw_rca_trade" in df.columns:
        df["raw_rca"] = pd.to_numeric(df["raw_rca_trade"], errors="coerce").fillna(0.0)
    # Backward compatibility with legacy metrics naming.
    if "country_current_exports" not in df.columns and "ecu_total_trade" in df.columns:
        df["country_current_exports"] = df["ecu_total_trade"]
    if "country_export_growth_5y" not in df.columns and "ecu_export_growth_5y" in df.columns:
        df["country_export_growth_5y"] = df["ecu_export_growth_5y"]
    if "country_exporter_rank" not in df.columns and "ecu_exporter_rank" in df.columns:
        df["country_exporter_rank"] = df["ecu_exporter_rank"]
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
        "alignment_weighted_percentile",
        "market_growth_5y",
        "market_size_share",
        "potential_market_size_share",
        "market_size",
        "potential_market_size",
        "potential_market_growth_5y",
        "potential_market_to_market_ratio",
        "total_trade",
        "country_current_exports",
        "country_export_growth_5y",
        "country_exporter_rank",
        "alignment_lead_unweighted",
        "alignment_lead_weighted",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Keep min-max normalized values for views that need bounded scales (e.g., sizes/ranking views).
    for col in ["raw_rca", "rca_transformed", "density", "eff_num_exp", "distance_travelled", "alignment_weighted_percentile", "pci", "cog", "market_growth_5y", "potential_market_growth_5y", "market_size_share", "potential_market_size_share"]:
        if col not in df.columns:
            df[col] = 0.0
        df[f"{col}_norm"] = normalize_0_1(df[col])

    # Use z-score normalization for feasibility/attractiveness index construction.
    for col in ["raw_rca", "rca_transformed", "density", "eff_num_exp", "distance_travelled", "alignment_weighted_percentile", "pci", "cog", "market_growth_5y", "potential_market_growth_5y", "market_size_share", "potential_market_size_share"]:
        if col not in df.columns:
            df[col] = 0.0
        df[f"{col}_z"] = normalize_zscore(df[col])

    df["feasibility_index"] = df[
        ["rca_transformed_z", "density_z", "eff_num_exp_z", "alignment_weighted_percentile_z"]
    ].mean(axis=1)
    df["attractiveness_index"] = df[
        ["pci_z", "cog_z", "potential_market_growth_5y_z", "potential_market_size_share_z"]
    ].mean(axis=1)
    df["combined_score"] = (df["feasibility_index"] + df["attractiveness_index"]) / 2
    df["potential_market_size_b"] = pd.to_numeric(df["potential_market_size"], errors="coerce").fillna(0.0) / 1_000_000_000
    df["market_size_b"] = pd.to_numeric(df["market_size"], errors="coerce").fillna(0.0) / 1_000_000_000
    df["total_trade_b"] = df["total_trade"] / 1_000_000_000
    df["country_current_exports_b"] = df["country_current_exports"] / 1_000_000_000
    return df
