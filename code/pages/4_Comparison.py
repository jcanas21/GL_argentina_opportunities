import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import load_anchor_proximity_dataset, load_opportunity_dataset, normalize_0_1


EXCLUDED_HS4 = {"2711", "2710", "7108", "2709", "2713", "2701", "2603", "2616"}
FEAS_COLS = ["rca_transformed_z", "density_z", "eff_num_exp_z", "dai_percentile_z"]
ATTR_COLS = ["pci_z", "cog_z", "accessible_market_growth_5y_z", "accessible_market_size_mm"]

PAGE2_PRESETS = {
    "Consolidadas": {
        "trade_min": 1.0,
        "market_growth_gt": 0.0,
        "rca_min": 1.00,
        "rca_max": None,
        "density_range": None,
        "strategic_balance": 0.80,
        "feas_weights": [0.00, 0.00, 0.00, 1.00],
        "attr_weights": [0.35, 0.00, 0.15, 0.15],
        "top_n": 50,
    },
    "Emergentes": {
        "trade_min": 1.0,
        "market_growth_gt": 0.0,
        "rca_min": 0.30,
        "rca_max": 0.99,
        "density_range": None,
        "strategic_balance": 0.30,
        "feas_weights": [0.35, 0.35, 0.00, 0.30],
        "attr_weights": [0.35, 0.35, 0.15, 0.15],
        "top_n": 50,
    },
    "Apuestas Estrategicas": {
        "trade_min": 1.0,
        "market_growth_gt": 0.0,
        "rca_min": 0.00,
        "rca_max": 0.30,
        "density_range": (0.50, None),
        "strategic_balance": 0.70,
        "feas_weights": [0.00, 0.70, 0.00, 0.30],
        "attr_weights": [0.30, 0.35, 0.15, 0.15],
        "top_n": 50,
    },
    "Balanceado": {
        "trade_min": 1.0,
        "market_growth_gt": 0.0,
        "rca_min": 0.00,
        "rca_max": 0.99,
        "density_range": (0.30, 0.40),
        "strategic_balance": 0.50,
        "feas_weights": [0.20, 0.30, 0.00, 0.30],
        "attr_weights": [0.30, 0.35, 0.15, 0.15],
        "top_n": 40,
    },
}

PAGE3_PRESETS = {
    "Top Anchored Candidates": {
        "trade_min": 1.0,
        "accessible_market_growth_gt": 0.0,
        "anchor_density_max": 50.0,
        "proximity_rank_range": (1, 10),
        "exclude_anchor_sections_prefixes": ("1.", "2.", "3."),
        "exclude_candidate_sections_prefixes": ("1.", "2.", "3."),
        "strategic_balance": 0.70,
        "w_dai": 1.0,
        "attr_weights": [0.35, 0.35, 0.15, 0.15],
        "top_n": 40,
    }
}

COMPARE_GROUP_COLORS = {
    "Overlap": "#2f5d74",
    "Only Page 2 preset": "#b07ac9",
    "Only Page 3 preset": "#74c5c6",
}


def weighted_index(frame: pd.DataFrame, cols: list[str], weights: list[float]) -> pd.Series:
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        return frame[cols].mean(axis=1)
    arr = np.column_stack([frame[c].to_numpy() for c in cols])
    w = np.array(weights, dtype=float)
    return pd.Series((arr * w).sum(axis=1) / weight_sum, index=frame.index)


def build_page2_recommendations(profile_name: str) -> pd.DataFrame:
    profile = PAGE2_PRESETS[profile_name]
    df = load_opportunity_dataset().copy()
    if df.empty:
        return df

    df["hs4"] = df["hs4"].astype(str).str.zfill(4)
    df["accessible_market_size_mm"] = normalize_0_1(pd.to_numeric(df["accessible_market_size"], errors="coerce").fillna(0.0))

    df["feasibility_raw"] = weighted_index(df, FEAS_COLS, profile["feas_weights"])
    df["attractiveness_raw"] = weighted_index(df, ATTR_COLS, profile["attr_weights"])
    df["feasibility_index"] = normalize_0_1(df["feasibility_raw"])
    df["attractiveness_index"] = normalize_0_1(df["attractiveness_raw"])
    df["combined_raw"] = (
        (1 - profile["strategic_balance"]) * df["feasibility_index"]
        + profile["strategic_balance"] * df["attractiveness_index"]
    )
    df["combined_score"] = normalize_0_1(df["combined_raw"])

    density_pct_2d = pd.to_numeric(df["density_percentile"], errors="coerce").fillna(0.0).round(2)
    raw_rca = pd.to_numeric(df["raw_rca"], errors="coerce").fillna(0.0)

    flt = df[
        (pd.to_numeric(df["accessible_market_size_b"], errors="coerce") >= profile["trade_min"])
        & (pd.to_numeric(df["market_growth_5y"], errors="coerce") > profile["market_growth_gt"])
        & (raw_rca >= profile["rca_min"])
        & (~df["hs4"].isin(EXCLUDED_HS4))
    ].copy()

    if profile["rca_max"] is not None:
        flt = flt[raw_rca.loc[flt.index] < profile["rca_max"]]

    if profile["density_range"] is not None:
        lo, hi = profile["density_range"]
        density_mask = density_pct_2d.loc[flt.index] >= lo
        if hi is not None:
            density_mask &= density_pct_2d.loc[flt.index] <= hi
        flt = flt[density_mask]

    flt = flt.sort_values(["combined_score", "accessible_market_size_b"], ascending=[False, False]).reset_index(drop=True)
    flt["page2_rank"] = np.arange(1, len(flt) + 1)
    return flt.head(profile["top_n"])[
        [
            "hs4",
            "product_name_short",
            "sector",
            "combined_score",
            "page2_rank",
            "raw_rca",
            "density_percentile",
            "dai_percentile",
            "accessible_market_size_b",
            "accessible_market_growth_5y",
            "pci",
            "cog",
        ]
    ].rename(
        columns={
            "combined_score": "page2_score",
            "sector": "page2_sector",
            "dai_percentile": "page2_dai_pct",
        }
    )


def build_page3_recommendations(profile_name: str) -> pd.DataFrame:
    profile = PAGE3_PRESETS[profile_name]
    df = load_anchor_proximity_dataset().copy()
    if df.empty:
        return df

    df["candidate_hs4"] = df["candidate_hs4"].astype(str).str.zfill(4)
    anchor_sections_ok = ~df["anchor_hs_section_name"].astype(str).str.startswith(profile["exclude_anchor_sections_prefixes"])
    candidate_sections_ok = ~df["candidate_hs_section_name"].astype(str).str.startswith(profile["exclude_candidate_sections_prefixes"])

    flt = df[
        (pd.to_numeric(df["accessible_market_size_b"], errors="coerce") >= profile["trade_min"])
        & (pd.to_numeric(df["accessible_market_growth_5y"], errors="coerce") > profile["accessible_market_growth_gt"])
        & (pd.to_numeric(df["anchor_density_percentile"], errors="coerce") <= profile["anchor_density_max"])
        & (pd.to_numeric(df["proximity_rank"], errors="coerce") >= profile["proximity_rank_range"][0])
        & (pd.to_numeric(df["proximity_rank"], errors="coerce") <= profile["proximity_rank_range"][1])
        & anchor_sections_ok
        & candidate_sections_ok
        & (~df["candidate_hs4"].isin(EXCLUDED_HS4))
    ].copy()

    candidate_scores = (
        flt.groupby(
            [
                "candidate_hs4",
                "candidate_product_name_short",
                "candidate_sector",
                "candidate_hs_section_name",
            ],
            as_index=False,
        )
        .agg(
            accessible_market_size=("accessible_market_size", "first"),
            accessible_market_size_b=("accessible_market_size_b", "first"),
            accessible_market_growth_5y=("accessible_market_growth_5y", "first"),
            dai_percentile=("dai_percentile", "first"),
            pci=("pci", "first"),
            cog=("cog", "first"),
            avg_proximity=("proximity", "mean"),
            anchor_count=("anchor_hs4", "nunique"),
        )
    )

    candidate_scores["dai_mm"] = normalize_0_1(candidate_scores["dai_percentile"])
    candidate_scores["pci_mm"] = normalize_0_1(candidate_scores["pci"])
    candidate_scores["cog_mm"] = normalize_0_1(candidate_scores["cog"])
    candidate_scores["accessible_market_growth_mm"] = normalize_0_1(candidate_scores["accessible_market_growth_5y"])
    candidate_scores["accessible_market_size_mm"] = normalize_0_1(candidate_scores["accessible_market_size"])
    candidate_scores["feasibility_index"] = candidate_scores["dai_mm"] if profile["w_dai"] > 0 else 0.0
    attr = candidate_scores[["pci_mm", "cog_mm", "accessible_market_growth_mm", "accessible_market_size_mm"]]
    attr_weights = np.array(profile["attr_weights"], dtype=float)
    candidate_scores["attractiveness_raw"] = (attr.to_numpy() * attr_weights).sum(axis=1) / float(attr_weights.sum())
    candidate_scores["attractiveness_index"] = normalize_0_1(candidate_scores["attractiveness_raw"])
    candidate_scores["combined_raw"] = (
        (1 - profile["strategic_balance"]) * candidate_scores["feasibility_index"]
        + profile["strategic_balance"] * candidate_scores["attractiveness_index"]
    )
    candidate_scores["combined_score"] = normalize_0_1(candidate_scores["combined_raw"])
    candidate_scores = candidate_scores.sort_values(
        ["combined_score", "accessible_market_size_b"], ascending=[False, False]
    ).reset_index(drop=True)
    candidate_scores["page3_rank"] = np.arange(1, len(candidate_scores) + 1)

    return candidate_scores.head(profile["top_n"])[
        [
            "candidate_hs4",
            "candidate_product_name_short",
            "candidate_sector",
            "candidate_hs_section_name",
            "combined_score",
            "page3_rank",
            "dai_percentile",
            "accessible_market_size_b",
            "accessible_market_growth_5y",
            "pci",
            "cog",
            "avg_proximity",
            "anchor_count",
        ]
    ].rename(
        columns={
            "candidate_hs4": "hs4",
            "candidate_product_name_short": "product_name_short",
            "candidate_sector": "page3_sector",
            "dai_percentile": "page3_dai_pct",
            "combined_score": "page3_score",
        }
    )


st.title("Comparison")
st.caption("Compare the recommendations produced by selected presets from Page 2 and Page 3.")

csel1, csel2 = st.columns(2)
page2_preset = csel1.selectbox("Page 2 preset", list(PAGE2_PRESETS.keys()), index=2)
page3_preset = csel2.selectbox("Page 3 preset", list(PAGE3_PRESETS.keys()), index=0)

page2_df = build_page2_recommendations(page2_preset)
page3_df = build_page3_recommendations(page3_preset)

if page2_df.empty and page3_df.empty:
    st.warning("No comparison data available for the selected presets.")
    st.stop()

comparison = page2_df.merge(
    page3_df,
    on="hs4",
    how="outer",
    suffixes=("_page2", "_page3"),
)

comparison["product_name_short"] = comparison["product_name_short_page2"].combine_first(comparison["product_name_short_page3"])
comparison["sector"] = comparison["page2_sector"].combine_first(comparison["page3_sector"]).fillna("Other")
comparison["accessible_market_size_b"] = comparison["accessible_market_size_b_page2"].combine_first(
    comparison["accessible_market_size_b_page3"]
)
comparison["accessible_market_growth_5y"] = comparison["accessible_market_growth_5y_page2"].combine_first(
    comparison["accessible_market_growth_5y_page3"]
)
comparison["pci"] = comparison["pci_page2"].combine_first(comparison["pci_page3"])
comparison["cog"] = comparison["cog_page2"].combine_first(comparison["cog_page3"])
comparison["dai_percentile"] = comparison["page2_dai_pct"].combine_first(comparison["page3_dai_pct"])

in_page2 = comparison["page2_rank"].notna()
in_page3 = comparison["page3_rank"].notna()
comparison["comparison_group"] = np.select(
    [in_page2 & in_page3, in_page2 & ~in_page3, ~in_page2 & in_page3],
    ["Overlap", "Only Page 2 preset", "Only Page 3 preset"],
    default="Unclassified",
)
comparison["best_rank"] = comparison[["page2_rank", "page3_rank"]].min(axis=1, skipna=True)
comparison = comparison.sort_values(
    ["comparison_group", "best_rank", "accessible_market_size_b"],
    ascending=[True, True, False],
).reset_index(drop=True)

summary = comparison["comparison_group"].value_counts().rename_axis("comparison_group").reset_index(name="count")

c1, c2, c3, c4 = st.columns(4)
c1.metric(page2_preset, f"{len(page2_df):,}")
c2.metric(page3_preset, f"{len(page3_df):,}")
c3.metric("Overlap", f"{int((comparison['comparison_group'] == 'Overlap').sum()):,}")
c4.metric("Union of recommendations", f"{len(comparison):,}")

fig = px.bar(
    summary,
    x="comparison_group",
    y="count",
    color="comparison_group",
    color_discrete_map=COMPARE_GROUP_COLORS,
    title=f"Recommendation overlap: {page2_preset} vs {page3_preset}",
)
fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Products")
st.plotly_chart(fig, width="stretch")

view = st.selectbox(
    "Comparison view",
    ["All recommendations", "Overlap only", f"Only {page2_preset}", f"Only {page3_preset}"],
    index=0,
)

view_map = {
    "All recommendations": comparison,
    "Overlap only": comparison[comparison["comparison_group"] == "Overlap"],
    f"Only {page2_preset}": comparison[comparison["comparison_group"] == "Only Page 2 preset"],
    f"Only {page3_preset}": comparison[comparison["comparison_group"] == "Only Page 3 preset"],
}
table = view_map[view].copy()
table["accessible_market_growth_5y"] = table["accessible_market_growth_5y"] * 100

st.subheader("Recommendation comparison table")
st.dataframe(
    table[
        [
            "comparison_group",
            "hs4",
            "product_name_short",
            "sector",
            "page2_rank",
            "page2_score",
            "page3_rank",
            "page3_score",
            "accessible_market_size_b",
            "accessible_market_growth_5y",
            "dai_percentile",
            "pci",
            "cog",
        ]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "comparison_group": st.column_config.TextColumn("Comparison Group"),
        "hs4": st.column_config.TextColumn("HS4"),
        "product_name_short": st.column_config.TextColumn("Product"),
        "sector": st.column_config.TextColumn("Sector"),
        "page2_rank": st.column_config.NumberColumn(f"{page2_preset} Rank", format="%.0f"),
        "page2_score": st.column_config.NumberColumn(f"{page2_preset} Score", format="%.4f"),
        "page3_rank": st.column_config.NumberColumn(f"{page3_preset} Rank", format="%.0f"),
        "page3_score": st.column_config.NumberColumn(f"{page3_preset} Score", format="%.4f"),
        "accessible_market_size_b": st.column_config.NumberColumn("Accessible Market Size (B USD)", format="%.3f"),
        "accessible_market_growth_5y": st.column_config.NumberColumn("Accessible Market Growth % (5y)", format="%.2f%%"),
        "dai_percentile": st.column_config.NumberColumn("DAI Percentile", format="%.1f"),
        "pci": st.column_config.NumberColumn("PCI", format="%.3f"),
        "cog": st.column_config.NumberColumn("COG", format="%.3f"),
    },
)

overlap = comparison[comparison["comparison_group"] == "Overlap"].copy()
if not overlap.empty:
    overlap["rank_gap"] = overlap["page3_rank"] - overlap["page2_rank"]
    st.subheader("Overlap diagnostics")
    st.dataframe(
        overlap[
            [
                "hs4",
                "product_name_short",
                "sector",
                "page2_rank",
                "page3_rank",
                "rank_gap",
                "accessible_market_size_b",
                "dai_percentile",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "hs4": st.column_config.TextColumn("HS4"),
            "product_name_short": st.column_config.TextColumn("Product"),
            "sector": st.column_config.TextColumn("Sector"),
            "page2_rank": st.column_config.NumberColumn(f"{page2_preset} Rank", format="%.0f"),
            "page3_rank": st.column_config.NumberColumn(f"{page3_preset} Rank", format="%.0f"),
            "rank_gap": st.column_config.NumberColumn(f"Rank Gap ({page3_preset} - {page2_preset})", format="%.0f"),
            "accessible_market_size_b": st.column_config.NumberColumn("Accessible Market Size (B USD)", format="%.3f"),
            "dai_percentile": st.column_config.NumberColumn("DAI Percentile", format="%.1f"),
        },
    )
else:
    st.info("No overlapping products between the two selected preset recommendation lists.")
