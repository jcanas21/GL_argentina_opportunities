import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re

from data_utils import load_anchor_proximity_dataset, normalize_0_1


SECTOR_COLORS = {
    "Services": "#b23c6f",
    "Textiles": "#7bc8a4",
    "Agriculture": "#e5c21a",
    "Stone": "#caa46b",
    "Minerals": "#a88b7d",
    "Metals": "#c9656b",
    "Chemicals": "#b07ac9",
    "Vehicles": "#7a6cc3",
    "Machinery": "#6e8fc3",
    "Electronics": "#74c5c6",
    "Other": "#2f5d74",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    hex_color = str(hex_color).lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(47,93,116,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _wrap_label(text: str, width: int = 18) -> str:
    words = str(text).split()
    if not words:
        return str(text)
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "<br>".join(lines)


def _section_sort_key(label: str) -> tuple[int, str]:
    text = str(label).strip()
    match = re.match(r"^(\d+)\.\s*(.*)$", text)
    if match:
        return (int(match.group(1)), match.group(2).lower())
    return (10_000, text.lower())


st.title("Anchored Proximity Analysis")
st.caption("Explore candidate products connected to anchor products through proximity links.")

df = load_anchor_proximity_dataset()
if df.empty:
    st.warning("No anchor proximity data available.")
    st.stop()

pci_series = pd.to_numeric(df["pci"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
if pci_series.empty:
    pci_color_min = 0.0
    pci_color_max = 1.0
else:
    pci_color_min = float(pci_series.quantile(0.02))
    pci_color_max = float(pci_series.quantile(0.98))
    if not np.isfinite(pci_color_min):
        pci_color_min = float(pci_series.min())
    if not np.isfinite(pci_color_max):
        pci_color_max = float(pci_series.max())
    if pci_color_max <= pci_color_min:
        pci_color_max = pci_color_min + 1e-6

anchor_sectors = sorted(df["anchor_sector"].dropna().astype(str).unique().tolist())
candidate_sectors = sorted(df["candidate_sector"].dropna().astype(str).unique().tolist())
anchor_sections = sorted(
    df["anchor_hs_section_name"].dropna().astype(str).unique().tolist(),
    key=_section_sort_key,
)
candidate_sections = sorted(
    df["candidate_hs_section_name"].dropna().astype(str).unique().tolist(),
    key=_section_sort_key,
)
candidate_products_df = (
    df[["candidate_hs4", "candidate_product_name_short"]]
    .dropna(subset=["candidate_hs4"])
    .copy()
    .assign(
        candidate_hs4=lambda d: d["candidate_hs4"].astype(str).str.zfill(4),
        candidate_product_name_short=lambda d: d["candidate_product_name_short"].fillna("").astype(str).str.strip(),
    )
    .drop_duplicates(subset=["candidate_hs4"])
    .sort_values("candidate_hs4")
)
candidate_products_df["candidate_label"] = (
    candidate_products_df["candidate_hs4"] + " - " + candidate_products_df["candidate_product_name_short"]
)
candidate_label_to_code = dict(
    zip(candidate_products_df["candidate_label"], candidate_products_df["candidate_hs4"])
)
proximity_rank_max = int(pd.to_numeric(df["proximity_rank"], errors="coerce").max())
accessible_market_max = float(df["accessible_market_size_b"].max()) if not df.empty else 0.0
anchor_density_min = float(pd.to_numeric(df["anchor_density_percentile"], errors="coerce").min()) if not df.empty else 0.0
anchor_density_max = float(pd.to_numeric(df["anchor_density_percentile"], errors="coerce").max()) if not df.empty else 1.0
if anchor_density_max == anchor_density_min:
    anchor_density_max = anchor_density_min + 0.01
top_n_default = min(40, max(10, int(df["candidate_hs4"].nunique())))
excluded_hs4_preset_codes = {"2711", "2710", "7108", "2709", "2713", "2701", "2603", "2616"}
excluded_labels_by_code = (
    candidate_products_df[candidate_products_df["candidate_hs4"].isin(excluded_hs4_preset_codes)]
    .sort_values("candidate_hs4")["candidate_label"]
    .tolist()
)
anchor_sections_excluding_123 = [
    s for s in anchor_sections if not str(s).startswith(("1.", "2.", "3."))
]
candidate_sections_excluding_123 = [
    s for s in candidate_sections if not str(s).startswith(("1.", "2.", "3."))
]

st.session_state.setdefault("p3_accessible_market_min", 0.0)
st.session_state.setdefault("p3_am_cagr_only", False)
st.session_state.setdefault("p3_strategic_balance", 0.50)
st.session_state.setdefault("p3_w_dai", 1.0)
st.session_state.setdefault("p3_w_pci", 0.35)
st.session_state.setdefault("p3_w_cog", 0.35)
st.session_state.setdefault("p3_w_growth", 0.15)
st.session_state.setdefault("p3_w_market", 0.15)
st.session_state.setdefault("p3_candidates_to_display", top_n_default)
st.session_state.setdefault("p3_selected_anchor_sectors", anchor_sectors)
st.session_state.setdefault("p3_selected_candidate_sectors", candidate_sectors)
st.session_state.setdefault("p3_selected_candidate_sections", candidate_sections)
st.session_state.setdefault("p3_selected_anchor_sections", anchor_sections)
st.session_state.setdefault("p3_excluded_product_labels", [])
st.session_state.setdefault("p3_proximity_rank_range", (1, min(100, max(1, proximity_rank_max))))
st.session_state.setdefault(
    "p3_anchor_density_range",
    (float(anchor_density_min), float(anchor_density_max)),
)
st.session_state.setdefault("p3_above_median_proximity_only", False)


def _apply_page3_profile(profile_name: str) -> None:
    st.session_state["p3_accessible_market_min"] = 1.0
    st.session_state["p3_am_cagr_only"] = True
    st.session_state["p3_w_dai"] = 1.0
    st.session_state["p3_selected_anchor_sectors"] = anchor_sectors
    st.session_state["p3_selected_candidate_sectors"] = candidate_sectors
    st.session_state["p3_selected_candidate_sections"] = candidate_sections
    st.session_state["p3_selected_anchor_sections"] = anchor_sections
    st.session_state["p3_excluded_product_labels"] = []
    st.session_state["p3_proximity_rank_range"] = (1, min(100, max(1, proximity_rank_max)))
    st.session_state["p3_anchor_density_range"] = (
        float(anchor_density_min),
        float(anchor_density_max),
    )
    st.session_state["p3_above_median_proximity_only"] = False

    if profile_name == "top_candidates":
        st.session_state["p3_strategic_balance"] = 0.70
        st.session_state["p3_w_pci"] = 0.35
        st.session_state["p3_w_cog"] = 0.35
        st.session_state["p3_w_growth"] = 0.15
        st.session_state["p3_w_market"] = 0.15
        st.session_state["p3_candidates_to_display"] = 40
        st.session_state["p3_selected_anchor_sections"] = anchor_sections_excluding_123
        st.session_state["p3_selected_candidate_sections"] = candidate_sections_excluding_123
        st.session_state["p3_excluded_product_labels"] = excluded_labels_by_code
        st.session_state["p3_anchor_density_range"] = (
            float(anchor_density_min),
            min(50.0, float(anchor_density_max)),
        )
        st.session_state["p3_proximity_rank_range"] = (1, min(10, max(1, proximity_rank_max)))


def _reset_page3_filters() -> None:
    st.session_state["p3_accessible_market_min"] = 0.0
    st.session_state["p3_am_cagr_only"] = False
    st.session_state["p3_selected_anchor_sectors"] = anchor_sectors
    st.session_state["p3_selected_candidate_sectors"] = candidate_sectors
    st.session_state["p3_selected_candidate_sections"] = candidate_sections
    st.session_state["p3_selected_anchor_sections"] = anchor_sections
    st.session_state["p3_excluded_product_labels"] = []
    st.session_state["p3_proximity_rank_range"] = (1, min(100, max(1, proximity_rank_max)))
    st.session_state["p3_anchor_density_range"] = (
        float(anchor_density_min),
        float(anchor_density_max),
    )
    st.session_state["p3_above_median_proximity_only"] = False

with st.sidebar:
    st.header("Preset Profiles")
    if st.button("Top Anchored Candidates", width="stretch"):
        _apply_page3_profile("top_candidates")
        st.rerun()

    st.header("Anchor Filters")
    if st.button("Reset all filters", width="stretch"):
        _reset_page3_filters()
        st.rerun()
    selected_anchor_sectors = st.multiselect(
        "Anchor sector",
        options=anchor_sectors,
        default=st.session_state["p3_selected_anchor_sectors"],
        key="p3_selected_anchor_sectors",
    )
    selected_candidate_sectors = st.multiselect(
        "Candidate sector",
        options=candidate_sectors,
        default=st.session_state["p3_selected_candidate_sectors"],
        key="p3_selected_candidate_sectors",
    )
    selected_candidate_sections = st.multiselect(
        "Candidate sections",
        options=candidate_sections,
        default=st.session_state["p3_selected_candidate_sections"],
        key="p3_selected_candidate_sections",
    )
    selected_anchor_sections = st.multiselect(
        "Anchor sections",
        options=anchor_sections,
        default=st.session_state["p3_selected_anchor_sections"],
        key="p3_selected_anchor_sections",
    )
    excluded_product_labels = st.multiselect(
        "Exclude products (HS4)",
        options=candidate_products_df["candidate_label"].tolist(),
        default=st.session_state["p3_excluded_product_labels"],
        key="p3_excluded_product_labels",
    )
    proximity_rank_range = st.slider(
        "Proximity rank range",
        min_value=1,
        max_value=max(1, proximity_rank_max),
        value=st.session_state["p3_proximity_rank_range"],
        step=1,
        key="p3_proximity_rank_range",
    )
    accessible_market_min = st.number_input(
        "Minimum accessible trade (Billion USD)",
        min_value=0.0,
        max_value=float(max(accessible_market_max, 0.1)),
        value=float(st.session_state["p3_accessible_market_min"]),
        step=0.1,
        format="%.2f",
        key="p3_accessible_market_min",
    )
    anchor_density_range = st.slider(
        "Anchor density percentile",
        min_value=float(anchor_density_min),
        max_value=float(anchor_density_max),
        value=st.session_state["p3_anchor_density_range"],
        step=0.01,
        format="%.2f",
        key="p3_anchor_density_range",
    )
    above_median_proximity_only = st.toggle(
        "Proximity above country median only",
        value=bool(st.session_state["p3_above_median_proximity_only"]),
        key="p3_above_median_proximity_only",
    )
    am_cagr_only = st.toggle("AM CAGR (5y) > 0", value=bool(st.session_state["p3_am_cagr_only"]), key="p3_am_cagr_only")

    st.header("Dimension Balance")
    strategic_balance = st.slider(
        "Feasibility (100%) = 0 | Attractiveness (100%) = 1",
        0.0,
        1.0,
        float(st.session_state["p3_strategic_balance"]),
        0.05,
        key="p3_strategic_balance",
    )
    st.header("Weight Controls")
    st.caption("Page 3 uses DAI for feasibility and lets you reweight attractiveness components.")
    with st.expander("Feasibility Components", expanded=True):
        st.caption("DAI is the sole feasibility component on this page.")
        w_dai = st.slider("DAI weight", 0.0, 1.0, float(st.session_state["p3_w_dai"]), 0.05, key="p3_w_dai")
    with st.expander("Attractiveness Components", expanded=True):
        w_pci = st.slider("PCI weight", 0.0, 1.0, float(st.session_state["p3_w_pci"]), 0.05, key="p3_w_pci")
        w_cog = st.slider("COG weight", 0.0, 1.0, float(st.session_state["p3_w_cog"]), 0.05, key="p3_w_cog")
        w_growth = st.slider(
            "Accessible market growth (5y) weight",
            0.0,
            1.0,
            float(st.session_state["p3_w_growth"]),
            0.05,
            key="p3_w_growth",
        )
        w_market = st.slider(
            "Accessible market size weight",
            0.0,
            1.0,
            float(st.session_state["p3_w_market"]),
            0.05,
            key="p3_w_market",
        )

flt = df.copy()
if selected_anchor_sectors:
    flt = flt[flt["anchor_sector"].isin(selected_anchor_sectors)]
if selected_candidate_sectors:
    flt = flt[flt["candidate_sector"].isin(selected_candidate_sectors)]
if selected_candidate_sections:
    flt = flt[flt["candidate_hs_section_name"].isin(selected_candidate_sections)]
if selected_anchor_sections:
    flt = flt[flt["anchor_hs_section_name"].isin(selected_anchor_sections)]
excluded_hs4_codes = {candidate_label_to_code[label] for label in excluded_product_labels}
if excluded_hs4_codes:
    flt = flt[~flt["candidate_hs4"].astype(str).str.zfill(4).isin(excluded_hs4_codes)]
flt = flt[
    (pd.to_numeric(flt["proximity_rank"], errors="coerce") >= proximity_rank_range[0])
    & (pd.to_numeric(flt["proximity_rank"], errors="coerce") <= proximity_rank_range[1])
]
if above_median_proximity_only:
    flt = flt[pd.to_numeric(flt["proximity_above_country_median"], errors="coerce").fillna(0).astype(int).eq(1)]
anchor_density_2d = pd.to_numeric(flt["anchor_density_percentile"], errors="coerce").fillna(0.0).round(2)
anchor_density_lo = round(min(anchor_density_range), 2)
anchor_density_hi = round(max(anchor_density_range), 2)
effective_anchor_density_hi = anchor_density_hi + 0.01 if anchor_density_hi >= round(float(anchor_density_max), 2) else anchor_density_hi
flt = flt[(anchor_density_2d >= anchor_density_lo) & (anchor_density_2d <= effective_anchor_density_hi)]
flt = flt[pd.to_numeric(flt["accessible_market_size_b"], errors="coerce") >= float(accessible_market_min)]
if am_cagr_only:
    flt = flt[pd.to_numeric(flt["accessible_market_growth_5y"], errors="coerce") > 0.0]

candidate_scores = (
    flt.groupby(["candidate_hs4", "candidate_product_name_short", "candidate_sector"], as_index=False)
    .agg(
        accessible_market_size=("accessible_market_size", "first"),
        accessible_market_size_b=("accessible_market_size_b", "first"),
        accessible_market_growth_5y=("accessible_market_growth_5y", "first"),
        dai_index=("dai_index", "first"),
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
candidate_scores["feasibility_index"] = candidate_scores["dai_mm"]
attr_cols = ["pci_mm", "cog_mm", "accessible_market_growth_mm", "accessible_market_size_mm"]
attr_weights = np.array([w_pci, w_cog, w_growth, w_market], dtype=float)
attr_weight_sum = float(attr_weights.sum())
if attr_weight_sum <= 0:
    candidate_scores["attractiveness_raw"] = candidate_scores[attr_cols].mean(axis=1)
else:
    candidate_scores["attractiveness_raw"] = (
        candidate_scores[attr_cols].to_numpy() * attr_weights
    ).sum(axis=1) / attr_weight_sum
candidate_scores["attractiveness_index"] = normalize_0_1(candidate_scores["attractiveness_raw"])
if float(w_dai) <= 0:
    candidate_scores["feasibility_index"] = 0.0
candidate_scores["combined_raw"] = (
    (1 - strategic_balance) * candidate_scores["feasibility_index"]
    + strategic_balance * candidate_scores["attractiveness_index"]
)
candidate_scores["combined_score"] = normalize_0_1(candidate_scores["combined_raw"])

flt = flt.merge(
    candidate_scores[
        [
            "candidate_hs4",
            "candidate_product_name_short",
            "feasibility_index",
            "attractiveness_index",
            "combined_score",
        ]
    ],
    on=["candidate_hs4", "candidate_product_name_short"],
    how="left",
)

c1, c2, c3 = st.columns(3)
c1.metric("Links shown", f"{len(flt):,}")
c2.metric("Unique anchors", f"{flt['anchor_hs4'].nunique():,}")
c3.metric("Unique candidates", f"{flt['candidate_hs4'].nunique():,}")

if flt.empty:
    st.info("No anchor-candidate links match the current filters.")
    st.stop()

anchor_labels = (
    flt[["anchor_hs4", "anchor_product_name_short", "anchor_sector"]]
    .drop_duplicates()
    .assign(node_label=lambda d: d["anchor_hs4"] + " - " + d["anchor_product_name_short"])
)
candidate_labels = (
    flt[["candidate_hs4", "candidate_product_name_short", "candidate_sector"]]
    .drop_duplicates()
    .assign(node_label=lambda d: d["candidate_hs4"] + " - " + d["candidate_product_name_short"])
)

anchor_node_ids = {k: i for i, k in enumerate(anchor_labels["node_label"].tolist())}
candidate_offset = len(anchor_node_ids)
candidate_node_ids = {k: candidate_offset + i for i, k in enumerate(candidate_labels["node_label"].tolist())}

links = flt.assign(
    anchor_node=lambda d: d["anchor_hs4"].astype(str).str.zfill(4) + " - " + d["anchor_product_name_short"].astype(str),
    candidate_node=lambda d: d["candidate_hs4"].astype(str).str.zfill(4) + " - " + d["candidate_product_name_short"].astype(str),
)

sankey_df = (
    links.groupby(
        ["anchor_node", "candidate_node", "anchor_sector", "candidate_sector"],
        as_index=False,
    )["proximity"]
    .sum()
)
sankey_df["source"] = sankey_df["anchor_node"].map(anchor_node_ids)
sankey_df["target"] = sankey_df["candidate_node"].map(candidate_node_ids)

node_labels = anchor_labels["node_label"].tolist() + candidate_labels["node_label"].tolist()
node_colors = (
    [SECTOR_COLORS.get(s, SECTOR_COLORS["Other"]) for s in anchor_labels["anchor_sector"].tolist()]
    + [SECTOR_COLORS.get(s, SECTOR_COLORS["Other"]) for s in candidate_labels["candidate_sector"].tolist()]
)
link_colors = [_hex_to_rgba(SECTOR_COLORS.get(s, SECTOR_COLORS["Other"]), alpha=0.4) for s in sankey_df["candidate_sector"].tolist()]

sankey = go.Figure(
    go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="rgba(15,23,42,0.15)", width=0.6),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sankey_df["source"],
            target=sankey_df["target"],
            value=sankey_df["proximity"].clip(lower=0.000001),
            color=link_colors,
            customdata=np.stack(
                [sankey_df["anchor_sector"], sankey_df["candidate_sector"], sankey_df["proximity"]],
                axis=-1,
            ),
            hovertemplate=(
                "Anchor sector: %{customdata[0]}<br>"
                "Candidate sector: %{customdata[1]}<br>"
                "Total proximity: %{customdata[2]:.4f}<extra></extra>"
            ),
        ),
    )
)
sankey.update_layout(
    title="Anchor-to-Candidate Proximity Sankey",
    font=dict(size=12),
    margin=dict(t=60, l=10, r=10, b=10),
    height=760,
)
st.plotly_chart(sankey, width="stretch")

top_n_max = max(10, int(candidate_scores["candidate_hs4"].nunique()))
st.session_state["p3_candidates_to_display"] = min(
    int(st.session_state.get("p3_candidates_to_display", top_n_default)),
    top_n_max,
)
top_n = st.slider(
    "Candidates to display",
    min_value=10,
    max_value=top_n_max,
    value=int(st.session_state["p3_candidates_to_display"]),
    step=1,
    key="p3_candidates_to_display",
)

candidate_table = candidate_scores.sort_values(["combined_score", "accessible_market_size"], ascending=[False, False]).reset_index(drop=True)
candidate_table["rank"] = np.arange(1, len(candidate_table) + 1)
candidate_display = (
    candidate_table.head(top_n).copy().assign(
        accessible_market_growth_5y=lambda d: d["accessible_market_growth_5y"] * 100,
    )
)
st.dataframe(
    candidate_display[
        [
            "rank",
            "candidate_hs4",
            "candidate_product_name_short",
            "candidate_sector",
            "combined_score",
            "attractiveness_index",
            "feasibility_index",
            "dai_index",
            "pci",
            "cog",
            "dai_percentile",
            "accessible_market_growth_5y",
            "accessible_market_size_b",
            "avg_proximity",
            "anchor_count",
        ]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("Rank", format="%.0f"),
        "candidate_hs4": st.column_config.TextColumn("HS4"),
        "candidate_product_name_short": st.column_config.TextColumn("Product"),
        "candidate_sector": st.column_config.TextColumn("Sector"),
        "combined_score": st.column_config.NumberColumn("Combined Opportunity Score", format="%.4f"),
        "attractiveness_index": st.column_config.NumberColumn("Attractiveness Index", format="%.4f"),
        "feasibility_index": st.column_config.NumberColumn("Feasibility Index", format="%.4f"),
        "dai_index": st.column_config.NumberColumn("DAI (Raw)", format="%.3f"),
        "pci": st.column_config.NumberColumn("PCI", format="%.3f"),
        "cog": st.column_config.NumberColumn("COG", format="%.3f"),
        "dai_percentile": st.column_config.NumberColumn("DAI Percentile", format="%.1f"),
        "accessible_market_growth_5y": st.column_config.NumberColumn("Accessible Market Growth % (5y)", format="%.2f%%"),
        "accessible_market_size_b": st.column_config.NumberColumn("Accessible Market Size (B USD)", format="%.3f"),
        "avg_proximity": st.column_config.NumberColumn("Average Proximity", format="%.4f"),
        "anchor_count": st.column_config.NumberColumn("Anchor Count", format="%.0f"),
    },
)

st.subheader("Anchor Proximity Candidates")
treemap_color_label = st.selectbox("Treemap color variable", ["Sector", "PCI (raw)"], key="anchor_treemap_color")

treemap_df = candidate_table.head(top_n).copy()
treemap_df["product_label"] = treemap_df["candidate_hs4"].astype(str).str.zfill(4) + " - " + treemap_df["candidate_product_name_short"].astype(str)
treemap_df["product_label_wrapped"] = treemap_df["product_label"].map(_wrap_label)
treemap_df["pci_for_color"] = pd.to_numeric(treemap_df["pci"], errors="coerce").clip(pci_color_min, pci_color_max)

st.metric(
    label="Accessible Market Size (B USD) shown",
    value=f"{treemap_df['accessible_market_size_b'].sum():,.3f}",
)

pci_colorscale = [
    (0.0, "rgb(227, 159, 96)"),
    (0.278697, "rgb(231, 173, 120)"),
    (0.338965, "rgb(235, 188, 143)"),
    (0.398272, "rgb(240, 202, 168)"),
    (0.448314, "rgb(244, 217, 191)"),
    (0.493999, "rgb(248, 231, 215)"),
    (0.494099, "rgb(192, 228, 225)"),
    (0.533691, "rgb(154, 211, 207)"),
    (0.571435, "rgb(116, 195, 189)"),
    (0.606597, "rgb(77, 178, 171)"),
    (0.661681, "rgb(40, 162, 153)"),
    (1.0, "rgb(2, 146, 135)"),
]
treemap_color_kwargs = (
    {"color": "candidate_sector", "color_discrete_map": SECTOR_COLORS}
    if treemap_color_label == "Sector"
    else {
        "color": "pci_for_color",
        "color_continuous_scale": pci_colorscale,
        "range_color": (pci_color_min, pci_color_max),
    }
)
treemap = px.treemap(
    treemap_df,
    path=["candidate_sector", "product_label_wrapped"],
    values="accessible_market_size_b",
    **treemap_color_kwargs,
    hover_data={
        "combined_score": ":.3f",
        "accessible_market_size_b": ":.3f",
        "accessible_market_growth_5y": ":.3%",
        "dai_index": ":.3f",
        "dai_percentile": ":.1f",
        "pci": ":.3f",
        "anchor_count": ":.0f",
        "avg_proximity": ":.4f",
        "candidate_sector": False,
        "product_label_wrapped": False,
    },
    title=(
        f"Anchor Proximity Candidates (n = {len(treemap_df)} candidates shown | "
        f"Accessible Market total = {treemap_df['accessible_market_size_b'].sum():,.3f} B USD) "
        f"| size = Accessible market size (B USD) | color = {treemap_color_label}"
    ),
)
treemap.update_traces(
    textinfo="label",
    textfont=dict(size=18, color="#ffffff"),
    marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
)
treemap.update_layout(margin=dict(t=60, l=10, r=10, b=95))
if treemap_color_label == "Sector":
    treemap.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            title_text="Sector",
        ),
    )
else:
    treemap.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="PCI", font=dict(color="#0f172a", size=16)),
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            len=0.7,
            bgcolor="rgba(255,255,255,0.96)",
            borderwidth=0,
            tickfont=dict(color="#0f172a", size=14),
            tickcolor="#0f172a",
            ticklen=6,
            tickwidth=1.2,
        ),
    )

st.plotly_chart(treemap, width="stretch")
