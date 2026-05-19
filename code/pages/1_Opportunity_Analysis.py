import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import (
    load_opportunity_dataset,
    normalize_0_1,
)


st.title("Opportunity Analysis")
st.caption("BACI-based HS92 opportunity model. Calibrate index components, rebalance feasibility vs attractiveness, and explore product opportunities.")

df = load_opportunity_dataset()
if df.empty:
    st.warning("No data available for year 2024 / ARG in complexity_calculations.csv.")
    st.stop()

FEAS_COLS = ["rca_transformed_z", "density_z", "eff_num_exp_z", "alignment_weighted_percentile_z"]
ATTR_COLS = ["pci_z", "cog_z", "potential_market_growth_5y_z", "potential_market_size_share_z"]
# Defensive schema guard for cached/legacy datasets.
for col in FEAS_COLS + ATTR_COLS + ["potential_market_growth_5y"]:
    if col not in df.columns:
        df[col] = 0.0
if "rca_cordoba_exports" not in df.columns:
    df["rca_cordoba_exports"] = np.nan
if "cordoba_rubro_description" not in df.columns:
    df["cordoba_rubro_description"] = ""
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
MANDATORY_EXCLUDED_HS4 = {
    "2701": "Coal",
    "2709": "Petroleum oils, crude",
    "2710": "Petroleum oils, refined",
    "2711": "Petroleum gases",
    "2713": "Petroleum coke",
    "7108": "Gold",
}


def weighted_index(frame: pd.DataFrame, cols: list[str], weights: list[float]) -> pd.Series:
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        return frame[cols].mean(axis=1)
    arr = np.column_stack([frame[c].to_numpy() for c in cols])
    w = np.array(weights, dtype=float)
    return pd.Series((arr * w).sum(axis=1) / weight_sum, index=frame.index)


defaults = {
    "w_rca": 0.00,
    "w_density": 0.70,
    "w_eff_num_exp": 0.00,
    "w_alignment_hv": 0.30,
    "w_pci": 0.35,
    "w_cog": 0.35,
    "w_growth": 0.15,
    "w_market_size": 0.15,
    "strategic_balance": 0.50,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

trade_max = float(df["total_trade_b"].max()) if not df.empty else 0.0
country_export_max = float(df["country_current_exports_b"].max()) if not df.empty else 0.0
country_export_max_m = country_export_max * 1000
density_pct_min_data = float(pd.to_numeric(df["density_percentile"], errors="coerce").min()) if not df.empty else 0.0
density_pct_max_data = float(pd.to_numeric(df["density_percentile"], errors="coerce").max()) if not df.empty else 1.0
density_pct_min_data = min(max(density_pct_min_data, 0.0), 1.0)
density_pct_max_data = min(max(density_pct_max_data, 0.0), 1.0)
if density_pct_max_data < density_pct_min_data:
    density_pct_min_data, density_pct_max_data = density_pct_max_data, density_pct_min_data
if density_pct_max_data == density_pct_min_data:
    if density_pct_max_data < 1.0:
        density_pct_max_data += 0.01
    else:
        density_pct_min_data = max(0.0, density_pct_min_data - 0.01)

rca_min_data = float(pd.to_numeric(df["raw_rca"], errors="coerce").min()) if not df.empty else 0.0
rca_max_data = float(pd.to_numeric(df["raw_rca"], errors="coerce").max()) if not df.empty else 1.0
if not np.isfinite(rca_min_data):
    rca_min_data = 0.0
if not np.isfinite(rca_max_data):
    rca_max_data = 1.0
if rca_max_data < rca_min_data:
    rca_min_data, rca_max_data = rca_max_data, rca_min_data
if rca_max_data == rca_min_data:
    rca_max_data = rca_min_data + 0.01
sector_options = sorted(x for x in df["sector"].dropna().astype(str).unique())
product_options_df = (
    df[["hs4", "product_name_short"]]
    .dropna(subset=["hs4"])
    .copy()
    .assign(
        hs4_code=lambda d: d["hs4"].astype(str).str.zfill(4),
        product_name_short=lambda d: d["product_name_short"].fillna("").astype(str).str.strip(),
    )
    .drop_duplicates(subset=["hs4_code"])
    .sort_values("hs4_code")
)
product_options_df["hs4_label"] = product_options_df["hs4_code"] + " - " + product_options_df["product_name_short"]
product_label_to_code = dict(zip(product_options_df["hs4_label"], product_options_df["hs4_code"]))
size_choices = {
    "Total trade (B USD)": "total_trade_b",
    "Accessible market size (B USD)": "potential_market_size_b",
    "Accessible market growth (5y)": "potential_market_growth_5y",
    "Raw RCA": "raw_rca",
    "Market growth (5y)": "market_growth_5y",
    "Country export growth (5y)": "country_export_growth_5y",
    "Effective exporters": "eff_num_exp",
    "Distance travelled": "distance_travelled",
}

if "trade_min" not in st.session_state:
    st.session_state["trade_min"] = 0.0
if "country_export_min_m" not in st.session_state:
    prev_b = float(st.session_state.get("country_export_min", 0.0))
    st.session_state["country_export_min_m"] = prev_b * 1000
if "rca_max_filter" not in st.session_state:
    prior = st.session_state.get("rca_range", (rca_min_data, rca_max_data))
    prior_max = float(prior[1]) if isinstance(prior, (tuple, list)) and len(prior) == 2 else float(rca_max_data)
    st.session_state["rca_max_filter"] = max(0.0, prior_max)
if "rca_min_filter" not in st.session_state:
    prior = st.session_state.get("rca_range", (rca_min_data, rca_max_data))
    prior_min = float(prior[0]) if isinstance(prior, (tuple, list)) and len(prior) == 2 else 0.0
    st.session_state["rca_min_filter"] = max(0.0, prior_min)
st.session_state["rca_min_filter"] = max(0.0, float(st.session_state.get("rca_min_filter", 0.0)))
st.session_state["rca_max_filter"] = max(
    float(st.session_state["rca_min_filter"]),
    float(st.session_state.get("rca_max_filter", rca_max_data)),
)
if "selected_sectors" not in st.session_state:
    st.session_state["selected_sectors"] = sector_options
if "above_median_only" not in st.session_state:
    st.session_state["above_median_only"] = False
if "above_export_median_only" not in st.session_state:
    st.session_state["above_export_median_only"] = False
if "above_potential_growth_only" not in st.session_state:
    st.session_state["above_potential_growth_only"] = False
if "size_label" not in st.session_state:
    st.session_state["size_label"] = "Accessible market growth (5y)"
if "density_pct_range" not in st.session_state:
    st.session_state["density_pct_range"] = (density_pct_min_data, density_pct_max_data)
if "top_n" not in st.session_state:
    st.session_state["top_n"] = 60

mandatory_excluded_labels = [
    label
    for label, code in product_label_to_code.items()
    if code in MANDATORY_EXCLUDED_HS4
]
mandatory_excluded_labels = sorted(mandatory_excluded_labels)
if "excluded_product_labels" not in st.session_state:
    st.session_state["excluded_product_labels"] = []


def _apply_preset(
    *,
    trade_min_value: float,
    rca_min_value: float,
    rca_max_value: float,
    density_range_value: tuple[float, float],
    feasibility_weights: dict[str, float],
    attractiveness_weight: float,
) -> None:
    st.session_state["trade_min"] = trade_min_value
    st.session_state["country_export_min_m"] = 0.0
    st.session_state["rca_min_filter"] = rca_min_value
    st.session_state["rca_max_filter"] = rca_max_value
    st.session_state["density_pct_range"] = density_range_value
    st.session_state["selected_sectors"] = sector_options
    st.session_state["excluded_product_labels"] = []
    st.session_state["above_median_only"] = True
    st.session_state["above_export_median_only"] = False
    st.session_state["above_potential_growth_only"] = False
    st.session_state["top_n"] = 50

    st.session_state["w_rca"] = feasibility_weights.get("w_rca", 0.0)
    st.session_state["w_density"] = feasibility_weights.get("w_density", 0.0)
    st.session_state["w_eff_num_exp"] = feasibility_weights.get("w_eff_num_exp", 0.0)
    st.session_state["w_alignment_hv"] = feasibility_weights.get("w_alignment_hv", 0.0)
    st.session_state["w_pci"] = 0.35
    st.session_state["w_cog"] = 0.35
    st.session_state["w_growth"] = 0.15
    st.session_state["w_market_size"] = 0.15
    st.session_state["strategic_balance"] = attractiveness_weight


st.sidebar.header("Preset Filters")
st.sidebar.caption("Each preset applies the algorithm thresholds, common exclusions, top 50 rows, and the corresponding score weights.")
preset_cols = st.sidebar.columns(1)
with preset_cols[0]:
    st.button(
        "Intensive Margin",
        width="stretch",
        on_click=_apply_preset,
        kwargs={
            "trade_min_value": 1.0,
            "rca_min_value": 1.0,
            "rca_max_value": max(float(rca_max_data), 1.0),
            "density_range_value": (density_pct_min_data, density_pct_max_data),
            "feasibility_weights": {
                "w_rca": 0.30,
                "w_density": 0.00,
                "w_eff_num_exp": 0.00,
                "w_alignment_hv": 0.70,
            },
            "attractiveness_weight": 0.70,
        },
    )
    st.button(
        "Extensive Margin: Low Hanging Fruits",
        width="stretch",
        on_click=_apply_preset,
        kwargs={
            "trade_min_value": 1.0,
            "rca_min_value": 0.30,
            "rca_max_value": 1.0,
            "density_range_value": (density_pct_min_data, density_pct_max_data),
            "feasibility_weights": {
                "w_rca": 0.35,
                "w_density": 0.35,
                "w_eff_num_exp": 0.00,
                "w_alignment_hv": 0.30,
            },
            "attractiveness_weight": 0.30,
        },
    )
    st.button(
        "Extensive Margin: Strategic Bets",
        width="stretch",
        on_click=_apply_preset,
        kwargs={
            "trade_min_value": 1.0,
            "rca_min_value": 0.0,
            "rca_max_value": 0.30,
            "density_range_value": (max(0.50, density_pct_min_data), density_pct_max_data),
            "feasibility_weights": {
                "w_rca": 0.00,
                "w_density": 0.70,
                "w_eff_num_exp": 0.00,
                "w_alignment_hv": 0.30,
            },
            "attractiveness_weight": 0.70,
        },
    )

st.sidebar.caption(
    "Always excluded: "
    + ", ".join(f"{code} {name}" for code, name in sorted(MANDATORY_EXCLUDED_HS4.items()))
)

st.sidebar.header("Filters")
size_label = st.sidebar.selectbox("Dot size variable", list(size_choices.keys()), key="size_label")
min_dot_size = 4
max_dot_size = 20

trade_min = st.sidebar.number_input(
    "Minimum total trade (Billion USD)",
    min_value=0.0,
    max_value=float(max(trade_max, 0.1)),
    value=float(st.session_state["trade_min"]),
    step=0.01,
    format="%.2f",
    key="trade_min",
    help="Type the minimum total trade threshold directly.",
)

country_export_min_m = st.sidebar.number_input(
    "Minimum Argentina exports (Million USD)",
    min_value=0.0,
    max_value=float(max(country_export_max_m, 1.0)),
    value=float(st.session_state["country_export_min_m"]),
    step=1.0,
    format="%.0f",
    key="country_export_min_m",
    help="Type the minimum Argentina exported value threshold directly.",
)
country_export_min_b = country_export_min_m / 1000

rca_step = 0.001
rca_min_filter = st.sidebar.number_input(
    "Minimum (Raw) RCA",
    min_value=0.0,
    value=min(float(st.session_state["rca_min_filter"]), float(st.session_state["rca_max_filter"])),
    step=float(rca_step),
    format="%.3f",
    key="rca_min_filter",
    help="Keep products with raw RCA greater than or equal to this value.",
)
rca_max_filter = st.sidebar.number_input(
    "Maximum (Raw) RCA",
    min_value=float(rca_min_filter),
    value=max(float(st.session_state["rca_max_filter"]), float(rca_min_filter)),
    step=float(rca_step),
    format="%.3f",
    key="rca_max_filter",
    help="Keep products with raw RCA less than or equal to this value.",
)

density_pct_range = st.sidebar.slider(
    "Density percentile range",
    min_value=float(density_pct_min_data),
    max_value=float(density_pct_max_data),
    value=st.session_state["density_pct_range"],
    step=0.01,
    key="density_pct_range",
)

selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=sector_options,
    default=st.session_state["selected_sectors"],
    key="selected_sectors",
)
excluded_product_labels = st.sidebar.multiselect(
    "Additional exclude products (HS4)",
    options=product_options_df["hs4_label"].tolist(),
    default=st.session_state["excluded_product_labels"],
    key="excluded_product_labels",
    help="Add one or more HS4 products to the mandatory exclusions listed above.",
)
excluded_hs4_codes = set(MANDATORY_EXCLUDED_HS4).union(
    {product_label_to_code[label] for label in excluded_product_labels}
)

above_median_only = st.sidebar.toggle(
    "CAGR 5y greater than 0",
    value=st.session_state["above_median_only"],
    key="above_median_only",
)
above_export_median_only = st.sidebar.toggle(
    "Above Argentina Export CAGR",
    value=st.session_state["above_export_median_only"],
    key="above_export_median_only",
)
above_potential_growth_only = st.sidebar.toggle(
    "Accessible Market Growth (5y) > 0",
    value=st.session_state["above_potential_growth_only"],
    key="above_potential_growth_only",
)

st.sidebar.header("Dimension Balance")
strategic_balance = st.sidebar.slider(
    "Feasibility = 0  |  Attractiveness = 1",
    0.0,
    1.0,
    float(st.session_state["strategic_balance"]),
    0.05,
    key="strategic_balance",
)

st.sidebar.header("Weight Controls")
st.sidebar.caption("Each index is a weighted average of normalized components.")
if st.sidebar.button("Reset Weights"):
    for k, v in defaults.items():
        st.session_state[k] = v

with st.sidebar.expander("Feasibility Components", expanded=True):
    w_rca = st.slider("Transformed RCA weight", 0.0, 1.0, float(st.session_state["w_rca"]), 0.05, key="w_rca")
    w_density = st.slider("Density weight", 0.0, 1.0, float(st.session_state["w_density"]), 0.05, key="w_density")
    w_eff_num_exp = st.slider(
        "Effective exporters weight", 0.0, 1.0, float(st.session_state["w_eff_num_exp"]), 0.05, key="w_eff_num_exp"
    )
    w_alignment_hv = st.slider(
        "WNAI weight",
        0.0,
        1.0,
        float(st.session_state["w_alignment_hv"]),
        0.05,
        key="w_alignment_hv",
    )

with st.sidebar.expander("Attractiveness Components", expanded=True):
    w_pci = st.slider("PCI weight", 0.0, 1.0, float(st.session_state["w_pci"]), 0.05, key="w_pci")
    w_cog = st.slider("COG weight", 0.0, 1.0, float(st.session_state["w_cog"]), 0.05, key="w_cog")
    w_growth = st.slider(
        "Accessible market growth (5y) weight", 0.0, 1.0, float(st.session_state["w_growth"]), 0.05, key="w_growth"
    )
    w_market_size = st.slider(
        "Accessible market size share weight", 0.0, 1.0, float(st.session_state["w_market_size"]), 0.05, key="w_market_size"
    )

flt = df.copy()
flt["feasibility_raw"] = weighted_index(
    flt,
    FEAS_COLS,
    [w_rca, w_density, w_eff_num_exp, w_alignment_hv],
)
flt["attractiveness_raw"] = weighted_index(
    flt,
    ATTR_COLS,
    [w_pci, w_cog, w_growth, w_market_size],
)
# Renormalize post-aggregation so final indexes remain bounded in [0, 1].
flt["feasibility_index"] = normalize_0_1(flt["feasibility_raw"])
flt["attractiveness_index"] = normalize_0_1(flt["attractiveness_raw"])
flt["combined_raw"] = ((1 - strategic_balance) * flt["feasibility_index"]) + (
    strategic_balance * flt["attractiveness_index"]
)
flt["combined_score"] = normalize_0_1(flt["combined_raw"])

flt = flt[flt["total_trade_b"] >= trade_min]
flt = flt[flt["country_current_exports_b"] >= country_export_min_b]
flt = flt[flt["raw_rca"] >= float(rca_min_filter)]
flt = flt[flt["raw_rca"] <= float(rca_max_filter)]
density_lo = min(density_pct_range)
density_hi = max(density_pct_range)
flt = flt[
    (pd.to_numeric(flt["density_percentile"], errors="coerce").fillna(0.0) >= density_lo)
    & (pd.to_numeric(flt["density_percentile"], errors="coerce").fillna(0.0) <= density_hi)
]
if selected_sectors:
    flt = flt[flt["sector"].isin(selected_sectors)]
if excluded_hs4_codes:
    flt = flt[~flt["hs4"].astype(str).str.zfill(4).isin(excluded_hs4_codes)]
if above_median_only:
    flt = flt[flt["market_growth_5y"] > 0]
if above_export_median_only:
    flt = flt[flt["above_median_export_cagr"]]
if above_potential_growth_only:
    flt = flt[flt["potential_market_growth_5y"] > 0]

if flt.empty:
    st.warning("No products match the current filters.")
    st.stop()

size_col = size_choices[size_label]
size_raw = pd.to_numeric(flt[size_col], errors="coerce").fillna(0).clip(lower=0)
size_norm = normalize_0_1(size_raw)
flt["dot_size"] = min_dot_size + (size_norm * (max_dot_size - min_dot_size))

c1, c2, c3 = st.columns(3)
c1.metric("Products shown", f"{len(flt):,} / {len(df):,}")
c2.metric("Avg Feasibility", f"{flt['feasibility_index'].mean():.3f}")
c3.metric("Avg Attractiveness", f"{flt['attractiveness_index'].mean():.3f}")
st.caption(
    f"Active filters: trade >= {trade_min:.2f} BUSD, "
    f"ARG exports >= {country_export_min_m:.0f} MUSD, "
    f"Raw RCA in [{float(rca_min_filter):.3f}, {float(rca_max_filter):.3f}], "
    f"density percentile in [{density_lo:.2f}, {density_hi:.2f}], "
    f"{len(selected_sectors)} sectors, "
    f"{len(excluded_hs4_codes)} excluded products, "
    f"CAGR>0 filter={above_median_only}, "
    f"above-export CAGR={above_export_median_only}, "
    f"accessible-market growth>0={above_potential_growth_only}."
)

fig = px.scatter(
    flt,
    x="feasibility_index",
    y="attractiveness_index",
    color="sector",
    color_discrete_map=SECTOR_COLORS,
    size="dot_size",
    size_max=max_dot_size,
    hover_name="product_name_short",
    hover_data={
        "raw_rca": ":.3f",
        "rca_transformed": ":.3f",
        "pci": ":.3f",
        "cog": ":.3f",
        "density": ":.3f",
        "eff_num_exp": ":.2f",
        "distance_travelled": ":.2f",
        "density_percentile": ":.3f",
        "market_growth_5y": ":.3%",
        "potential_market_growth_5y": ":.3%",
        "country_export_growth_5y": ":.3%",
        "market_size_share": ":.3%",
        "market_size_b": ":.3f",
        "potential_market_size_b": ":.3f",
        "potential_market_to_market_ratio": ":.3%",
        "total_trade_b": ":.3f",
        "dot_size": False,
    },
    labels={
        "raw_rca": "Raw RCA",
        "rca_transformed": "Transformed RCA",
        "pci": "PCI",
        "cog": "COG",
        "density": "Density",
        "eff_num_exp": "Effective Exporters",
        "distance_travelled": "Distance Travelled",
        "density_percentile": "Density Percentile",
        "market_growth_5y": "Global Market Growth (5y)",
        "potential_market_growth_5y": "Accessible Market Growth (5y)",
        "country_export_growth_5y": "Country Export Growth (5y)",
        "market_size_share": "Global Market Share",
        "market_size_b": "Global Market Size (B USD)",
        "potential_market_size_b": "Accessible Market Size (B USD)",
        "potential_market_to_market_ratio": "Accessible/Market Ratio",
        "total_trade_b": "Total Trade (B USD)",
        "feasibility_index": "Feasibility",
        "attractiveness_index": "Attractiveness",
        "sector": "Sector",
    },
    custom_data=["hs4", "product_name_short"],
    title="Feasibility vs Attractiveness (HS92, 2024)",
    template="plotly_white",
)
fig.update_traces(marker=dict(opacity=0.78, line=dict(width=0.5, color="#1f2f46")))
fig.update_layout(
    xaxis_title="Feasibility",
    yaxis_title="Attractiveness",
    legend_title="Sector",
    margin=dict(t=60, l=20, r=20, b=20),
)
diag_min = float(min(flt["feasibility_index"].min(), flt["attractiveness_index"].min()))
diag_max = float(max(flt["feasibility_index"].max(), flt["attractiveness_index"].max()))
fig.add_shape(
    type="line",
    x0=diag_min,
    y0=diag_min,
    x1=diag_max,
    y1=diag_max,
    line=dict(color="rgba(220, 38, 38, 0.55)", width=2, dash="dash"),
)
st.plotly_chart(
    fig,
    width="stretch",
)

st.subheader("Top Product Rankings")
search_text = st.text_input("Search product name", placeholder="Type to filter...")
top_n = st.slider(
    "Rows to display",
    min_value=10,
    max_value=300,
    value=int(st.session_state["top_n"]),
    step=10,
    key="top_n",
)

table = flt.copy()
if search_text:
    mask = table["product_name_short"].astype(str).str.contains(search_text, case=False, na=False)
    table = table[mask]

table = table.sort_values("combined_score", ascending=False).reset_index(drop=True)
table["rank"] = np.arange(1, len(table) + 1)

display_cols = [
    "rank",
    "hs4",
    "product_name_short",
    "sector",
    "cordoba_rubro_description",
    "rca_cordoba_exports",
    "combined_score",
    "attractiveness_index",
    "feasibility_index",
    "raw_rca",
    "density",
    "density_percentile",
    "pci",
    "cog",
    "eff_num_exp",
    "country_exporter_rank",
    "alignment_weighted_percentile",
    "alignment_lead_weighted",
    "distance_travelled",
    "market_growth_5y",
    "potential_market_growth_5y",
    "country_export_growth_5y",
    "country_current_exports",
    "market_share_change_abs",
    "market_size_share",
    "potential_market_size",
    "potential_market_to_market_ratio",
    "total_trade_b",
]
table_display = (
    table[display_cols]
    .head(top_n)
    .assign(
        cordoba_rubro_description=lambda d: d["cordoba_rubro_description"].fillna(""),
        market_growth_5y=lambda d: d["market_growth_5y"] * 100,
        potential_market_growth_5y=lambda d: d["potential_market_growth_5y"] * 100,
        country_export_growth_5y=lambda d: d["country_export_growth_5y"] * 100,
        country_current_exports=lambda d: d["country_current_exports"] / 1_000_000,
        market_share_change_abs=lambda d: d["market_share_change_abs"] * 100,
        market_size_share=lambda d: d["market_size_share"] * 100,
        potential_market_size=lambda d: d["potential_market_size"] / 1_000_000_000,
        potential_market_to_market_ratio=lambda d: d["potential_market_to_market_ratio"] * 100,
    )
)


def _lead_color(value: float) -> str:
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v):
        return ""
    if float(v) > 0:
        return "color: #16a34a;"
    if float(v) < 0:
        return "color: #dc2626;"
    return ""


table_styler = table_display.style.map(
    _lead_color,
    subset=["alignment_lead_weighted"],
)

st.dataframe(
    table_styler,
    width="stretch",
    hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("Rank", format="%.0f"),
        "hs4": st.column_config.TextColumn("HS4"),
        "product_name_short": st.column_config.TextColumn("Product"),
        "sector": st.column_config.TextColumn("Sector"),
        "cordoba_rubro_description": st.column_config.TextColumn(
            "Cordoba NCM/Rubro Description",
            help="Spanish description of the mapped Cordoba export rubro used for the contextual RCA. Blank when there is no mapped rubro.",
        ),
        "rca_cordoba_exports": st.column_config.NumberColumn(
            "Cordoba Export RCA (contextual)",
            format="%.2f",
            help="RCA for the mapped Cordoba export rubro/NCM parent. Repeated across HS4 children when the mapping is broader than HS4.",
        ),
        "combined_score": st.column_config.NumberColumn("Combined Opportunity Score", format="%.3f"),
        "attractiveness_index": st.column_config.NumberColumn("Attractiveness Index", format="%.3f"),
        "feasibility_index": st.column_config.NumberColumn("Feasibility Index", format="%.3f"),
        "raw_rca": st.column_config.NumberColumn("Raw RCA", format="%.3f"),
        "density": st.column_config.NumberColumn("Density (Raw)", format="%.6f"),
        "density_percentile": st.column_config.NumberColumn("Density Percentile", format="%.3f"),
        "distance_travelled": st.column_config.NumberColumn("Distance Travelled", format="%.2f"),
        "pci": st.column_config.NumberColumn("PCI", format="%.3f"),
        "cog": st.column_config.NumberColumn("COG", format="%.3f"),
        "eff_num_exp": st.column_config.NumberColumn("Effective Exporters", format="%.2f"),
        "country_exporter_rank": st.column_config.NumberColumn("Country Exporter Rank (2024)", format="%.0f"),
        "alignment_weighted_percentile": st.column_config.NumberColumn("WNAI Percentile", format="%.1f"),
        "alignment_lead_weighted": st.column_config.NumberColumn("WNAI Lead", format="%.1f"),
        "market_growth_5y": st.column_config.NumberColumn("Global Market Growth % (5y)", format="%.2f%%"),
        "potential_market_growth_5y": st.column_config.NumberColumn("Accessible Market Growth % (5y)", format="%.2f%%"),
        "country_export_growth_5y": st.column_config.NumberColumn("Country Export Growth % (5y)", format="%.2f%%"),
        "country_current_exports": st.column_config.NumberColumn("Country Current Exports (M USD)", format="%.2f"),
        "market_share_change_abs": st.column_config.NumberColumn("Absolute Market Share Change (pp, 2024-2020)", format="%.2f"),
        "market_size_share": st.column_config.NumberColumn("Global Market Share", format="%.2f%%"),
        "potential_market_size": st.column_config.NumberColumn("Accessible Market Size (B USD)", format="%.3f"),
        "potential_market_to_market_ratio": st.column_config.NumberColumn("Accessible-to-Market Ratio", format="%.2f%%"),
        "total_trade_b": st.column_config.NumberColumn("Total Trade (B USD)", format="%.3f"),
    },
)

st.subheader("Opportunity Summary by Sector")
treemap_size_options = {
    "Accessible market size (B USD)": "potential_market_size",
    "Accessible market growth (5y)": "potential_market_growth_5y",
    "Market size (B USD)": "total_trade_b",
    "Combined Opportunity Score": "combined_score",
    "Density Percentile": "density_percentile",
    "Frequency": "frequency",
}
st.session_state.setdefault("treemap_size_metric_v1", "Accessible market size (B USD)")
treemap_size_label = st.selectbox(
    "Treemap size variable",
    options=list(treemap_size_options.keys()),
    key="treemap_size_metric_v1",
)
treemap_value_col = treemap_size_options[treemap_size_label]

treemap_df = table_display[
    ["sector", "hs4", "product_name_short", "combined_score", "total_trade_b", "potential_market_size", "potential_market_growth_5y", "density_percentile"]
].copy()
treemap_df["sector"] = treemap_df["sector"].fillna("Sin sector")
treemap_df["product_label"] = (
    treemap_df["hs4"].astype(str).str.zfill(4) + " - " + treemap_df["product_name_short"].astype(str)
)


def _wrap_treemap_label(text: str, width: int = 18) -> str:
    words = str(text).split()
    if not words:
        return str(text)
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        if len(current) + 1 + len(w) <= width:
            current = f"{current} {w}"
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return "<br>".join(lines)


treemap_df["product_label_wrapped"] = treemap_df["product_label"].map(_wrap_treemap_label)
treemap_df["frequency"] = 1.0

if treemap_df.empty:
    st.info("No products available for the treemap under the current filters.")
else:
    treemap = px.treemap(
        treemap_df,
        path=["sector", "product_label_wrapped"],
        values=treemap_value_col,
        color="sector",
        color_discrete_map=SECTOR_COLORS,
        hover_data={
            "combined_score": ":.3f",
            "total_trade_b": ":.3f",
            "potential_market_size": ":.3f",
            "potential_market_growth_5y": ":.3f",
            "density_percentile": ":.3f",
            "product_label": True,
            "sector": False,
            "product_label_wrapped": False,
        },
        title=f"Opportunity treemap (n = {len(treemap_df)} products shown) | size = {treemap_size_label}",
    )
    treemap.update_traces(
        textinfo="label",
        textfont=dict(size=18, color="#ffffff"),
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.45)")),
    )
    treemap.update_layout(
        margin=dict(t=60, l=10, r=10, b=10),
    )
    st.plotly_chart(treemap, width="stretch")
