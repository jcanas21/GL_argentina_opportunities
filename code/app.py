from pathlib import Path

import pandas as pd
import streamlit as st


def render_guide_and_glossary() -> None:
    st.title("Argentina Export Opportunities Dashboard")
    st.caption("Guide and glossary for the V1 Opportunity Analysis page (HS92 4-digit).")

    st.markdown("## What This Dashboard Does")
    st.markdown(
        """
The **Opportunity Analysis** page ranks HS92 4-digit products by combining two dimensions:

- **Feasibility**: how realistic it is for Argentina to compete now.
- **Attractiveness**: how valuable the opportunity is if Argentina expands in that product.

You can:
- Filter products by market size, RCA, sector, growth, density percentile, and Argentina export floor.
- Exclude specific HS4 products from the analysis.
- Reweight each component of feasibility and attractiveness.
- Rebalance overall strategy between feasibility and attractiveness.
- Explore a ranked product table and sector treemap.
"""
    )

    st.markdown("## How Scores Are Built")
    st.markdown(
        """
- Component variables are normalized (z-score) for score construction.
- **Feasibility Index** combines: transformed RCA, density, effective exporters, WNAI percentile.
- **Attractiveness Index** combines: PCI, COG, accessible market growth (5y), accessible market size share.
- **Combined Opportunity Score** rebalances feasibility and attractiveness by your strategic slider, then rescales to 0-1 for ranking.
"""
    )

    st.markdown("## Algebra and Interpretation")
    st.markdown(
        r"""
### Distance Travelled (by product)
\[
\mathrm{DistanceTravelled}_i
=
\sum_y
\left(
Distance_y \times \frac{M_{i,y}}{\sum_y M_{i,y}}
\right)
\]

- \(M_{i,y}\): imports of product \(i\) by destination \(y\).
- Interpretation: value-weighted average shipping distance of product \(i\). Larger destinations matter more.

### Accessible Market Size
\[
\mathrm{AccessibleMarket}_i
=
\sum_{y:\,Distance_y \le \mathrm{DistanceTravelled}_i} M_{i,y}
\]

- Uses Distance Travelled as the product-specific cutoff.
- Interpretation: total demand in destinations that are within the product's "reachable" distance profile.

### WNAI (Weighted Network Alignment Index)
\[
\mathrm{WNAI}_{z,i}
=
\sum_y
\left[
\left(
\frac{X_{z,y}/M_y}{X_z/WT}
\right)
\times
\left(
\frac{M_{i,y}}{M_y}
\right)
\times
GDPShare_y
\right]
\]

- \(z\): exporter (Argentina in this dashboard), \(i\): product, \(y\): partner market.
- First term: relative pipe thickness (where exporter over-indexes vs world average).
- Second term: product relevance in each partner's import basket.
- Third term: partner economic weight.
- Interpretation: higher WNAI means Argentina's strongest trade links are better aligned with large, product-relevant markets.
"""
    )

    st.markdown("## Variable Glossary (Main Page)")
    st.caption("Brief definitions plus interpretation guidance for the key technical variables.")
    glossary = pd.DataFrame(
        [
            {"Variable": "Raw RCA", "Brief Definition": "Revealed Comparative Advantage: Argentina's export share in a product divided by the world's export share in that product.", "How to Read It": "Greater than 1 = Argentina is relatively specialized; less than 1 = weaker specialization.", "Unit / Scale": "Ratio"},
            {"Variable": "Transformed RCA", "Brief Definition": "Min-max version of Raw RCA used for complexity calculations.", "How to Read It": "0 = lowest relative RCA in the sample, 1 = highest.", "Unit / Scale": "0-1"},
            {"Variable": "Density (Raw)", "Brief Definition": "Product-space proximity to Argentina's current capabilities.", "How to Read It": "Higher means the product is closer to what Argentina already knows how to export.", "Unit / Scale": "Continuous"},
            {"Variable": "Density Percentile", "Brief Definition": "Relative rank of density across products.", "How to Read It": "0.80 means denser than 80% of products in the sample.", "Unit / Scale": "0-1"},
            {"Variable": "Effective Exporters", "Brief Definition": "Effective number of competing exporters in that product (competition breadth).", "How to Read It": "Higher usually implies a broader competitive field.", "Unit / Scale": "Count-like index"},
            {"Variable": "WNAI Percentile", "Brief Definition": "Weighted Network Alignment Index percentile for Argentina versus major exporters in each product.", "How to Read It": "Higher percentile = Argentina's trade network is better aligned with high-value demand hubs.", "Unit / Scale": "0-100"},
            {"Variable": "WNAI Lead", "Brief Definition": "Argentina's WNAI percentile minus the median percentile of top competitors.", "How to Read It": "Positive = Argentina leads peers; negative = Argentina trails peers.", "Unit / Scale": "Percentile points"},
            {"Variable": "PCI", "Brief Definition": "Product Complexity Index: sophistication level of the product based on global export structures.", "How to Read It": "Higher often signals stronger long-term upgrading potential.", "Unit / Scale": "Continuous"},
            {"Variable": "COG", "Brief Definition": "Complexity Outlook Gain proxy: potential capability gain from moving into the product.", "How to Read It": "Higher suggests larger strategic learning/upgrading potential.", "Unit / Scale": "Continuous"},
            {"Variable": "Global Market Growth % (5y)", "Brief Definition": "5-year compound annual growth of world trade in the product.", "How to Read It": "Positive and higher values indicate faster-expanding global demand.", "Unit / Scale": "Percent per year"},
            {"Variable": "Country Export Growth % (5y)", "Brief Definition": "5-year compound annual growth of Argentina exports in the product.", "How to Read It": "Higher means Argentina is scaling faster in that product.", "Unit / Scale": "Percent per year"},
            {"Variable": "Country Current Exports (M USD)", "Brief Definition": "Argentina's export value in 2024 for the product.", "How to Read It": "Higher means a larger current export base.", "Unit / Scale": "Million USD"},
            {"Variable": "Country Exporter Rank (2024)", "Brief Definition": "Argentina's global rank among exporters of that product by value.", "How to Read It": "Lower rank number is better (e.g., 3 is better than 20).", "Unit / Scale": "Rank"},
            {"Variable": "Absolute Market Share Change (pp)", "Brief Definition": "Change in Argentina's world market share from 2020 to 2024.", "How to Read It": "Positive = Argentina gained share; negative = lost share.", "Unit / Scale": "Percentage points"},
            {"Variable": "Global Market Share", "Brief Definition": "Product's share in total world trade (2024).", "How to Read It": "Higher means the product is more important in global trade.", "Unit / Scale": "Percent"},
            {"Variable": "Total Trade (B USD)", "Brief Definition": "Total world trade value of the product in 2024.", "How to Read It": "Higher means a larger global market.", "Unit / Scale": "Billion USD"},
            {"Variable": "Accessible Market Size (B USD)", "Brief Definition": "Accessible demand proxy based on Argentina's network-positioned market reach.", "How to Read It": "Higher suggests more demand is realistically reachable.", "Unit / Scale": "Billion USD"},
            {"Variable": "Accessible-to-Market Ratio", "Brief Definition": "Accessible market size divided by total global market size.", "How to Read It": "Higher means a larger share of world demand appears structurally reachable.", "Unit / Scale": "Percent"},
            {"Variable": "Feasibility Index", "Brief Definition": "Composite score from transformed RCA, density, effective exporters, and WNAI percentile.", "How to Read It": "Higher means easier/less risky entry given current capabilities and network.", "Unit / Scale": "0-1"},
            {"Variable": "Attractiveness Index", "Brief Definition": "Composite score from PCI, COG, global growth, and accessible market size share.", "How to Read It": "Higher means stronger upside and strategic value.", "Unit / Scale": "0-1"},
            {"Variable": "Combined Opportunity Score", "Brief Definition": "Final score that blends feasibility and attractiveness using user-defined balance and weights.", "How to Read It": "Higher = better overall opportunity under current strategy settings.", "Unit / Scale": "0-1"},
        ]
    )
    st.dataframe(glossary, use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="Argentina Opportunities",
    page_icon=":bar_chart:",
    layout="wide",
)

pages = [
    st.Page(render_guide_and_glossary, title="Guide and glossary", icon=":material/menu_book:", default=True),
    st.Page(Path("pages/1_Opportunity_Analysis.py"), title="Opportunity Analysis", icon=":material/insights:"),
]
pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
