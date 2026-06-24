from pathlib import Path

import pandas as pd
import streamlit as st


def render_guide_and_glossary() -> None:
    st.title("Argentina Export Opportunities Dashboard")
    st.caption("Guide and glossary for the BACI-based Argentina opportunity dashboard (HS92 4-digit).")

    st.markdown("## What This Dashboard Does")
    st.markdown(
        """
The dashboard ranks HS92 4-digit products by combining two dimensions:

- **Feasibility**: how realistic it is for Argentina to compete now.
- **Attractiveness**: how valuable the opportunity is if Argentina expands in that product.

V2 uses CEPII BACI HS92 trade flows for commercial calculations, while product
names and sectors still come from the HS92 reference tables.

You can:
- Filter products by market size, RCA, sector, growth, density percentile, and Argentina export floor.
- Exclude specific HS4 products from the analysis.
- Reweight each component of feasibility and attractiveness.
- Rebalance overall strategy between feasibility and attractiveness.
- Explore a ranked product table, sector treemap, anchored proximity network, and comparison view.
"""
    )

    st.markdown("## How Scores Are Built")
    st.markdown(
        """
- Component variables are normalized (z-score) for score construction.
- **Feasibility Index** combines: transformed RCA, density, effective exporters, DAI percentile.
- **Attractiveness Index** combines: PCI, COG, accessible market growth (5y), accessible market size share.
- **Combined Opportunity Score** rebalances feasibility and attractiveness by your strategic slider, then rescales to 0-1 for ranking.
"""
    )

    st.markdown("## Variable Glossary (Main Page)")
    st.caption("Brief definitions plus interpretation guidance for the key technical variables.")
    glossary = pd.DataFrame(
        [
            {"Variable": "Raw RCA", "Brief Definition": "Revealed Comparative Advantage: Argentina's export share in a product divided by the world's export share in that product.", "How to Read It": "Greater than 1 = Argentina is relatively specialized; less than 1 = weaker specialization.", "Unit / Scale": "Ratio"},
            {"Variable": "Transformed RCA", "Brief Definition": "Bounded RCA transformation used for complexity calculations: RCA / (RCA + 1).", "How to Read It": "Closer to 1 means stronger relative specialization; closer to 0 means weaker specialization.", "Unit / Scale": "0-1"},
            {"Variable": "Density (Raw)", "Brief Definition": "Product-space proximity to Argentina's current capabilities.", "How to Read It": "Higher means the product is closer to what Argentina already knows how to export.", "Unit / Scale": "Continuous"},
            {"Variable": "Density Percentile", "Brief Definition": "Product-specific 2024 percentile of Argentina's density compared with other countries for the same HS4 product.", "How to Read It": "0.80 means Argentina's density is higher than roughly 80% of countries for that product.", "Unit / Scale": "0-1"},
            {"Variable": "Distance Travelled", "Brief Definition": "Weighted average bilateral distance travelled by product, using bilateral export values between origin x and destination y as weights.", "How to Read It": "Higher means exports of that product are concentrated in farther destination markets.", "Unit / Scale": "Distance units (from bilateral distance file)"},
            {"Variable": "Effective Exporters", "Brief Definition": "Effective number of competing exporters in that product (competition breadth).", "How to Read It": "Higher usually implies a broader competitive field.", "Unit / Scale": "Count-like index"},
            {"Variable": "DAI Percentile", "Brief Definition": "Demand Alignment Index percentile for Argentina versus major exporters in each product.", "How to Read It": "Higher percentile = Argentina's export network is better aligned with countries whose import baskets are relatively important for that product.", "Unit / Scale": "0-100"},
            {"Variable": "DAI Lead", "Brief Definition": "Argentina's DAI percentile minus the median percentile of top competitors.", "How to Read It": "Positive = Argentina leads peers; negative = Argentina trails peers.", "Unit / Scale": "Percentile points"},
            {"Variable": "PCI", "Brief Definition": "Product Complexity Index: sophistication level of the product based on global export structures.", "How to Read It": "Higher often signals stronger long-term upgrading potential.", "Unit / Scale": "Continuous"},
            {"Variable": "COG", "Brief Definition": "Complexity Outlook Gain proxy: potential capability gain from moving into the product.", "How to Read It": "Higher suggests larger strategic learning/upgrading potential.", "Unit / Scale": "Continuous"},
            {"Variable": "Global Market Growth % (5y)", "Brief Definition": "5-year compound annual growth of world trade in the product.", "How to Read It": "Positive and higher values indicate faster-expanding global demand.", "Unit / Scale": "Percent per year"},
            {"Variable": "Country Export Growth % (5y)", "Brief Definition": "5-year compound annual growth of Argentina exports in the product.", "How to Read It": "Higher means Argentina is scaling faster in that product.", "Unit / Scale": "Percent per year"},
            {"Variable": "Country Current Exports (M USD)", "Brief Definition": "Argentina's export value in 2024 for the product.", "How to Read It": "Higher means a larger current export base.", "Unit / Scale": "Million USD"},
            {"Variable": "Country Exporter Rank (2024)", "Brief Definition": "Argentina's global rank among exporters of that product by value.", "How to Read It": "Lower rank number is better (e.g., 3 is better than 20).", "Unit / Scale": "Rank"},
            {"Variable": "Absolute Market Share Change (pp)", "Brief Definition": "Change in Argentina's world market share from 2020 to 2024.", "How to Read It": "Positive = Argentina gained share; negative = lost share.", "Unit / Scale": "Percentage points"},
            {"Variable": "Global Market Share", "Brief Definition": "Product's share in total world trade (2024).", "How to Read It": "Higher means the product is more important in global trade.", "Unit / Scale": "Percent"},
            {"Variable": "Total Trade (B USD)", "Brief Definition": "Total world trade value of the product in 2024.", "How to Read It": "Higher means a larger global market.", "Unit / Scale": "Billion USD"},
            {"Variable": "Accessible Market Size (B USD)", "Brief Definition": "Import demand in destinations considered accessible for each product: destinations within the product's travelled-distance threshold, plus destinations where Argentina already exports at least USD 100M in that product.", "How to Read It": "Higher suggests more demand is realistically reachable under the product-specific access rule.", "Unit / Scale": "Billion USD"},
            {"Variable": "Accessible-to-Market Ratio", "Brief Definition": "Accessible market size divided by total global market size.", "How to Read It": "Higher means a larger share of world demand appears structurally reachable.", "Unit / Scale": "Percent"},
            {"Variable": "Feasibility Index", "Brief Definition": "Composite score from transformed RCA, density, effective exporters, and DAI percentile.", "How to Read It": "Higher means easier/less risky entry given current capabilities and demand alignment.", "Unit / Scale": "0-1"},
            {"Variable": "Attractiveness Index", "Brief Definition": "Composite score from PCI, COG, global growth, and accessible market size share.", "How to Read It": "Higher means stronger upside and strategic value.", "Unit / Scale": "0-1"},
            {"Variable": "Combined Opportunity Score", "Brief Definition": "Final score that blends feasibility and attractiveness using user-defined balance and weights.", "How to Read It": "Higher = better overall opportunity under current strategy settings.", "Unit / Scale": "0-1"},
        ]
    )
    st.dataframe(glossary, width="stretch", hide_index=True)

    st.markdown("## Algebra and Interpretation")
    st.markdown("### Distance Travelled (by product)")
    st.latex(r"\mathrm{DistanceTravelled}_i = \sum_y \left( Distance_{x,y} \times \frac{X_{x,y,i}}{\sum_y X_{x,y,i}} \right)")
    st.markdown("- `X_{x,y,i}`: bilateral exports of product `i` from origin `x` to destination `y`.")
    st.markdown("- Interpretation: weighted average distance travelled by product, where bilateral export value is the weight.")

    st.markdown("### Accessible Market Size")
    st.latex(r"\mathrm{AccessibleMarket}_{i,t} = \sum_y M_{i,y,t} \cdot \mathbf{1}\{Distance_y \le \mathrm{DistanceTravelled}_i \;\lor\; X_{\mathrm{ARG},y,i,t} \ge 100M\}")
    st.markdown("- `M_{i,y,t}`: imports of product `i` by destination `y` in year `t`.")
    st.markdown("- `X_{ARG,y,i,t}`: Argentina exports of product `i` to destination `y` in year `t`.")
    st.markdown("- Interpretation: total import demand in destinations reachable by the product's distance profile, plus destinations where Argentina already has a large product-specific export presence.")

    st.markdown("### DAI (Demand Alignment Index)")
    st.latex(
        r"\mathrm{DAI}_{z,i} = \sum_y \left[ \left( \frac{X_{z,y}/X_z}{M_y/WT} \right) \times \left( \frac{M_{i,y}}{\sum_y M_{i,y}} \right) \right]"
    )
    st.markdown("- `z`: exporter (Argentina in this dashboard), `i`: product, `y`: partner market.")
    st.markdown("- First term: bilateral demand alignment, comparing Argentina's export orientation toward partner `y` with `y`'s weight in world imports.")
    st.markdown("- Second term: partner `y`'s share of world imports for product `i`.")
    st.markdown("- Interpretation: higher DAI means Argentina's export network is better aligned with product-relevant demand markets.")


st.set_page_config(
    page_title="Argentina Opportunities",
    page_icon=":bar_chart:",
    layout="wide",
)

pages = [
    st.Page(render_guide_and_glossary, title="Guide and glossary", icon=":material/menu_book:", default=True),
    st.Page(Path("pages/1_Opportunity_Analysis.py"), title="Opportunity Analysis", icon=":material/insights:"),
    st.Page(Path("pages/3_Anchored_Proximity_Analysis.py"), title="Anchored Proximity Analysis", icon=":material/hub:"),
    st.Page(Path("pages/4_Comparison.py"), title="Comparison", icon=":material/compare_arrows:"),
]
pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
