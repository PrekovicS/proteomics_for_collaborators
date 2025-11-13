import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
import plotly.express as px
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import base64

# -----------------------------------------------------------------------------------------
# Utility - save matplotlib figure as SVG
# -----------------------------------------------------------------------------------------
def fig_to_svg(fig):
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf

# -----------------------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------------------

st.title("Proteomics Analysis Tool — DE + Heatmaps")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose page:", ["Volcano Plot (DE Results)", 
                                         "Heatmap (Imputed Proteomics)"])

# =========================================================================================
# PAGE 1 — VOLCANO PLOTS
# =========================================================================================

if page == "Volcano Plot (DE Results)":
    st.header("Volcano Plot Generator")

    de_file = st.file_uploader("Upload DE results CSV", type=["csv"])

    if de_file:
        df = pd.read_csv(de_file)

        # Ensure column names exist
        if not {"logFC", "p_value"}.issubset(df.columns):
            st.error("CSV must contain at least: logFC, p_value")
        else:
            # Select p-value or adjusted
            p_col = st.selectbox("Select p-value column:", 
                                 [c for c in df.columns if "p" in c.lower()])

            # Volcano customization
            st.subheader("Volcano Controls")
            logfc_thresh = st.number_input("logFC threshold", 0.1, 5.0, 1.0)
            p_thresh = st.number_input("p-value threshold", 1e-10, 0.1, 0.05)

            col_up = st.color_picker("Colour (Upregulated)", "#d62728")
            col_down = st.color_picker("Colour (Downregulated)", "#1f77b4")
            col_ns = st.color_picker("Colour (Not significant)", "#a8a8a8")

            point_size = st.slider("Point size", 10, 200, 50)
            width = st.slider("Figure width", 4, 16, 8)
            height = st.slider("Figure height", 4, 16, 6)

            # Gene labeling
            top_n = st.selectbox("Label top significant genes:", [0, 5, 10, 25, 50])
            gene_to_highlight = st.text_input("Highlight a specific gene (optional):")

            df["neglog10p"] = -np.log10(df[p_col])
            df["sig"] = "NS"
            df.loc[(df["logFC"] > logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "UP"
            df.loc[(df["logFC"] < -logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "DOWN"

            color_map = {
                "UP": col_up,
                "DOWN": col_down,
                "NS": col_ns
            }

            fig, ax = plt.subplots(figsize=(width, height))

            for group in ["NS", "UP", "DOWN"]:
                temp = df[df["sig"] == group]
                ax.scatter(temp["logFC"], temp["neglog10p"],
                           c=color_map[group], s=point_size, label=group, alpha=0.8)

            # highlight gene
            if gene_to_highlight:
                gdf = df[df["Gene"] == gene_to_highlight]
                if not gdf.empty:
                    ax.scatter(gdf["logFC"], gdf["neglog10p"],
                               c="yellow", s=point_size*2,
                               edgecolor="black", linewidth=1)

            # label top N
            if top_n > 0:
                top_genes = df.nsmallest(top_n, p_col)
                for _, row in top_genes.iterrows():
                    ax.text(row["logFC"], row["neglog10p"], row["Gene"], fontsize=9)

            ax.axhline(-np.log10(p_thresh), linestyle="--", color="black")
            ax.axvline(logfc_thresh, linestyle="--", color="black")
            ax.axvline(-logfc_thresh, linestyle="--", color="black")

            ax.set_xlabel("logFC")
            ax.set_ylabel(f"-log10({p_col})")
            ax.set_title("Volcano Plot")
            ax.legend()

            st.pyplot(fig)
            st.download_button("Download SVG", data=fig_to_svg(fig),
                               file_name="volcano.svg", mime="image/svg+xml")

# =========================================================================================
# PAGE 2 — HEATMAPS
# =========================================================================================

if page == "Heatmap (Imputed Proteomics)":
    st.header("Heatmap Generator")

    prot_file = st.file_uploader("Upload proteomics_imputed_for_streamlit.csv", type=["csv"])

    if prot_file:
        df = pd.read_csv(prot_file)

        # Extract metadata + expression matrix
        meta = df.iloc[:, :3]
        expr = df.iloc[:, 3:].copy()

        sample_names = list(expr.columns)

        st.subheader("Heatmap Options")

        # Select samples
        selected_samples = st.multiselect(
            "Select samples to include:",
            sample_names,
            default=sample_names
        )

        expr_sel = expr[selected_samples]

        # Cluster options
        cluster_rows = st.checkbox("Cluster rows", True)
        cluster_cols = st.checkbox("Cluster columns", True)

        cluster_method = st.selectbox("Clustering method:",
                                      ["average", "ward", "complete", "single"])

        # Viridis options
        cmap_choice = st.selectbox("Colour map (viridis family):",
                                   ["viridis", "plasma", "inferno", "magma", "cividis"])

        # Use group averages instead of replicates
        group_average = st.checkbox("Plot group averages instead of replicates?", False)

        if group_average:
            # infer groups from names
            groups = [s.rsplit("_", 1)[0] for s in selected_samples]
            df_groups = pd.DataFrame(expr_sel.T.groupby(groups).mean().T)
            expr_heat = df_groups
        else:
            expr_heat = expr_sel

        width = st.slider("Figure width", 4, 20, 10)
        height = st.slider("Figure height", 4, 20, 12)

        fig, ax = plt.subplots(figsize=(width, height))
        sns.clustermap(
            expr_heat,
            cmap=cmap_choice,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            method=cluster_method
        )

        st.pyplot()

        st.download_button("Download Heatmap (SVG)",
                           data=fig_to_svg(fig),
                           file_name="heatmap.svg",
                           mime="image/svg+xml")
