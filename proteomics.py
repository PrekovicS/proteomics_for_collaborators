import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------------------------------------------------------------------
# Utility — Save Matplotlib figure as SVG
# -----------------------------------------------------------------------------------------
def fig_to_svg(fig):
    buf = BytesIO()
    fig.savefig(buf, format="svg", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

# -----------------------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------------------

st.title("Proteomics Analysis Tool — Volcano Plots & Heatmaps")

page = st.sidebar.radio(
    "Navigation:",
    ["Volcano Plot (DE Results)", "Heatmap (Imputed Proteomics)"]
)

# =========================================================================================
# PAGE 1 — VOLCANO PLOTS
# =========================================================================================

if page == "Volcano Plot (DE Results)":

    st.header("Volcano Plot Generator")

    de_file = st.file_uploader("Upload DE results CSV", type=["csv"])

    if de_file:

        df = pd.read_csv(de_file)

        st.write("Columns detected:", list(df.columns))

        # Identify usable numeric p-value columns
        pval_candidates = [c for c in df.columns if "p" in c.lower()]
        if len(pval_candidates) == 0:
            st.error("No p-value columns found. Must contain 'p' in the column name.")
            st.stop()

        # User chooses p-value or adjusted p-value
        p_col = st.selectbox("Select p-value column:", pval_candidates)

        # Ensure p-value column numeric
        df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=[p_col])

        # Ensure logFC numeric
        if "logFC" not in df.columns:
            st.error("CSV must contain a 'logFC' column.")
            st.stop()

        df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
        df = df.dropna(subset=["logFC"])

        # Volcano settings
        st.subheader("Volcano Settings")

        logfc_thresh = st.number_input("logFC threshold:", 0.1, 10.0, 1.0)
        p_thresh = st.number_input("p-value threshold:", 1e-15, 0.1, 0.05)

        col_up = st.color_picker("Upregulated colour", "#d62728")
        col_down = st.color_picker("Downregulated colour", "#1f77b4")
        col_ns = st.color_picker("Non-significant colour", "#a8a8a8")

        point_size = st.slider("Point size", 10, 200, 40)
        width = st.slider("Figure width", 4, 20, 10)
        height = st.slider("Figure height", 4, 20, 6)

        label_top = st.selectbox("Label top significant genes:", 
                                 [0, 5, 10, 25, 50])

        gene_highlight = st.text_input("Highlight a specific gene:")

        # Compute -log10(p)
        df["neglog10p"] = -np.log10(df[p_col].replace(0, np.nan))

        # Significance classification
        df["sig"] = "NS"
        df.loc[(df["logFC"] > logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "UP"
        df.loc[(df["logFC"] < -logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "DOWN"

        color_map = {"UP": col_up, "DOWN": col_down, "NS": col_ns}

        # Plot
        fig, ax = plt.subplots(figsize=(width, height))

        for s in ["NS", "UP", "DOWN"]:
            sub = df[df["sig"] == s]
            ax.scatter(sub["logFC"], sub["neglog10p"],
                       c=color_map[s], s=point_size, alpha=0.8, label=s)

        # Highlight gene
        if gene_highlight and "Gene" in df.columns:
            gdf = df[df["Gene"].astype(str) == gene_highlight]
            if not gdf.empty:
                ax.scatter(gdf["logFC"], gdf["neglog10p"], 
                           c="yellow", edgecolor="black",
                           s=point_size*2, linewidth=1.5)

        # Label top N
        if label_top > 0:
            if "p_value" in df.columns:
                df_sort = df.sort_values(p_col).head(label_top)
            else:
                df_sort = df.sort_values("neglog10p", ascending=False).head(label_top)

            for _, row in df_sort.iterrows():
                if "Gene" in df.columns:
                    ax.text(row["logFC"], row["neglog10p"],
                            str(row["Gene"]), fontsize=9)

        ax.axhline(-np.log10(p_thresh), color="black", ls="--")
        ax.axvline(logfc_thresh, color="black", ls="--")
        ax.axvline(-logfc_thresh, color="black", ls="--")

        ax.set_xlabel("logFC")
        ax.set_ylabel(f"-log10({p_col})")
        ax.set_title("Volcano Plot")
        ax.legend()

        st.pyplot(fig)

        st.download_button(
            "Download SVG",
            data=fig_to_svg(fig),
            file_name="volcano.svg",
            mime="image/svg+xml"
        )

# =========================================================================================
# PAGE 2 — HEATMAPS
# =========================================================================================

if page == "Heatmap (Imputed Proteomics)":

    st.header("Heatmap Generator")

    prot_file = st.file_uploader("Upload proteomics_imputed_for_streamlit.csv", type=["csv"])

    if prot_file:

        df = pd.read_csv(prot_file)

        # Metadata + expression values
        meta = df.iloc[:, :3]
        expr = df.iloc[:, 3:]

        sample_names = list(expr.columns)

        st.subheader("Sample Selection")

        selected_samples = st.multiselect(
            "Select samples to include:",
            sample_names,
            default=sample_names
        )

        expr_sel = expr[selected_samples]

        # Replicate vs group-average
        use_group_avg = st.checkbox("Use group averages (collapse replicates)?", False)

        if use_group_avg:
            # infer group IDs from sample name prefix
            groups = [name.rsplit("_", 1)[0] for name in selected_samples]
            expr_sel = expr_sel.T.groupby(groups).mean().T

        st.subheader("Clustering & Colour")

        row_cluster = st.checkbox("Cluster rows", True)
        col_cluster = st.checkbox("Cluster columns", True)

        cluster_method = st.selectbox(
            "Clustering method",
            ["average", "complete", "ward", "single"]
        )

        cmap_choice = st.selectbox(
            "Viridis palette:",
            ["viridis", "plasma", "inferno", "magma", "cividis"]
        )

        width = st.slider("Width", 4, 20, 10)
        height = st.slider("Height", 4, 20, 12)

        # Draw heatmap
        fig = sns.clustermap(
            expr_sel,
            cmap=cmap_choice,
            method=cluster_method,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            figsize=(width, height)
        ).fig

        st.pyplot(fig)

        st.download_button(
            "Download Heatmap (SVG)",
            data=fig_to_svg(fig),
            file_name="heatmap.svg",
            mime="image/svg+xml"
        )
