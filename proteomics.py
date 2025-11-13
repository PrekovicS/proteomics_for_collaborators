import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

def fig_to_svg(fig):
    buf = BytesIO()
    fig.savefig(buf, format="svg", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

st.title("Proteomics Analysis Tool — Volcano Plots & Heatmaps")

page = st.sidebar.radio(
    "Navigation:",
    ["Volcano Plot (DE Results)", "Heatmap (Imputed Proteomics)"]
)

# =========================================================================================
# VOLCANO PLOT PAGE
# =========================================================================================

if page == "Volcano Plot (DE Results)":

    st.header("Volcano Plot Generator")
    de_file = st.file_uploader("Upload DE results CSV", type=["csv"])

    if de_file:
        df = pd.read_csv(de_file)
        st.write("Columns detected:", list(df.columns))

        pval_candidates = [c for c in df.columns if "p" in c.lower()]
        p_col = st.selectbox("Select p-value column:", pval_candidates)

        df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=[p_col])

        df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
        df = df.dropna(subset=["logFC"])

        logfc_thresh = st.number_input("logFC threshold:", 0.1, 10.0, 1.0)
        p_thresh = st.number_input("p-value threshold:", 1e-15, 0.1, 0.05)

        col_up   = st.color_picker("Upregulated colour", "#d62728")
        col_down = st.color_picker("Downregulated colour", "#1f77b4")
        col_ns   = st.color_picker("Non-significant colour", "#a8a8a8")

        point_size = st.slider("Point size", 10, 200, 40)
        width = st.slider("Width", 4, 20, 10)
        height = st.slider("Height", 4, 20, 6)

        label_top = st.selectbox("Label top significant genes:", [0, 5, 10, 25, 50])
        gene_highlight = st.text_input("Highlight a specific gene (optional):")

        df["neglog10p"] = -np.log10(df[p_col].replace(0, np.nan))

        df["sig"] = "NS"
        df.loc[(df["logFC"] > logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "UP"
        df.loc[(df["logFC"] < -logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "DOWN"

        color_map = {"UP": col_up, "DOWN": col_down, "NS": col_ns}

        fig, ax = plt.subplots(figsize=(width, height))

        for s in ["NS", "UP", "DOWN"]:
            sub = df[df["sig"] == s]
            ax.scatter(sub["logFC"], sub["neglog10p"], c=color_map[s], s=point_size, alpha=0.8)

        if gene_highlight and "Gene" in df.columns:
            gdf = df[df["Gene"].astype(str).str.upper() == gene_highlight.upper()]
            if len(gdf) > 0:
                ax.scatter(gdf["logFC"], gdf["neglog10p"],
                           c="yellow", edgecolor="black", s=point_size*2, linewidth=1.5)

        if label_top > 0 and "Gene" in df.columns:
            df_sort = df.sort_values(p_col).head(label_top)
            for _, row in df_sort.iterrows():
                ax.text(row["logFC"], row["neglog10p"], str(row["Gene"]), fontsize=9)

        ax.axhline(-np.log10(p_thresh), color="black", ls="--")
        ax.axvline(logfc_thresh,       color="black", ls="--")
        ax.axvline(-logfc_thresh,      color="black", ls="--")

        ax.set_xlabel("logFC")
        ax.set_ylabel(f"-log10({p_col})")
        ax.set_title("Volcano Plot")

        st.pyplot(fig)

        st.download_button(
            "Download SVG",
            data=fig_to_svg(fig),
            file_name="volcano.svg",
            mime="image/svg+xml"
        )


# =========================================================================================
# HEATMAP PAGE (ROBUST, NO CRASHES, PLOT ONLY FOUND GENES)
# =========================================================================================

if page == "Heatmap (Imputed Proteomics)":

    st.header("Heatmap Generator")
    prot_file = st.file_uploader("Upload proteomics_imputed_for_streamlit.csv", type=["csv"])

    if prot_file:
        df = pd.read_csv(prot_file)

        meta = df.iloc[:, :3]
        expr = df.iloc[:, 3:].copy()  # all abundance columns

        # ---------------------------------------------------------
        # GENE INPUT
        # ---------------------------------------------------------
        st.subheader("Gene Selection")
        gene_text = st.text_area(
            "Enter genes (comma, space, newline separated):",
            placeholder="FOXA2, HNF4A, ABCB1"
        ).strip()

        if gene_text:
            gene_list = [
                g.strip().upper()
                for g in gene_text.replace(",", "\n").split("\n")
                if g.strip() != ""
            ]
            gene_list = list(set(gene_list))
        else:
            gene_list = []

        df["__gene_upper__"] = df["Gene_name"].str.upper()

        df_found = df[df["__gene_upper__"].isin(gene_list)]

        if len(df_found) == 0 and len(gene_list) > 0:
            st.warning("None of the entered genes were found. Showing nothing.")
            st.stop()

        # If user entered nothing → show full matrix
        if len(gene_list) == 0:
            df_found = df.copy()

        expr_sel = expr.loc[df_found.index]
        expr_sel.index = df_found["Gene_name"].values

        # ---------------------------------------------------------
        # SAMPLE / CONDITION SELECTION
        # ---------------------------------------------------------
        st.subheader("Samples or Conditions")
        sample_names = list(expr_sel.columns)

        selected_samples = st.multiselect(
            "Select samples:",
            sample_names,
            default=sample_names
        )
        expr_sel = expr_sel[selected_samples]

        # ---------------------------------------------------------
        # SCALING (SAFE)
        # ---------------------------------------------------------
        st.subheader("Scaling")
        scale_mode = st.selectbox("Scale:", ["None", "Row", "Column"])

        expr_scaled = expr_sel.copy()

        if scale_mode == "Row":
            means = expr_scaled.mean(axis=1)
            stds = expr_scaled.std(axis=1).replace(0, np.nan)
            expr_scaled = expr_scaled.sub(means, axis=0).div(stds, axis=0)
        elif scale_mode == "Column":
            means = expr_scaled.mean(axis=0)
            stds = expr_scaled.std(axis=0).replace(0, np.nan)
            expr_scaled = expr_scaled.sub(means).div(stds)

        expr_scaled = expr_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

        # ---------------------------------------------------------
        # SAFE CHECKS
        # ---------------------------------------------------------
        if expr_scaled.shape[0] == 0 or expr_scaled.shape[1] == 0:
            st.error("No valid data left to plot.")
            st.stop()

        # Avoid zero-range crash
        if np.all(expr_scaled.to_numpy() == 0):
            vmin, vmax = -1, 1
        else:
            vmin = np.nanmin(expr_scaled.to_numpy())
            vmax = np.nanmax(expr_scaled.to_numpy())
            if vmin == vmax:
                vmin -= 1
                vmax += 1

        # ---------------------------------------------------------
        # CLUSTERING
        # ---------------------------------------------------------
        st.subheader("Clustering")
        row_cluster = st.checkbox("Cluster rows?", True)
        col_cluster = st.checkbox("Cluster columns?", True)
        cluster_method = st.selectbox("Clustering method:",
                                      ["average", "complete", "ward", "single"])
        cmap_choice = st.selectbox("Colormap:", ["viridis", "plasma", "inferno", "magma", "cividis"])
        width = st.slider("Heatmap Width", 4, 20, 10)
        height = st.slider("Heatmap Height", 4, 20, 12)

        safe_row = row_cluster and expr_scaled.shape[0] > 1
        safe_col = col_cluster and expr_scaled.shape[1] > 1

        # ---------------------------------------------------------
        # HEATMAP (NEVER CRASHES)
        # ---------------------------------------------------------
        g = sns.clustermap(
            expr_scaled,
            cmap=cmap_choice,
            method=cluster_method,
            row_cluster=safe_row,
            col_cluster=safe_col,
            figsize=(width, height),
            vmin=vmin,
            vmax=vmax
        )

        st.pyplot(g.fig)

        st.download_button(
            "Download Heatmap (SVG)",
            data=fig_to_svg(g.fig),
            file_name="heatmap.svg",
            mime="image/svg+xml"
        )
