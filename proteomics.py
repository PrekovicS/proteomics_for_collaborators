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
# PAGE 1 — VOLCANO PLOTS  (UNCHANGED)
# =========================================================================================

if page == "Volcano Plot (DE Results)":

    st.header("Volcano Plot Generator")

    de_file = st.file_uploader("Upload DE results CSV", type=["csv"])

    if de_file:

        df = pd.read_csv(de_file)
        st.write("Columns detected:", list(df.columns))

        pval_candidates = [c for c in df.columns if "p" in c.lower()]
        if not pval_candidates:
            st.error("No p-value columns found.")
            st.stop()

        p_col = st.selectbox("Select p-value column:", pval_candidates)

        df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=[p_col])

        if "logFC" not in df.columns:
            st.error("CSV must contain logFC.")
            st.stop()

        df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
        df = df.dropna(subset=["logFC"])

        st.subheader("Volcano Settings")

        logfc_thresh = st.number_input("logFC threshold:", 0.1, 10.0, 1.0)
        p_thresh = st.number_input("p-value threshold:", 1e-15, 0.1, 0.05)

        col_up   = st.color_picker("Upregulated colour", "#d62728")
        col_down = st.color_picker("Downregulated colour", "#1f77b4")
        col_ns   = st.color_picker("Non-significant colour", "#a8a8a8")

        point_size = st.slider("Point size", 10, 200, 40)
        width = st.slider("Figure width", 4, 20, 10)
        height = st.slider("Figure height", 4, 20, 6)

        label_top = st.selectbox("Label top significant genes:", [0, 5, 10, 25, 50])
        gene_highlight = st.text_input("Highlight a specific gene:")

        df["neglog10p"] = -np.log10(df[p_col].replace(0, np.nan))
        df["sig"] = "NS"
        df.loc[(df["logFC"] > logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "UP"
        df.loc[(df["logFC"] < -logfc_thresh) & (df[p_col] < p_thresh), "sig"] = "DOWN"

        color_map = {"UP": col_up, "DOWN": col_down, "NS": col_ns}

        fig, ax = plt.subplots(figsize=(width, height))

        for s in ["NS", "UP", "DOWN"]:
            sub = df[df["sig"] == s]
            ax.scatter(sub["logFC"], sub["neglog10p"],
                       c=color_map[s], s=point_size, alpha=0.8, label=s)

        if gene_highlight and "Gene" in df.columns:
            gdf = df[df["Gene"].astype(str).str.upper() == gene_highlight.upper()]
            if not gdf.empty:
                ax.scatter(gdf["logFC"], gdf["neglog10p"],
                           c="yellow", edgecolor="black",
                           s=point_size*2, linewidth=1.5)

        if label_top > 0:
            df_sort = df.sort_values(p_col).head(label_top)
            for _, row in df_sort.iterrows():
                if "Gene" in df.columns:
                    ax.text(row["logFC"], row["neglog10p"],
                            str(row["Gene"]), fontsize=9)

        ax.axhline(-np.log10(p_thresh), color="black", ls="--")
        ax.axvline(logfc_thresh,       color="black", ls="--")
        ax.axvline(-logfc_thresh,      color="black", ls="--")

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
# PAGE 2 — HEATMAP (UPDATED WITH MULTIGENE TEXT INPUT + GROUP SELECTION)
# =========================================================================================

if page == "Heatmap (Imputed Proteomics)":

    st.header("Heatmap Generator")

    prot_file = st.file_uploader("Upload proteomics_imputed_for_streamlit.csv", type=["csv"])

    if prot_file:

        df = pd.read_csv(prot_file)
        meta = df.iloc[:, :3]
        expr = df.iloc[:, 3:]

        # -----------------------------------------------------
        # ⭐ MULTIGENE TEXT INPUT
        # -----------------------------------------------------
        st.subheader("Gene Selection")

        gene_text = st.text_area(
            "Enter one or more gene names (comma or newline separated). Leave empty to plot ALL genes:",
            ""
        )

        gene_text = gene_text.strip()

        if gene_text:
            gene_list = [
                g.strip().upper() for g in
                gene_text.replace(",", "\n").split("\n") 
                if g.strip() != ""
            ]
            gene_list = list(set(gene_list))
            df["Gene_upper"] = df["Gene_name"].str.upper()

            df = df[df["Gene_upper"].isin(gene_list)]
            expr = expr.loc[df.index]
        else:
            df["Gene_upper"] = df["Gene_name"].str.upper()

        expr.index = df["Gene_name"].values  # row labels

        # -----------------------------------------------------
        # Sample / Condition selection
        # -----------------------------------------------------
        st.subheader("Samples or Conditions")

        sample_names = list(expr.columns)

        use_group_avg = st.checkbox("Use group averages instead of replicates?", False)

        if use_group_avg:

            condition_ids = sorted({name.rsplit("_",1)[0] for name in sample_names})

            selected_conditions = st.multiselect(
                "Select conditions to include:",
                condition_ids,
                default=condition_ids
            )

            df_groups = {}
            for cond in selected_conditions:
                reps = [c for c in sample_names if c.startswith(cond)]
                df_groups[cond] = expr[reps].mean(axis=1)

            expr_sel = pd.DataFrame(df_groups)

        else:
            selected_samples = st.multiselect(
                "Select samples to include:",
                sample_names,
                default=sample_names
            )
            expr_sel = expr[selected_samples]


        # -----------------------------------------------------
        # Scaling
        # -----------------------------------------------------
        st.subheader("Scaling")

        scale_mode = st.selectbox("Scale:", ["None", "Row", "Column"])

        if scale_mode == "Row":
            expr_scaled = (expr_sel - expr_sel.mean(axis=1).values[:,None]) / expr_sel.std(axis=1).values[:,None]
        elif scale_mode == "Column":
            expr_scaled = (expr_sel - expr_sel.mean()) / expr_sel.std()
        else:
            expr_scaled = expr_sel.copy()

        # -----------------------------------------------------
        # Clustering
        # -----------------------------------------------------
        st.subheader("Clustering")

        row_cluster = st.checkbox("Cluster rows?", True)
        col_cluster = st.checkbox("Cluster columns?", True)
        cluster_method = st.selectbox("Clustering method:",
                                      ["average", "complete", "ward", "single"])

        cmap_choice = st.selectbox("Viridis palette:",
                                   ["viridis", "plasma", "inferno", "magma", "cividis"])

        width = st.slider("Heatmap Width", 4, 20, 10)
        height = st.slider("Heatmap Height", 4, 20, 12)

        # -----------------------------------------------------
        # Draw heatmap
        # -----------------------------------------------------
        g = sns.clustermap(
            expr_scaled,
            cmap=cmap_choice,
            method=cluster_method,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            figsize=(width, height)
        )

        st.pyplot(g.fig)

        st.download_button(
            "Download Heatmap (SVG)",
            data=fig_to_svg(g.fig),
            file_name="heatmap.svg",
            mime="image/svg+xml"
        )
