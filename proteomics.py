import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def impute_min(df_numeric):
    """Replace NA with per-column minimum."""
    return df_numeric.apply(lambda x: x.fillna(x.min()), axis=0)

def impute_bottom5(df_numeric):
    """Impute NA with random draws from bottom 5% of global values."""
    low = df_numeric.stack().quantile(0.05)
    low_global_min = df_numeric.min().min()
    return df_numeric.apply(
        lambda x: x.fillna(np.random.uniform(low_global_min, low)), axis=0
    )

def download_svg(fig):
    """Return a vector SVG file for download."""
    buffer = BytesIO()
    fig.savefig(buffer, format="svg", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer

def differential_analysis(df_imp, df_meta, cond1, cond2):
    """Compute logFC and p-values comparing cond2 vs cond1."""

    group1 = df_imp.loc[:, df_imp.columns.str.contains(cond1)]
    group2 = df_imp.loc[:, df_imp.columns.str.contains(cond2)]

    pvals = []
    logfc = []

    for i in range(len(df_imp)):
        x = group1.iloc[i].dropna()
        y = group2.iloc[i].dropna()
        if len(x) > 1 and len(y) > 1:
            p = ttest_ind(x, y, equal_var=False).pvalue
        else:
            p = np.nan
        pvals.append(p)
        logfc.append(np.mean(y) - np.mean(x))

    out = df_meta.copy()
    out["logFC"] = logfc
    out["pvalue"] = pvals
    return out


# -----------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------

st.title("Proteomics Explorer — USP8 / CDAN1 response in OE vs ecDNA")

uploaded = st.file_uploader("Upload proteomics matrix (.xlsx)", type=["xlsx"])

if uploaded is not None:

    # -----------------------------
    # Load & preprocess
    # -----------------------------
    df = pd.read_excel(uploaded)
    metadata_cols = 4

    # Convert to numeric safely
    df_numeric = df.iloc[:, metadata_cols:].apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )

    # Keep metadata separately
    df_meta = df.iloc[:, :metadata_cols]

    # ALWAYS define df_imp (fixes NameErrors)
    df_imp = df_numeric.copy()

    # ==============================
    # STEP 1: IMPUTATION
    # ==============================
    st.subheader("Step 1 — Imputation")

    imp_method = st.selectbox("Imputation method:", ["Minimum", "Random (bottom 5%)"])

    if st.button("Apply imputation"):
        if imp_method == "Minimum":
            df_imp = impute_min(df_numeric)
        else:
            df_imp = impute_bottom5(df_numeric)
        st.success("Imputation applied.")

    # ==============================
    # STEP 2: PCA
    # ==============================
    st.subheader("Step 2 — PCA")

    n_genes = st.number_input(
        "Number of top variable proteins:",
        min_value=50,
        max_value=3000,
        value=500,
        step=50,
    )

    if st.button("Run PCA"):
        variances = df_imp.var(axis=1)
        selected = df_imp.loc[variances.sort_values(ascending=False).index[:n_genes]]

        X = selected.transpose()
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pcs[:, 0], pcs[:, 1], s=40)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA (top variable proteins)")

        st.pyplot(fig)
        st.download_button("Download PCA (SVG)", download_svg(fig), "pca.svg")

    # ==============================
    # STEP 3: DIFF ANALYSIS
    # ==============================
    st.subheader("Step 3 — Differential Expression")

    all_cols = df_numeric.columns
    all_conditions = sorted(
        set([" ".join(col.replace("2Log Abundance", "").strip().split("_")[:-1])
             for col in all_cols])
    )

    cond1 = st.selectbox("Baseline (NT)", all_conditions)
    cond2 = st.selectbox("Compare", all_conditions)

    if st.button("Run differential analysis"):
        de = differential_analysis(df_imp, df_meta, cond1, cond2)

        st.write(de.head())

        # Volcano parameters
        st.subheader("Volcano Plot")
        p_thresh = st.number_input("p-value threshold", 0.0001, 0.1, 0.05)
        logfc_thresh = st.number_input("logFC threshold", 0.1, 5.0, 1.0)

        col_up = st.color_picker("UP-regulated colour", "#d62728")
        col_dn = st.color_picker("DOWN-regulated colour", "#1f77b4")
        col_ns = st.color_picker("Non-significant colour", "#bdbdbd")

        # Prepare volcano
        de["-log10p"] = -np.log10(de["pvalue"])

        is_up = (de["pvalue"] < p_thresh) & (de["logFC"] > logfc_thresh)
        is_dn = (de["pvalue"] < p_thresh) & (de["logFC"] < -logfc_thresh)

        fig2, ax2 = plt.subplots(figsize=(6, 5))

        ax2.scatter(de.loc[~(is_up | is_dn), "logFC"],
                    de.loc[~(is_up | is_dn), "-log10p"],
                    s=20, color=col_ns)

        ax2.scatter(de.loc[is_up, "logFC"],
                    de.loc[is_up, "-log10p"],
                    s=30, color=col_up)

        ax2.scatter(de.loc[is_dn, "logFC"],
                    de.loc[is_dn, "-log10p"],
                    s=30, color=col_dn)

        ax2.axvline(logfc_thresh, color="black", linestyle="--")
        ax2.axvline(-logfc_thresh, color="black", linestyle="--")
        ax2.axhline(-np.log10(p_thresh), color="black", linestyle="--")

        ax2.set_xlabel("logFC")
        ax2.set_ylabel("-log10(p)")
        ax2.set_title(f"{cond2} vs {cond1}")

        st.pyplot(fig2)
        st.download_button("Download Volcano (SVG)", download_svg(fig2), "volcano.svg")

    # ==============================
    # STEP 4: HEATMAP
    # ==============================
    st.subheader("Step 4 — Heatmap")

    cmap = st.selectbox("Colour map", ["viridis", "plasma", "magma", "inferno", "cividis"])
    cluster = st.selectbox("Clustering", ["both", "rows", "columns", "none"])
    n_hm = st.number_input("Top variable proteins:", 50, 2000, 250, 50)

    if st.button("Generate heatmap"):
        variances = df_imp.var(axis=1)
        selected = df_imp.loc[variances.sort_values(ascending=False).index[:n_hm]]

        fig3 = plt.figure(figsize=(10, 8))

        if cluster == "none":
            cg = sns.clustermap(selected, cmap=cmap, row_cluster=False, col_cluster=False)
        elif cluster == "rows":
            cg = sns.clustermap(selected, cmap=cmap, row_cluster=True, col_cluster=False)
        elif cluster == "columns":
            cg = sns.clustermap(selected, cmap=cmap, row_cluster=False, col_cluster=True)
        else:
            cg = sns.clustermap(selected, cmap=cmap)

        st.pyplot(cg.fig)
        st.download_button("Download Heatmap (SVG)", download_svg(cg.fig), "heatmap.svg")
