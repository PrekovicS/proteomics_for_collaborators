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
    """Replace NA with minimum value (per column)."""
    return df_numeric.apply(lambda x: x.fillna(x.min()), axis=0)

def impute_bottom5(df_numeric):
    """Impute NA with random draws from bottom 5% of distribution."""
    bottom_vals = df_numeric.stack().quantile(0.05)
    return df_numeric.apply(lambda x: x.fillna(np.random.uniform(df_numeric.min().min(), bottom_vals)), axis=0)

def download_svg(fig):
    """Return a vector graphic for download."""
    output = BytesIO()
    fig.savefig(output, format="svg", dpi=300, bbox_inches="tight")
    output.seek(0)
    return output

def differential_analysis(df, cond1, cond2, metadata_cols=4):
    """Compute p-values + logFC for cond2 vs cond1."""
    group1 = df.loc[:, df.columns.str.contains(cond1)]
    group2 = df.loc[:, df.columns.str.contains(cond2)]
    
    pvals = []
    logfc = []

    for i in range(len(df)):
        x = group1.iloc[i].dropna()
        y = group2.iloc[i].dropna()
        if len(x) > 1 and len(y) > 1:
            p = ttest_ind(x, y, equal_var=False).pvalue
        else:
            p = np.nan
        pvals.append(p)
        logfc.append(np.mean(y) - np.mean(x))

    out = df.iloc[:, :metadata_cols].copy()
    out["logFC"] = logfc
    out["pvalue"] = pvals
    return out


# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------

st.title("Proteomics Explorer — USP8 / CDAN1 response across ecDNA vs OE backgrounds")

uploaded = st.file_uploader("Upload proteomics matrix (.xlsx)", type=["xlsx"])

if uploaded is not None:
    df = pd.read_excel(uploaded)
    metadata_cols = 4
    df_numeric = df.iloc[:, metadata_cols:].apply(
    lambda col: pd.to_numeric(col, errors="coerce")
)

    st.subheader("Step 1 — Imputation")
    imp_method = st.selectbox("Imputation method:", ["Minimum", "Random from bottom 5%"])

    if st.button("Apply imputation"):
        if imp_method == "Minimum":
            df_imp = impute_min(df_numeric)
        else:
            df_imp = impute_bottom5(df_numeric)

        st.success("Imputation complete.")
        df_final = pd.concat([df.iloc[:, :metadata_cols], df_imp], axis=1)
    else:
        df_final = pd.concat([df.iloc[:, :metadata_cols], df_numeric], axis=1)

    # -----------------------------------------------------------
    # PCA
    # -----------------------------------------------------------
    st.subheader("Step 2 — PCA")

    n_genes = st.number_input("Number of top variable proteins to use:", 100, 2000, 500)

    if st.button("Run PCA"):
        variances = df_imp.var(axis=1)
        top_genes = df_imp.loc[variances.sort_values(ascending=False).index[:n_genes]]

        X = top_genes.transpose()
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pca_res[:, 0], pca_res[:, 1])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA")

        st.pyplot(fig)
        st.download_button("Download PCA as SVG", download_svg(fig), "pca.svg", "image/svg+xml")

    # -----------------------------------------------------------
    # Differential expression
    # -----------------------------------------------------------
    st.subheader("Step 3 — Differential Analysis")

    all_conditions = [c.replace("2Log Abundance ", "") for c in df.columns[metadata_cols:]]
    unique_conditions = sorted(set([" ".join(c.split()[:-1]) for c in all_conditions]))

    cond1 = st.selectbox("Baseline (NT)", unique_conditions)
    cond2 = st.selectbox("Compare:", unique_conditions)

    if st.button("Run differential analysis"):
        de = differential_analysis(df_final, cond1, cond2)

        st.write(de.head())

        # Volcano controls
        st.subheader("Volcano Plot")

        p_thresh = st.number_input("p-value threshold:", 0.0001, 0.1, 0.05, step=0.001)
        logfc_thresh = st.number_input("logFC threshold:", 0.1, 5.0, 1.0)

        col_up = st.color_picker("Colour for UP genes", "#D62728")
        col_down = st.color_picker("Colour for DOWN genes", "#1F77B4")
        col_ns = st.color_picker("Colour for NS", "#BBBBBB")

        # Volcano plot
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        de["-log10(p)"] = -np.log10(de["pvalue"])

        # significance classification
        sig_up = (de["pvalue"] < p_thresh) & (de["logFC"] > logfc_thresh)
        sig_down = (de["pvalue"] < p_thresh) & (de["logFC"] < -logfc_thresh)

        ax2.scatter(de.loc[~(sig_up | sig_down), "logFC"],
                    de.loc[~(sig_up | sig_down), "-log10(p)"], c=col_ns, s=20)
        ax2.scatter(de.loc[sig_up, "logFC"], de.loc[sig_up, "-log10(p)"], c=col_up, s=25)
        ax2.scatter(de.loc[sig_down, "logFC"], de.loc[sig_down, "-log10(p)"], c=col_down, s=25)

        ax2.axvline(logfc_thresh, color="black", linestyle="--")
        ax2.axvline(-logfc_thresh, color="black", linestyle="--")
        ax2.axhline(-np.log10(p_thresh), color="black", linestyle="--")

        ax2.set_xlabel("logFC")
        ax2.set_ylabel("-log10(p)")
        ax2.set_title(f"{cond2} vs {cond1}")

        st.pyplot(fig2)
        st.download_button("Download Volcano as SVG", download_svg(fig2),
                           "volcano.svg", "image/svg+xml")

    # -----------------------------------------------------------
    # Heatmap
    # -----------------------------------------------------------
    st.subheader("Step 4 — Heatmap")

    cmap_opt = st.selectbox("Colour map:", ["viridis", "plasma", "inferno", "magma", "cividis"])
    clustering = st.selectbox("Clustering:", ["both", "rows", "columns", "none"])
    num_genes_hm = st.number_input("Number of top variable genes:", 50, 2000, 250)

    if st.button("Generate heatmap"):
        variances = df_imp.var(axis=1)
        top_hm = df_imp.loc[variances.sort_values(ascending=False).index[:num_genes_hm]]

        if clustering == "none":
            cg = sns.clustermap(top_hm, cmap=cmap_opt, row_cluster=False, col_cluster=False)
        elif clustering == "rows":
            cg = sns.clustermap(top_hm, cmap=cmap_opt, row_cluster=True, col_cluster=False)
        elif clustering == "columns":
            cg = sns.clustermap(top_hm, cmap=cmap_opt, row_cluster=False, col_cluster=True)
        else:
            cg = sns.clustermap(top_hm, cmap=cmap_opt)

        st.pyplot(cg.fig)
        st.download_button("Download Heatmap as SVG",
                           download_svg(cg.fig), "heatmap.svg", "image/svg+xml")
