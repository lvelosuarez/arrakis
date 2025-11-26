import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo 
    from pathlib import Path
    import json 
    import numpy as np 
    import polars as pl 
    import pandas as pd 
    from sklearn.decomposition import PCA 
    import plotly.express as px
    return PCA, Path, json, mo, np, pl, px


@app.cell
def _(Path):
    BASE_DIR = Path("/mnt/san/microbio/database/triplet_semihard_loss-ranger-0.5-hq-256-CNNFCGR_Levels-level1-clip80") / "index"
    EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
    ID_EMBEDDINGS_PATH = BASE_DIR / "id_embeddings.json"
    LABELS_PATH = BASE_DIR / "labels.json"

    BASE_DIR, EMBEDDINGS_PATH, ID_EMBEDDINGS_PATH, LABELS_PATH
    return EMBEDDINGS_PATH, ID_EMBEDDINGS_PATH, LABELS_PATH


@app.cell
def _(EMBEDDINGS_PATH, ID_EMBEDDINGS_PATH, LABELS_PATH, json, np, pl):
    # 3.1 Load embeddings (PanSpace embedding matrix)
    E = np.load(EMBEDDINGS_PATH)  # shape (N, D)
    N, D = E.shape

    # 3.2 Load mapping: embedding index -> genome_id
    with open(ID_EMBEDDINGS_PATH) as f:
        id_map = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    records = []
    for emb_idx_str, path in id_map.items(): 
        emb_idx = int(emb_idx_str) 
        label_str = labels.get(emb_idx_str, "") 
        records.append( { "emb_idx": emb_idx, 
                          "genome_id": path,  
                          "raw_label": label_str,}) 
    df = pl.DataFrame(records).sort("emb_idx")
    return E, df


@app.cell
def _(df, pl):
    df_tax = df.clone() 
    df_tax = df_tax.with_columns( pl.col("genome_id").str.split("/").alias("path_parts") )
    df_tax = df_tax.with_columns( pl.col("path_parts").list.get(-2).alias("dir_name"), pl.col("path_parts").list.get(-1).alias("filename"), )
    df_tax = df_tax.with_columns( pl.col("filename") .str.replace(".npy", "") .alias("sample_id") )
    df_tax = df_tax.with_columns( pl.col("raw_label") .cast(pl.Utf8) .str.split("_") .alias("label_tokens") )
    df_tax = df_tax.with_columns( pl.when(pl.col("label_tokens").list.len() >= 1) .then(pl.col("label_tokens").list.get(0)) .otherwise(None) .alias("genus") )
    df_tax = df_tax.with_columns( pl.when(pl.col("label_tokens").list.len() >= 2) .then( pl.concat_str( [ pl.col("label_tokens").list.get(0), pl.lit(" "), pl.col("label_tokens").list.slice(1).list.join(" ") ] ) ) .otherwise(None) .alias("species") )
    return (df_tax,)


@app.cell
def _():
    BENCHMARK_GENERA = { 
        "bacillus": r"^bacillus[a-z]*$",
        "bacillusS": r"^bacillus$", 
        "bacillusA": r"^bacillusa", 
        "streptococcus": r"^streptococcus$", 
        "escherichia": r"^escherichia$",
        "wohlfahrtiimonas": r"^wohlfahrtiimonas$"
    }
    return (BENCHMARK_GENERA,)


@app.cell
def _(df_tax, pl):
    present = ( df_tax .select(pl.col("genus")) .drop_nulls() .unique() .to_series() .to_list() )
    present_series = pl.Series("genus", present).cast(pl.Utf8)
    return (present_series,)


@app.cell
def _(BENCHMARK_GENERA, mo, present_series):
    available_benchmarks = [] 
    for name, pattern in BENCHMARK_GENERA.items(): 
        if present_series.str.contains(pattern, literal=False).any(): 
            available_benchmarks.append(name) 
            genus_selector = mo.ui.dropdown(options=available_benchmarks,
                                            value=available_benchmarks[0] if available_benchmarks else None, 
                                            label="Select benchmark genus", ) 

    ui = mo.vstack( [ mo.md("## Choose test genus"), genus_selector, ] ) 
    ui
    return (genus_selector,)


@app.cell
def _(BENCHMARK_GENERA, E, df_tax, genus_selector, pl):
    selected_genus = genus_selector.value
    regex = BENCHMARK_GENERA[selected_genus]
    df_genus = df_tax.filter( pl.col("genus").cast(pl.Utf8).str.contains(regex, literal=False))
    idxs = df_genus["emb_idx"].to_numpy().astype(int)
    E_genus = E[idxs]
    genus_counts = ( df_genus .group_by("species") .agg(pl.len().alias("n")) .sort("n", descending=True) )
    return E_genus, df_genus, idxs


@app.cell
def _(E, E_genus, PCA, df_genus, idxs, pl, px):
    import builtins

    n_components = min(50, E_genus.shape[0])
    pca = PCA(n_components=n_components) 
    X_pca = pca.fit_transform(E[idxs])
    explained_variance_ratio = pca.explained_variance_ratio_

    pca3d = X_pca[:, :3]
    df_3d = df_genus.with_columns(
        [
            pl.Series("pc1", pca3d[:, 0]),
            pl.Series("pc2", pca3d[:, 1]),
            pl.Series("pc3", pca3d[:, 2]),
        ]
    )

    N_PER_SPECIES = 300

    dfs = []
    for sp in df_3d["species"].unique().to_list():
        sub = df_3d.filter(pl.col("species") == sp)
        sub_n = sub.shape[0]          # <— avoids shadowed height()
        n = builtins.min(N_PER_SPECIES, sub_n)
        dfs.append(sub.sample(n=n, shuffle=True))
    df_plot = pl.concat(dfs)
    df_plot_pd = df_plot.to_pandas()

    fig_pca = px.scatter_3d(
        df_plot_pd,
        x="pc1", y="pc2", z="pc3",
        color="species",
        hover_data=["genome_id", "sample_id"],
        opacity=0.7,
    )
    fig_pca.update_layout(scene_dragmode="orbit")
    fig_pca

    return


@app.cell
def _(E_genus, faiss):
    faiss_vectors = E_genus.astype("float32").copy()
    faiss.normalize_L2(faiss_vectors)
    return (faiss_vectors,)


@app.cell
def _(faiss_vectors):
    faiss_vectors
    return


if __name__ == "__main__":
    app.run()
