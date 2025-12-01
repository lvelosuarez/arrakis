import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import json
    import numpy as np
    import polars as pl
    import pandas as pd
    import umap
    import faiss
    from itertools import islice
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import ConvexHull, QhullError
    from scipy.ndimage import gaussian_filter
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image
    import colorcet as cc
    return (
        NearestNeighbors,
        PCA,
        Path,
        csr_matrix,
        faiss,
        gaussian_filter,
        go,
        json,
        mo,
        np,
        pl,
        plt,
        px,
        umap,
    )


@app.cell
def paths(Path):
    BASE_DIR = Path("/Users/lourdes/Desktop/arrakis/triplet_semihard_loss-ranger-0.5-hq-256-CNNFCGR_Levels-level1-clip80") / "index"
    EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
    ID_EMBEDDINGS_PATH = BASE_DIR / "id_embeddings.json"
    LABELS_PATH = BASE_DIR / "labels.json"
    taxonomy = "/Users/lourdes/Dropbox/Curro/Projects/arrakis/taxonomy.tsv.gz"
    BASE_DIR, EMBEDDINGS_PATH, ID_EMBEDDINGS_PATH, LABELS_PATH, taxonomy
    return EMBEDDINGS_PATH, ID_EMBEDDINGS_PATH, LABELS_PATH, taxonomy


@app.cell(hide_code=True)
def loading(
    EMBEDDINGS_PATH,
    ID_EMBEDDINGS_PATH,
    LABELS_PATH,
    json,
    mo,
    np,
    pl,
):
    # 3.1 Load embeddings (PanSpace embedding matrix)
    E = np.load(EMBEDDINGS_PATH)   # shape (N, D)
    N, D = E.shape
    # 3.2 Load mapping: embedding index -> genome_id
    with open(ID_EMBEDDINGS_PATH) as f:
        id_map = json.load(f)
    # 3.3 Load labels / metadata per genome
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    records = []
    for emb_idx_str, path in id_map.items():
        emb_idx = int(emb_idx_str)
        label_str = labels.get(emb_idx_str, "")
        records.append(
            {
                "emb_idx": emb_idx,
                "genome_id": path,      # path like data_2m/fcgr/8mer/...
                 "raw_label": label_str, # string like: "salmonella_enterica" SAMN14516922 is ...
            }
        )

    df = pl.DataFrame(records).sort("emb_idx")

    summary = mo.md(
        f"""
    ## 1. Loading data

    - Embeddings shape: **{E.shape}** (N × D)  
    - Number of rows in df: **{df.shape[0]}**  
    - Columns: `{df.columns}`  

    Currently each row has:

    - `emb_idx`: integer index in the embedding matrix  
    - `genome_id`: the path to the original FCGR file  
    - `raw_label`: a descriptive string from `labels.json`  
            """
    )

    summary
    return E, df


@app.cell
def create_df_tax(df, pl):
    df_tax = df.clone()
    df_tax = df_tax.with_columns(
        pl.col("genome_id").str.split("/").alias("path_parts")
    )
    df_tax = df_tax.with_columns(
        pl.col("path_parts").list.get(-2).alias("dir_name"),
        pl.col("path_parts").list.get(-1).alias("filename"),
    )
    df_tax = df_tax.with_columns(
        pl.col("filename")
        .str.replace(".npy", "")
        .alias("sample_accession")
    )
    df_tax = df_tax.with_columns(
        pl.col("raw_label")
        .cast(pl.Utf8)
        .str.split("_")
        .alias("label_tokens")
    )
    df_tax = df_tax.with_columns(
        pl.when(pl.col("label_tokens").list.len() >= 1)
        .then(pl.col("label_tokens").list.get(0))
        .otherwise(None)
        .alias("genus")
    )
    df_tax = df_tax.with_columns(
        pl.when(pl.col("label_tokens").list.len() >= 2)
        .then(
            pl.concat_str(
                [
                    pl.col("label_tokens").list.get(0),                # genus token
                    pl.lit(" "),                                      # space
                    pl.col("label_tokens").list.slice(1).list.join(" ")  # remaining tokens
                ]
            )
        )
        .otherwise(None)
        .alias("species")
    )
    return (df_tax,)


@app.cell
def _(df_tax, pl, taxonomy):
    df_tree = pl.read_csv(
        taxonomy,
        separator="\t"
    )
    df_f = df_tree.join(df_tax, on = "sample_accession")
    df_f.columns
    return (df_f,)


@app.cell
def _(df_f, pl):
    df_f.group_by("phylum_name").agg([
        pl.n_unique("class_name").alias("n_class"),
        pl.n_unique("order_name").alias("n_order"),
        pl.n_unique("family_name").alias("n_family"),
        pl.n_unique("genus_name").alias("n_genus"),
        pl.n_unique("species_right").alias("n_species"),
        pl.n_unique("sample_accession").alias("n_accession"),
    ]).sort("phylum_name")
    return


@app.cell(hide_code=True)
def _(df_tax, mo, pl):
    BENCHMARK_GENERA = {
        "bacillus":      r"^bacillus[a-z]*$",
        "bacillusS":   r"^bacillus$",
        "bacillusA":   r"^bacillusa",
        "streptococcus": r"^streptococcus$",
        "escherichia":   r"^escherichia$",
        # matches bacillus, bacillu, bacillusa, etc.

    }

    # Actual genera present in the dataset (raw strings)
    present = (
        df_tax
        .select(pl.col("genus"))
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )

    present_series = pl.Series("genus", present).cast(pl.Utf8)

    # Keep only benchmark genera whose regex matches at least one present genus
    available_benchmarks = []
    for name, pattern in BENCHMARK_GENERA.items():
        if present_series.str.contains(pattern, literal=False).any():
            available_benchmarks.append(name)

    genus_selector = mo.ui.dropdown(
        options=available_benchmarks,  # e.g. ["streptococcus", "escherichia", "bacillus"]
        value=available_benchmarks[0] if available_benchmarks else None,
        label="Select benchmark genus",
    )

    ui = mo.vstack(
        [
            mo.md("## 2. Choose test genus"),
            genus_selector,
        ]
    )

    ui
    return BENCHMARK_GENERA, genus_selector


@app.cell(hide_code=True)
def _(BENCHMARK_GENERA, E, df_tax, genus_selector, mo, pl):
    selected_genus = genus_selector.value  # <- this will be e.g. "salmonella"
    regex = BENCHMARK_GENERA[selected_genus]
    # Filter with regex on the genus column
    df_genus = df_tax.filter(
        pl.col("genus").cast(pl.Utf8).str.contains(regex, literal=False))
    # Extract embedding indices and subset E
    idxs = df_genus["emb_idx"].to_numpy().astype(int)
    E_genus = E[idxs]
    # Small debug: show how many of each raw genus we captured
    genus_counts = (
    df_genus
    .group_by("species")
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
    )
    summary1 = mo.vstack(
    [
        mo.md(
            f"""
    ## 3. Embedding subset for genus group `{selected_genus}`

    - Regex used: `{BENCHMARK_GENERA.get(selected_genus, '')}`
    - Number of genomes: **{E_genus.shape[0]}**
    - Embedding dimension: **{E_genus.shape[1]}**
            """
        ),
        mo.md("**Genus labels included in this group:**"),
        mo.ui.table(genus_counts.to_pandas()) if df_genus.height > 0 else mo.md("_No rows selected._"),
    ]
    )

    summary1
    return E_genus, df_genus, idxs, selected_genus


@app.cell(hide_code=True)
def _(mo, selected_genus):
    mo.md(f"""
    ### Table des {selected_genus}

    | Species                      | Genomes | Key Characteristics |
    |:-----------------------------|--------:|:---------------------|
    | bacillusa anthracis          | 1355    | Agent of anthrax; carries pXO1/pXO2 plasmids; extremely clonal; very low genomic diversity; obligate mammalian pathogen. |
    | bacillus subtilis            | 1429    | Model organism; soil and plant-associated; high competence; many biosynthetic gene clusters (BGCs); robust sporulation. |
    | bacillusa paranthracis       | 576     | Member of Bacillus cereus s.l.; often food/environment-associated; genetically close to *B. cereus* group. |
    | bacillusa bombysepticus      | 320     | Entomopathogenic; associated with silkworm septicemia; part of B. cereus s.l.; carries virulence plasmids. |
    | bacillusa cereus             | 312     | Classic foodborne pathogen; diarrheal & emetic toxins; highly diverse accessory genome; environmental & clinical strains. |
    | bacillusa thuringiensis      | 288     | Entomopathogenic; produces Cry/Cyt toxins; plasmid-rich; widely used as biopesticide; part of B. cereus s.l. |
    | bacillus velezensis          | 151     | Plant-growth-promoting rhizobacterium; strong antimicrobial BGC repertoire (surfactin, fengycin, etc.). |
    | bacillusa wiedmannii         | 147     | Recently described B. cereus s.l. species; opportunistic pathogen; often isolated from environment and foods. |
    | bacillusa mobilis            | 123     | B. cereus s.l.; environmental species; mobile genetic elements common; variable pathogenic potential. |
    | bacillusa mycoides           | 117     | Rhizoid colony morphology; soil-associated; part of B. cereus s.l.; lower virulence but ecological versatility. |
    | bacillusa toyonensis         | 109     | B. cereus s.l.; frequently used as probiotic in animal feed; genome encodes unique antimicrobial modules. |
    | bacillus pumilus             | 81      | Widespread environmental species; UV-resistant; produces peptide antimicrobials; plant-associated roles. |
    | bacillus licheniformis       | 72      | Food/environment isolate; protease + enzyme producer; often used industrially; close to *B. paralicheniformis*. |
    | bacillusa tropicus           | 41      | Member of B. cereus s.l.; tropical habitat associations; can show mild pathogenicity. |
    | bacillus safensis            | 39      | Known from spacecraft clean rooms; high stress resistance; notable for extremotolerance. |
    | bacillus altitudinis         | 36      | Isolated from high-altitude environments; environmental microbe; stress tolerant. |
    | bacillus paralicheniformis   | 34      | Sister species to *licheniformis*; enzyme and antimicrobial production; environmental. |
    | bacillusa cytotoxicus        | 23      | Thermotolerant B. cereus s.l. lineage; produces cytotoxin K variants; associated with food poisoning. |
    | bacillus spizizenii          | 16      | Close to *B. subtilis*; natural competence; similar core genome; model for sporulation studies. |
    | bacillus halotolerans        | 12      | Salt-tolerant species; environmental; moderate halophily. |
    | bacillusa luti               | 14      | Environmental member of B. cereus s.l.; associated with soil; moderately thermotolerant traits. |
    | bacillusa pseudomycoides     | 11      | B. cereus s.l.; genetically close to *mycoides*; often soil-associated; low virulence. |
    | bacillusa albus              | 11      | Recently described member of B. cereus s.l.; environmental origin; limited data available. |
    """)
    return


@app.cell(hide_code=True)
def _(E, E_genus, PCA, idxs, mo, selected_genus):
    n_components = min(50, E_genus.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(E[idxs])
    explained_variance_ratio = pca.explained_variance_ratio_
    mo.md(
        f"""
    ## 4. PCA sur le sous-ensemble {selected_genus}

    - Nombre de génomes dans df_genus: **{E[idxs].shape[0]}**  
    - Dimension d'origine: **{E[idxs].shape[1]}**  
    - PCA calculé avec **{n_components}** composantes
    """
    )
    return X_pca, explained_variance_ratio, n_components


@app.cell(hide_code=True)
def _(explained_variance_ratio, n_components, np, plt, selected_genus):
    cum_var = np.cumsum(explained_variance_ratio)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, n_components + 1), cum_var, marker="o")
    ax.set_xlabel("Nombre de composantes principales")
    ax.set_ylabel("Variance expliquée cumulée")
    ax.set_title(f"PCA ({selected_genus}) — variance expliquée cumulée")
    ax.grid(True)
    fig
    return


@app.cell
def _(X_pca, df_genus, pl, px, selected_genus):
    pca3d = X_pca[:, :3]
    df_3d = df_genus.with_columns(
        [
            pl.Series("pc1", pca3d[:, 0]),
            pl.Series("pc2", pca3d[:, 1]),
            pl.Series("pc3", pca3d[:, 2]),
        ]
    )
    df_3d_pd = df_3d.select(
        ["pc1", "pc2", "pc3", "species","genus","genome_id", "sample_id"]
    ).to_pandas()

    fig_pca = px.scatter_3d(
        df_3d_pd,
        x="pc1",
        y="pc2",
        z="pc3",
        color="species",
        hover_data=["genome_id", "sample_id"],
        opacity=0.7,
    )

    fig_pca.update_layout(scene_dragmode="orbit",
        title= f"PCA ({selected_genus})",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(font=dict(size=9)),
    )

    fig_pca
    return df_3d, df_3d_pd


@app.cell(hide_code=True)
def _(df_3d_pd, gaussian_filter, go, np, selected_genus):
    N_ = 60 # try 50–100, not 150+
    x_bins = np.linspace(df_3d_pd.pc1.min(), df_3d_pd.pc1.max(), N_)
    y_bins = np.linspace(df_3d_pd.pc2.min(), df_3d_pd.pc2.max(), N_)
    z_bins = np.linspace(df_3d_pd.pc3.min(), df_3d_pd.pc3.max(), N_)

    # Histogramdd
    H, edges = np.histogramdd(
        df_3d_pd[['pc1','pc2','pc3']].values,
        bins=[x_bins, y_bins, z_bins]
    )
    H = gaussian_filter(H, sigma=1)
    H_norm = H / H.max()
    X, Y, Z = np.meshgrid(
        x_bins[:-1], 
        y_bins[:-1], 
        z_bins[:-1],
        indexing='ij'
    )
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    Vf = H_norm.ravel()
    gamma = 0.20   # try values between 0.3–0.6
    Vf_gamma = Vf ** gamma
    fig_render = go.Figure(
        go.Volume(
            x=Xf,
            y=Yf,
            z=Zf,
            value=Vf_gamma,
            opacity=0.18,
            surface_count=12,
            colorscale="Hot"
        )
    )

    fig_render.update_layout(scene_dragmode="orbit",
        title= f" 3D PCA density ({selected_genus})",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(font=dict(size=9)),
    )

    fig_render
    return


@app.cell(hide_code=True)
def _(E, idxs, np, umap):
    X_pca_input = np.asarray(E[idxs]) 
    print("UMAP input:", X_pca_input.shape)

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=3,     # 3D output
        metric="cosine",    # best for embeddings
        random_state=42
    )

    # Compute UMAP
    X_pca = reducer.fit_transform(X_pca_input)   # (3447, 3)
    return (X_pca,)


@app.cell(hide_code=True)
def _(X_pca, df_3d, pl, px, selected_genus):
    df_umap = df_3d.with_columns([
        pl.Series("u1", X_pca[:, 0]),
        pl.Series("u2", X_pca[:, 1]),
        pl.Series("u3", X_pca[:, 2]),
    ])
    df_umap_pd = df_umap.select(
        ["u1", "u2", "u3", "species","genus","genome_id", "sample_id"]
    ).to_pandas()

    fig_umap = px.scatter_3d(
        df_umap_pd,
        x="u1",
        y="u2",
        z="u3",
        color="species",
        hover_data=["genome_id", "sample_id"],
        opacity=0.7,
    )

    fig_umap.update_layout(scene_dragmode="orbit",
        title= f"PCA ({selected_genus})",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(font=dict(size=9)),
    )

    fig_umap
    return


@app.cell(hide_code=True)
def _(mo):
    k_selector = mo.ui.slider(
    start=5, stop=200, step=5, value=30,
    label="k for kNN graph"
    )
    mo.vstack([
    mo.md("## 5. kNN graph construction"),
    k_selector
    ])
    return (k_selector,)


@app.cell(hide_code=True)
def _(E_genus, NearestNeighbors, k_selector, mo):
    k = int(k_selector.value)

    knn = NearestNeighbors(
        n_neighbors=k + 1,   # +1 because first neighbor is self
        metric="euclidean",     # consistent with your UMAP input
        n_jobs=-1
    )
    knn.fit(E_genus)

    dists, nbrs = knn.kneighbors(E_genus, return_distance=True)

    # drop self-neighbor in column 0
    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]

    mo.md(
        f"""
    ### kNN computed on **PanSpace high-D embeddings**

    - N genomes: **{E_genus.shape[0]}**
    - k (neighbors per genome): **{k}**
    - distances shape: {dists.shape}
    - neighbors shape: {nbrs.shape}
        """
    )
    return dists, k, nbrs


@app.cell(hide_code=True)
def _(csr_matrix, dists, k, mo, nbrs, np):
    N_1 = nbrs.shape[0]

    # rows: each point repeated k times
    rows = np.repeat(np.arange(N_1), k)
    cols = nbrs.ravel()

    # weights: similarity = 1 - cosine distance
    weights = (1.0 - dists).ravel()

    A = csr_matrix((weights, (rows, cols)), shape=(N_1, N_1))

    # symmetrize: keep strongest edge in either direction
    A = A.maximum(A.T)

    density = A.nnz / (N_1 * N_1)

    mo.md(
        f"""
    ### kNN adjacency (sparse)

    - Nodes: **{N_1}**
    - Edges (non-zeros): **{A.nnz}**
    - Graph density: **{density:.5f}**
        """
    )
    return (A,)


@app.cell(hide_code=True)
def _(A, mo, np):
    degrees = np.array(A.getnnz(axis=1))

    mo.vstack([
        mo.md("### kNN degree distribution (should be ~k, after symmetrization a bit higher)"),
        mo.md(
            f"""
    - min degree: **{degrees.min()}**
    - median degree: **{np.median(degrees)}**
    - max degree: **{degrees.max()}**
            """
        )
    ])
    return


@app.cell
def _(A, X_pca, go, np):
    # sample a subset of edges for plotting
    coo = A.tocoo()
    n_edges = coo.nnz
    sample_size = min(5000, n_edges)

    idx = np.random.choice(n_edges, size=sample_size, replace=False)

    edge_x = np.vstack([X_pca[coo.row[idx], 0], X_pca[coo.col[idx], 0]]).T
    edge_y = np.vstack([X_pca[coo.row[idx], 1], X_pca[coo.col[idx], 1]]).T
    edge_z = np.vstack([X_pca[coo.row[idx], 2], X_pca[coo.col[idx], 2]]).T

    fig_knn = go.Figure()

    # edges
    for ex, ey, ez in zip(edge_x, edge_y, edge_z):
        fig_knn.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="lines",
            line=dict(width=1),
            opacity=0.15,
            showlegend=False
        ))

    # nodes
    fig_knn.add_trace(go.Scatter3d(
        x=X_pca[:,0], y=X_pca[:,1], z=X_pca[:,2],
        mode="markers",
        marker=dict(size=3, opacity=0.7),
        showlegend=False
    ))

    fig_knn.update_layout(
        title="kNN graph edges over PCA projection (sampled)",
        scene_dragmode="orbit"
    )
    fig_knn
    return


@app.cell
def _(E_genus, faiss, time):

    X = E_genus.astype("float32").copy()

    t0 = time.time()
    fc_index = faiss.IndexFlatIP(X.shape[1])
    print("Index init:", time.time() - t0, "sec")

    t0 = time.time()
    fc_index.add(X)
    print("Add:", time.time() - t0, "sec")

    t0 = time.time()
    sim, nbrs = fc_index.search(X, 51)
    print("Search:", time.time() - t0, "sec")
    return (nbrs,)


if __name__ == "__main__":
    app.run()
