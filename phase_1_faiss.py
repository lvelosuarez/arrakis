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
    from sklearn.decomposition import PCA 
    import plotly.express as px
    import faiss
    return PCA, Path, faiss, json, mo, np, pl, px


@app.cell
def _():
    import warnings
    import sklearn

    # Ignore all FutureWarnings from sklearn
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="sklearn"
    )
    return


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
        "mycobacterium": r"^mycobacterium$"
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
def _():
     # k values you want to test (including self, as FAISS returns)
    k_values = [2, 5, 6,7,8,9,10, 15, 20, 30, 50, 100]

    # HDBSCAN hyperparameters
    hdb_min_cluster_size = 50    # tune per genus
    hdb_min_samples = None       # or e.g. 10

    # Leiden hyperparameters
    leiden_resolution = 1.0      # higher → more clusters
    return hdb_min_cluster_size, hdb_min_samples, k_values, leiden_resolution


@app.cell
def _(E_genus, faiss, k_values):
    # Ensure float32 for FAISS
    X_genus = E_genus.astype("float32").copy()

    # Cosine-like similarity via L2 normalization + inner product
    faiss.normalize_L2(X_genus)

    n_samples, dim = X_genus.shape

    # Use the largest k we need
    max_k = max(k_values)

    # Build GPU index
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    gpu_index.add(X_genus)

    # Self kNN search for max_k
    sims_knn_full, idx_knn_full = gpu_index.search(X_genus, max_k)

    # sims_knn[i, :]  -> similarity scores of neighbors of point i
    # idx_knn[i, :]   -> indices of neighbors of point i
    return X_genus, idx_knn_full, sims_knn_full


@app.cell
def _(np):
    from scipy import sparse
    from scipy.sparse import csgraph
    import hdbscan
    import igraph as ig
    import leidenalg
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    def build_knn_graph(idx_knn, sims_knn):
        """
        idx_knn, sims_knn: shape (n_samples, k) including self as first column.
        Returns symmetric sparse adjacency.
        """
        n_samples, k = idx_knn.shape

        # Drop the self-neighbor (column 0)
        neighbor_idx = idx_knn[:, 1:]
        neighbor_sims = sims_knn[:, 1:]
        k_eff = k - 1

        sources = np.repeat(np.arange(n_samples), k_eff)
        targets = neighbor_idx.reshape(-1)
        weights = neighbor_sims.reshape(-1)

        A = sparse.csr_matrix(
            (weights, (sources, targets)),
            shape=(n_samples, n_samples),
        )
        A_sym = A.maximum(A.T)
        return A_sym

    def run_hdbscan(X, min_cluster_size, min_samples):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(X)
        return labels, clusterer

    def run_leiden_from_adj(A_sym, resolution):
        A_coo = A_sym.tocoo()
        n_nodes = A_sym.shape[0]
        edges = list(zip(A_coo.row.tolist(), A_coo.col.tolist()))
        weights = A_coo.data.astype(float)

        g = ig.Graph(n=n_nodes, edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution,
        )
        labels = np.array(partition.membership)
        return labels, g, partition

    def graph_connectivity_stats(A_sym):
        """
        Connected components and giant component size.
        """
        n_components, labels = csgraph.connected_components(
            A_sym, directed=False, connection='weak'
        )
        component_sizes = np.bincount(labels)
        giant = component_sizes.max()
        giant_fraction = giant / A_sym.shape[0]
        return n_components, giant_fraction

    def clustering_metrics(X, labels_hdb, labels_leiden, A_sym):
        results = {}

        # --- Cluster counts / noise ---
        mask_hdb = labels_hdb >= 0
        unique_hdb = np.unique(labels_hdb[mask_hdb]) if mask_hdb.any() else []
        n_clusters_hdb = len(unique_hdb)
        noise_fraction_hdb = 1.0 - (mask_hdb.sum() / len(labels_hdb))

        n_clusters_leiden = len(np.unique(labels_leiden))

        results["n_clusters_hdb"] = n_clusters_hdb
        results["n_clusters_leiden"] = n_clusters_leiden
        results["noise_fraction_hdb"] = noise_fraction_hdb

        # --- Silhouette scores ---
        try:
            if mask_hdb.sum() > 1 and n_clusters_hdb > 1:
                sil_hdb = silhouette_score(X[mask_hdb], labels_hdb[mask_hdb])
            else:
                sil_hdb = np.nan
        except Exception:
            sil_hdb = np.nan

        try:
            if n_clusters_leiden > 1:
                sil_leiden = silhouette_score(X, labels_leiden)
            else:
                sil_leiden = np.nan
        except Exception:
            sil_leiden = np.nan

        results["silhouette_hdb"] = sil_hdb
        results["silhouette_leiden"] = sil_leiden

        # --- Agreement between methods (ARI) ---
        try:
            if mask_hdb.sum() > 1:
                ari = adjusted_rand_score(
                    labels_hdb[mask_hdb],
                    labels_leiden[mask_hdb]
                )
            else:
                ari = np.nan
        except Exception:
            ari = np.nan
        results["ari_hdb_vs_leiden"] = ari

        # --- Graph connectivity ---
        n_components, giant_fraction = graph_connectivity_stats(A_sym)
        results["n_components"] = n_components
        results["giant_component_fraction"] = giant_fraction

        return results
    return (
        build_knn_graph,
        clustering_metrics,
        run_hdbscan,
        run_leiden_from_adj,
    )


@app.cell
def _(
    X_genus,
    build_knn_graph,
    clustering_metrics,
    hdb_min_cluster_size,
    hdb_min_samples,
    idx_knn_full,
    k_values,
    leiden_resolution,
    pl,
    run_hdbscan,
    run_leiden_from_adj,
    sims_knn_full,
):
    sweep_records = []

    for k in k_values:
        # Use first k neighbors (including self)
        sims_k = sims_knn_full[:, :k]
        idx_k = idx_knn_full[:, :k]

        # 1) Build kNN graph
        A_sym_k = build_knn_graph(idx_k, sims_k)

        # 2) HDBSCAN clustering on X_genus
        labels_hdb, _ = run_hdbscan(
            X_genus,
            min_cluster_size=hdb_min_cluster_size,
            min_samples=hdb_min_samples,
        )

        # 3) Leiden clustering on the kNN graph
        labels_leiden, _, _ = run_leiden_from_adj(
            A_sym_k,
            resolution=leiden_resolution,
        )

        # 4) Metrics
        metrics = clustering_metrics(
            X_genus,
            labels_hdb,
            labels_leiden,
            A_sym_k,
        )

        record = {
            "k": k,
            **metrics,
        }
        sweep_records.append(record)

    df_k_sweep = pl.DataFrame(sweep_records)

    df_k_sweep
    return labels_hdb, labels_leiden


@app.cell
def _(df_genus, labels_hdb, labels_leiden, pl):
    df_genus_clusters = df_genus.with_columns([
        pl.Series("cluster_hdbscan_k30", labels_hdb),
        pl.Series("cluster_leiden_k30", labels_leiden),
    ])
    df_genus_clusters
    return


@app.cell
def _(
    X_genus,
    build_knn_graph,
    idx_knn_full,
    np,
    run_hdbscan,
    run_leiden_from_adj,
    sims_knn_full,
):
    # Choose k inside stability plateau
    k_bacillus = 30

    sims_k_bacillus = sims_knn_full[:, :k_bacillus]
    idx_k_bacillus = idx_knn_full[:, :k_bacillus]

    # kNN graph for Bacillus at this k
    A_sym_k_bacillus = build_knn_graph(idx_k_bacillus, sims_k_bacillus)

    # HDBSCAN on embeddings (macro basins)
    labels_hdb_bacillus, hdbscan_model_bacillus = run_hdbscan(
        X_genus,
        min_cluster_size=50,   # tune if you like
        min_samples=None,
    )

    # Leiden on kNN graph (species candidates)
    labels_leiden_bacillus, g_leiden_bacillus, partition_leiden_bacillus = run_leiden_from_adj(
        A_sym_k_bacillus,
        resolution=1.0,
    )

    unique_hdb_bacillus = np.unique(labels_hdb_bacillus[labels_hdb_bacillus >= 0])
    unique_leiden_bacillus = np.unique(labels_leiden_bacillus)

    print(f"[Bacillus] HDBSCAN clusters (k={k_bacillus}): {len(unique_hdb_bacillus)} + noise")
    print(f"[Bacillus] Leiden clusters (k={k_bacillus}): {len(unique_leiden_bacillus)}")
    return labels_hdb_bacillus, labels_leiden_bacillus


@app.cell
def _(X_genus, labels_hdb_bacillus, labels_leiden_bacillus, np):
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt
    from collections import Counter

    leiden_ids_bacillus = np.unique(labels_leiden_bacillus)

    centroids_bacillus = []
    dominant_hdb_bacillus = []
    sizes_bacillus = []

    for cid in leiden_ids_bacillus:
        mask = labels_leiden_bacillus == cid
        X_c = X_genus[mask]
        centroids_bacillus.append(X_c.mean(axis=0))

        sizes_bacillus.append(mask.sum())

        hdb_labels_in_cluster = labels_hdb_bacillus[mask]
        hdb_non_noise = hdb_labels_in_cluster[hdb_labels_in_cluster >= 0]
        if len(hdb_non_noise) == 0:
            dominant_hdb_bacillus.append(-1)
        else:
            dom = Counter(hdb_non_noise).most_common(1)[0][0]
            dominant_hdb_bacillus.append(dom)

    centroids_bacillus = np.vstack(centroids_bacillus)

    # Hierarchical clustering on Leiden centroids
    Z_bacillus = linkage(centroids_bacillus, method="ward")

    # Color map for HDBSCAN basins
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map_bacillus = {}
    for h in set(dominant_hdb_bacillus):
        if h == -1:
            color_map_bacillus[h] = "lightgray"
        else:
            color_map_bacillus[h] = palette[h % len(palette)]

    leaf_colors_bacillus = {
        str(cid): color_map_bacillus[dom]
        for cid, dom in zip(leiden_ids_bacillus, dominant_hdb_bacillus)
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z_bacillus,
        labels=[str(cid) for cid in leiden_ids_bacillus],
        leaf_rotation=90,
        leaf_font_size=8,
        link_color_func=lambda k: "black",
        ax=ax,
    )

    # Color tick labels by dominant HDBSCAN basin
    for tick_label in ax.get_xmajorticklabels():
        cid_str = tick_label.get_text()
        tick_label.set_color(leaf_colors_bacillus.get(cid_str, "black"))

    ax.set_title(
        "Bacillus — Leiden cluster dendrogram\n(leaves colored by dominant HDBSCAN basin)"
    )
    ax.set_ylabel("Linkage distance")
    fig.tight_layout()
    fig
    return


@app.cell
def _(df_genus, labels_hdb_bacillus, labels_leiden_bacillus, pl):
    assert df_genus.height == len(labels_hdb_bacillus) == len(labels_leiden_bacillus)

    df_bacillus_with_clusters = df_genus.with_columns(
        [
            pl.Series("cluster_hdbscan_bacillus", labels_hdb_bacillus),
            pl.Series("cluster_leiden_bacillus", labels_leiden_bacillus),
        ]
    )

    # counts of species per Leiden cluster
    df_bacillus_cluster_species = (
        df_bacillus_with_clusters
        .group_by(["cluster_leiden_bacillus", "species"])
        .len()
        .rename({"len": "count_in_cluster"})
    )

    # summary per Leiden cluster
    df_bacillus_cluster_totals = (
        df_bacillus_cluster_species
        .group_by("cluster_leiden_bacillus")
        .agg(
            [
                pl.col("count_in_cluster").sum().alias("cluster_size"),
                pl.col("species").n_unique().alias("n_species_in_cluster"),
                pl.struct(["species", "count_in_cluster"])
                .sort_by("count_in_cluster", descending=True)
                .first()
                .alias("dominant_pair"),
            ]
        )
    )

    df_bacillus_cluster_summary = (
        df_bacillus_cluster_totals
        .with_columns(
            [
                pl.col("dominant_pair").struct.field("species").alias("dominant_species"),
                pl.col("dominant_pair").struct.field("count_in_cluster").alias("dominant_count"),
            ]
        )
        .with_columns(
            (pl.col("dominant_count") / pl.col("cluster_size")).alias("dominant_purity")
        )
        .drop("dominant_pair")
        .sort("cluster_size", descending=True)
    )

    df_bacillus_cluster_summary
    return


if __name__ == "__main__":
    app.run()
