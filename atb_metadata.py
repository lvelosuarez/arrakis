import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import os
    import duckdb
    from pathlib import Path
    import polars as pl
    return Path, mo, pl


@app.cell
def _(Path):
    BASE_DIR = Path("/Users/lourdes/Desktop/arrakis/triplet_semihard_loss-ranger-0.5-hq-256-CNNFCGR_Levels-level1-clip80") / "index"
    EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
    ID_EMBEDDINGS_PATH = BASE_DIR / "id_embeddings.json"
    LABELS_PATH = BASE_DIR / "labels.json"
    sample_id = BASE_DIR / "samples_tsv"
    SQLITE_PATH = "/Users/lourdes/Desktop/arrakis/atb.metadata.202408.sqlite"
    tax = "/Users/lourdes/Desktop/arrakis/bac120_taxonomy_r214.tsv"
    metadata = "/Users/lourdes/Desktop/arrakis/bac120_metadata_r214.tsv"
    return SQLITE_PATH, metadata, sample_id, tax


@app.cell
def _(mo):
    mo.sql("SET sqlite_all_varchar=true")
    return


@app.cell
def _(SQLITE_PATH, mo):
    mo.sql(f"ATTACH '{SQLITE_PATH}' AS db (TYPE sqlite)")
    mo.sql("USE db")
    return


@app.cell
def _(mo):
    df_tables = mo.sql("""
        SHOW TABLES
    """)
    df_tables
    return


@app.cell
def _(mo, run, sample_id):
    mo.sql(f"""
        CREATE OR REPLACE TEMP TABLE t_sample_species AS
        SELECT *
        FROM read_csv_auto('{sample_id}', delim='\t', header=True)
    """)
    df_join = mo.sql(f"""
        SELECT
            s.sample_accession,
            s.species AS species_short,
            t.*
        FROM t_sample_species s
        JOIN  run t
        ON s.sample_accession = t.sample_accession
    """)
    return (df_join,)


@app.cell
def _(df_join):
    df_join.shape
    return


@app.cell
def _(mo, pl):
    TABLE = "ena_20240801"
    df_head = mo.sql(f"""
        SELECT *
        FROM {TABLE}
        LIMIT 10
    """)
    df_head.select(pl.col("taxonomic_classification"))
    return


@app.cell
def _(checkm2, mo):
    mo.sql("""
        SELECT *
        FROM checkm2
        WHERE sample_accession= 'SAMN14908305'
    """)
    return


@app.cell(hide_code=True)
def _(metadata, pl):
    df_meta = pl.read_csv(
        metadata,
        separator="\t",
        has_header=True,       # or False if needed
        infer_schema_length=0, # don't try to guess types; everything becomes Utf8
    )
    df_meta.filter(pl.col("accession") == "RS_GCF_000007825.1")
    df_meta.columns
    return (df_meta,)


@app.cell
def _(mo, sylph):
    mo.sql("""
        SELECT *
        FROM sylph
        WHERE sample_accession= 'SAMN14908305'
    """)
    return


@app.cell(hide_code=True)
def _(pl, tax):
    df_tax = pl.read_csv(tax, separator="\t", has_header=False)
    df_tax = df_tax.rename({
        "column_1": "accession",
        "column_2": "taxonomy"
    })
    df_tax = df_tax.with_columns(
        pl.col("taxonomy").str.split(";").alias("tax_list")
    ).with_columns([
        pl.col("tax_list").list.get(0).alias("kingdom"),
        pl.col("tax_list").list.get(1).alias("phylum"),
        pl.col("tax_list").list.get(2).alias("class"),
        pl.col("tax_list").list.get(3).alias("order"),
        pl.col("tax_list").list.get(4).alias("family"),
        pl.col("tax_list").list.get(5).alias("genus"),
        pl.col("tax_list").list.get(6).alias("species"),
    ])
    df_tax.filter(pl.col("accession") == "RS_GCF_000007825.1")
    return (df_tax,)


@app.cell
def _(df_tax):
    df_tax.shape
    return


@app.cell
def _(df_tax, pl):
    df_tax_clean = (
        df_tax
        .with_columns([
            pl.col("kingdom")
              .str.replace(r"^[a-z]__", "")
              .str.replace_all("_", " ")
              .str.to_lowercase()
              .alias("kingdom_name"),

            pl.col("phylum")
              .str.replace(r"^[a-z]__", "")
              .str.replace_all("_", " ")
              .str.to_lowercase()
              .alias("phylum_name"),

            pl.col("class")
              .str.replace(r"^[a-z]__", "")
              .str.replace_all("_", " ")
              .str.to_lowercase()
              .alias("class_name"),

            pl.col("order")
              .str.replace(r"^[a-z]__", "")
              .str.replace_all("_", " ")
              .str.to_lowercase()
              .alias("order_name"),

            pl.col("family")
              .str.replace(r"^[a-z]__", "")
              .str.replace_all("_", " ")
              .str.to_lowercase()
              .alias("family_name"),

            # intermediate cleaned genus/species
            pl.col("genus")
              .str.replace(r"^[a-z]__", "")
              .str.replace_all("_", " ")
              .str.to_lowercase()
              .alias("genus_clean"),

            pl.col("species")
              .str.replace(r"^[a-z]__", "")
              .str.replace_all("_", " ")
              .str.to_lowercase()
              .alias("species_clean"),
        ])
        # extract species epithet (last word)
        .with_columns(
            pl.col("species_clean")
              .str.split(" ")
              .list.get(-1)
              .alias("species_epithet")
        )
        # glue ANY trailing single letter onto genus: "paenibacillus h" -> "paenibacillush"
        .with_columns(
            pl.when(
                pl.col("genus_clean").str.contains(r" [a-z]{1,2}$", literal=False)
            )
            .then(
                pl.col("genus_clean").str.replace(
                    r"\s([a-z]{1,2})$",
                    "$1",
                    literal=False,   # regex mode so $1 is the captured letter
                )
            )
            .otherwise(pl.col("genus_clean"))
            .alias("genus_name")
        )
        # now that genus_name exists, build species_name
        .with_columns(
            (pl.col("genus_name") + " " + pl.col("species_epithet")).alias("species_name")
        )
        .select([
            "kingdom_name",
            "phylum_name",
            "class_name",
            "order_name",
            "family_name",
            "genus_name",
            "species_name",
        ])
        .unique()
    )

    return (df_tax_clean,)


@app.cell
def _(df_tax_clean, pl):
    df_tax_clean.filter(
        pl.col("species_name").str.contains("^clostridium ", literal=False)).sort("species_name")
    return


@app.cell
def _(df_meta, pl, sample_id):
    df_panspace = pl.read_csv(
        sample_id,
        separator="\t",
        has_header=True
    )

    df_joined = df_meta.join(
        df_panspace,
        left_on="ncbi_biosample",
        right_on="sample_accession",
        how="left"    # or "left" depending on your needs
    )
    return (df_panspace,)


@app.cell
def _(df_panspace, pl):
    df_panspace_clean = df_panspace.with_columns(
        pl.col("species")
          .str.replace_all("_", " ")
          .str.to_lowercase()
          .alias("species_name")
    )
    return (df_panspace_clean,)


@app.cell
def _(df_panspace_clean, df_tax_clean):
    df_panspace_tax = df_panspace_clean.join(
        df_tax_clean,
        on="species_name",
        how="left"
    )
    return (df_panspace_tax,)


@app.cell
def _(df_panspace_tax):
    df_panspace_tax.write_csv("/Users/lourdes/Desktop/arrakis/taxonomy.tsv", separator="\t", include_header=True)
    return


if __name__ == "__main__":
    app.run()
