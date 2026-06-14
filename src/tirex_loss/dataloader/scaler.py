import polars as pl

def scale_series(df: pl.DataFrame) -> pl.DataFrame:
    """
    Z-score scales each time series (grouped by series_index) in-place.
    Adds 'value_scaled', 'loc', 'scale' columns.
    """
    return (
        df
        .with_columns([
            pl.col("value").mean().over("series_index").alias("loc"),
            pl.col("value").std().over("series_index").alias("scale"),
        ])
        .with_columns([
            # guard: if std == 0, use abs(mean) + eps as scale
            pl.when(pl.col("scale") == 0)
              .then((pl.col("loc").abs() + 1e-5))
              .otherwise(pl.col("scale"))
              .alias("scale")
        ])
        .with_columns([
            ((pl.col("value") - pl.col("loc")) / pl.col("scale")).alias("value_scaled")
        ])
    )


def rescale_series(df: pl.DataFrame) -> pl.DataFrame:
    """
    Inverse transform: reconstructs original values from value_scaled + stored loc/scale.
    """
    return df.with_columns([
        (pl.col("value_scaled") * pl.col("scale") + pl.col("loc")).alias("value_rescaled")
    ])