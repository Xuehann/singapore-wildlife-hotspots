#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")
warnings.filterwarnings("ignore", message=".*keep_geom_type=True.*")

TARGET_CRS = 3414
ANIMAL_ENCODINGS = ("latin1", "cp1252", "utf-8")
MODEL_FAMILIES = ("random_forest", "poisson_regression")
ENVIRONMENT_BASES = ("realized", "planned")
FORECAST_HORIZONS = ("next_quarter", "next_year")


@dataclass
class Config:
    base_dir: Path
    output_dir: Path
    hex_radius_m: float = 500.0
    density_radius_m: float = 500.0
    kernel_radius_m: float = 1500.0
    train_fraction: float = 0.8
    random_state: int = 42
    top_landuse_categories: int = 12
    backtest_start_year: int = 2018
    min_training_periods: int = 8
    rf_n_estimators: int = 250
    rf_min_samples_leaf: int = 2
    poisson_max_iter: int = 2000


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message: str) -> None:
    print(f"[wildlife] {message}", flush=True)


def minmax(series: pd.Series) -> pd.Series:
    values = series.fillna(0.0).astype(float)
    lower = float(values.min())
    upper = float(values.max())
    if math.isclose(lower, upper):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lower) / (upper - lower)


def safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    value = left.corr(right)
    return None if pd.isna(value) else float(value)


def read_vector(path: Path, *, animal_layer: bool = False) -> gpd.GeoDataFrame:
    if animal_layer:
        for encoding in ANIMAL_ENCODINGS:
            try:
                return gpd.read_file(path, engine="fiona", encoding=encoding)
            except Exception:
                continue
        raise RuntimeError(f"Unable to read animal layer: {path}")
    return gpd.read_file(path)


def unify_projection(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input layer is missing CRS information.")
    if gdf.crs.to_epsg() == TARGET_CRS:
        return gdf
    return gdf.to_crs(TARGET_CRS)


def read_boundary(config: Config) -> tuple[gpd.GeoDataFrame, shapely.Geometry]:
    boundary = read_vector(config.base_dir / "SGBoundary" / "SGBoundary.shp")
    boundary = unify_projection(boundary)
    boundary = boundary[boundary.geometry.notna()].copy()
    boundary_union = boundary.geometry.union_all()
    return boundary, boundary_union


def clip_to_boundary(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary_union = boundary.geometry.union_all()
    return gdf[gdf.geometry.intersects(boundary_union)].copy()


def standardize_animal_columns(gdf: gpd.GeoDataFrame, species: str) -> gpd.GeoDataFrame:
    rename_map = {
        "common_nam": "common_name",
        "descriptio": "description",
        "quality_gr": "quality_grade",
        "observatio": "observation_id",
    }
    gdf = gdf.rename(columns=rename_map)
    gdf["Date"] = pd.to_datetime(gdf["Date"], errors="coerce")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf["Date"].notna()].copy()
    gdf["species"] = species
    gdf["x"] = gdf.geometry.x.round(3)
    gdf["y"] = gdf.geometry.y.round(3)
    gdf = gdf.drop_duplicates(subset=["x", "y", "Date"])
    keep = ["species", "Date", "common_name", "description", "geometry"]
    optional = [col for col in ("quality_grade", "observation_id") if col in gdf.columns]
    return gdf[keep + optional].copy()


def load_animal_points(config: Config, boundary: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    outputs: dict[str, gpd.GeoDataFrame] = {}
    for species in ("Monkeys", "Otters"):
        gdf = read_vector(config.base_dir / species / f"{species}.shp", animal_layer=True)
        gdf = unify_projection(gdf)
        gdf = clip_to_boundary(gdf, boundary)
        outputs[species.lower()] = standardize_animal_columns(gdf, species.lower())
    return outputs


def load_context_layers(config: Config, boundary: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    names = [
        "Roads",
        "Waterways",
        "Masterplan2025",
        "Parkconnector",
        "ParksNaturereserves",
    ]
    layers: dict[str, gpd.GeoDataFrame] = {}
    for name in names:
        gdf = read_vector(config.base_dir / name / f"{name}.shp")
        gdf = unify_projection(gdf)
        gdf = clip_to_boundary(gdf, boundary)
        layers[name.lower()] = gdf[gdf.geometry.notna()].copy()
    for optional in ("Buildings", "SubzonePopulation2019"):
        path = config.base_dir / optional / f"{optional}.shp"
        if not path.exists():
            continue
        gdf = read_vector(path)
        gdf = unify_projection(gdf)
        gdf = clip_to_boundary(gdf, boundary)
        layers[optional.lower()] = gdf[gdf.geometry.notna()].copy()
    return layers


def make_hexagon(cx: float, cy: float, radius: float) -> Polygon:
    angles = [math.radians(angle) for angle in range(0, 360, 60)]
    points = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
    return Polygon(points)


def build_hex_grid(boundary_union: shapely.Geometry, radius: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = boundary_union.bounds
    dx = 1.5 * radius
    dy = math.sqrt(3) * radius
    geometries = []
    ids = []
    column = 0
    x = minx - 2 * radius
    idx = 0
    while x <= maxx + 2 * radius:
        y_offset = dy / 2 if column % 2 else 0.0
        y = miny - 2 * radius
        while y <= maxy + 2 * radius:
            hexagon = make_hexagon(x, y + y_offset, radius)
            if hexagon.intersects(boundary_union):
                geometries.append(hexagon)
                ids.append(f"hex_{idx:05d}")
                idx += 1
            y += dy
        x += dx
        column += 1
    return gpd.GeoDataFrame({"hex_id": ids}, geometry=geometries, crs=TARGET_CRS).reset_index(drop=True)


def nearest_distance(centroids: gpd.GeoDataFrame, target: gpd.GeoDataFrame, field: str) -> pd.DataFrame:
    if target.empty:
        return pd.DataFrame({"hex_id": centroids["hex_id"], field: np.nan})
    nearest = gpd.sjoin_nearest(
        centroids[["hex_id", "geometry"]],
        target[["geometry"]],
        how="left",
        distance_col=field,
    )
    nearest = nearest.groupby("hex_id", as_index=False)[field].min()
    return nearest


def buffered_line_density(
    centroids: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    radius_m: float,
    prefix: str,
) -> pd.DataFrame:
    out = pd.DataFrame({"hex_id": centroids["hex_id"]})
    out[f"{prefix}_length_m_in_buffer"] = 0.0
    out[f"{prefix}_density_km_per_km2"] = 0.0
    if lines.empty:
        return out

    buffers = centroids[["hex_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius_m)
    joined = gpd.sjoin(lines[["geometry"]], buffers[["hex_id", "geometry"]], predicate="intersects", how="inner")
    if joined.empty:
        return out

    buffer_lookup = buffers.set_index("hex_id").geometry
    clipped = shapely.intersection(joined.geometry.array, buffer_lookup.loc[joined["hex_id"]].array)
    joined["segment_length_m"] = shapely.length(clipped)
    length_sum = joined.groupby("hex_id")["segment_length_m"].sum()
    area_km2 = math.pi * (radius_m**2) / 1_000_000
    out = out.set_index("hex_id")
    out.loc[length_sum.index, f"{prefix}_length_m_in_buffer"] = length_sum
    out.loc[length_sum.index, f"{prefix}_density_km_per_km2"] = (length_sum / 1000.0) / area_km2
    return out.reset_index()


def polygon_intersection_share(
    hexes: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    field_name: str,
) -> pd.DataFrame:
    out = pd.DataFrame({"hex_id": hexes["hex_id"], field_name: 0.0})
    if polygons.empty:
        return out
    overlay = gpd.overlay(hexes[["hex_id", "geometry"]], polygons[["geometry"]], how="intersection", keep_geom_type=False)
    if overlay.empty:
        return out
    overlay["inter_area_m2"] = overlay.geometry.area
    share = overlay.groupby("hex_id")["inter_area_m2"].sum() / hexes.set_index("hex_id").geometry.area
    out = out.set_index("hex_id")
    out.loc[share.index, field_name] = share.clip(upper=1.0)
    return out.reset_index()


def population_weighted_feature(
    hexes: gpd.GeoDataFrame,
    population_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame({"hex_id": hexes["hex_id"], "population_weighted_density": 0.0})
    if population_gdf.empty:
        return out
    numeric_candidates = [
        col
        for col in population_gdf.columns
        if col != "geometry" and "pop" in col.lower() and pd.api.types.is_numeric_dtype(population_gdf[col])
    ]
    if not numeric_candidates:
        return out
    pop_field = numeric_candidates[0]
    source = population_gdf[[pop_field, "geometry"]].copy().reset_index(names="source_id")
    overlay = gpd.overlay(hexes[["hex_id", "geometry"]], source, how="intersection", keep_geom_type=False)
    if overlay.empty:
        return out
    overlay["inter_area_m2"] = overlay.geometry.area
    source_area = source.geometry.area.replace(0, np.nan)
    density_lookup = (source[pop_field] / source_area).fillna(0.0)
    overlay["population_density"] = overlay["source_id"].map(density_lookup)
    weighted = overlay.groupby("hex_id").apply(
        lambda frame: float((frame["population_density"] * frame["inter_area_m2"]).sum() / max(frame["inter_area_m2"].sum(), 1.0))
    )
    out = out.set_index("hex_id")
    out.loc[weighted.index, "population_weighted_density"] = weighted
    return out.reset_index()


def landuse_features(
    hexes: gpd.GeoDataFrame,
    masterplan: gpd.GeoDataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, list[str]]:
    overlay = gpd.overlay(
        hexes[["hex_id", "geometry"]],
        masterplan[["LU_DESC", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if overlay.empty:
        empty = pd.DataFrame({"hex_id": hexes["hex_id"], "dominant_landuse": "UNKNOWN", "dominant_landuse_share": 0.0})
        return empty, []

    overlay["inter_area_m2"] = overlay.geometry.area
    total_area = overlay.groupby("hex_id")["inter_area_m2"].sum().rename("hex_area")
    by_lu = overlay.groupby(["hex_id", "LU_DESC"])["inter_area_m2"].sum().reset_index()
    by_lu = by_lu.merge(total_area, on="hex_id")
    by_lu["share"] = by_lu["inter_area_m2"] / by_lu["hex_area"]

    dominant = by_lu.sort_values(["hex_id", "share"], ascending=[True, False]).drop_duplicates("hex_id")
    dominant = dominant.rename(columns={"LU_DESC": "dominant_landuse", "share": "dominant_landuse_share"})

    top_categories = list(
        by_lu.groupby("LU_DESC")["inter_area_m2"].sum().sort_values(ascending=False).head(top_n).index
    )
    by_lu["lu_group"] = np.where(by_lu["LU_DESC"].isin(top_categories), by_lu["LU_DESC"], "OTHER")
    pivot = by_lu.groupby(["hex_id", "lu_group"])["share"].sum().unstack(fill_value=0.0)
    pivot = pivot.rename(columns={col: f"lu_share_{slugify(col)}" for col in pivot.columns})

    features = hexes[["hex_id"]].merge(
        dominant[["hex_id", "dominant_landuse", "dominant_landuse_share"]],
        on="hex_id",
        how="left",
    )
    features = features.merge(pivot.reset_index(), on="hex_id", how="left").fillna(0.0)
    green_columns = [col for col in pivot.columns if any(key in col for key in ["park", "open_space", "sports_recreation", "reserve", "agriculture", "beach_area"])]
    water_columns = [col for col in pivot.columns if "waterbody" in col]
    road_columns = [col for col in pivot.columns if "road" in col]
    features["green_landuse_share"] = features[green_columns].sum(axis=1) if green_columns else 0.0
    features["water_landuse_share"] = features[water_columns].sum(axis=1) if water_columns else 0.0
    features["road_landuse_share"] = features[road_columns].sum(axis=1) if road_columns else 0.0
    return features, top_categories


def compute_static_features(
    config: Config,
    hexes: gpd.GeoDataFrame,
    layers: dict[str, gpd.GeoDataFrame],
) -> tuple[gpd.GeoDataFrame, list[str]]:
    centroids = hexes[["hex_id", "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid

    features = pd.DataFrame({"hex_id": hexes["hex_id"]})
    log("static features: nearest distances")
    features = features.merge(nearest_distance(centroids, layers["roads"], "dist_to_road_m"), on="hex_id", how="left")
    features = features.merge(nearest_distance(centroids, layers["waterways"], "dist_to_waterway_m"), on="hex_id", how="left")
    features = features.merge(nearest_distance(centroids, layers["parkconnector"], "dist_to_parkconnector_m"), on="hex_id", how="left")

    log("static features: line densities")
    features = features.merge(buffered_line_density(centroids, layers["roads"], config.density_radius_m, "road"), on="hex_id", how="left")
    features = features.merge(buffered_line_density(centroids, layers["waterways"], config.density_radius_m, "waterway"), on="hex_id", how="left")
    features = features.merge(polygon_intersection_share(hexes, layers["parksnaturereserves"], "park_reserve_share"), on="hex_id", how="left")

    if "buildings" in layers:
        features = features.merge(polygon_intersection_share(hexes, layers["buildings"], "building_share"), on="hex_id", how="left")
    else:
        features["building_share"] = 0.0
    if "subzonepopulation2019" in layers:
        features = features.merge(population_weighted_feature(hexes, layers["subzonepopulation2019"]), on="hex_id", how="left")
    else:
        features["population_weighted_density"] = 0.0

    scale_m = max(config.hex_radius_m, 1.0)
    features["in_park_reserve"] = (features["park_reserve_share"] > 0).astype(int)
    features["park_connector_access"] = 1.0 / (1.0 + features["dist_to_parkconnector_m"] / scale_m)
    features["road_access"] = 1.0 / (1.0 + features["dist_to_road_m"] / scale_m)
    features["waterway_access"] = 1.0 / (1.0 + features["dist_to_waterway_m"] / scale_m)
    features["human_pressure_score"] = 0.6 * minmax(features["building_share"]) + 0.4 * minmax(features["population_weighted_density"])

    features["realized_monkey_context_score"] = (
        0.45 * minmax(features["park_reserve_share"])
        + 0.25 * minmax(features["park_connector_access"])
        + 0.20 * (1.0 - minmax(features["road_access"]))
        + 0.10 * (1.0 - minmax(features["human_pressure_score"]))
    )
    features["realized_otter_context_score"] = (
        0.45 * minmax(features["waterway_access"])
        + 0.30 * minmax(features["waterway_density_km_per_km2"])
        + 0.15 * minmax(features["park_reserve_share"])
        + 0.10 * (1.0 - minmax(features["road_access"]))
    )

    top_categories: list[str] = []
    if "masterplan2025" in layers:
        log("static features: land-use overlay")
        landuse, top_categories = landuse_features(hexes, layers["masterplan2025"], config.top_landuse_categories)
        features = features.merge(landuse, on="hex_id", how="left")
    else:
        features["dominant_landuse"] = "UNKNOWN"
        features["dominant_landuse_share"] = 0.0
        features["green_landuse_share"] = 0.0
        features["water_landuse_share"] = 0.0
        features["road_landuse_share"] = 0.0

    features["planned_monkey_context_score"] = (
        0.35 * minmax(features["green_landuse_share"])
        + 0.25 * minmax(features["park_reserve_share"])
        + 0.20 * minmax(features["park_connector_access"])
        + 0.20 * (1.0 - minmax(features["road_landuse_share"]))
    )
    features["planned_otter_context_score"] = (
        0.35 * minmax(features["water_landuse_share"])
        + 0.30 * minmax(features["waterway_access"])
        + 0.20 * minmax(features["waterway_density_km_per_km2"])
        + 0.15 * minmax(features["park_reserve_share"])
    )

    merged = hexes.merge(features, on="hex_id", how="left").drop_duplicates("hex_id")
    return merged, top_categories


def build_hex_neighbors(hexes: gpd.GeoDataFrame) -> dict[str, list[str]]:
    left = hexes[["hex_id", "geometry"]].rename(columns={"hex_id": "src_hex"})
    right = hexes[["hex_id", "geometry"]].rename(columns={"hex_id": "dst_hex"})
    touching = gpd.sjoin(left, right, predicate="touches", how="inner")
    touching = touching[touching["src_hex"] != touching["dst_hex"]].copy()
    neighbors: dict[str, set[str]] = {hex_id: set() for hex_id in hexes["hex_id"]}
    for row in touching[["src_hex", "dst_hex"]].itertuples(index=False):
        neighbors[row.src_hex].add(row.dst_hex)
    return {key: sorted(value) for key, value in neighbors.items()}


def build_spatial_kernel_weights(hexes: gpd.GeoDataFrame, radius_m: float) -> dict[str, list[tuple[str, float]]]:
    centroids = hexes[["hex_id", "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid
    source_buffers = centroids.rename(columns={"hex_id": "src_hex"})
    source_buffers["geometry"] = source_buffers.geometry.buffer(radius_m)
    targets = centroids.rename(columns={"hex_id": "dst_hex"})
    pairs = gpd.sjoin(targets, source_buffers[["src_hex", "geometry"]], predicate="within", how="inner")
    if pairs.empty:
        return {hex_id: [(hex_id, 1.0)] for hex_id in hexes["hex_id"]}
    source_points = centroids.set_index("hex_id").geometry
    pairs["distance_m"] = shapely.distance(pairs.geometry.array, source_points.loc[pairs["src_hex"]].array)
    pairs["weight"] = 1.0 / (1.0 + (pairs["distance_m"] / max(radius_m, 1.0)))
    kernel_weights: dict[str, list[tuple[str, float]]] = {}
    for src_hex, frame in pairs.groupby("src_hex"):
        weights = frame[["dst_hex", "weight"]].copy()
        total = float(weights["weight"].sum())
        if total <= 0:
            kernel_weights[src_hex] = [(src_hex, 1.0)]
            continue
        kernel_weights[src_hex] = [(row.dst_hex, float(row.weight / total)) for row in weights.itertuples(index=False)]
    return kernel_weights


def matrix_to_long(matrix: pd.DataFrame, value_name: str, period_col: str) -> pd.DataFrame:
    long_df = matrix.stack(future_stack=True).rename(value_name).reset_index()
    return long_df.rename(columns={"level_0": period_col, "level_1": "hex_id"})


def build_panel(
    hexes: gpd.GeoDataFrame,
    animal_points: gpd.GeoDataFrame,
    static_features: gpd.GeoDataFrame,
    species: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = gpd.sjoin(animal_points, hexes[["hex_id", "geometry"]], predicate="within", how="inner")
    joined["quarter"] = joined["Date"].dt.to_period("Q").dt.start_time
    quarterly_obs = joined.groupby(["hex_id", "quarter"]).agg(observation_count=("species", "size")).reset_index()

    quarters = pd.date_range(joined["quarter"].min(), joined["quarter"].max(), freq="QS")
    panel = pd.MultiIndex.from_product([hexes["hex_id"], quarters], names=["hex_id", "quarter"]).to_frame(index=False)
    panel = panel.merge(quarterly_obs, on=["hex_id", "quarter"], how="left")
    panel["observation_count"] = panel["observation_count"].fillna(0).astype(int)

    panel = panel.merge(static_features.drop(columns="geometry"), on="hex_id", how="left")
    panel["species"] = species
    panel["quarter_num"] = panel["quarter"].dt.quarter
    panel["year"] = panel["quarter"].dt.year
    panel["year_index"] = panel["year"] - int(panel["year"].min())
    panel["quarter_sin"] = np.sin(2 * np.pi * panel["quarter_num"] / 4.0)
    panel["quarter_cos"] = np.cos(2 * np.pi * panel["quarter_num"] / 4.0)
    panel["season_index"] = panel["quarter_num"] - 1
    if species == "otters":
        panel["realized_species_context_score"] = panel["realized_otter_context_score"]
        panel["planned_species_context_score"] = panel["planned_otter_context_score"]
    else:
        panel["realized_species_context_score"] = panel["realized_monkey_context_score"]
        panel["planned_species_context_score"] = panel["planned_monkey_context_score"]
    return panel, quarterly_obs


def add_temporal_features(
    panel: pd.DataFrame,
    hex_ids: list[str],
    periods: pd.DatetimeIndex,
    neighbors: dict[str, list[str]],
    kernel_weights: dict[str, list[tuple[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.sort_values(["hex_id", "quarter"]).reset_index(drop=True)
    by_hex = panel.groupby("hex_id", sort=False)

    panel["intensity_lag_1_quarter"] = by_hex["observation_count"].shift(1).fillna(0.0)
    panel["intensity_lag_2_quarters_sum"] = by_hex["observation_count"].transform(
        lambda s: s.shift(1).rolling(2, min_periods=1).sum()
    ).fillna(0.0)
    panel["intensity_lag_4_quarters_sum"] = by_hex["observation_count"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).sum()
    ).fillna(0.0)

    panel["target_next_quarter_intensity"] = by_hex["observation_count"].shift(-1)
    panel["target_next_year_intensity"] = (
        by_hex["observation_count"].shift(-1).fillna(0.0)
        + by_hex["observation_count"].shift(-2).fillna(0.0)
        + by_hex["observation_count"].shift(-3).fillna(0.0)
        + by_hex["observation_count"].shift(-4).fillna(0.0)
    )
    panel["target_next_year_available"] = (
        by_hex["observation_count"].shift(-1).notna()
        & by_hex["observation_count"].shift(-2).notna()
        & by_hex["observation_count"].shift(-3).notna()
        & by_hex["observation_count"].shift(-4).notna()
    )
    panel["forecast_target_quarter"] = panel["quarter"] + pd.offsets.QuarterBegin(1)
    panel["forecast_target_year_start"] = panel["quarter"] + pd.offsets.QuarterBegin(1)

    count_matrix = (
        panel.pivot(index="quarter", columns="hex_id", values="observation_count")
        .reindex(index=periods, columns=hex_ids, fill_value=0)
        .astype(float)
    )
    count_prev = count_matrix.shift(1).fillna(0.0)
    recent2 = count_matrix.rolling(2, min_periods=1).sum().fillna(0.0)

    count_prev_arr = count_prev.to_numpy()
    recent2_arr = recent2.to_numpy()
    neighbor_prev = np.zeros_like(count_prev_arr, dtype=float)
    kernel_recent = np.zeros_like(recent2_arr, dtype=float)
    index_lookup = {hex_id: idx for idx, hex_id in enumerate(hex_ids)}

    for hex_id, idx in index_lookup.items():
        nbrs = neighbors.get(hex_id, [])
        nbr_idx = [index_lookup[n] for n in nbrs if n in index_lookup]
        neighbor_prev[:, idx] = count_prev_arr[:, nbr_idx].mean(axis=1) if nbr_idx else 0.0

        kernel_pairs = kernel_weights.get(hex_id, [(hex_id, 1.0)])
        kernel_idx = [index_lookup[n] for n, _ in kernel_pairs if n in index_lookup]
        kernel_w = np.array([weight for n, weight in kernel_pairs if n in index_lookup], dtype=float)
        kernel_recent[:, idx] = (recent2_arr[:, kernel_idx] * kernel_w).sum(axis=1) if len(kernel_idx) else 0.0

    panel = panel.merge(
        matrix_to_long(pd.DataFrame(neighbor_prev, index=periods, columns=hex_ids), "neighbor_intensity_lag_1_quarter", "quarter"),
        on=["quarter", "hex_id"],
        how="left",
    )
    panel = panel.merge(
        matrix_to_long(pd.DataFrame(kernel_recent, index=periods, columns=hex_ids), "species_kernel_recent_intensity", "quarter"),
        on=["quarter", "hex_id"],
        how="left",
    )
    panel["used_current_quarter"] = (panel["observation_count"] > 0).astype(int)
    return panel, panel.copy()


def build_feature_columns(panel: pd.DataFrame, environment_basis: str) -> list[str]:
    excluded = {
        "hex_id",
        "quarter",
        "species",
        "forecast_target_quarter",
        "forecast_target_year_start",
        "target_next_quarter_intensity",
        "target_next_year_intensity",
        "target_next_year_available",
        "used_current_quarter",
    }
    feature_cols = [col for col in panel.columns if col not in excluded]
    planned_only = {
        "dominant_landuse",
        "dominant_landuse_share",
        "green_landuse_share",
        "water_landuse_share",
        "road_landuse_share",
        "planned_monkey_context_score",
        "planned_otter_context_score",
        "planned_species_context_score",
    }
    planned_only.update({col for col in panel.columns if col.startswith("lu_share_")})
    if environment_basis == "realized":
        feature_cols = [col for col in feature_cols if col not in planned_only]
    elif environment_basis == "planned":
        pass
    else:
        raise ValueError(f"Unknown environment basis: {environment_basis}")
    return feature_cols


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    other_frames: Iterable[pd.DataFrame],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    categorical = [col for col in ("dominant_landuse",) if col in feature_cols]
    train_X = pd.get_dummies(train_df[feature_cols].copy(), columns=categorical, dummy_na=True).fillna(0.0)
    transformed = []
    for frame in other_frames:
        X = pd.get_dummies(frame[feature_cols].copy(), columns=categorical, dummy_na=True)
        X = X.reindex(columns=train_X.columns, fill_value=0.0).fillna(0.0)
        transformed.append(X)
    return train_X, transformed


def choose_backtest_periods(panel: pd.DataFrame, config: Config, target_col: str) -> list[pd.Timestamp]:
    if target_col == "target_next_year_intensity":
        available = panel[panel["target_next_year_available"]]
    else:
        available = panel[panel[target_col].notna()]
    periods = sorted(pd.Timestamp(p) for p in available["quarter"].unique())
    if not periods:
        return []
    preferred_start = pd.Timestamp(config.backtest_start_year, 1, 1)
    min_start_idx = min(config.min_training_periods, max(len(periods) - 1, 0))
    fallback_start = periods[min_start_idx] if min_start_idx < len(periods) else periods[-1]
    start_period = max(preferred_start, fallback_start)
    selected = [period for period in periods if period >= start_period]
    if len(selected) < 3:
        split_idx = max(config.min_training_periods, int(len(periods) * config.train_fraction))
        split_idx = min(split_idx, len(periods) - 1)
        selected = periods[split_idx:]
    return selected


def build_model(model_family: str, config: Config):
    if model_family == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=config.random_state,
        )
    if model_family == "poisson_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=0.001, max_iter=config.poisson_max_iter)),
            ]
        )
    raise ValueError(f"Unknown model family: {model_family}")


def extract_model_explanation(model, model_family: str, feature_names: list[str]) -> dict[str, dict[str, float]]:
    if model_family == "random_forest":
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return {"feature_importance_top20": {k: float(v) for k, v in importances.head(20).items()}}
    estimator = model.named_steps["model"]
    coefficients = pd.Series(estimator.coef_, index=feature_names).sort_values(ascending=False)
    return {
        "top_positive_coefficients": {k: float(v) for k, v in coefficients.head(20).items()},
        "top_negative_coefficients": {k: float(v) for k, v in coefficients.tail(20).sort_values().items()},
    }


def intensity_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {
            "mae": None,
            "rmse": None,
            "rank_correlation": None,
        }
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    truth = y_true.astype(float).to_numpy()
    rank_corr = pd.Series(truth).rank(method="average").corr(pd.Series(pred).rank(method="average"))
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "rank_correlation": None if pd.isna(rank_corr) else float(rank_corr),
    }


def hotspot_capture_metrics(y_true: pd.Series, y_pred: np.ndarray, share: float = 0.10) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {
            "top_5_percent_capture_rate": None,
            "top_10_percent_capture_rate": None,
            "top_10_percent_mean_actual_intensity": None,
        }
    pred = np.asarray(y_pred, dtype=float)
    truth = y_true.astype(float).to_numpy()
    total_actual = float(truth.sum())
    results: dict[str, float | None] = {}
    for pct in (0.05, 0.10):
        k = max(1, int(math.ceil(len(truth) * pct)))
        idx = np.argpartition(pred, -k)[-k:]
        captured = float(truth[idx].sum())
        results[f"top_{int(pct * 100)}_percent_capture_rate"] = None if math.isclose(total_actual, 0.0) else float(captured / total_actual)
        if math.isclose(pct, 0.10):
            results["top_10_percent_mean_actual_intensity"] = float(truth[idx].mean())
    return results


def metric_value(metrics: dict[str, object], key: str, *, lower_is_better: bool = False) -> float:
    value = metrics.get(key)
    if value is None:
        return float("-inf")
    numeric = float(value)
    return -numeric if lower_is_better else numeric


def choose_recommended_model(model_results: dict[str, dict[str, object]]) -> dict[str, object]:
    flattened = []
    for model_family, payload in model_results.items():
        flattened.append(
            {
                "model_family": model_family,
                "metrics": payload["intensity_metrics"],
                "hotspot_metrics": payload["hotspot_metrics"],
            }
        )
    ranked = sorted(
        flattened,
        key=lambda item: (
            metric_value(item["hotspot_metrics"], "top_10_percent_capture_rate"),
            metric_value(item["hotspot_metrics"], "top_5_percent_capture_rate"),
            metric_value(item["metrics"], "rank_correlation"),
            metric_value(item["metrics"], "rmse", lower_is_better=True),
            metric_value(item["metrics"], "mae", lower_is_better=True),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        "best_by_top_10_percent_capture_rate": max(flattened, key=lambda item: metric_value(item["hotspot_metrics"], "top_10_percent_capture_rate"))["model_family"],
        "best_by_rank_correlation": max(flattened, key=lambda item: metric_value(item["metrics"], "rank_correlation"))["model_family"],
        "best_by_rmse": min(flattened, key=lambda item: float("inf") if item["metrics"]["rmse"] is None else float(item["metrics"]["rmse"]))["model_family"],
        "recommended_model": best["model_family"],
    }


def compare_forecast_surfaces(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    intensity_col: str,
) -> dict[str, float | None]:
    merged = left[["hex_id", intensity_col, "hotspot_flag"]].merge(
        right[["hex_id", intensity_col, "hotspot_flag"]],
        on="hex_id",
        suffixes=("_left", "_right"),
    )
    corr = safe_corr(merged[f"{intensity_col}_left"], merged[f"{intensity_col}_right"])
    left_hot = set(merged[merged["hotspot_flag_left"] == 1]["hex_id"])
    right_hot = set(merged[merged["hotspot_flag_right"] == 1]["hex_id"])
    denom = max(len(left_hot | right_hot), 1)
    overlap = len(left_hot & right_hot) / float(denom)
    mean_delta = float((merged[f"{intensity_col}_right"] - merged[f"{intensity_col}_left"]).mean())
    return {
        "intensity_correlation": corr,
        "hotspot_overlap_jaccard": overlap,
        "mean_intensity_delta_planned_minus_realized": mean_delta,
    }


def assign_hotspot_fields(forecast_hexes: gpd.GeoDataFrame, intensity_col: str) -> gpd.GeoDataFrame:
    forecast_hexes = forecast_hexes.copy()
    forecast_hexes["hotspot_rank"] = forecast_hexes[intensity_col].rank(method="first", ascending=False).astype(int)
    k = max(1, int(math.ceil(len(forecast_hexes) * 0.10)))
    forecast_hexes["hotspot_flag"] = 0
    top_idx = forecast_hexes.nlargest(k, intensity_col).index
    forecast_hexes.loc[top_idx, "hotspot_flag"] = 1
    return forecast_hexes


def classify_habitat_context(forecast_hexes: gpd.GeoDataFrame, environment_basis: str) -> pd.Series:
    realized_natural = (
        0.45 * minmax(forecast_hexes["park_reserve_share"])
        + 0.35 * minmax(forecast_hexes["waterway_access"])
        + 0.20 * (1.0 - minmax(forecast_hexes["road_access"]))
    )
    realized_urban = (
        0.45 * minmax(forecast_hexes["road_access"])
        + 0.35 * minmax(forecast_hexes["human_pressure_score"])
        + 0.20 * (1.0 - minmax(forecast_hexes["park_reserve_share"]))
    )
    if environment_basis == "planned":
        natural = (
            0.40 * realized_natural
            + 0.35 * minmax(forecast_hexes["green_landuse_share"])
            + 0.25 * minmax(forecast_hexes["water_landuse_share"])
        )
        urban = (
            0.45 * realized_urban
            + 0.35 * minmax(forecast_hexes["road_landuse_share"])
            + 0.20 * (1.0 - minmax(forecast_hexes["green_landuse_share"]))
        )
    else:
        natural = realized_natural
        urban = realized_urban

    labels = np.where(
        natural >= urban + 0.15,
        "natural",
        np.where(urban >= natural + 0.15, "urban", "mixed"),
    )
    return pd.Series(labels, index=forecast_hexes.index)


def make_intensity_map(
    species: str,
    environment_basis: str,
    forecast_horizon: str,
    boundary: gpd.GeoDataFrame,
    forecast_hexes: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    waterways: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    output_path: Path,
    target_period: pd.Timestamp,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    boundary.boundary.plot(ax=axes[0], color="black", linewidth=0.6)
    forecast_hexes.plot(ax=axes[0], column="predicted_intensity", cmap="YlGnBu", legend=True, linewidth=0)
    axes[0].set_title(f"{species.title()} {environment_basis} {forecast_horizon}\npredicted intensity\n{target_period.date()}")

    boundary.boundary.plot(ax=axes[1], color="black", linewidth=0.6)
    forecast_hexes.plot(
        ax=axes[1],
        column="habitat_context",
        categorical=True,
        legend=True,
        linewidth=0,
        cmap="Set2",
    )
    axes[1].set_title("Habitat context")

    boundary.boundary.plot(ax=axes[2], color="black", linewidth=0.6)
    forecast_hexes.plot(ax=axes[2], column="predicted_intensity", cmap="Greys", alpha=0.3, linewidth=0)
    hotspots = forecast_hexes[forecast_hexes["hotspot_flag"] == 1]
    if species == "otters":
        waterways.plot(ax=axes[2], color="#1d4ed8", linewidth=0.3, alpha=0.4)
    else:
        parks.boundary.plot(ax=axes[2], color="#166534", linewidth=0.4, alpha=0.5)
    if not roads.empty:
        roads.plot(ax=axes[2], color="#444444", linewidth=0.15, alpha=0.15)
    if not hotspots.empty:
        hotspots.boundary.plot(ax=axes[2], color="#d97706", linewidth=0.8)
    axes[2].set_title("Top predicted hotspot cells")

    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_predict_model(
    model_family: str,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: Config,
) -> tuple[np.ndarray, dict[str, dict[str, float]], list[str]]:
    train_X, (predict_X,) = prepare_feature_matrices(train_df, [predict_df], feature_cols)
    estimator = build_model(model_family, config)
    estimator.fit(train_X, train_df[target_col].astype(float))
    prediction = np.clip(np.asarray(estimator.predict(predict_X), dtype=float), 0.0, None)
    explanation = extract_model_explanation(estimator, model_family, list(train_X.columns))
    return prediction, explanation, list(train_X.columns)


def rolling_backtest(
    panel: pd.DataFrame,
    feature_cols: list[str],
    config: Config,
    model_family: str,
    target_col: str,
    target_period_col: str,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    periods = choose_backtest_periods(panel, config, target_col)
    predictions = []

    for period in periods:
        train_df = panel[panel["quarter"] < period].copy()
        test_df = panel[panel["quarter"] == period].copy()
        if target_col == "target_next_year_intensity":
            train_df = train_df[train_df["target_next_year_available"]].copy()
            test_df = test_df[test_df["target_next_year_available"]].copy()
        else:
            train_df = train_df[train_df[target_col].notna()].copy()
            test_df = test_df[test_df[target_col].notna()].copy()
        if train_df.empty or test_df.empty:
            continue

        pred, _, _ = fit_predict_model(model_family, train_df, test_df, feature_cols, target_col, config)
        frame = test_df[["hex_id", "quarter", target_period_col, target_col]].copy()
        frame["model_family"] = model_family
        frame["predicted_intensity"] = pred
        frame["actual_intensity"] = frame[target_col].astype(float)
        predictions.append(frame)

    if predictions:
        all_predictions = pd.concat(predictions, ignore_index=True)
        intensity = intensity_metrics(all_predictions["actual_intensity"], all_predictions["predicted_intensity"].to_numpy())
        hotspot = hotspot_capture_metrics(all_predictions["actual_intensity"], all_predictions["predicted_intensity"].to_numpy())
        intensity["n_backtest_periods"] = int(all_predictions["quarter"].nunique())
        intensity["backtest_period_start"] = str(pd.Timestamp(all_predictions["quarter"].min()).date())
        intensity["backtest_period_end"] = str(pd.Timestamp(all_predictions["quarter"].max()).date())
    else:
        all_predictions = pd.DataFrame(
            columns=["hex_id", "quarter", target_period_col, target_col, "model_family", "predicted_intensity", "actual_intensity"]
        )
        intensity = {
            "mae": None,
            "rmse": None,
            "rank_correlation": None,
            "n_backtest_periods": 0,
            "backtest_period_start": None,
            "backtest_period_end": None,
        }
        hotspot = {
            "top_5_percent_capture_rate": None,
            "top_10_percent_capture_rate": None,
            "top_10_percent_mean_actual_intensity": None,
        }
    return intensity, hotspot, all_predictions


def select_target_metadata(forecast_horizon: str) -> tuple[str, str]:
    if forecast_horizon == "next_quarter":
        return "target_next_quarter_intensity", "forecast_target_quarter"
    if forecast_horizon == "next_year":
        return "target_next_year_intensity", "forecast_target_year_start"
    raise ValueError(f"Unknown forecast horizon: {forecast_horizon}")


def forecast_surface(
    panel: pd.DataFrame,
    hex_features: gpd.GeoDataFrame,
    feature_cols: list[str],
    model_family: str,
    forecast_horizon: str,
    config: Config,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, dict[str, float]], pd.Timestamp]:
    target_col, target_period_col = select_target_metadata(forecast_horizon)
    train_df = panel.copy()
    if forecast_horizon == "next_year":
        train_df = train_df[train_df["target_next_year_available"]].copy()
    else:
        train_df = train_df[train_df[target_col].notna()].copy()

    forecast_template = panel[panel["quarter"] == panel["quarter"].max()].copy()
    pred, explanation, _ = fit_predict_model(model_family, train_df, forecast_template, feature_cols, target_col, config)
    forecast_template["predicted_intensity"] = pred
    target_period = pd.Timestamp(forecast_template[target_period_col].iloc[0])

    cols = [
        "hex_id",
        "quarter",
        target_period_col,
        "observation_count",
        "intensity_lag_1_quarter",
        "intensity_lag_2_quarters_sum",
        "intensity_lag_4_quarters_sum",
        "neighbor_intensity_lag_1_quarter",
        "species_kernel_recent_intensity",
        "predicted_intensity",
    ]
    forecast_hexes = hex_features.merge(forecast_template[cols], on="hex_id", how="left")
    forecast_hexes = assign_hotspot_fields(forecast_hexes, "predicted_intensity")
    forecast_frame = forecast_template[cols].copy()
    return forecast_hexes, forecast_frame, explanation, target_period


def run_environment_horizon(
    config: Config,
    species: str,
    environment_basis: str,
    forecast_horizon: str,
    panel: pd.DataFrame,
    hex_features: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    waterways: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    output_dir: Path,
) -> dict[str, object]:
    target_col, target_period_col = select_target_metadata(forecast_horizon)
    feature_cols = build_feature_columns(panel, environment_basis)
    model_results: dict[str, dict[str, object]] = {}
    backtest_frames = []

    for model_family in MODEL_FAMILIES:
        log(f"{species}: {environment_basis} {forecast_horizon} backtest {model_family}")
        intensity, hotspot, backtest = rolling_backtest(panel, feature_cols, config, model_family, target_col, target_period_col)
        model_results[model_family] = {
            "intensity_metrics": intensity,
            "hotspot_metrics": hotspot,
        }
        if not backtest.empty:
            backtest_frames.append(backtest)

    comparison_summary = choose_recommended_model(model_results)
    recommended_model = comparison_summary["recommended_model"]
    log(f"{species}: {environment_basis} {forecast_horizon} final forecast {recommended_model}")
    forecast_hexes, forecast_frame, explanation, target_period = forecast_surface(
        panel,
        hex_features,
        feature_cols,
        recommended_model,
        forecast_horizon,
        config,
    )
    forecast_hexes["habitat_context"] = classify_habitat_context(forecast_hexes, environment_basis)
    forecast_hexes["forecast_horizon"] = forecast_horizon
    forecast_hexes["environment_basis"] = "realized_plus_planned" if environment_basis == "planned" else "realized"
    forecast_frame["hotspot_rank"] = forecast_hexes.set_index("hex_id").loc[forecast_frame["hex_id"], "hotspot_rank"].to_numpy()
    forecast_frame["hotspot_flag"] = forecast_hexes.set_index("hex_id").loc[forecast_frame["hex_id"], "hotspot_flag"].to_numpy()
    forecast_frame["habitat_context"] = forecast_hexes.set_index("hex_id").loc[forecast_frame["hex_id"], "habitat_context"].to_numpy()
    forecast_frame["forecast_horizon"] = forecast_horizon
    forecast_frame["environment_basis"] = forecast_hexes["environment_basis"].iloc[0]

    stem = f"{species}_{environment_basis}_{forecast_horizon}_intensity"
    forecast_hexes.to_file(output_dir / f"{stem}.gpkg", layer="forecast_intensity", driver="GPKG")
    forecast_frame.to_csv(output_dir / f"{stem}.csv", index=False)
    make_intensity_map(
        species,
        environment_basis,
        forecast_horizon,
        boundary,
        forecast_hexes,
        roads,
        waterways,
        parks,
        output_dir / f"{stem}.png",
        target_period,
    )

    if backtest_frames:
        pd.concat(backtest_frames, ignore_index=True).to_csv(output_dir / f"{species}_{environment_basis}_{forecast_horizon}_model_comparison_backtest_predictions.csv", index=False)

    model_results[recommended_model]["selected_for_forecast"] = True
    model_results[recommended_model]["final_model_explanation"] = explanation
    report = {
        "environment_basis": environment_basis,
        "forecast_horizon": forecast_horizon,
        "target_definition": target_col,
        "forecast_target_period": str(target_period.date()),
        "feature_columns": feature_cols,
        "model_results": model_results,
        "recommended_model": recommended_model,
        "comparison_summary": comparison_summary,
        "output_files": {
            "gpkg": f"{stem}.gpkg",
            "csv": f"{stem}.csv",
            "png": f"{stem}.png",
        },
    }
    return {
        "report": report,
        "forecast_hexes": forecast_hexes,
    }


def model_species(
    config: Config,
    species: str,
    animal_points: gpd.GeoDataFrame,
    hex_features: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    waterways: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    neighbors: dict[str, list[str]],
    kernel_weights: dict[str, list[tuple[str, float]]],
) -> dict[str, object]:
    log(f"{species}: building quarterly intensity panel")
    panel, quarterly_obs = build_panel(hex_features[["hex_id", "geometry"]], animal_points, hex_features, species)
    hex_ids = list(hex_features["hex_id"])
    periods = pd.DatetimeIndex(sorted(panel["quarter"].unique()))
    full_panel, model_panel = add_temporal_features(panel, hex_ids, periods, neighbors, kernel_weights)

    output_dir = ensure_dir(config.output_dir / species)
    animal_points.to_file(output_dir / f"{species}_observations.gpkg", layer="observations", driver="GPKG")

    environment_reports: dict[str, dict[str, object]] = {}
    forecast_surfaces: dict[str, dict[str, gpd.GeoDataFrame]] = {}
    for environment_basis in ENVIRONMENT_BASES:
        environment_reports[environment_basis] = {}
        forecast_surfaces[environment_basis] = {}
        for forecast_horizon in FORECAST_HORIZONS:
            result = run_environment_horizon(
                config,
                species,
                environment_basis,
                forecast_horizon,
                model_panel,
                hex_features,
                roads,
                waterways,
                parks,
                boundary,
                output_dir,
            )
            environment_reports[environment_basis][forecast_horizon] = result["report"]
            forecast_surfaces[environment_basis][forecast_horizon] = result["forecast_hexes"]

    scenario_comparison = {
        forecast_horizon: compare_forecast_surfaces(
            forecast_surfaces["realized"][forecast_horizon],
            forecast_surfaces["planned"][forecast_horizon],
            "predicted_intensity",
        )
        for forecast_horizon in FORECAST_HORIZONS
    }

    species_report = {
        "species": species,
        "analysis_unit": "500m_hex_by_quarter",
        "primary_model_basis": "realized",
        "scenario_model_basis": "realized_plus_planned",
        "primary_forecast_horizon": "next_quarter",
        "secondary_forecast_horizon": "next_year",
        "n_hexes": int(hex_features["hex_id"].nunique()),
        "n_quarters": int(full_panel["quarter"].nunique()),
        "n_observations": int(len(animal_points)),
        "positive_hex_quarters": int((full_panel["observation_count"] > 0).sum()),
        "environment_models": environment_reports,
        "scenario_comparison": scenario_comparison,
        "quarterly_observations_head": [
            {
                "hex_id": row["hex_id"],
                "quarter": str(pd.Timestamp(row["quarter"]).date()),
                "observation_count": int(row["observation_count"]),
            }
            for row in quarterly_obs.head(10).to_dict(orient="records")
        ],
    }
    (output_dir / f"{species}_metrics.json").write_text(json.dumps(species_report, indent=2))
    return species_report


def save_global_outputs(
    config: Config,
    boundary: gpd.GeoDataFrame,
    hex_features: gpd.GeoDataFrame,
    reports: list[dict[str, object]],
    top_landuse_categories: list[str],
    layers: dict[str, gpd.GeoDataFrame],
) -> None:
    ensure_dir(config.output_dir)
    boundary.to_file(config.output_dir / "sg_boundary.gpkg", layer="boundary", driver="GPKG")
    hex_features.to_file(config.output_dir / "hex_features.gpkg", layer="hex_features", driver="GPKG")
    summary = {
        "target_crs": TARGET_CRS,
        "analysis_unit": "500m_hex_by_quarter",
        "forecast_horizons": list(FORECAST_HORIZONS),
        "environment_bases": ["realized", "realized_plus_planned"],
        "hex_radius_m": config.hex_radius_m,
        "density_radius_m": config.density_radius_m,
        "kernel_radius_m": config.kernel_radius_m,
        "backtest_start_year": config.backtest_start_year,
        "min_training_periods": config.min_training_periods,
        "top_landuse_categories": top_landuse_categories,
        "optional_layers_present": [name for name in ("buildings", "subzonepopulation2019") if name in layers],
        "species_reports": reports,
    }
    (config.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Singapore wildlife group-level future presence intensity model")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Directory containing the wildlife data folders")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where outputs will be written")
    parser.add_argument("--hex-radius", type=float, default=500.0, help="Hexagon radius in meters")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of periods used for training")
    args = parser.parse_args()
    output_dir = args.output_dir or args.base_dir / "outputs"
    return Config(
        base_dir=args.base_dir.resolve(),
        output_dir=output_dir.resolve(),
        hex_radius_m=args.hex_radius,
        density_radius_m=args.hex_radius,
        train_fraction=args.train_fraction,
    )


def main() -> None:
    config = parse_args()
    ensure_dir(config.output_dir)
    ensure_dir(Path(os.environ["XDG_CACHE_HOME"]))
    ensure_dir(Path(os.environ["MPLCONFIGDIR"]))

    log("loading boundary")
    boundary, boundary_union = read_boundary(config)
    log("loading animal observations")
    animal_points = load_animal_points(config, boundary)
    log("loading context layers")
    layers = load_context_layers(config, boundary)
    log("building hex grid")
    hexes = build_hex_grid(boundary_union, config.hex_radius_m)
    log(f"hex grid ready with {len(hexes)} cells")
    log("computing static features")
    hex_features, top_landuse_categories = compute_static_features(config, hexes, layers)
    log("building hex neighbors")
    neighbors = build_hex_neighbors(hexes)
    log("building spatial kernel weights")
    kernel_weights = build_spatial_kernel_weights(hexes, config.kernel_radius_m)

    reports = []
    for species in ("monkeys", "otters"):
        log(f"modeling {species}")
        report = model_species(
            config,
            species,
            animal_points[species],
            hex_features,
            layers["roads"],
            layers["waterways"],
            layers["parksnaturereserves"],
            boundary,
            neighbors,
            kernel_weights,
        )
        reports.append(report)

    log("writing shared outputs")
    save_global_outputs(config, boundary, hex_features, reports, top_landuse_categories, layers)
    log("run complete")
    print(json.dumps({"status": "ok", "output_dir": str(config.output_dir), "species": [r["species"] for r in reports]}, indent=2))


if __name__ == "__main__":
    main()
