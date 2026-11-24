from pathlib import Path
import yaml
import os
import pandas as pd
import numpy as np
import xarray as xr
from shapely.geometry import Point
import geopandas as gpd


def load_project(project_path: Path | str) -> tuple[dict, pd.DataFrame]:
    """
    Loads configuration parameters and metadata from an existing project.

    Parameters
    ----------
    project_path: Path | str
        The path to the project directory containing the config.yaml and cohort_metadata.csv files.

    Returns
    -------
    config: dict
        A dictionary containing configuration parameters.
    metadata_df: pd.DataFrame
        A DataFrame containing cohort metadata.
    """
    # Load config.yaml
    project_path = Path(project_path).resolve()
    config_file_path = project_path / "config.yaml"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Load metadata CSV
    metadata_file_path = project_path / "cohort_metadata.csv"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")

    metadata_df = pd.read_csv(metadata_file_path)

    register_genotype_sets(config.get("genotype_sets"))
    return (config, metadata_df)


def load_cohort_metadata(config: dict) -> pd.DataFrame:
    """
    Loads cohort metadata from the CSV file specified in the project configuration.

    Parameters
    ----------
    config: dict
        The project configuration dictionary containing the path to the cohort metadata CSV.

    Returns
    -------
    metadata_df: pd.DataFrame
        A DataFrame containing cohort metadata.
    """
    project_path = Path(config["project_path_full"]).resolve()
    metadata_file_path = project_path / "cohort_metadata.csv"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")

    metadata_df = pd.read_csv(metadata_file_path)
    return metadata_df


def load_session_dataset(config: dict, session_name: str) -> xr.Dataset:
    """
    Loads the xarray Dataset for a specific session.

    Parameters
    ----------
    config: dict
        The project configuration dictionary.
    session_name: str
        The name of the session to load.

    Returns
    -------
    ds: xr.Dataset
        The xarray Dataset for the specified session.
    """
    if session_name not in config["session_names"]:
        raise ValueError(f"Session name {session_name} not found in project configuration.")
    project_path = Path(config["project_path_full"]).resolve()
    nc_file_path = project_path / "data" / "dlc_results" / f"{session_name}.nc"
    if not nc_file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {nc_file_path}")

    ds = xr.load_dataset(nc_file_path)
    return ds


def save_figure(
    config: dict,
    fig_name: str,
    subdir: str = "results/task_performance",
    dpi: int = 300,
    ext: str = "pdf",
):
    """
    Save the current matplotlib figure to a standardized results folder.

    Parameters
    ----------
    config : dict
        Project's configuration dictionary.
    fig_name : str
        Name of the figure file, e.g., 'Shannons_entropy' or 'Bout_Success'.
        Extension is automatically appended as defined by `ext`.
    subdir : str
        Subfolder path under BASE_PATH to save the figure.
    dpi : int
        Resolution of saved figure.
    ext : str
        File extension, e.g., 'pdf', 'png', etc.
    """
    import matplotlib.pyplot as plt

    base_path = config["project_path_full"]
    os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    save_path = os.path.join(base_path, subdir, f"{fig_name}.{ext}")
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved: {save_path}")


def update_dataset_with_grid_positions(
    ds: xr.Dataset,
    grid_file_path: Path | str,
) -> xr.Dataset:
    """
    Add grid position numbers to an xarray Dataset containing pose estimation data.
    This function performs spatial joins between tracked body positions and a grid
    shapefile to determine which grid cell each position falls within at each time point.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing pose estimation data.
    grid_file_path : Path | str
        Path to the grid shapefile (e.g., '{session}_grid.shp')

    Returns
    -------
    xr.Dataset
        Updated xarray Dataset with an additional data variable 'grid_number'
        indicating the grid cell number for each tracked position.
    """
    # Load the Grid shapefile
    grid_file_path = Path(grid_file_path)
    if not grid_file_path.exists():
        raise FileNotFoundError(f"Grid file not found at {grid_file_path}")
    grid = gpd.read_file(str(grid_file_path))

    # Initialize grid numbers array with NaNs
    n_time = len(ds.time)
    n_keypoints = len(ds.keypoints)
    n_individuals = 1  # Assuming single individual 'individual_0'
    grid_numbers_array = np.full((n_time, n_keypoints, n_individuals), np.nan)

    # Get keypoint names
    keypoints = ds.keypoints.values

    # Process each keypoint
    for kp_idx, keypoint in enumerate(keypoints):
        # Extract x,y positions for this keypoint
        # position has shape (time, space, keypoints, individuals)
        xy = ds.sel(individuals="individual_0", keypoints=keypoint).position.values

        # Create Point geometries for each time point
        points = []
        for x, y in xy:
            if pd.notna(x) and pd.notna(y):
                points.append(Point(x, y))
            else:
                points.append(None)

        # Create GeoDataFrame of points
        pnt_gpd = gpd.GeoDataFrame(
            geometry=points,
            index=np.arange(len(points)),
            crs=grid.crs,
        )

        # Find which polygon each point is in
        point_in_polys = gpd.tools.sjoin(
            left_df=pnt_gpd,
            right_df=grid,
            predicate="within",
            how="left",
        )

        # Extract grid numbers
        # Use 'FID' column from grid or index_right if FID doesn't exist
        if "FID" in point_in_polys.columns:
            grid_nums = point_in_polys["FID"].values
        else:
            grid_nums = point_in_polys["index_right"].values

        # Store in the array
        grid_numbers_array[:, kp_idx, 0] = grid_nums

    # Add grid_number as a new data variable to the dataset
    ds["grid_number"] = xr.DataArray(
        grid_numbers_array,
        dims=["time", "keypoints", "individuals"],
        coords={"time": ds.time, "keypoints": ds.keypoints, "individuals": ds.individuals},
        attrs={
            "description": "Grid cell number for each tracked position",
            "grid_file": str(grid_file_path),
        },
    )

    return ds


def group_sample_sizes(df: pd.DataFrame, group_col: str, session_col: str = "session") -> dict:
    """{group label: number of unique sessions} for each value of group_col in df.

    Pass the result to append_n_to_legend() so a figure's legend states each
    group's N (sessions contributing to the dataframe actually being
    plotted, not the whole project), e.g. "E3" -> "E3 (n=12)".
    """
    return df.groupby(group_col)[session_col].nunique().to_dict()


def append_n_to_legend(legend, *counts: dict) -> None:
    """Appends " (n=<count>)" to an existing legend's text labels.

    legend : a matplotlib Legend object -- e.g. ax.get_legend(), the return
        value of plt.legend()/ax.legend()/fig.legend(), or a seaborn
        FacetGrid's `._legend` attribute. None is a no-op, so call sites
        that may not have a legend don't need their own guard.
    *counts : one or more {label: n} dicts from group_sample_sizes() -- pass
        several when one legend mixes groupings (e.g. hue=genotype and
        style=sex). A label matching none of them (a seaborn section header
        in a combined hue+style legend, or an unrelated label like
        "Excluded") is left as-is.

    Mutates the legend's already-rendered Text objects directly, so it
    works regardless of how the legend was built without needing to redo
    its original kwargs (loc, frameon, fontsize, title, ...).
    """
    if legend is None:
        return
    merged: dict = {}
    for c in counts:
        merged.update(c)
    for text in legend.get_texts():
        label = text.get_text()
        if label in merged:
            text.set_text(f"{label} (n={merged[label]})")


_GENOTYPE_SETS: dict = {}


def register_genotype_sets(genotype_sets: dict | None) -> None:
    """Remember config.yaml's genotype_sets, so genotype_order() can put any
    figure's groups in their configured order (load_project() and
    build_genotype_groups() call this)."""
    if genotype_sets:
        _GENOTYPE_SETS.update(genotype_sets)


def genotype_order(labels) -> list:
    """The genotype labels present in a figure, in config.yaml order: the
    order of the smallest genotype_sets entry containing all of them (e.g.
    WT, NLF / APOE3, APOE4 / E3, E3NLF, E4, E4NLF). Labels in no set, or
    nothing registered, fall back to alphabetical. Pass as seaborn hue_order."""
    present = [str(l) for l in dict.fromkeys(labels) if pd.notna(l)]
    fits = [list(m) for m in _GENOTYPE_SETS.values() if set(present) <= set(m)]
    if not fits:
        return sorted(present)
    order = min(fits, key=len)
    return [g for g in order if g in present]


def build_genotype_groups(df: pd.DataFrame, set_name: str, genotype_sets: dict) -> pd.DataFrame:
    """Relabel a dataframe's `genotype` column per genotype_sets[set_name].

    Rows whose raw genotype isn't in any label's list are dropped; rows that
    are get `genotype` replaced by the label (e.g. E3 and E3NLF both become
    "APOE3" for set_name="apoe_pair"). Works the same whether the set keeps
    genotypes separate ("pair") or pools several into one label ("wt_nlf").

    df : dataframe with a `genotype` column
    set_name : key of genotype_sets to use
    genotype_sets : set_name -> {label: [raw genotype strings]}, e.g. the
        project notebook's own GENOTYPE_SETS -- passed in explicitly (not
        imported) so this function, and any cache keyed on its arguments,
        stays correct for a different project's genotype labels.
    """
    register_genotype_sets(genotype_sets)
    mapping = genotype_sets[set_name]
    genotype_lookup = {raw: label for label, raw_genotypes in mapping.items() for raw in raw_genotypes}
    df_grouped = df[df["genotype"].isin(genotype_lookup)].copy()
    df_grouped["genotype"] = df_grouped["genotype"].map(genotype_lookup)
    return df_grouped


def build_zone_column(
    df: pd.DataFrame,
    zone_names: list,
    zones: dict,
    column_name: str = "custom_zone",
) -> pd.DataFrame:
    """Add a `column_name` column labeling each row's zone by `zones` membership.

    zone_names : list of zone names (keys of `zones`) to include -- zones may
        overlap (share grid cells); a shared cell's rows are duplicated (once
        per zone it belongs to) rather than assigned to just one.
    zones : zone_name -> list of grid cell numbers, e.g. the project
        notebook's own ZONES -- passed in explicitly so this function stays
        correct for a different project's zone layout.
    column_name : name of the new column to add (default "custom_zone")
    """
    zones_subset = {name: zones[name] for name in zone_names}
    labeled = []
    for zone, cells in zones_subset.items():
        subset = df[df["grid_number"].isin(cells)].copy()
        subset[column_name] = zone
        labeled.append(subset)
    if not labeled:
        return df.iloc[0:0].assign(**{column_name: []})
    # Concat WITHOUT ignore_index, then sort_index to restore each session's
    # original chronological row order.
    return pd.concat(labeled).sort_index(kind="stable").reset_index(drop=True)
