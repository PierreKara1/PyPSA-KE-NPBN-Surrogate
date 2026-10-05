# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Creates electric demand profile csv.

Relevant Settings
-----------------

.. code:: yaml

    load:
        scale:
        ssp:
        weather_year:
        prediction_year:
        region_load:

Inputs
------

- ``networks/base.nc``: confer :ref:`base`, a base PyPSA Network
- ``resources/bus_regions/regions_onshore.geojson``: confer :mod:`build_bus_regions`
- ``load_data_paths``: paths to load profiles, e.g. hourly country load profiles produced by GEGIS
- ``resources/shapes/gadm_shapes.geojson``: confer :ref:`shapes`, file containing the gadm shapes

Outputs
-------

- ``resources/demand_profiles.csv``: the content of the file is the electric demand profile associated to each bus. The file has the snapshots as rows and the buses of the network as columns.

Description
-----------

The rule :mod:`build_demand` creates load demand profiles in correspondence of the buses of the network.
It creates the load paths for GEGIS outputs by combining the input parameters of the countries, weather year, prediction year, and SSP scenario.
Then with a function that takes in the PyPSA network "base.nc", region and gadm shape data, the countries of interest, a scale factor, and the snapshots,
it returns a csv file called "demand_profiles.csv", that allocates the load to the buses of the network according to GDP and population.
"""
import os
import os.path
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import scipy.sparse as sparse
import xarray as xr
from _helpers import (
    BASE_DIR,
    configure_logging,
    create_logger,
    read_csv_nafix,
    read_osm_config,
)
from shapely.prepared import prep
from shapely.validation import make_valid
from Reshape_load_curve import (reshape_load_curve_lit)

logger = create_logger(__name__)


def normed(s):
    return s / s.sum()


def get_gegis_regions(countries):
    """
    Get the GEGIS region from the config file.

    Parameters
    ----------
    region : str
        The region of the bus

    Returns
    -------
    str
        The GEGIS region
    """
    gegis_dict, world_iso = read_osm_config("gegis_regions", "world_iso")

    regions = []

    for d_region in [gegis_dict, world_iso]:
        for key, value in d_region.items():
            # ignore if the key is already in the regions list
            if key not in regions:
                # if a country is in the regions values, then load it
                cintersect = set(countries).intersection(set(value.keys()))
                if cintersect:
                    regions.append(key)
    return regions


def get_load_paths_gegis(ssp_parentfolder, config):
    """
    Create load paths for GEGIS outputs.

    The paths are created automatically according to included country,
    weather year, prediction year and ssp scenario

    Example
    -------
    ["/data/ssp2-2.6/2030/era5_2013/Africa.nc", "/data/ssp2-2.6/2030/era5_2013/Africa.nc"]
    """
    countries = config.get("countries")
    region_load = get_gegis_regions(countries)
    weather_year = config.get("load_options")["weather_year"]
    prediction_year = config.get("load_options")["prediction_year"]
    ssp = config.get("load_options")["ssp"]

    scenario_path = os.path.join(ssp_parentfolder, ssp)

    load_paths = []
    load_dir = os.path.join(
        ssp_parentfolder,
        str(ssp),
        str(prediction_year),
        "era5_" + str(weather_year),
    )

    file_names = []
    for continent in region_load:
        sel_ext = ".nc" # change from .nc
        for ext in [".nc", ".csv"]:
            load_path = os.path.join(BASE_DIR, str(load_dir), str(continent) + str(ext))
            if os.path.exists(load_path):
                sel_ext = ext
                break
        file_name = str(continent) + str(sel_ext)
        load_path = os.path.join(str(load_dir), file_name)
        load_paths.append(load_path)
        file_names.append(file_name)

    logger.info(
        f"Demand data folder: {load_dir}, load path is {load_paths}.\n"
        + f"Expected files: "
        + "; ".join(file_names)
    )

    return load_paths


def shapes_to_shapes(orig, dest):
    """
    Adopted from vresutils.transfer.Shapes2Shapes()
    """
    orig_prepped = list(map(prep, orig))
    transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=float)

    for i, j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest[i]):
            area = orig[j].intersection(dest[i]).area
            transfer[i, j] = area / dest[i].area

    return transfer


def load_demand_csv(path):
    df = read_csv_nafix(path, sep=";")
    df.time = pd.to_datetime(df.time, format="%Y-%m-%d %H:%M:%S")
    load_regions = {c: n for c, n in zip(df.region_code, df.region_name)}

    gegis_load = df.set_index(["region_code", "time"]).to_xarray()
    gegis_load = gegis_load.assign_coords(
        {
            "region_name": (
                "region_code",
                [name for (code, name) in load_regions.items()],
            )
        }
    )
    return gegis_load

import pandas as pd

import pandas as pd

import pandas as pd

import pandas as pd

import geopandas as gpd

def assign_epra_weight_to_buses(group, shapes_cntry):
    """
    Assign EPRA_demand_weight from county polygons (shapes_cntry) to each bus in group,
    using centroids for spatial inclusion, then restore original bus geometries.

    Parameters:
    - group: GeoDataFrame with polygon geometries for buses.
    - shapes_cntry: GeoDataFrame with 'EPRA_demand_weight' and polygon geometries.

    Returns:
    - A GeoDataFrame with original geometries and a new column 'EPRA_demand_weight'.
    """
    # Ensure both are in the same CRS
    shapes_cntry = shapes_cntry.to_crs("EPSG:4326")
    group = group.to_crs("EPSG:4326")

    # Store original polygon geometry
    group["geometry_poly"] = group.geometry

    # Replace geometry with centroid for spatial join
    group["geometry"] = group.geometry.centroid

    # Perform spatial join using centroids
    buses_with_epra = gpd.sjoin(
        group,
        shapes_cntry[["EPRA_demand_weight", "geometry"]],
        how="left",
        predicate="within"
    )

    # Restore original polygon geometry and drop intermediate columns
    buses_with_epra["geometry"] = buses_with_epra["geometry_poly"]
    buses_with_epra = buses_with_epra.drop(columns=["geometry_poly", "index_right"])

    return buses_with_epra



def build_demand_profiles(
    n,
    load_paths,
    regions,
    admin_shapes,
    countries,
    scale,
    start_date,
    end_date,
    out_path,
    config
):
    """
    Create csv file of electric demand time series.

    Parameters
    ----------
    n : pypsa network
    load_paths: paths of the load files
    regions : .geojson
        Contains bus_id of low voltage substations and
        bus region shapes (voronoi cells)
    admin_shapes : .geojson
        contains subregional gdp, population and shape data
    countries : list
        List of countries that is config input
    scale : float
        The scale factor is multiplied with the load (1.3 = 30% more load)
    start_date: parameter
        The start_date is the first hour of the first day of the snapshots
    end_date: parameter
        The end_date is the last hour of the last day of the snapshots

    Returns
    -------
    demand_profiles.csv : csv file containing the electric demand time series
    """
    substation_lv_i = n.buses.index[n.buses["substation_lv"]]
    regions = gpd.read_file(regions).set_index("name").reindex(substation_lv_i)
    load_paths = load_paths

    gegis_load_list = []

    for path in load_paths:
        if str(path).endswith(".csv"):
            gegis_load_xr = load_demand_csv(path)
        else:
            # Merge load .nc files: https://stackoverflow.com/questions/47226429/join-merge-multiple-netcdf-files-using-xarray
            gegis_load_xr = xr.open_mfdataset(path, combine="nested")
        gegis_load_list.append(gegis_load_xr)

    logger.info(f"Merging demand data from paths {load_paths} into the load data frame")
    gegis_load = xr.merge(gegis_load_list)
    gegis_load = gegis_load.to_dataframe().reset_index().set_index("time")

    # filter load for analysed countries
    gegis_load = gegis_load.loc[gegis_load.region_code.isin(countries)]
    
    # Replace the year in the datetime index from 2013 to 2023
    if config['enable']['change_weather_data_year']:
        gegis_load.index = gegis_load.index.map(lambda dt: dt.replace(year=2023))

    if isinstance(scale, dict):
        logger.info(f"Using custom scaling factor for load data.")
        DEFAULT_VAL = scale.get("DEFAULT", 1.0)
        for country in countries:
            scale.setdefault(country, DEFAULT_VAL)

        for country, scale_country in scale.items():
            gegis_load.loc[
                gegis_load.region_code == country, "Electricity demand"
            ] *= scale_country

    elif isinstance(scale, (int, float)):
        logger.info(f"Load data scaled with scaling factor {scale}.")
        gegis_load["Electricity demand"] *= scale

    shapes = gpd.read_file(admin_shapes).set_index("GADM_ID")
    shapes["geometry"] = shapes["geometry"].apply(lambda x: make_valid(x))

    def upsample(cntry, group):
        """
        Distributes load in country according to population and gdp.
        """
        l = gegis_load.loc[gegis_load.region_code == cntry]["Electricity demand"]
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            shapes_cntry = shapes.loc[shapes.country == cntry]
            transfer = shapes_to_shapes(group, shapes_cntry.geometry).T.tocsr()
            gdp_n = pd.Series(
                transfer.dot(shapes_cntry["gdp"].fillna(1.0).values), index=group.index
            )
            pop_n = pd.Series(
                transfer.dot(shapes_cntry["pop"].fillna(1.0).values), index=group.index
            )

            # relative factors 0.6 and 0.4 have been determined from a linear
            # regression on the country to EU continent load data
            # (refer to vresutils.load._upsampling_weights)
            # TODO: require adjustment for Africa
            if config['load_options']['impose_EPRA']:
                # Assuming `group` is your GeoSeries or GeoDataFrame with polygon geometries
                gdf = gpd.GeoDataFrame(geometry=group)

                # Compute centroids
                gdf["centroid"] = gdf.geometry.centroid

                # Create a new DataFrame with bus name/index and centroid coordinates
                centroid_df = pd.DataFrame({
                    "bus": gdf.index,
                    "x": gdf["centroid"].x,
                    "y": gdf["centroid"].y
                })
                centroid_gdf = gpd.GeoDataFrame(
                    centroid_df.reset_index(drop=True),
                    geometry=gpd.points_from_xy(centroid_df["x"], centroid_df["y"]),
                    crs="EPSG:4326"
                )

                # Step 2: Ensure matching CRS for spatial join
                shapes_cntry = shapes_cntry.to_crs("EPSG:4326")

                # Step 3: Spatial join to get EPRA_demand_weight from counties
                joined = gpd.sjoin(centroid_gdf, shapes_cntry[["EPRA_demand_weight", "geometry"]], how="left", predicate="within")

                # Step 4: Create a Series or DataFrame to assign weights to buses
                bus_weights = joined.set_index("bus")["EPRA_demand_weight"]

                # Merge the two Series into a DataFrame
                merged = pd.concat([
                    bus_weights,
                    gdp_n.reindex(bus_weights.index),
                    pop_n.reindex(bus_weights.index)
                ], axis=1)

                # Rename columns for clarity
                merged.columns = ["EPRA_demand_weight", "gdp_n", "pop_n"]

                # # Optional: sort by bus index if needed
                # merged = merged.sort_index()

                # Step 1: Normalize gdp and pop across all buses
                gdp_norm = merged["gdp_n"] / merged["gdp_n"].sum()
                pop_norm = merged["pop_n"] / merged["pop_n"].sum()

                result_list = []

                for epra_value in merged["EPRA_demand_weight"].unique():
                    region = merged[merged["EPRA_demand_weight"] == epra_value]
                    region.reset_index(inplace=True, drop=False)
                    raw = 0.8 * region["gdp_n"] + 0.2 * region["pop_n"]
                    raw /= raw.sum()  # Normalize within the region
                    scaled = raw * epra_value  # Scale to regional demand weight

                    region_result = pd.DataFrame({
                        "bus": region["bus"], 
                        "EPRA_demand_weight": epra_value,
                        "computed_factor": scaled.values
                    })

                    result_list.append(region_result)

                bus_demand_factors = pd.concat(result_list).set_index("bus")
                bus_demand_factors.index = bus_demand_factors.index.astype(int)
                bus_demand_factors = bus_demand_factors.sort_index()
                # Step 4: Normalize overall to ensure total demand sums to 1
                factors = bus_demand_factors["computed_factor"] #/ bus_demand_factors["computed_factor"].sum()

            else:
                factors = normed(0.8 * normed(gdp_n) + 0.2 * normed(pop_n))

            return pd.DataFrame(
                factors.values * l.values[:, np.newaxis],
                index=l.index,
                columns=factors.index,
            )

    demand_profiles = pd.concat(
        [
            upsample(cntry, group)
            for cntry, group in regions.geometry.groupby(regions.country)
        ],
        axis=1,
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
    demand_profiles = demand_profiles.loc[start_date:end_date]

    if config['enable']['reshape_load_curve'] == True:
        demand_profiles_reshaped = reshape_load_curve_lit(config, demand_profiles)
        demand_profiles_reshaped.to_csv(out_path, header=True)
        logger.info(f"✅ Demand_profiles fitted to literature data.")
    else:
        demand_profiles.to_csv(out_path, header=True)
        logger.info(f"✅ Demand_profiles NOT fitted to literature data.")

    logger.info(f"Demand_profiles csv file created for the corresponding snapshots.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_demand_profiles")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)

    # Snakemake imports:
    config = snakemake.config
    regions = snakemake.input.regions
    load_paths = snakemake.input["load"]
    countries = snakemake.params.countries
    admin_shapes = snakemake.input.gadm_shapes
    scale = snakemake.params.load_options.get("scale", 1.0)
    start_date = snakemake.params.snapshots["start"]
    end_date = snakemake.params.snapshots["end"]
    out_path = snakemake.output[0]

    build_demand_profiles(
        n,
        load_paths,
        regions,
        admin_shapes,
        countries,
        scale,
        start_date,
        end_date,
        out_path,
        config
    )