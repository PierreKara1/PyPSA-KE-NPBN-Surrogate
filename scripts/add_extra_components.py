# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        rooftop_share:
        USD2013_to_EUR2013:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import os

import numpy as np
import pandas as pd
import pypsa
import networkx as nx
from geopy.distance import geodesic
from _helpers import (
    configure_logging,
    create_logger,
    lossy_bidirectional_links,
    override_component_attrs,
    set_length_based_efficiency,
)
from add_electricity import (
    _add_missing_carriers_from_costs,
    add_nice_carrier_names,
    load_costs,
)

from custom_phs_units import (
    add_and_connect_phs_msr_clusters
)
                                
idx = pd.IndexSlice

logger = create_logger(__name__)


def attach_storageunits(n, costs, config):
    elec_opts = config["electricity"]
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    max_hours = elec_opts["max_hours"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        n.madd(
            "StorageUnit",
            buses_i,
            " " + carrier,
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[lookup_store[carrier], "efficiency"],
            efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"],
            max_hours=max_hours[carrier],
            cyclic_state_of_charge=True,
        )


def attach_stores(n, costs, config):
    elec_opts = config["electricity"]
    carriers = elec_opts["extendable_carriers"]["Store"]


    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "H2" in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd(
            "Store",
            h2_buses_i,
            bus=h2_buses_i,
            carrier="H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen storage tank", "capital_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Electrolysis",
            bus0=buses_i,
            bus1=h2_buses_i,
            carrier="H2 electrolysis",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Fuel Cell",
            bus0=h2_buses_i,
            bus1=buses_i,
            carrier="H2 fuel cell",
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            capital_cost=costs.at["fuel cell", "capital_cost"]
            * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
        )

    if "battery" in carriers:
        # 1. Filter out PHS virtual locations first
        all_potential_buses = [bus for bus in buses_i if not bus.startswith("PHS_")]
        
        if config['custom_data']['limit_battery_buses'] == 'top10':
            mean_load_by_bus = (
                n.loads_t.p_set[all_potential_buses]
                .mean(axis=0)
            )

            # 3. Safely pick the top 10 buses based on that capacity
            top_10_demand_buses = (
                mean_load_by_bus
                .nlargest(10)
                .index.tolist()
            )
            
            # Override your active loop array with only the top 10 nodes
            battery_base_buses = top_10_demand_buses
            
            print(f"🔋 Successfully isolated Top 10 demand nodes: {battery_base_buses}")
        else:
            battery_base_buses = all_potential_buses
        
        if config['enable']['set_e_boundary_batteries'] == True:
            e_max_batteries = config['custom_data']['e_nom_max_batteries']
            avg_loads = (
                n.loads_t.p_set[battery_base_buses]
                .mean(axis=0)
                .clip(lower=0)
            )
            load_weights = avg_loads / avg_loads.sum()

            # Dict of e_nom_max per bus
            e_max_batteries_each = {
                bus: e_max_batteries * load_weights[bus]
                for bus in battery_base_buses
            }

            # List of max capacities in the same order as battery_base_buses
            e_nom_max_values = [e_max_batteries_each[bus] for bus in battery_base_buses]

        else:
            # Set to 1 if it's a removed bus, else np.inf
            buses_to_remove = ['KE0 4', 'KE0 7','KE0 11', 'KE0 12', 'KE0 14', 'KE0 15', 'KE0 16', 'KE0 19', 'KE0 21', 'KE0 23', 'KE0 24']
            e_nom_max_values = [0 if config['custom_data']['limit_battery_buses'] == 'spec_buses' and bus in buses_to_remove else np.inf for bus in battery_base_buses]

        if config['enable']['set_p_boundary_batteries'] == True:
            p_max_batteries = config['custom_data']['p_nom_max_batteries']

            avg_loads = (
                n.loads_t.p_set[battery_base_buses]
                .mean(axis=0)
                .clip(lower=0)
            )
            load_weights = avg_loads / avg_loads.sum()

            # Dict of e_nom_max per bus
            p_max_batteries_each = {
                bus: p_max_batteries * load_weights[bus]
                for bus in battery_base_buses
            }

            # List of max capacities in the same order as battery_base_buses
            p_nom_max_values = [p_max_batteries_each[bus] for bus in battery_base_buses]
            e_nom_max_values = [p * elec_opts['max_hours']['battery'] for p in p_nom_max_values]

        else:
            buses_to_remove = ['KE0 4', 'KE0 7','KE0 11', 'KE0 12', 'KE0 14', 'KE0 15', 'KE0 16', 'KE0 19', 'KE0 21', 'KE0 23', 'KE0 24']
            p_nom_max_values = [0 if config['custom_data']['limit_battery_buses'] == 'spec_buses' and bus in buses_to_remove else np.inf for bus in battery_base_buses]
            e_nom_max_values = [0 if config['custom_data']['limit_battery_buses'] == 'spec_buses' and bus in buses_to_remove else np.inf for bus in battery_base_buses]

        # Add battery buses
        b_buses_i = n.madd(
            "Bus", [bus + " battery" for bus in battery_base_buses],
            carrier="battery", **{k: n.buses.loc[battery_base_buses][k].values for k in ["x", "y", "country"]}
        )

        # Add battery stores with per-bus max capacities
        n.madd(
            "Store",
            b_buses_i,
            bus=b_buses_i,
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            e_nom_max=e_nom_max_values,  # 👈 list of values aligned with b_buses_i
            capital_cost=costs.at["battery storage", "capital_cost"],
            marginal_cost=costs.at["battery", "marginal_cost"],
        )


        n.madd(
            "Link",
            [b + " charger" for b in battery_base_buses],
            bus0=battery_base_buses,
            bus1=b_buses_i,
            carrier="battery charger",
            efficiency=costs.at["battery inverter", "efficiency"],
            capital_cost=costs.at["battery inverter", "capital_cost"],
            p_nom_max=p_nom_max_values,
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

        n.madd(
            "Link",
            [b + " discharger" for b in battery_base_buses],
            bus0=b_buses_i,
            bus1=battery_base_buses,
            carrier="battery discharger",
            efficiency=costs.at["battery inverter", "efficiency"],
            p_nom_max=p_nom_max_values,
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

    if ("csp" in elec_opts["renewable_carriers"]) and (
        config["renewable"]["csp"]["csp_model"] == "advanced"
    ):
        # add separate buses for csp
        main_buses = n.generators.query("carrier == 'csp'").bus
        csp_buses_i = n.madd(
            "Bus",
            main_buses + " csp",
            carrier="csp",
            x=n.buses.loc[main_buses, "x"].values,
            y=n.buses.loc[main_buses, "y"].values,
            country=n.buses.loc[main_buses, "country"].values,
        )
        n.generators.loc[main_buses.index, "bus"] = csp_buses_i

        # add stores for csp
        n.madd(
            "Store",
            csp_buses_i,
            bus=csp_buses_i,
            carrier="csp",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["csp-tower TES", "capital_cost"],
            marginal_cost=costs.at["csp-tower TES", "marginal_cost"],
        )

        # add links for csp
        n.madd(
            "Link",
            csp_buses_i,
            bus0=csp_buses_i,
            bus1=main_buses,
            carrier="csp",
            efficiency=costs.at["csp-tower", "efficiency"],
            capital_cost=costs.at["csp-tower", "capital_cost"],
            p_nom_extendable=True,
            marginal_cost=costs.at["csp-tower", "marginal_cost"],
        )


def attach_hydrogen_pipelines(n, costs, config, transmission_efficiency):
    elec_opts = config["electricity"]
    ext_carriers = elec_opts["extendable_carriers"]
    as_stores = ext_carriers.get("Store", [])

    if "H2 pipeline" not in ext_carriers.get("Link", []):
        return

    assert "H2" in as_stores, (
        "Attaching hydrogen pipelines requires hydrogen "
        "storage to be modelled as Store-Link-Bus combination. See "
        "`config.yaml` at `electricity: extendable_carriers: Store:`."
    )

    # determine bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines[attrs], n.links.query('carrier=="DC"')[attrs]]
    ).reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[
        ~pd.DataFrame(np.sort(candidates[["bus0", "bus1"]])).duplicated()
    ]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.madd(
        "Link",
        h2_links.index,
        bus0=h2_links.bus0.values + " H2",
        bus1=h2_links.bus1.values + " H2",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=h2_links.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * h2_links.length,
        carrier="H2 pipeline",
    )

    # split the pipeline into two unidirectional links to properly apply transmission losses in both directions.
    lossy_bidirectional_links(n, "H2 pipeline")

    # set the pipelines efficiency and the electricity required by the pipeline for compression
    set_length_based_efficiency(n, "H2 pipeline", " H2", transmission_efficiency)

def add_ethiopia_import_generator(n):
    """Add a dummy generator representing imports from Ethiopia."""
    USDtoEUR = config["costs"]["USD2013_to_EUR2013"]

    # Filter out buses that start with "PHS"
    valid_buses = n.buses.loc[~n.buses.index.str.startswith("PHS")]

    # Select the most northern (highest y-coordinate) bus
    most_northern_bus = valid_buses.loc[valid_buses.y.idxmax()].name

    # Make sure the carrier exists, or add it
    if "ethiopia_import" not in n.carriers.index:
        n.add("Carrier", name="ethiopia_import", nice_name="Ethiopia Import", color="#2E86C1")
        n.carriers.at["ethiopia_import", "emissions"] = 0.0
        n.carriers.at["ethiopia_import", "nice_name"] = "Ethiopia Import"
        n.carriers.at["ethiopia_import", "color"] = "#2E86C1"
        n.carriers.at["ethiopia_import", "emissions"] = 0.0
        n.carriers.at["ethiopia_import", "marginal_cost"] = 0.0
        print("✅ Ethiopia import generator successfully added.")

    # Now add the generator
    n.add("Generator",
        name="KE_ET_generator",
        bus=most_northern_bus,
        carrier="ethiopia_import",
        p_nom=200,                # Installed capacity: 200 MW
        p_nom_extendable=config['enable']['extend_ethiopia'],
        p_nom_max=400,    # Fixed capacity, not expandable
        marginal_cost=70*USDtoEUR, # Marginal cost: 70 €/MWh (adjust if needed) - value from PGTMP (2016)
        capital_cost = 0.02*1269           # Transmission line already built, so not additional capital cost - only FOM from https://kcv.co.ke/wp-content/uploads/2020/07/Kenya-PGTMP-Final-LTP-Vol-I-Main-Report-October-2016.pdf
    )

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_extra_components", simpl="", clusters=25)

    configure_logging(snakemake)

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    transmission_efficiency = snakemake.params.transmission_efficiency
    config = snakemake.config

    costs = load_costs(
        snakemake.input.tech_costs,
        config["costs"],
        config["electricity"],
        Nyears,
    )

    attach_storageunits(n, costs, config)
    attach_stores(n, costs, config)
    if config['enable']['add_PHS'] == True:
        add_and_connect_phs_msr_clusters(n,config)
    attach_hydrogen_pipelines(n, costs, config, transmission_efficiency)
    if config['enable']['add_Ethiopia'] == True:
        add_ethiopia_import_generator(n)

    add_nice_carrier_names(n, config=snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
