# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Solves linear optimal power flow for a network iteratively while updating
reactances.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: /img/results.png
        :width: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning)
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`prepare_network` and :mod:`solve_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances on values of corresponding flows.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors introduced through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
"""
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging, create_logger, override_component_attrs
from linopy import merge
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.abstract import optimize_transmission_expansion_iteratively
from pypsa.optimization.optimize import optimize
from linopy import LinearExpression
from pypsa.linopt import define_constraints, linexpr
logger = create_logger(__name__)
logging.getLogger("gurobipy").propagate = False
# logging.getLogger("linopy").propagate = False
pypsa.pf.logger.setLevel(logging.WARNING)


def prepare_network(n, solve_opts, config):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if "lv_limit" in n.global_constraints.index:
        n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
        n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load")
        n.madd(
            "Generator",
            n.buses.index,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1,
            marginal_cost=solve_opts.get("load_shedding") * 1000,  # convert to Eur/MWh
            p_nom=1e5,  # MW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if snakemake.config["foresight"] == "myopic":
        add_land_use_constraint(n)

    return n

def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep

def add_CCL_constraints(n, config):
    """
    Add CCL (carrier limit only) constraint to the network.

    Assumes a single-country model.
    """

    agg_p_nom_limits = config["electricity"].get("agg_p_nom_limits")

    try:
        agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col="carrier")
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity limits per carrier in "
            "config['electricity']['agg_p_nom_limits']."
        )
        raise

    logger.info("✅ Adding generation capacity constraints per carrier (no country split)")

    # Only extendable generators
    extendable_generators = n.generators.query("p_nom_extendable")
    capacity_variable = n.model["Generator-p_nom"]

    lhs_terms = []
    carriers = extendable_generators.carrier.unique()

    for carrier in carriers:
        idx = extendable_generators.index[extendable_generators.carrier == carrier]
        if len(idx) > 0:
            # 🛠️ THIS IS THE CORRECT WAY:
            carrier_sum = capacity_variable.sel({"Generator-ext": idx}).sum()
            lhs_terms.append((carrier, carrier_sum))

    # Build lhs into a pandas Series for easier handling
    lhs = pd.Series(dict(lhs_terms))

    # Match min and max constraints
    min_matrix = agg_p_nom_minmax["min"].reindex(lhs.index)
    max_matrix = agg_p_nom_minmax["max"].reindex(lhs.index)

    # Add constraints to the model
    for carrier in lhs.index:
        if pd.notnull(min_matrix[carrier]):
            n.model.add_constraints(lhs[carrier] >= min_matrix[carrier], name=f"agg_p_nom_min_{carrier}")
        if pd.notnull(max_matrix[carrier]):
            n.model.add_constraints(lhs[carrier] <= max_matrix[carrier], name=f"agg_p_nom_max_{carrier}")


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    dispatch_variable = n.model["Generator-p"]
    lhs_gen = (
        (dispatch_variable * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # the current formulation implies that the available hydro power is (inflow - spillage)
    # it implies efficiency_dispatch is 1 which is not quite general
    # see https://github.com/pypsa-meets-earth/pypsa-earth/issues/1245 for possible improvements
    if not n.storage_units_t.inflow.empty:
        spillage_variable = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage_variable * (-n.snapshot_weightings.stores * scaling))
            .groupby_sum(sgrouper)
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    capacity_variable = n.model["Generator-p_nom"]
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin_constraint(n, sns, config):
    """
    Build reserve margin constraints based on the formulation
    as suggested in GenX
    https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf
    It implies that the reserve margin also accounts for optimal
    dispatch of distributed energy resources (DERs) and demand response
    which is a novel feature of GenX.
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (
            p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
        ).sum("Generator")

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

def add_phs_constraints(n, min_hours, max_hours, max_share_of_peak, config):
    """
    Add constraints to:
    1. Enforce: 1 / min_hours ≤ p_nom / e_nom ≤ 1 / max_hours
    2. Enforce: p_nom (generation) = p_nom (pumping)
    3. Enforce: p_nom ≤ max_share_of_peak * peak_demand
    4. Enforce: pumping * efficiency = generation power
    """

    peak_demand = n.loads_t.p_set.sum(axis=1).max()
    if pd.isna(peak_demand) or peak_demand == 0:
        return  # can't apply constraint if peak demand is missing

    phs_stores = n.stores[n.stores.carrier == "Phs_Store"]
    phs_gen_links = n.links[n.links.carrier == "Phs_Generation"]
    phs_pump_links = n.links[n.links.carrier == "Phs_Pumping"]

    p_nom_cap = min(max_share_of_peak * peak_demand, phs_stores.e_nom_max.iloc[0]/max_hours)

    for store_name, store in phs_stores.iterrows():
        bus = store.bus

        gen_match = phs_gen_links[phs_gen_links.bus0 == bus]
        pump_match = phs_pump_links[phs_pump_links.bus1 == bus]
        if gen_match.empty or pump_match.empty:
            continue

        gen_name = gen_match.index[0]
        pump_name = pump_match.index[0]

        if not (n.links.at[gen_name, "p_nom_extendable"]
                and n.links.at[pump_name, "p_nom_extendable"]
                and n.stores.at[store_name, "e_nom_extendable"]):
            continue

        # Constraint 1: p_nom generation ratio lower bound
        n.model.add_constraints(
            n.model.variables["Link-p_nom"][gen_name]
            - (1 / min_hours) * n.model.variables["Store-e_nom"][store_name]
            >= 0,
            name=f"phs_min_ratio_{store_name}"
        )

        # Constraint 2: p_nom generation ratio upper bound
        n.model.add_constraints(
            n.model.variables["Link-p_nom"][gen_name]
            - (1 / max_hours) * n.model.variables["Store-e_nom"][store_name]
            <= 0,
            name=f"phs_max_ratio_{store_name}"
        )

        if config['enable']['N1_PHS']['N1_PHS_activate']:
            # Constraint 3: p_nom max cap for generation link
            n.model.add_constraints(
                n.model.variables["Link-p_nom"][gen_name]
                <= p_nom_cap,
                name=f"phs_gen_p_nom_cap_{store_name}"
            )
            print(f"✅ Max. PHS storage capacity imposed: {p_nom_cap} MW")    
        
        # Constraint 4: pumping = generation
        eff = n.links.at[gen_name, "efficiency"]  # typically generation link has the efficiency
        n.model.add_constraints(
            eff * n.model.variables["Link-p_nom"][pump_name]
            - n.model.variables["Link-p_nom"][gen_name]
            == 0,
            name=f"phs_power_efficiency_{store_name}"
        )
    print(f"✅ Added PHS storage hours constraint:")
    print(f"PHS storage hours between {min_hours} and {max_hours} hours.")
    print(f"✅ PHS generation power must be equal to (efficiency-corrected) pumping power")

def equalize_battery_charger_discharger(n):
    p_nom = n.model["Link-p_nom"]
    link_labels = p_nom.coords["Link-ext"].values

    chargers = n.links.query('carrier == "battery charger"').index
    for charger in chargers:
        discharger = charger.replace("charger", "discharger")
        if discharger not in n.links.index:
            continue

        eff = n.links.at[discharger, "efficiency"]  # assume discharger efficiency is what matters

        i_charger = np.where(link_labels == charger)[0]
        i_discharger = np.where(link_labels == discharger)[0]

        if i_charger.size == 0 or i_discharger.size == 0:
            continue

        n.model.add_constraints(
            eff * p_nom.isel(**{"Link-ext": i_charger[0]}) == p_nom.isel(**{"Link-ext": i_discharger[0]}),
            name=f"equal_coupled_battery_{charger}"
        )
    print(f"✅ Battery discharger capacity forced to {eff} of charger.")


def ethiopia_constraints(n):
    # Get generator index of Ethiopia imports
    ethiopia_gens = n.generators.query("carrier == 'ethiopia_import'").index

    if ethiopia_gens.empty:
        print("❌ No Ethiopia import generators found.")
        return

    # Get variable (Linopy Variable object)
    p = n.model["Generator-p"]

    # Convert generator names to integer positions in the model dimension
    gen_pos = p.coords["Generator"].values
    ethiopia_pos = [i for i, g in enumerate(gen_pos) if g in ethiopia_gens]

    # Subset using isel
    p_ethiopia = p.isel(Generator=ethiopia_pos)

    # Weight dispatch by snapshot weightings
    weights = n.snapshot_weightings.generators
    total_dispatch = (p_ethiopia * weights).sum()
 
    # Minimum required dispatch
    if config['enable']['extend_ethiopia']:
        required_dispatch = 1.8 * n.generators.loc[ethiopia_gens, "p_nom"].sum() * len(n.snapshots)
    else:
        required_dispatch = 0.9 * n.generators.loc[ethiopia_gens, "p_nom"].sum() * len(n.snapshots)

    # Add the constraint
    n.model.add_constraints(
        total_dispatch >= required_dispatch,
        name="ethiopia_import_min_dispatch"
    )

    print(f"✅ Applied minimum dispatch constraint: Ethiopia ≥ {required_dispatch:.2f} MWh")


def enforce_ethiopia_dispatch(n, target_gwh=419):
    """
    Enforces a fixed total generation constraint on the Ethiopia import carrier,
    safely adjusting for PyPSA's internal annualization weightings.
    """
    # Find the generator names for this carrier
    ethiopia_gens = n.generators.query("carrier == 'ethiopia_import'").index
    if ethiopia_gens.empty:
        print("⚠️ No Ethiopia import generators found. Constraint not applied.")
        return

    snapshots = n.snapshots

    # 1. Pull the mathematical variable directly from the Linopy model
    p = n.model["Generator-p"]

    # 2. Select only the Ethiopia generators and the active snapshots
    p_eth = p.sel(Generator=ethiopia_gens, snapshot=snapshots)

    # 3. CRITICAL FIX: Strip PyPSA's annualization multiplier.
    # We want the true physical hours per snapshot (usually 1.0 for hourly scale).
    # n.snapshot_weightings.generators contains: (physical_hours * objective_weight)
    # We divide by the objective weight to isolate the true physical hours.
    objective_weight = n.snapshot_weightings.objective.loc[snapshots].iloc[0]
    
    # Calculate the true physical duration per snapshot step
    physical_weights = n.snapshot_weightings.generators.loc[snapshots] / objective_weight
    
    # Convert true physical weights to an xarray DataArray
    weights_xr = xr.DataArray(
        physical_weights.values[:, np.newaxis],  # Shape: (len(snapshots), 1)
        coords={"snapshot": snapshots, "Generator": ethiopia_gens},
        dims=("snapshot", "Generator")
    )

    # 4. Multiply by physical weights (hours) and sum up
    total_dispatch = (p_eth * weights_xr).sum(dim=["snapshot", "Generator"])

    # 5. Add the equality constraint to the model
    n.model.add_constraints(
        total_dispatch == target_gwh * 1000,  # GWh to MWh
        name="ethiopia_fixed_dispatch"
    )

    print(f"✅ Ethiopia dispatch constrained to exactly {target_gwh} physical GWh.")
    print(f"   (Stripped internal PyPSA temporal scaling factor of {objective_weight:.2f})")
    
    return n

def enforce_wind_dispatch(n, target_gwh=972):

    wind_gens = n.generators.query("carrier == 'onwind'").index
    if wind_gens.empty:
        print("⚠️ No wind generators found. Constraint not applied.")
        return

    # Time window
    start = pd.Timestamp("2023-07-01")
    end = pd.Timestamp("2023-12-31 23:59:59")
    mask = (n.snapshots >= start) & (n.snapshots <= end)
    snapshots = n.snapshots[mask]

    # Dispatch variable
    p = n.model["Generator-p"]

    # Get coordinate labels
    gen_labels = p.coords["Generator"].values
    snap_labels = p.coords["snapshot"].values

    # Index positions
    wind_idx = [i for i, g in enumerate(gen_labels) if g in wind_gens]
    snap_idx = [i for i, s in enumerate(snap_labels) if s in snapshots]

    # Subset dispatch variable
    p_wind = p[dict(Generator=wind_idx, snapshot=snap_idx)]

    # Align snapshot weights
    weights = n.snapshot_weightings.generators.loc[snapshots]
    weights_array = np.repeat(weights.values[:, np.newaxis], len(wind_idx), axis=1)

    # Multiply elementwise (shape matches [snapshot, Generator])
    total_dispatch = (p_wind * weights_array).sum()

    # Add constraint
    n.model.add_constraints(
        total_dispatch == target_gwh * 1000,
        name="wind_fixed_dispatch"
    )

    print(f"✅ Wind dispatch constrained to exactly {target_gwh} GWh between July–Dec 2023.")

def add_reserve_margin_constraint(n, epsilon_peak_load):
    """
    Enforces a reserve margin constraint:
    Sum of dispatchable generator capacity + battery charger capacity + PHS pumping capacity
    must be at least (1 + ε) * peak load.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    epsilon_peak_load : float
        Reserve margin percentage (e.g., 0.15 = 15%)
    """

    # Compute peak system demand
    peak_load = n.loads_t.p_set.sum(axis=1).max()
    print(f"Peak load: {peak_load:.2f} MW")

    # --- Dispatchable generators ---
    mask_dispatchable = (
        ~n.generators.carrier.isin(["solar", "onwind", "load", "ethiopia_import"]) &
        n.generators.p_nom_extendable.fillna(False)
    )
    gen_indices = n.generators.index[mask_dispatchable]
    gen_lhs = n.model.variables["Generator-p_nom"].loc[gen_indices].sum()

    # --- Battery charger links ---
    battery_indices = n.links.index[
        (n.links.carrier == "battery discharger") &
        n.links.p_nom_extendable.fillna(False)
    ]
    battery_lhs = n.model.variables["Link-p_nom"].loc[battery_indices].sum()

    # --- PHS pumping links ---
    phs_indices = n.links.index[
        (n.links.carrier == "Phs_Generation") &
        n.links.p_nom_extendable.fillna(False)
    ]
    phs_lhs = n.model.variables["Link-p_nom"].loc[phs_indices].sum()

    # --- Total LHS ---
    lhs = gen_lhs + battery_lhs + phs_lhs

    # --- RHS ---
    rhs = (1 + epsilon_peak_load) * peak_load

    # Add constraint
    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    print(f"✅ Added reserve margin constraint:")
    print(f"   Σ p_nom (dispatchable + battery + PHS) ≥ {(1 + epsilon_peak_load):.2f} × peak_load = {rhs:.2f} MW")

def impose_bus_capacity_limit(n, max_share):
    """
    Impose that no bus has a total generation capacity greater than
    max_share * peak load, excluding CCGT generators.

    Parameters
    ----------
    n : pypsa.Network
    max_share : float
        Maximum allowed share (e.g., 0.3 for 30%) of peak load per bus.
    """
    max_p_nom = n.generators.query("carrier != ['load', 'rain']").groupby('bus').p_nom.sum().max() # Max. existing bus capacity
    peak_load = n.loads_t.p_set.sum(axis=1).max() # Peak (hourly) system load
    bus_cap_limit = max(max_share * peak_load, max_p_nom) # Imposed bus capacity limit
    if bus_cap_limit == max_p_nom:
        print("❌ Imposed peak load fraction is lower than existing maximum bus capacity.")

    # Only consider extendable generators and exclude CCGT
    ext_gens = n.generators.query("p_nom_extendable")
    if ext_gens.empty:
        print("No extendable generators to constrain.")
        return

    # Group by bus
    for bus, gens_at_bus in ext_gens.groupby("bus"):
        gen_indices = gens_at_bus.index
        if len(gen_indices) == 0:
            continue
        lhs = n.model["Generator-p_nom"].loc[gen_indices].sum()
        n.model.add_constraints(lhs <= bus_cap_limit, name=f"bus_capacity_limit_{bus}")

    print(f"✅ Imposed per-bus generation capacity limit (excluding CCGT): {bus_cap_limit:.2f} MW")


def update_capacity_constraint(n):
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve

    # TODO check if `p_max_pu[ext_i]` is safe for empty `ext_i` and drop if cause in case
    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].rename(
            {"Generator-ext": "Generator"}
        )
        lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="gen_updated_capacity_constraint")


def add_operational_reserve_margin(n, sns, config):
    """
    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """

    add_operational_reserve_margin_constraint(n, sns, config)

    update_capacity_constraint(n)


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext] * eff
        - n.model["Link-p_nom"].loc[dischargers_ext] # changed * eff here
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")

    print(f"✅ Added battery constraint. Discharger is max. {eff} of charger.")

def add_RES_constraints(n, res_share, config):
    """
    The constraint ensures that a predefined share of power is generated
    by renewable sources

    Parameters
    ----------
        n : pypsa.Network
        res_share: float
        config : dict
    """

    logger.warning(
        "The add_RES_constraints() is still work in progress. "
        "Unexpected results might be incurred, particularly if "
        "temporal clustering is applied or if an unexpected change of technologies "
        "is subject to future improvements."
    )

    renew_techs = config["electricity"]["renewable_carriers"]

    charger = ["H2 electrolysis", "battery charger"]
    discharger = ["H2 fuel cell", "battery discharger"]

    ren_gen = n.generators.query("carrier in @renew_techs")
    ren_stores = n.storage_units.query("carrier in @renew_techs")
    ren_charger = n.links.query("carrier in @charger")
    ren_discharger = n.links.query("carrier in @discharger")

    gens_i = ren_gen.index
    stores_i = ren_stores.index
    charger_i = ren_charger.index
    discharger_i = ren_discharger.index

    stores_t_weights = n.snapshot_weightings.stores

    lgrouper = n.loads.bus.map(n.buses.country)
    ggrouper = ren_gen.bus.map(n.buses.country)
    sgrouper = ren_stores.bus.map(n.buses.country)
    cgrouper = ren_charger.bus0.map(n.buses.country)
    dgrouper = ren_discharger.bus0.map(n.buses.country)

    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    rhs = res_share * load

    # Generators
    lhs_gen = (
        (n.model["Generator-p"].loc[:, gens_i] * n.snapshot_weightings.generators)
        .groupby(ggrouper.to_xarray())
        .sum()
    )

    # StorageUnits
    store_disp_expr = (
        n.model["StorageUnit-p_dispatch"].loc[:, stores_i] * stores_t_weights
    )
    store_expr = n.model["StorageUnit-p_store"].loc[:, stores_i] * stores_t_weights
    charge_expr = n.model["Link-p"].loc[:, charger_i] * stores_t_weights.apply(
        lambda r: r * n.links.loc[charger_i].efficiency
    )
    discharge_expr = n.model["Link-p"].loc[:, discharger_i] * stores_t_weights.apply(
        lambda r: r * n.links.loc[discharger_i].efficiency
    )

    lhs_dispatch = store_disp_expr.groupby(sgrouper).sum()
    lhs_store = store_expr.groupby(sgrouper).sum()

    # Stores (or their resp. Link components)
    # Note that the variables "p0" and "p1" currently do not exist.
    # Thus, p0 and p1 must be derived from "p" (which exists), taking into account the link efficiency.
    lhs_charge = charge_expr.groupby(cgrouper).sum()

    lhs_discharge = discharge_expr.groupby(cgrouper).sum()

    lhs = lhs_gen + lhs_dispatch - lhs_store - lhs_charge + lhs_discharge

    n.model.add_constraints(lhs == rhs, name="res_share")


def add_land_use_constraint(n):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = (
            n.generators.loc[n.generators.carrier == carrier, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def add_h2_network_cap(n, cap):
    h2_network = n.links.loc[n.links.carrier == "H2 pipeline"]
    if h2_network.index.empty:
        return
    h2_network_cap = n.model["Link-p_nom"]
    h2_network_cap_index = h2_network_cap.indexes["Link-ext"]
    subset_index = h2_network.index.intersection(h2_network_cap_index)
    diff_index = h2_network_cap_index.difference(subset_index)
    if len(diff_index) > 0:
        logger.warning(
            f"Impossible to set a limit for H2 pipelines extension for the following links: {diff_index}"
        )
    lhs = (
        h2_network_cap.loc[subset_index] * h2_network.loc[subset_index, "length"]
    ).sum()
    rhs = cap * 1000
    n.model.add_constraints(lhs <= rhs, name="h2_network_cap")


def H2_export_yearly_constraint(n):
    res = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "offwind",
        "offwind2",
        "ror",
    ]
    res_index = n.generators.loc[n.generators.carrier.isin(res)].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_index)),
        index=n.snapshots,
        columns=res_index,
    )
    capacity_variable = n.model["Generator-p"]

    # single line sum
    res = (weightings * capacity_variable.loc[res_index]).sum()

    load_ind = n.loads[n.loads.carrier == "AC"].index.intersection(
        n.loads_t.p_set.columns
    )

    load = (
        n.loads_t.p_set[load_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    h2_export = n.loads.loc["H2 export load"].p_set * 8760

    lhs = res

    include_country_load = snakemake.config["policy_config"]["yearly"][
        "re_country_load"
    ]

    if include_country_load:
        elec_efficiency = (
            n.links.filter(like="Electrolysis", axis=0).loc[:, "efficiency"].mean()
        )
        rhs = (
            h2_export * (1 / elec_efficiency) + load
        )  # 0.7 is approximation of electrloyzer efficiency # TODO obtain value from network
    else:
        rhs = h2_export * (1 / 0.7)

    n.model.add_constraints(lhs >= rhs, name="H2ExportConstraint-RESproduction")


def monthly_constraints(n, n_ref):
    res_techs = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "offwind",
        "offwind2",
        "ror",
    ]
    allowed_excess = snakemake.config["policy_config"]["hydrogen"]["allowed_excess"]

    res_index = n.generators.loc[n.generators.carrier.isin(res_techs)].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_index)),
        index=n.snapshots,
        columns=res_index,
    )
    capacity_variable = n.model["Generator-p"]

    # single line sum
    res = (weightings * capacity_variable[res_index]).sum(axis=1)
    res = res.groupby(res.index.month).sum()

    link_p = n.model["Link-p"]
    electrolysis = link_p.loc[
        n.links.index[n.links.index.str.contains("H2 Electrolysis")]
    ]

    weightings_electrolysis = pd.DataFrame(
        np.outer(
            n.snapshot_weightings["generators"], [1.0] * len(electrolysis.columns)
        ),
        index=n.snapshots,
        columns=electrolysis.columns,
    )

    elec_input = ((-allowed_excess * weightings_electrolysis) * electrolysis).sum(
        axis=1
    )

    elec_input = elec_input.groupby(elec_input.index.month).sum()

    if snakemake.config["policy_config"]["hydrogen"]["additionality"]:
        res_ref = n_ref.generators_t.p[res_index] * weightings
        res_ref = res_ref.groupby(n_ref.generators_t.p.index.month).sum().sum(axis=1)

        elec_input_ref = (
            n_ref.links_t.p0.loc[
                :, n_ref.links_t.p0.columns.str.contains("H2 Electrolysis")
            ]
            * weightings_electrolysis
        )
        elec_input_ref = (
            -elec_input_ref.groupby(elec_input_ref.index.month).sum().sum(axis=1)
        )

        for i in range(len(res.index)):
            lhs = res.iloc[i] + "\n" + elec_input.iloc[i]
            rhs = res_ref.iloc[i] + elec_input_ref.iloc[i]
            n.model.add_constraints(
                lhs >= rhs, name=f"RESconstraints_{i}-REStarget_{i}"
            )

    else:
        for i in range(len(res.index)):
            lhs = res.iloc[i] + "\n" + elec_input.iloc[i]

            n.model.add_constraints(
                lhs >= 0.0, name=f"RESconstraints_{i}-REStarget_{i}"
            )
    # else:
    #     logger.info("ignoring H2 export constraint as wildcard is set to 0")


def add_chp_constraints(n):
    electric_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links[electric_bool].query("p_nom_extendable").index
    heat_ext = n.links[heat_bool].query("p_nom_extendable").index

    electric_fix = n.links[electric_bool].query("~p_nom_extendable").index
    heat_fix = n.links[heat_bool].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_co2_sequestration_limit(n, sns):
    co2_stores = n.stores.loc[n.stores.carrier == "co2 stored"].index

    if co2_stores.empty:
        return

    vars_final_co2_stored = n.model["Store-e"].loc[sns[-1], co2_stores]

    lhs = (1 * vars_final_co2_stored).sum()
    rhs = (
        n.config["sector"].get("co2_sequestration_potential", 5) * 1e6
    )  # TODO change 200 limit (Europe)

    name = "co2_sequestration_limit"

    n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def set_h2_colors(n):
    blue_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("blue H2")]
    ]

    pink_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("pink H2")]
    ]

    fuelcell_ind = n.loads[n.loads.carrier == "land transport fuel cell"].index

    other_ind = n.loads[
        (n.loads.carrier == "H2 for industry")
        | (n.loads.carrier == "H2 for shipping")
        | (n.loads.carrier == "H2")
    ].index

    load_fuelcell = (
        n.loads_t.p_set[fuelcell_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    load_other_h2 = n.loads.loc[other_ind].p_set.sum() * 8760

    load_h2 = load_fuelcell + load_other_h2

    weightings_blue = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(blue_h2.columns)),
        index=n.snapshots,
        columns=blue_h2.columns,
    )

    weightings_pink = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(pink_h2.columns)),
        index=n.snapshots,
        columns=pink_h2.columns,
    )

    total_blue = (weightings_blue * blue_h2).sum().sum()

    total_pink = (weightings_pink * pink_h2).sum().sum()

    rhs_blue = load_h2 * snakemake.config["sector"]["hydrogen"]["blue_share"]
    rhs_pink = load_h2 * snakemake.config["sector"]["hydrogen"]["pink_share"]

    n.model.add_constraints(total_blue == rhs_blue, name="blue_h2_share")

    n.model.add_constraints(total_pink == rhs_pink, name="pink_h2_share")


def add_existing(n):
    if snakemake.wildcards["planning_horizons"] == "2050":
        directory = (
            "results/"
            + "Existing_capacities/"
            + snakemake.config["run"].replace("2050", "2030")
        )
        n_name = (
            snakemake.input.network.split("/")[-1]
            .replace(str(snakemake.config["scenario"]["clusters"][0]), "")
            .replace(str(snakemake.config["costs"]["discountrate"][0]), "")
            .replace("_presec", "")
            .replace(".nc", ".csv")
        )
        df = pd.read_csv(directory + "/electrolyzer_caps_" + n_name, index_col=0)
        existing_electrolyzers = df.p_nom_opt.values

        h2_index = n.links[n.links.carrier == "H2 Electrolysis"].index
        n.links.loc[h2_index, "p_nom_min"] = existing_electrolyzers

        # n_name = snakemake.input.network.split("/")[-1].replace(str(snakemake.config["scenario"]["clusters"][0]), "").\
        #     replace(".nc", ".csv").replace(str(snakemake.config["costs"]["discountrate"][0]), "")
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)

        for tech in snakemake.config["custom_data"]["renewables"]:
            # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
            existing_res = df.loc[tech]
            existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
            tech_index = n.generators[n.generators.carrier == tech].index
            n.generators.loc[tech_index, tech] = existing_res

def disable_expansion_calibration(n, config):
    if config['enable']['calibration_run']:
        n.generators["p_nom_extendable"] = False
        n.links["p_nom_extendable"] = False
        n.storage_units["e_nom_extendable"] = False
        print("✅ Expansion calibration mode: All p_nom and e_nom set to non-extendable.")

def add_lossy_bidirectional_link_constraints(n: pypsa.components.Network) -> None:
    """
    Ensures that the two links simulating a bidirectional_link are extended the same amount.
    """

    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    # ensure that the 'reversed' column is boolean and identify all link carriers that have 'reversed' links
    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    # get the indices of all forward links (non-reversed), that have a reversed counterpart
    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    # function to get backward (reversed) indices corresponding to forward links
    # this function is required to properly interact with the myopic naming scheme
    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    # get the indices of all backward links (reversed)
    backward_i = get_backward_i(forward_i)

    # get the p_nom optimization variables for the links using the get_var function
    links_p_nom = n.model["Link-p_nom"]

    # only consider forward and backward links that are present in the optimization variables
    subset_forward = forward_i.intersection(links_p_nom.indexes["Link-ext"])
    subset_backward = backward_i.intersection(links_p_nom.indexes["Link-ext"])

    # ensure we have a matching number of forward and backward links
    if len(subset_forward) != len(subset_backward):
        raise ValueError("Mismatch between forward and backward links.")

    # define the lefthand side of the constrain p_nom (forward) - p_nom (backward) = 0
    # this ensures that the forward links always have the same maximum nominal power as their backward counterpart
    lhs = links_p_nom.loc[backward_i] - links_p_nom.loc[forward_i]

    # add the constraint to the PySPA model
    n.model.add_constraints(lhs == 0, name="Link-bidirectional_sync")

def add_battery_duration_constraint(n, max_hours=6.0):
    """
    Adds constraints to enforce an upper duration bound on extendable battery setups.
    Formulation: P_discharger - (1 / max_hours) * E_store <= 0
    """

    # 1. Isolate components explicitly flag-filtered for investment expansion
    battery_stores = n.stores[(n.stores.carrier == "battery") & (n.stores.e_nom_extendable)]
    discharger_links = n.links[(n.links.carrier == "battery discharger") & (n.links.p_nom_extendable)]

    if battery_stores.empty:
        logger.info("ℹ️ No extendable battery stores detected. Skipping battery constraint binding.")
        return

    # 2. Dynamic optimization variable identification safely matching Linopy standards
    store_var_name = "Store-e_nom" if "Store-e_nom" in n.model.variables else "Store-e_nom_extendable"
    link_var_name = "Link-p_nom" if "Link-p_nom" in n.model.variables else "Link-p_nom_extendable"

    # 3. Extract Linopy variables and their coordinate arrays (Logic from function 1)
    e_nom = n.model.variables[store_var_name]
    p_nom = n.model.variables[link_var_name]
    
    # Dynamically grab the correct dimension names (usually "Store-ext" and "Link-ext")
    store_dim = e_nom.dims[0]
    link_dim = p_nom.dims[0]
    
    store_labels = e_nom.coords[store_dim].values
    link_labels = p_nom.coords[link_dim].values

    bound_count = 0

    # 4. Step through assets using prefix naming strings to link elements
    for store_name in battery_stores.index:
        base_prefix = store_name.replace(" battery", "")
        
        dis_match = discharger_links[discharger_links.index.str.startswith(base_prefix)]
        if dis_match.empty:
            continue
            
        dis_name = dis_match.index[0]

        # 5. Find positional indices using np.where() to avoid KeyErrors
        i_store = np.where(store_labels == store_name)[0]
        i_discharger = np.where(link_labels == dis_name)[0]

        if i_store.size == 0 or i_discharger.size == 0:
            logger.warning(f"⚠️ Skipped {store_name}: Optimization variables missing from Linopy matrix slice.")
            continue

        # 6. Apply Constraint: Maximum energy-to-power storage duration ceiling
        n.model.add_constraints(
            p_nom.isel(**{link_dim: i_discharger[0]}) - (1 / max_hours) * e_nom.isel(**{store_dim: i_store[0]}) >= 0,
            name=f"battery_max_ratio_{store_name}"
        )

        bound_count += 1

    logger.info(f"🎯 Successfully appended duration limits to {bound_count} batteries.")

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.linopf.network_lopf``.

    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    if config['enable']['add_PHS']:
        if config['enable']['impose_PHS_max_hours']:
            add_phs_constraints(n, config['electricity']['max_hours']['CL_PHS']+2, config['electricity']['max_hours']['CL_PHS']-2,config['enable']['N1_PHS']['max_PHS_cap'], config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "RES" in o:
            res_share = float(re.findall("[0-9]*\.?[0-9]+$", o)[0])
            add_RES_constraints(n, res_share, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    if config['enable']['ethiopia_import_minimum_dispatch']:
        ethiopia_constraints(n)
    else:
        print("❌ Ethiopia minimum import constraint not applied.")
    if config["custom_data"]["impose_charger_equals_discharger"]:
        equalize_battery_charger_discharger(n)
    if config['enable'].get('force_ethiopia_dispatch', False):
        enforce_ethiopia_dispatch(n,419)
    else:
        print("❌ Ethiopia force dispatch constraint not applied.")
    if config['enable'].get('force_wind_dispatch', False):
        enforce_wind_dispatch(n,972)
    else:
        print("❌ Wind force dispatch constraint not applied.")
    if not config['enable']['impose_dec_gens'] == False:
        impose_bus_capacity_limit(n, max_share = config['enable']['impose_dec_gens'])
    if config['enable'].get('battery_duration_constraint', False):
        add_battery_duration_constraint(n, max_hours=config['enable'].get('battery_duration_constraint', 6.0))
    else:
        print("❌ Battery duration constraint not applied.")

    # add_battery_constraints(n) # Commented out myself to use my own battery constraint instead
    add_lossy_bidirectional_link_constraints(n)

    disable_expansion_calibration(n, config) # Disables expansion for calibration runs

    if snakemake.config["sector"]["chp"]:
        logger.info("setting CHP constraints")
        add_chp_constraints(n)

    if (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_yearly_matching"
    ):
        if snakemake.config["policy_config"]["hydrogen"]["additionality"] == True:
            logger.info(
                "additionality is currently not supported for yearly constraints, proceeding without additionality"
            )
        logger.info("setting h2 export to yearly greenness constraint")
        H2_export_yearly_constraint(n)

    elif (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_monthly_matching"
    ):
        if not snakemake.config["policy_config"]["hydrogen"]["is_reference"]:
            logger.info("setting h2 export to monthly greenness constraint")
            monthly_constraints(n, n_ref)
        else:
            logger.info("preparing reference case for additionality constraint")

    elif (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "no_res_matching"
    ):
        logger.info("no h2 export constraint set")

    else:
        raise ValueError(
            'H2 export constraint is invalid, check config["policy_config"]'
        )

    if snakemake.config["sector"]["hydrogen"]["network"]:
        if snakemake.config["sector"]["hydrogen"]["network_limit"]:
            add_h2_network_cap(
                n, snakemake.config["sector"]["hydrogen"]["network_limit"]
            )

    if snakemake.config["sector"]["hydrogen"]["set_color_shares"]:
        logger.info("setting H2 color mix")
        set_h2_colors(n)

    add_co2_sequestration_limit(n, snapshots)


def configure_transmission_losses(n, config):
    """
    Enable PyPSA's piecewise-linear transmission loss approximation.

    PyPSA requires finite s_nom_max values for extendable passive branches when
    transmission losses are enabled. Some scenarios leave them infinite, so this
    function applies a configurable fallback bound before optimisation.
    """

    if not config["enable"].get("piecewise_transmission_losses", False):
        return 0

    tangents = int(config["enable"].get("transmission_loss_tangents", 3))
    if tangents <= 0:
        logger.warning("Transmission losses requested, but tangents <= 0. Skipping.")
        return 0

    factor = float(config["enable"].get("transmission_loss_s_nom_max_factor", 3.0))
    floor = float(config["enable"].get("transmission_loss_s_nom_max_floor", 1000.0))

    for component in n.passive_branch_components:
        branches = n.df(component)
        if branches.empty or "s_nom_extendable" not in branches:
            continue

        extendable = branches.index[branches.s_nom_extendable.astype(bool)]
        if extendable.empty:
            continue

        s_nom_max = branches.loc[extendable, "s_nom_max"]
        unbounded = extendable[~np.isfinite(s_nom_max.astype(float))]
        if unbounded.empty:
            continue

        base = branches.loc[unbounded, "s_nom"].astype(float)
        if "s_nom_min" in branches:
            base = pd.concat(
                [base, branches.loc[unbounded, "s_nom_min"].astype(float)], axis=1
            ).max(axis=1)

        fallback = (base * factor).clip(lower=floor)
        n.df(component).loc[unbounded, "s_nom_max"] = fallback
        logger.info(
            "Set finite s_nom_max for %s %s branches for transmission losses "
            "(factor=%s, floor=%s MW).",
            len(unbounded),
            component,
            factor,
            floor,
        )

    logger.info(
        "✅ Enabled PyPSA transmission losses with %s piecewise-linear tangents.",
        tangents,
    )
    return tangents

def solve_network(n, config, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = configure_transmission_losses(n, config)

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = (cf_solving.get("min_iterations", 4),)
        kwargs["max_iterations"] = (cf_solving.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if status != "ok":  # and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            simpl="",
            clusters="25",
            ll="c1.0",
            opts="Ep-1h-fix",
            planning_horizons="2030",
            discountrate="0.12",
            demand="AB",
            sopts="144H",
            h2export="0",
            configfile="config_2023_calibrated.yaml",
        )

    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]
    config = snakemake.config

    is_sector_coupled = "sopts" in snakemake.wildcards.keys()

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    if snakemake.params.augmented_line_connection.get("add_to_snakefile"):
        n.lines.loc[n.lines.index.str.contains("new"), "s_nom_min"] = (
            snakemake.params.augmented_line_connection.get("min_expansion")
        )

    if (
        snakemake.config["custom_data"]["add_existing"]
        and snakemake.wildcards.planning_horizons == "2050"
        and is_sector_coupled
    ):
        add_existing(n)

    for o in opts:
        if "Ep" in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                logger.info("Setting emission prices according to wildcard value.")
                add_emission_prices(n, dict(co2=float(m[0])))
            else:
                logger.info("Setting emission prices according to config value.")
                add_emission_prices(n, snakemake.params.costs["emission_prices"])
                logger.info(f"Emission prices: {n.generators[n.generators.carrier == 'CCGT'].marginal_cost}")
            break

    if (
        snakemake.config["policy_config"]["hydrogen"]["additionality"]
        and not snakemake.config["policy_config"]["hydrogen"]["is_reference"]
        and snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        != "no_res_matching"
        and is_sector_coupled
    ):
        n_ref_path = snakemake.config["policy_config"]["hydrogen"]["path_to_ref"]
        n_ref = pypsa.Network(n_ref_path)
    else:
        n_ref = None

    n = prepare_network(n, solve_opts, config=solve_opts)

    if config["lines"].get("resistance_multiplier", 1.0) != 1.0:
        n.line_types["r_per_length"] *= config["lines"]["resistance_multiplier"] # This inflates the resistance by 50% of lines to account for underestimation of losses due to tortuosity, temperature and absence of transformers
        print("✅ Applied resistance multiplier of {} to line types".format(config["lines"]["resistance_multiplier"]))

    if config["enable"]["impose_geothermal_ramp_limits"]:
        n.generators.loc[n.generators.carrier == "geothermal", "ramp_limit_up"] = 0.0
        n.generators.loc[n.generators.carrier == "geothermal", "ramp_limit_down"] = 0.0
        print("✅ Imposed 80% ramp limits on geothermal plants.")

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        log_fn=snakemake.log.solver,
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Objective function: {n.objective}")
    logger.info(f"Objective constant: {n.objective_constant}")
