import numpy as np
import pandas as pd
from _helpers import BASE_DIR, configure_logging, create_logger
logger = create_logger(__name__)
logger.propagate = False  # Add this to stop the duplicate echo

def reservoir_storage_capacity(volume, head):
    """Calculate theoretical potential energy capacity of a hydro reservoir.

    Converts volumetric water storage (V in million cubic meters, Mm^3) and
    effective hydraulic head (h in meters) into equivalent energy capacity in MWh
    using gravitational potential energy at 100% nominal water-to-wire recovery.

    Parameters
    ----------
    volume : float
        Gross or active reservoir storage volume in million cubic meters (Mm^3 / MCM).
    head : float
        Gross hydraulic head or dam height in meters (m).

    Returns
    -------
    float
        Theoretical potential energy capacity in megawatt-hours (MWh).

    Notes
    -----
    The formulation computes:
        E = (V * 10^6 m^3 * rho * g * h) / (3.6 * 10^9 J/MWh)
    where:
        - rho = 1000 kg/m^3 (density of water)
        - g = 9.81 m/s^2 (standard gravity)
        - 3.6 * 10^9 J = 1 MWh
    """
    e_nominal = volume * 10**6 * 9.81 * 1000 * head / (3.6e9) # MWh
    return e_nominal

def build_seven_forks_cascade(n, config):
    """Construct a chained multi-port hydro cascade for Kenya's Seven Forks scheme.

    Replaces aggregated, single-bus `StorageUnit` components created during network
    clustering with an explicit hydrological network of linked reservoirs, turbines,
    spillways, and natural inflow generators. This implementation is inspired by and
    adapts the chained reservoir formulation from the official PyPSA documentation:
    https://docs.pypsa.org/v1.0.2/examples/chained-hydro-reservoirs/

    Hydrological & Electrical Topology:
    -----------------------------------
    For each dam in the cascade sequence:

        [Rain Generator] -> (Natural run-off profile via p_max_pu)
               |
               v
        [Water Bus: '{name} water bus'] <---> [Store: '{name} reservoir']
          |                       |
          | (Turbine Link)        | (Spillage Link)
          | eff1 = 0.90           | eff = transit_loss * (head_down / head_up)
          |                       |
          +--------> [bus1: Clustered AC Substation]
          |
          +--(bus2: eff2 = transit_loss * head_down / head_up)
          |
          v
        [Downstream Water Bus] (or 'Cascade Outflow' sink for the terminal dam)

    Methodological Steps:
    ---------------------
    1. Schema Extension:
       Dynamically registers multi-port link attributes (`bus2`, `efficiency2`, `p2`)
       into `n.component_attrs["Link"]` to allow turbine discharge to serve as an
       energy source for downstream reservoirs.
    2. Asset Parameterization & Geospatial Matching:
       Reads dam coordinates, head, and volume from `custom_powerplants_choose` CSV.
       Computes theoretical potential energy and assigns each dam to the nearest
       non-PHS AC grid substation via Euclidean minimum distance.
    3. Inflow Re-apportionment:
       Harvests ERA5/weather-derived hydro inflows previously pooled onto regional
       clustered `StorageUnit` assets, weights them proportionally by each reservoir's
       energy storage bound ($e_{\text{nom}}$), and deletes the old clustered assets.
    4. Component Generation:
       - Adds discrete carrier types: `'reservoir'`, `'sink'`, `'rain'`.
       - Instantiates water buses and `Store` units with cyclic constraints.
       - Connects 3-port turbine `Link` elements (power to AC bus, discharge to downstream).
       - Instantiates spillage bypass `Link` elements with head-ratio conversion.
       - Attaches a non-cyclic terminal sink `Store` (`Cascade Outflow Sink`) to capture
         outflows from the final reservoir (e.g., Kiambere or High Grand Falls).
    5. Natural Inflow Mapping:
       Injects apportioned time-series inflows as `'rain'` `Generator` units bound to
       the water buses with tight bounds (`p_min_pu = normalized - 0.01`, `p_max_pu = normalized`).

    Parameters
    ----------
    n : pypsa.Network
        The clustered PyPSA network instance to be modified in-place.
    config : dict
        PyPSA configuration dictionary containing:
        - `config["electricity"]["custom_powerplants_choose"]`: Path to the hydro
          specifications CSV file.
        - `config["renewable"]["hydro"]["limit_hydro_capacity"]`: Optional float
          scaling factor applied to usable reservoir capacity (default: 1.0).
        - `config["renewable"]["hydro"]["multiplier"]`: Optional float scalar
          applied to natural inflow profiles (default: 1.0).

    Returns
    -------
    pypsa.Network
        The network instance with the cascade scheme integrated.

    Credits
    -----------------
    Pierre Karamountzos (pierre.karamountzos25@imperial.ac.uk)

    References
    ----------
    [1] PyPSA Developers, "Chained Hydro Reservoirs Example",
    https://docs.pypsa.org/v1.0.2/examples/chained-hydro-reservoirs/
    """
    n.add("Carrier", "reservoir")
    n.add("Carrier", "sink")
    n.add("Carrier", "rain")

    # 1. Enable multi-port link attributes in the existing network schema
    n.component_attrs["Link"].loc["bus2"] = [
        "string", np.nan, np.nan, "2nd bus", "Input"
    ]
    n.component_attrs["Link"].loc["efficiency2"] = [
        "static or series", "per unit", 1.0, "2nd bus efficiency", "Input"
    ]
    n.component_attrs["Link"].loc["p2"] = [
        "series", "MW", 0.0, "2nd bus output", "Output"
    ]

    # 2. Dynamically build specifications from custom powerplants CSV
    csv_path = config["electricity"].get("custom_powerplants_choose")
    if not csv_path:
        logger.error("⚠️ No custom powerplants CSV path found in config! Aborting cascade.")
        return n
        
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
    except Exception as e:
        logger.error(f"⚠️ Failed to read CSV at {csv_path}: {e}")
        return n

    # if a reservoir with the name Karura exists in df, include it in the 7 Forks cascade
    if "karura" in df["name"].str.lower().values:
        cascade_order = ["Masinga", "Kamburu", "Gitaru", "Kindaruma", "Karura", "Kiambere", "High Grand Falls"]
    else:
        cascade_order = ["Masinga", "Kamburu", "Gitaru", "Kindaruma", "Kiambere"]
    
    cascade_specs = {}
    # Fetch all AC buses
    ac_buses_raw = n.buses[n.buses.carrier == "AC"]
    
    # ADJUSTMENT: Filter out any virtual buses that start with 'PHS' or 'phs'
    is_phs_bus = ac_buses_raw.index.str.strip().str.upper().str.startswith("PHS")
    ac_buses = ac_buses_raw[~is_phs_bus]
    
    logger.info(f"Filtered out {is_phs_bus.sum()} PHS buses from geographic matching pool. {len(ac_buses)} true AC network nodes remaining.")
    
    usable_res_volume = config['renewable']['hydro'].get('limit_hydro_capacity', 1.0)
    transit_losses = 0.85 # Assume 15% water losses between reservoirs

    for i, plant_name in enumerate(cascade_order):
        row = df[df["name"].str.contains(plant_name, case=False, na=False)]
        
        if row.empty:
            logger.error(f"⚠️ Could not find '{plant_name}' in {csv_path}. Aborting cascade.")
            return n
            
        row = row.iloc[0]
        p_nom = float(row["capacity"])
        volume_mcm = float(row["volume_mm3"]) 
        head = float(row["damheight_m"])         
        
        # Safely extract coordinates from your CSV schema
        lat_col = [c for c in ["latitude", "lat"] if c in row.index]
        lon_col = [c for c in ["longitude", "lon"] if c in row.index]
        if not lat_col or not lon_col:
            logger.error(f"⚠️ Missing coordinate columns (lat/lon) in CSV for {plant_name}!")
            return n
            
        lat = float(row[lat_col[0]])
        lon = float(row[lon_col[0]])
        
        # Calculate theoretical potential energy (MWh) at 1.0 efficiency
        e_nom = reservoir_storage_capacity(volume_mcm, head)
        downstream = cascade_order[i+1] if i + 1 < len(cascade_order) else None
        
        # GEOGRAPHIC MATCHING: Find the closest clustered AC bus to the dam
        distances = np.sqrt((ac_buses.x - lon)**2 + (ac_buses.y - lat)**2)
        closest_elec_bus = distances.idxmin()
        
        # FIXED: Added x, y, lon, lat to specs so we can pass them to the buses later
        cascade_specs[plant_name] = {
            "p_nom": p_nom,
            "e_nom": e_nom,
            "head": head,
            "downstream": downstream,
            "elec_bus": closest_elec_bus,
            "x": lon,
            "y": lat
        }
        logger.info(f"Geomapped {plant_name} to clustered electrical bus '{closest_elec_bus}'")

    # ==============================================================================
    # 3. FIXED: Harvest inflows using the verified geographic mappings
    # ==============================================================================
    harvested_inflows = {} 
    
    # Group our cascade plants by their assigned clustered electrical bus
    from collections import defaultdict
    bus_to_plants = defaultdict(list)
    for plant_name, specs in cascade_specs.items():
        bus_to_plants[specs["elec_bus"]].append(plant_name)
        
    # Process the inflows exactly once per unique clustered bus
    for target_bus, plants in bus_to_plants.items():
        # Find the consolidated clustered hydro asset sitting on this regional node
        matching_su = n.storage_units[
            (n.storage_units.bus == target_bus) & 
            (n.storage_units.carrier.str.lower().str.contains("hydro", na=False))
        ].index.tolist()
        
        if matching_su:
            su_id = matching_su[0]
            clustered_inflow = n.storage_units_t.inflow[su_id].copy()
            logger.info(f"Harvesting pooled weather cutout '{su_id}' for shared cluster plants: {plants}")
            
            # Calculate the total combined potential energy capacity (MWh) sharing this bus
            total_energy_on_bus = sum(cascade_specs[p]["e_nom"] for p in plants)
            
            # Apportion the pooled water among the sharing reservoirs based on storage weight
            for plant_name in plants:
                if total_energy_on_bus > 0:
                    weight = cascade_specs[plant_name]["e_nom"] / total_energy_on_bus
                else:
                    weight = 1.0 / len(plants)  # Safety fallback to even split if energy bounds are zero
                
                # Assign exactly to the cleanly mapped reservoir key
                harvested_inflows[f"{plant_name} reservoir"] = clustered_inflow * weight
                logger.info(f"  -> Assigned {weight*100:.1f}% of '{su_id}' inflow to {plant_name} reservoir based on storage size")
                
            # Safely remove the clustered asset now that its data has been fully distributed
            n.remove("StorageUnit", su_id)
        else:
            logger.warning(f"No clustered hydro asset found at bus '{target_bus}' for plants {plants}")

    # ==============================================================================
    # 4. FIXED: Build Nodes & Stores with unique names to prevent collisions
    # ==============================================================================
    
    # Stage A: Create reservoirs
    for name, specs in cascade_specs.items():
        water_bus = f"{name} water bus"
        
        # FIXED: Added geographic coordinates and country tags directly to the bus kwargs
        n.add("Bus", water_bus, 
              carrier="reservoir", 
              v_nom=300.0,
              x=specs["x"],
              y=specs["y"],
              country="KE")
        
        # 2. Force-inject the custom PyPSA-Earth columns via Pandas
        n.buses.loc[water_bus, ["lon", "lat", "country"]] = [specs["x"], specs["y"], "KE"]
        
        n.add("Store", f"{name} reservoir",
              bus=water_bus,
              e_nom=specs["e_nom"] * usable_res_volume,
              e_cyclic=True)
              
    # Stage B: Connect the turbine and spillage links

    # FIXED: Place the final "Cascade Outflow" sink slightly downstream/east of Kiambere 
    last_plant = cascade_order[-1]
    outflow_x = cascade_specs[last_plant]["x"] + 0.05
    outflow_y = cascade_specs[last_plant]["y"] - 0.02
    
    # 1. Add the outflow bus natively
    n.add("Bus", "Cascade Outflow", 
          carrier="sink", 
          v_nom=300.0,
          x=outflow_x,
          y=outflow_y)
          
    # 2. Force-inject the custom columns here too
    n.buses.loc["Cascade Outflow", ["lon", "lat", "country"]] = [outflow_x, outflow_y, "KE"]

    for name, specs in cascade_specs.items():
        water_bus = f"{name} water bus"
        
        # Connect turbine links (incorporating your 90% powertrain efficiency parameter)
        link_kwargs = {
            "bus0": water_bus,
            "bus1": specs["elec_bus"],
            "efficiency": 0.9, 
            "p_nom": specs["p_nom"] 
        }
        
        if specs["downstream"]:
            downstream_name = specs["downstream"]
            link_kwargs["bus2"] = f"{downstream_name} water bus"
            link_kwargs["efficiency2"] = transit_losses * cascade_specs[downstream_name]["head"] / specs["head"]
            
        n.add("Link", f"{name} turbine", **link_kwargs)

        # Connect spillage links
        if specs["downstream"]:
            downstream_name = specs["downstream"]

            n.add("Link", f"{name} spillage",
                  bus0=water_bus,
                  bus1=f"{downstream_name} water bus",
                  efficiency=transit_losses * cascade_specs[downstream_name]["head"] / specs["head"],
                  p_nom=1e6,
                  p_nom_min=1e6-10,
                  p_nom_extendable=False)
        else:
            # FIXED: The final reservoir (Kiambere) spills out of the system
            n.add("Link", f"{name} spillage",
                  bus0=water_bus,
                  bus1="Cascade Outflow",
                  carrier='sink',
                  efficiency=1.0,  # Just dumping volume, no energy conversion needed
                  p_nom=1e6)
            n.buses.loc["Cascade Outflow", ["lon", "lat", "country"]] = [outflow_x, outflow_y, "KE"]

            # 2. FIXED: Add an infinite dummy storage bucket to absorb the spilled water safely
            n.add("Store", "Cascade Outflow Sink",
                bus="Cascade Outflow",
                carrier="sink",
                e_nom=1e6,          # Fictive reservoir to absorb spillage of last reservoir
                e_initial=0.0,      # Start empty
                e_cyclic=False)     # Let it fill up infinitely without forcing it to empty

    # ==============================================================================
    # 5. FIXED: Route localized natural inflows via Rain Generators
    # ==============================================================================
    multiplier = config['renewable']['hydro'].get('multiplier', 1.0)
    
    # Map each saved profile to its corresponding water bus using a Generator
    for target_store, inflow_series in harvested_inflows.items():
        # Clean up the name string properly to just the plant name (e.g., "Masinga")
        base_name = target_store.replace(" reservoir", "") 
        water_bus = f"{base_name} water bus"  # Maps perfectly to Stage A
        
        # Apply config multiplier
        adjusted_inflow = inflow_series * multiplier
        
        # PyPSA best practice: Define p_nom as the max capacity, and p_max_pu as a 0-1 normalized profile
        p_nom_max = adjusted_inflow.max()
        
        if p_nom_max > 0:
            normalized_inflow = adjusted_inflow / p_nom_max
        else:
            p_nom_max = 0.0
            normalized_inflow = 0.0
            
        n.add("Generator", f"{base_name} rain inflow",
              bus=water_bus,
              carrier="rain",
              p_nom=p_nom_max,
              p_min_pu=normalized_inflow-0.01,
              p_max_pu=normalized_inflow,
              marginal_cost=0.0) # Free energy
              
        logger.info(f"Added rain generator for {base_name} with peak {p_nom_max:.2f} MW")
    
    # Clean pipeline compatibility adjustments (prevents downstream script crashes)
    if "underwater_fraction" not in n.links.columns:
        n.links["underwater_fraction"] = 0.0
    else:
        n.links["underwater_fraction"] = n.links["underwater_fraction"].fillna(0.0)

    if "dc" not in n.links.columns:
        n.links["dc"] = False
    else:
        n.links["dc"] = n.links["dc"].fillna(False)

    logger.info("✅ Successfully established Seven Forks Cascade constraints post-clustering.")
    return n