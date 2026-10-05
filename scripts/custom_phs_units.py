import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import numpy as np

def add_and_connect_phs_msr_clusters(n, config):
    
    """Integrate Pumped Hydro Storage (PHS) clusters into the network.

    Reads geospatial and technical specifications of candidate PHS clusters from
    a CSV file, models each cluster as a dual-bus store/link topology, calculates
    annualized capital expenditure and operational costs, identifies the nearest
    transmission bus on the main synchronized AC grid, and connects the facility
    using an extendable transmission line.

    Topology Layout per Cluster:
        [Main AC Grid Bus]
                 |
                 | (Extendable AC Line: length = geodesic distance)
                 v
        [Dummy Bus: PHS_dummy_{cluster}]  <-- AC bus at 380 kV
          |                            ^
          | (Pumping Link: eff=0.90)   | (Generation Link: eff=0.90)
          v                            |
        [Reservoir Bus: PHS_{cluster}] <-- Energy carrier bus
                 |
                 | (Direct bus attachment)
                 v
        [Store: PHS_{cluster}_store]   <-- Cyclic MWh reservoir

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance to which carriers, buses, stores, links,
        and transmission lines will be attached.
    config : dict
        A nested dictionary representing the PyPSA workflow configuration.
        Required keys:
        - `config["costs"]["USD2013_to_EUR2013"]`: float
            Currency conversion factor.
        - `config["costs"]["discountrate"][0]`: float
            Discount rate (weighted average cost of capital) used for the
            capital recovery factor / annuality factor.

    Input Data Requirements
    -----------------------
    Reads `data/phs_clusters.csv` which must contain the following columns:
    - `Name` (str): Unique site or cluster identifier.
    - `Bus_X` (float): Longitude of the PHS site.
    - `Bus_Y` (float): Latitude of the PHS site.
    - `Energy_Storage_Capacity(MWh)` (float): Upper reservoir bound (`e_nom_max`).
    - `Unit Cost of Storage($/kWh)` (float): Overnight energy capital expenditure.
    - `Unit Cost of Power ($/kW)` (float): Overnight power rating capital expenditure
      (pumping and turbine machinery).

    Financial Formulation
    ---------------------
    Annualized capital costs ($C_{ann}$) are derived using the Capital Recovery
    Factor ($CRF$) and Fixed Operations & Maintenance percentage ($FoM$):
        CRF = r / (1 - (1 + r)^(-lifetime))
        C_ann = Unit_Cost * 1000 * (CRF + FoM / 100) * USDtoEUR
    Where:
        - `lifetime` = 80 years
        - `FoM` = 1.5% per year
        - `r` = discount_rate

    Notes
    -----
    - **Round-trip Efficiency**: Symmetrical link efficiencies of $\eta = 0.90$
      yield a total cycle efficiency of $0.90 \times 0.90 = 81.0\%$.
    - **Cost Allocation**: Turbine generation capex is assigned $0, attributing
      the combined powerhouse machinery capex entirely to the pumping link to
      prevent double-counting during capacity expansion.
    - **Grid Connection**: Candidate clusters connect exclusively to the
      largest connected AC component (`main_component`) to avoid linking into
      isolated or orphaned sub-networks.

    Credits
    -----------------
    Pierre Karamountzos (pierre.karamountzos25@imperial.ac.uk)
    """
    
    USDtoEUR = config["costs"]["USD2013_to_EUR2013"]
    discount_rate = config["costs"]["discountrate"][0]

    # Step 1: Add necessary carriers
    n.add("Carrier", "Phs_Reservoir", color="#EE4B2B", nice_name="CL PHS storage")
    n.add("Carrier", "Phs_Pumping", color="#000000", nice_name="CL PHS pump")
    n.add("Carrier", "Phs_Generation", color="#808080", nice_name="CL PHS gen")
    n.add("Carrier", "Phs_Store", color="#1742AF", nice_name="CL PHS store")

    # Step 2: Load PHS cluster sites
    PHS_sites_path = "data/phs_clusters.csv"
    phs_clusters = pd.read_csv(
        PHS_sites_path
    )

    # Step 3: Build grid graph for finding closest buses
    G = nx.Graph()
    G.add_edges_from(zip(n.lines.bus0[n.lines.carrier == "AC"], n.lines.bus1[n.lines.carrier == "AC"]))
    components = list(nx.connected_components(G))
    main_component = max(components, key=len)
    grid_buses = list(main_component)

    # Step 4: Add PHS reservoirs, stores, links, and candidate lines
    for idx, row in phs_clusters.iterrows():
        cluster_name = row["Name"]
        site_lon = row["Bus_X"]
        site_lat = row["Bus_Y"]
        storage_capacity_mwh = row['Energy_Storage_Capacity(MWh)']

        site_x = site_lon
        site_y = site_lat

        # ➡️ Find closest grid bus
        other_buses = n.buses.loc[grid_buses]
        distances = other_buses.apply(
            lambda r: geodesic((site_lat, site_lon), (r["y"], r["x"])).km,
            axis=1
        )
        closest_grid_bus = distances.idxmin()
        dist_km = distances.min()

        # Default costs
        lifetime = 80 # https://ease-storage.eu/wp-content/uploads/2016/07/EASE_TD_Mechanical_PHS.pdf
        FoM = 1.5 # %/year - https://www.mdpi.com/1996-1073/16/11/4516
        annuality_factor = discount_rate/(1-(1+discount_rate)**(-lifetime))
        unit_cost_kWh = row["Unit Cost of Storage($/kWh)"]  # $/kWh
        link_capital_cost = row["Unit Cost of Power ($/kW)"]  # $/kW for pumping/turbine - 0 if we consider that the powerhouse costs are included in the capital cost of the reservoir
        reservoir_bus = f"PHS_{cluster_name}"
        reservoir_bus_dummy = f"PHS_dummy_{cluster_name}"

        # Add PHS reservoir bus
        n.add("Bus",
            name=reservoir_bus,
            carrier="Phs_Reservoir",
            x=site_x,
            y=site_y,
            v_nom=380,
            control="PQ",
            country = 'KE'
        )
        n.buses.loc[reservoir_bus, "lat"] = site_lat
        n.buses.loc[reservoir_bus, "lon"] = site_lon

        n.add("Bus",
            name=reservoir_bus_dummy,
            carrier="AC",
            x=site_x,
            y=site_y,
            v_nom=380,
            control="PQ",
            country = 'KE'
        )
        n.buses.loc[reservoir_bus_dummy, "lat"] = site_lat
        n.buses.loc[reservoir_bus_dummy, "lon"] = site_lon

        # Add Store (energy reservoir)
        n.add("Store",
            name=f"PHS_{cluster_name}_store",
            bus=reservoir_bus,
            carrier="Phs_Store",
            e_cyclic=True,
            e_nom_extendable=True,
            e_nom_max=storage_capacity_mwh,
            capital_cost= unit_cost_kWh * 1000 * (annuality_factor + FoM/100) * USDtoEUR # €/MWh 
        )

        # Add Links for pumping (grid → reservoir) and generation (reservoir → grid)
        n.add("Link",
            name=f"PHS_{cluster_name}_pumping",
            bus0=reservoir_bus_dummy,
            bus1=reservoir_bus,
            carrier="Phs_Pumping",
            efficiency=0.9,
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max= np.inf,
            capital_cost= (link_capital_cost) * 1000 * (annuality_factor + FoM/100) * USDtoEUR # €/MW
        )

        n.add("Link",
            name=f"PHS_{cluster_name}_generation",
            bus0=reservoir_bus,
            bus1=reservoir_bus_dummy,
            carrier="Phs_Generation",
            efficiency=0.9,
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max = np.inf,
            capital_cost=0
        )

        # Add candidate Line (PHS bus ↔️ grid bus)
        line_name = f"{reservoir_bus_dummy}--{closest_grid_bus}"
        if line_name not in n.lines.index:
            n.add("Line",
                name=line_name,
                bus0=reservoir_bus_dummy,
                bus1=closest_grid_bus,
                length=dist_km,
                s_nom_extendable=True,
                carrier="AC"
            )

        n.lines["dc"] = n.lines["dc"].fillna(0.0).astype(float)
        n.lines.loc[n.lines.index.str.startswith("PHS_"), "type"] = "Al/St 240/40 2-bundle 220.0"
        n.lines.loc[n.lines.index.str.startswith("PHS_"), "s_nom_min"] = 0.0
        n.lines.loc[n.lines.index.str.startswith("PHS_"), "num_parallel"] = 1.0 # 0.260526 Based on minimum values for pre-existing lines (see prepare_network.py)

