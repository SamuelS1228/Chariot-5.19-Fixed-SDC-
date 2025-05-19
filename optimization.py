
import numpy as np
from sklearn.cluster import KMeans
from utils import haversine, warehousing_cost

ROAD_FACTOR = 1.3  # inflate straight‑line distance to approximate road miles

def _dist_matrix(lon, lat, centers):
    d = np.empty((len(lon), len(centers)))
    for j, (clon, clat) in enumerate(centers):
        d[:, j] = haversine(lon, lat, clon, clat) * ROAD_FACTOR
    return d

def _assign(df, centers):
    lon = df["Longitude"].values
    lat = df["Latitude"].values
    dmat = _dist_matrix(lon, lat, centers)
    idx = dmat.argmin(axis=1)
    dmin = dmat[np.arange(len(df)), idx]
    return idx, dmin

def _greedy(df, k, fixed, sites, rate_out):
    chosen = fixed.copy()
    pool = [s for s in sites if s not in chosen]
    while len(chosen) < k and pool:
        best_site = None
        best_cost = None
        for cand in pool:
            cost, _, _ = _outbound(df, chosen + [cand], rate_out)
            if best_cost is None or cost < best_cost:
                best_site, best_cost = cand, cost
        chosen.append(best_site)
        pool.remove(best_site)
    return chosen

def _outbound(df, centers, rate_out):
    idx, dmin = _assign(df, centers)
    cost = (df["DemandLbs"] * dmin * rate_out).sum()
    return cost, idx, dmin

def optimize(
    df, k_vals, rate_out,
    sqft_per_lb, cost_sqft, fixed_cost,
    consider_inbound=False, inbound_rate_mile=0.0, inbound_pts=None,
    fixed_centers=None, rdc_list=None, transfer_rate_mile=0.0,
    rdc_sqft_per_lb=None, rdc_cost_per_sqft=None,
    candidate_sites=None, restrict_cand=False, candidate_costs=None
):
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    rdc_list = rdc_list or []
    candidate_costs = candidate_costs or {}

    def _cost_sqft(lon, lat):
        if restrict_cand:
            key = (round(float(lon), 6), round(float(lat), 6))
            return candidate_costs.get(key, cost_sqft)
        return cost_sqft

    best = None
    for k in k_vals:
        k_eff = max(k, len(fixed_centers))

        # choose centers
        if candidate_sites and len(candidate_sites) >= k_eff:
            centers = _greedy(df, k_eff, fixed_centers, candidate_sites, rate_out)
        else:
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(df[["Longitude", "Latitude"]])
            centers = km.cluster_centers_.tolist()
            for i, fc in enumerate(fixed_centers):
                centers[i] = fc

        idx, dmin = _assign(df, centers)
        assigned = df.copy()
        assigned["Warehouse"] = idx
        assigned["DistMi"] = dmin

        out_cost = (assigned["DemandLbs"] * dmin * rate_out).sum()

        demand_per_wh = []
        wh_cost = 0.0
        for i, (clon, clat) in enumerate(centers):
            dem = assigned.loc[assigned["Warehouse"] == i, "DemandLbs"].sum()
            demand_per_wh.append(dem)
            wh_cost += warehousing_cost(dem, sqft_per_lb, _cost_sqft(clon, clat), fixed_cost)

        # hub sets
        sdc_only = [r for r in rdc_list if r["is_sdc"]]
        rdc_only = [r for r in rdc_list if not r["is_sdc"]]
        hub_set = sdc_only or rdc_only  # prefer SDC if present

        # inbound
        in_cost = 0.0
        if consider_inbound and inbound_pts:
            if hub_set:
                total_dem = sum(demand_per_wh)
                share = 1.0 / len(hub_set)
                for slon, slat, pct in inbound_pts:
                    for hub in hub_set:
                        hx, hy = hub["coords"]
                        dist = haversine(slon, slat, hx, hy) * ROAD_FACTOR
                        in_cost += dist * total_dem * pct * share * inbound_rate_mile
            else:
                for lon, lat, pct in inbound_pts:
                    dists = np.array([haversine(lon, lat, cx, cy) * ROAD_FACTOR for cx, cy in centers])
                    in_cost += (dists * np.array(demand_per_wh) * pct * inbound_rate_mile).sum()

        # transfers
        trans_cost = 0.0
        if hub_set:
            share = 1.0 / len(hub_set)
            for hub in hub_set:
                hx, hy = hub["coords"]
                dists = np.array([haversine(hx, hy, wx, wy) * ROAD_FACTOR for wx, wy in centers])
                trans_cost += (dists * np.array(demand_per_wh) * share * transfer_rate_mile).sum()

        total_cost = out_cost + wh_cost + in_cost + trans_cost
        if best is None or total_cost < best["total_cost"]:
            best = dict(
                centers=centers,
                assigned=assigned,
                demand_per_wh=demand_per_wh,
                total_cost=total_cost,
                out_cost=out_cost,
                in_cost=in_cost,
                trans_cost=trans_cost,
                wh_cost=wh_cost,
                rdc_list=rdc_list,
            )
    return best
