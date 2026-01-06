import pandas as pd

FIRMS_API_KEY = "002d242100ffdb6932ac2f3d41135492"
FIRMS_SOURCE = "MODIS_NRT"

# Preset regions for convenience
REGIONS = {
    "usa": {"min_lon": -125, "max_lon": -66, "min_lat": 24, "max_lat": 49},
    "europe": {"min_lon": -10, "max_lon": 40, "min_lat": 35, "max_lat": 71},
    "australia": {"min_lon": 113, "max_lon": 154, "min_lat": -44, "max_lat": -10},
    "brazil": {"min_lon": -74, "max_lon": -34, "min_lat": -34, "max_lat": 5},
    "canada": {"min_lon": -141, "max_lon": -52, "min_lat": 42, "max_lat": 83},
    "world": {"min_lon": -180, "max_lon": 180, "min_lat": -90, "max_lat": 90},
}


def get_firms_hotspots(min_lon=-180, max_lon=180, min_lat=-90, max_lat=90, days=1):
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_API_KEY}/{FIRMS_SOURCE}/{bbox}/{days}"
    )

    try:
        df = pd.read_csv(url)

        if df.empty:
            print("Aucun hotspot trouvé.")
            return pd.DataFrame()

        useful_cols = [
            "latitude",
            "longitude",
            "brightness",
            "scan",
            "track",
            "acq_date",
            "acq_time",
            "satellite",
            "confidence",
            "version",
            "bright_t31",
            "frp",
            "daynight"
        ]
        df = df[[c for c in useful_cols if c in df.columns]]

        return df

    except Exception as e:
        print("Erreur FIRMS:", e)
        return pd.DataFrame()


def get_hotspots_by_region(region_name, days=1):
    if region_name not in REGIONS:
        raise ValueError(f"Region inconnue: {region_name}. Disponibles: {list(REGIONS.keys())}")
    
    region = REGIONS[region_name]
    return get_firms_hotspots(
        region["min_lon"],
        region["max_lon"],
        region["min_lat"],
        region["max_lat"],
        days
    )


def save_hotspots_to_csv(min_lon=-180, max_lon=180, min_lat=-90, max_lat=90, days=1, filename="hotspots.csv"):
    df = get_firms_hotspots(min_lon, max_lon, min_lat, max_lat, days)
    
    if df.empty:
        print("Pas de données à sauvegarder.")
        return
    
    df.to_csv(filename, index=False)
    print(f"{len(df)} hotspots enregistrés dans {filename}")


def save_region_hotspots_to_csv(region_name, days=1, filename=None):
    if filename is None:
        filename = f"{region_name}_hotspots.csv"
    
    df = get_hotspots_by_region(region_name, days)
    
    if df.empty:
        print("Pas de données à sauvegarder.")
        return
    
    df.to_csv(filename, index=False)
    print(f"{len(df)} hotspots enregistrés dans {filename}")


if __name__ == "__main__":
    save_hotspots_to_csv(days=1, filename="./world_hotspots.csv")
