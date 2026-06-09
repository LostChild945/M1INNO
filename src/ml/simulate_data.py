"""
Simulation de données agricoles réalistes pour l'entraînement ML.

Génère pour 20 pays (top utilisateurs de pesticides) :
  - 3 parcelles par pays = 60 parcelles
  - Relevés capteurs annuels (sol + météo) pour 2010-2016
  - Rendements corrélés aux capteurs et à l'utilisation de pesticides

Lancement :
  python3 src/ml/simulate_data.py
  ou depuis le conteneur Airflow :
  python3 /opt/airflow/src/ml/simulate_data.py
"""
import os
import random
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://agritech:agritech_secret@localhost:5432/agritech",
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# lat, lon, avg_temp_c, avg_rain_mm
COUNTRY_PROFILES = {
    "China, mainland":           (35.9, 104.2, 12,  600),
    "United States of America":  (37.1, -95.7, 13,  750),
    "Brazil":                    (-14.2,-51.9, 25, 1500),
    "Argentina":                 (-38.4,-63.6, 14,  600),
    "France":                    (46.2,   2.2, 11,  650),
    "Italy":                     (41.9,  12.6, 13,  750),
    "Japan":                     (36.2, 138.3, 13, 1700),
    "Colombia":                  (4.6,  -74.1, 24, 3000),
    "India":                     (20.6,  78.9, 25, 1100),
    "Canada":                    (56.1,-106.3,  3,  550),
    "Ukraine":                   (48.4,  31.2,  8,  550),
    "Mexico":                    (23.6,-102.6, 21,  750),
    "Malaysia":                  (4.2,  109.5, 27, 2500),
    "Spain":                     (40.5,  -3.7, 14,  650),
    "Thailand":                  (15.9, 100.9, 28, 1400),
    "Germany":                   (51.2,  10.5,  9,  620),
    "Australia":                 (-25.3, 133.8, 22,  450),
    "Turkey":                    (38.9,  35.2, 12,  600),
    "United Kingdom":            (55.4,  -3.4, 10,  900),
    "Russian Federation":        (61.5, 105.3,  1,  500),
}

CROPS = {
    "wheat":     {"base": 3.5, "opt_temp": 15, "opt_rain": 500},
    "corn":      {"base": 5.5, "opt_temp": 22, "opt_rain": 650},
    "rice":      {"base": 4.2, "opt_temp": 28, "opt_rain": 1200},
    "soybean":   {"base": 2.8, "opt_temp": 24, "opt_rain": 700},
    "sunflower": {"base": 2.0, "opt_temp": 20, "opt_rain": 450},
}

YEARS = list(range(2010, 2017))
PARCELS_PER_COUNTRY = 3


def simulate_yield(crop, pesticide_norm, soil_ph, soil_moisture,
                   soil_nitrogen, air_temp, rainfall):
    cfg = CROPS[crop]
    base = cfg["base"]

    pest_effect  = base * 0.20 * np.log1p(pesticide_norm * 4)
    ph_dev       = soil_ph - 6.5
    soil_effect  = base * (-0.06 * ph_dev**2 + 0.04 * soil_moisture + 0.0008 * soil_nitrogen)
    temp_dev     = air_temp - cfg["opt_temp"]
    rain_dev     = rainfall - cfg["opt_rain"]
    climate_effect = base * (-0.008 * temp_dev**2 - 0.000008 * rain_dev**2 + 0.05)
    noise        = np.random.normal(0, 0.12 * base)

    return round(max(0.1, base + pest_effect + soil_effect + climate_effect + noise), 3)


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Vérifie si la simulation a déjà été lancée
    cur.execute("SELECT COUNT(*) FROM parcels")
    if cur.fetchone()[0] > 0:
        print("[simulate] Données déjà présentes — skip.")
        cur.close()
        conn.close()
        return

    # ── Parcelles ──────────────────────────────────────────────────
    parcels = []
    for country, (lat, lon, _, _) in COUNTRY_PROFILES.items():
        for i in range(PARCELS_PER_COUNTRY):
            crop = random.choice(list(CROPS.keys()))
            parcels.append((
                f"{country} — parcelle {i + 1}",
                crop,
                round(random.uniform(50, 400), 1),
                round(lat + random.uniform(-2, 2), 4),
                round(lon + random.uniform(-2, 2), 4),
                random.choice(["loam", "clay", "sandy loam", "silt loam"]),
                country,
            ))

    execute_values(
        cur,
        """INSERT INTO parcels (name, crop_type, area_ha, latitude, longitude, soil_type, country)
           VALUES %s RETURNING id""",
        parcels,
    )
    parcel_ids = [row[0] for row in cur.fetchall()]
    conn.commit()
    print(f"[simulate] {len(parcel_ids)} parcelles créées")

    # ── Capteurs & rendements ───────────────────────────────────────
    sensor_rows = []
    yield_rows  = []

    cur.execute("""
        SELECT area, year, value_normalized
        FROM pesticide_use
        WHERE year BETWEEN 2010 AND 2016
    """)
    pest_lookup = {(r[0], r[1]): r[2] for r in cur.fetchall()}

    for idx, (parcel_id, (country, (_, _, avg_temp, avg_rain), crop)) in enumerate(
        zip(parcel_ids, [
            (c, (COUNTRY_PROFILES[c], crop))
            for c in COUNTRY_PROFILES
            for crop in [p[1] for p in parcels if p[6] == c]
        ])
    ):
        for year in YEARS:
            pest_norm = pest_lookup.get((country, year), 0.3)

            soil_ph       = round(random.gauss(6.5, 0.4), 2)
            soil_moisture = round(random.gauss(0.45, 0.10), 3)
            soil_nitrogen = round(random.gauss(120, 30), 1)
            air_temp      = round(avg_temp + random.gauss(0, 1.5), 1)
            rainfall      = round(avg_rain * random.gauss(1, 0.15), 0)
            humidity      = round(random.gauss(65, 10), 1)
            solar_rad     = round(random.gauss(180, 30), 1)

            sensor_rows.append((
                parcel_id,
                datetime(year, 7, 1),   # relevé annuel en milieu de saison
                soil_moisture,
                air_temp,
                soil_ph,
                soil_nitrogen,
                air_temp,
                humidity,
                rainfall,
                solar_rad,
            ))

            yld = simulate_yield(
                crop, pest_norm, soil_ph, soil_moisture,
                soil_nitrogen, air_temp, rainfall,
            )
            yield_rows.append((parcel_id, year, yld))

    execute_values(
        cur,
        """INSERT INTO sensor_readings
           (parcel_id, recorded_at, soil_moisture, soil_temp_c, soil_ph,
            nitrogen_ppm, air_temp_c, humidity_pct, rainfall_mm, solar_rad_wm2)
           VALUES %s""",
        sensor_rows,
    )
    execute_values(
        cur,
        "INSERT INTO yield_records (parcel_id, harvest_year, yield_t_per_ha) VALUES %s",
        yield_rows,
    )
    conn.commit()
    print(f"[simulate] {len(sensor_rows)} relevés capteurs, {len(yield_rows)} rendements créés")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
