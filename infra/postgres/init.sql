-- Schéma AgriTech

CREATE TABLE IF NOT EXISTS parcels (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    crop_type   VARCHAR(50)  NOT NULL,
    area_ha     FLOAT        NOT NULL,
    latitude    FLOAT        NOT NULL,
    longitude   FLOAT        NOT NULL,
    soil_type   VARCHAR(50),
    country     VARCHAR(150),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    id              BIGSERIAL PRIMARY KEY,
    parcel_id       INT REFERENCES parcels(id),
    recorded_at     TIMESTAMP NOT NULL,
    soil_moisture   FLOAT,
    soil_temp_c     FLOAT,
    soil_ph         FLOAT,
    nitrogen_ppm    FLOAT,
    air_temp_c      FLOAT,
    humidity_pct    FLOAT,
    rainfall_mm     FLOAT,
    solar_rad_wm2   FLOAT
);

CREATE TABLE IF NOT EXISTS ndvi_observations (
    id          BIGSERIAL PRIMARY KEY,
    parcel_id   INT REFERENCES parcels(id),
    observed_at DATE NOT NULL,
    ndvi_value  FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS yield_records (
    id              BIGSERIAL PRIMARY KEY,
    parcel_id       INT REFERENCES parcels(id),
    harvest_year    INT NOT NULL,
    yield_t_per_ha  FLOAT NOT NULL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id              BIGSERIAL PRIMARY KEY,
    parcel_id       INT REFERENCES parcels(id),
    predicted_at    TIMESTAMP DEFAULT NOW(),
    model_name      VARCHAR(100),
    model_version   VARCHAR(50),
    predicted_yield FLOAT,
    irrigation_rec_mm FLOAT,
    confidence      FLOAT
);

CREATE INDEX IF NOT EXISTS idx_sensor_parcel_time ON sensor_readings(parcel_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_ndvi_parcel_date   ON ndvi_observations(parcel_id, observed_at);

-- Données FAO : utilisation de pesticides par pays et par année
CREATE TABLE IF NOT EXISTS pesticide_use (
    id                  SERIAL PRIMARY KEY,
    area                VARCHAR(150)  NOT NULL,
    year                INT           NOT NULL,
    value_tonnes        FLOAT         NOT NULL,
    yoy_growth_pct      FLOAT,
    ma5_tonnes          FLOAT,
    cagr_5y_pct         FLOAT,
    value_normalized    FLOAT,
    pct_vs_global_avg   FLOAT,
    UNIQUE (area, year)
);

CREATE INDEX IF NOT EXISTS idx_pesticide_area_year ON pesticide_use(area, year);
CREATE INDEX IF NOT EXISTS idx_pesticide_year      ON pesticide_use(year);
