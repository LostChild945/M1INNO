-- Migration 001 : ajout de la table pesticide_use (données FAO)
-- À appliquer sur un conteneur déjà démarré :
--   docker exec agritech-postgres psql -U agritech -d agritech -f /tmp/001_add_pesticide_use.sql

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
