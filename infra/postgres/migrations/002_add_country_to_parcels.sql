-- Migration 002 : ajout colonne country sur parcels (lien avec pesticide_use)
ALTER TABLE parcels ADD COLUMN IF NOT EXISTS country VARCHAR(150);
