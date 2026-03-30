"""
Generate a synthetic car market dataset for the analytics dashboard.
Illustrative data for portfolio / coursework — not real sales figures.
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "car_market_data.csv")

RNG = np.random.default_rng(42)
N = 2400

BRANDS = [
    "Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Hyundai", "Kia",
    "Volkswagen", "Nissan", "Mazda", "Subaru", "Chevrolet", "Tesla", "Lexus",
    "Porsche", "Volvo", "Jeep", "Ram", "Genesis",
    "Mitsubishi", "Mini", "Suzuki", "Cadillac", "Buick", "Alfa Romeo", "Ferrari", "Aston Martin",
]

# Start with even weights and boost commonly seen brands so the generated
# dataset better reflects a mix of mainstream and premium marques.
BRAND_WEIGHTS = np.ones(len(BRANDS), dtype=float)
boost = {
    "Toyota": 1.8,
    "Honda": 1.6,
    "Ford": 1.5,
    "Tesla": 1.3,
    "BMW": 1.1,
}
for k, m in boost.items():
    if k in BRANDS:
        BRAND_WEIGHTS[BRANDS.index(k)] *= m
BRAND_WEIGHTS = BRAND_WEIGHTS / BRAND_WEIGHTS.sum()

BRAND_MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot"],
    "Ford": ["F-150", "Mustang", "Explorer", "Escape"],
    "BMW": ["3 Series", "5 Series", "X3", "X5"],
    "Mercedes": ["C-Class", "E-Class", "GLC", "GLE"],
    "Audi": ["A4", "A6", "Q5", "Q7"],
    "Hyundai": ["Elantra", "Tucson", "Santa Fe", "Sonata"],
    "Kia": ["Sportage", "Telluride", "Sorento", "K5"],
    "Volkswagen": ["Golf", "Jetta", "Tiguan", "Passat"],
    "Nissan": ["Altima", "Sentra", "Rogue", "Murano"],
    "Mazda": ["Mazda3", "Mazda6", "CX-5", "CX-50"],
    "Subaru": ["Forester", "Outback", "Crosstrek", "WRX"],
    "Chevrolet": ["Silverado", "Equinox", "Malibu", "Tahoe"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
    "Lexus": ["ES", "RX", "NX", "IS"],
    "Porsche": ["911", "Cayenne", "Macan", "Panamera"],
    "Volvo": ["XC60", "XC90", "S60", "S90"],
    "Jeep": ["Wrangler", "Grand Cherokee", "Compass", "Cherokee"],
    "Ram": ["1500", "2500", "3500", "ProMaster"],
    "Genesis": ["G70", "G80", "GV70", "GV80"],
    "Mitsubishi": ["Outlander", "Eclipse Cross", "Lancer"],
    "Mini": ["Cooper", "Countryman"],
    "Suzuki": ["Swift", "Vitara"],
    "Cadillac": ["XT5", "Escalade"],
    "Buick": ["Enclave", "Encore"],
    "Alfa Romeo": ["Giulia", "Stelvio"],
    "Ferrari": ["488 GTB", "Portofino"],
    "Aston Martin": ["DB11", "Vantage"],
}

FUELS = ["Petrol", "Diesel", "Hybrid", "Electric"]
FUEL_P = [0.52, 0.18, 0.18, 0.12]

TRANSMISSIONS = ["Manual", "Automatic", "CVT"]
TRANS_P = [0.12, 0.68, 0.20]

WHEELS = ["FWD", "RWD", "AWD"]
WHEELS_P = [0.48, 0.22, 0.30]

CATEGORIES = ["Sedan", "SUV", "Coupe", "Hatchback", "Truck", "Van"]
CAT_P = [0.28, 0.38, 0.10, 0.12, 0.09, 0.03]

DOORS = [2, 3, 4, 5]
DOORS_P = [0.08, 0.02, 0.62, 0.28]

BRAND_SCORE = {b: 420 + int(RNG.integers(-40, 80)) for b in BRANDS}


def main():
    rows = []
    for i in range(1, N + 1):
        brand = str(RNG.choice(BRANDS, p=BRAND_WEIGHTS))
        model = str(RNG.choice(BRAND_MODELS[brand]))
        year = int(RNG.integers(2015, 2025))
        cylinders = int(RNG.choice([4, 4, 4, 6, 6, 8, 12], p=[0.52, 0, 0, 0.28, 0, 0.15, 0.05]))
        fuel = str(RNG.choice(FUELS, p=FUEL_P))
        transmission = str(RNG.choice(TRANSMISSIONS, p=TRANS_P))
        driven_wheels = str(RNG.choice(WHEELS, p=WHEELS_P))
        doors = int(RNG.choice(DOORS, p=DOORS_P))
        market_category = str(RNG.choice(CATEGORIES, p=CAT_P))

        hp_base = {4: 155, 6: 285, 8: 400, 12: 620}.get(cylinders, 200)
        engine_hp = int(np.clip(hp_base + RNG.normal(0, 35), 90, 720))

        price_base = 18000 + (year - 2015) * 2200 + engine_hp * 45
        brand_mult = 0.85 + (BRAND_SCORE[brand] - 400) / 900
        price = int(np.clip(price_base * brand_mult + RNG.normal(0, 3500), 12000, 185000))

        fuel_bonus = {"Petrol": 0, "Diesel": 12, "Hybrid": 45, "Electric": 55}
        cat_bonus = {"SUV": 25, "Sedan": 10, "Coupe": 5, "Hatchback": 8, "Truck": 30, "Van": -15}
        trans_bonus = {"CVT": 8, "Automatic": 5, "Manual": -5}
        wheel_bonus = {"AWD": 18, "RWD": 10, "FWD": 0}

        score = (
            BRAND_SCORE[brand]
            + (year - 2015) * 14
            + engine_hp * 0.22
            + fuel_bonus[fuel]
            + cat_bonus[market_category]
            + trans_bonus[transmission]
            + wheel_bonus[driven_wheels]
            + float(RNG.normal(0, 48))
        )
        popularity_score = int(np.clip(round(score), 100, 995))

        rows.append(
            {
                "car_id": i,
                "brand": brand,
                "model": model,
                "year": year,
                "engine_hp": engine_hp,
                "engine_cylinders": cylinders,
                "fuel_type": fuel,
                "transmission": transmission,
                "driven_wheels": driven_wheels,
                "doors": doors,
                "market_category": market_category,
                "price": price,
                "popularity_score": popularity_score,
            }
        )

    df = pd.DataFrame(rows)
    q1, q2 = df["popularity_score"].quantile([0.34, 0.67]).tolist()

    def level(s):
        if s <= q1:
            return "Low"
        if s <= q2:
            return "Medium"
        return "High"

    df["popularity_level"] = df["popularity_score"].apply(level)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(df)} rows). Level cutoffs: Low <= {q1:.0f}, Medium <= {q2:.0f}.")


if __name__ == "__main__":
    main()
