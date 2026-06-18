# python-gee-ai/carbon_pricing.py
"""
GeoESG — Indonesia Carbon Pricing Module
=========================================
Calculates carbon value using Indonesian carbon market frameworks.

References:
  [1] Wijayanto et al. (2026), Indonesian J. Energy & Policy
      — AI-Driven Carbon Pricing: Geospatial Analysis for Indonesia's Energy Transition
  [2] Perpres 98/2021 — Nilai Ekonomi Karbon (NEK) / Carbon Economic Value
  [3] SRN (Sistem Registri Nasional) — National Registry System
  [4] IDXCarbon — Indonesia Carbon Exchange pricing data
  [5] SNI 7724:2011 — Faktor konversi karbon hutan tropis

Indonesian Carbon Market Context:
  - IDXCarbon launched Sep 2023, price range Rp 50,000-100,000/ton CO2e
  - Perpres 98/2021 mandates carbon pricing for power sector
  - SRN tracks carbon credits from forestry (REDD+), energy, waste
  - I-ETS (Indonesia ETS) covers coal power plants >100MW
"""


# ─── Carbon Pricing Data (2024-2026 market range) ────────────────

# IDXCarbon average prices (Rp per ton CO2e)
IDXCARBON_PRICES = {
    "forestry_redd": 65000,      # REDD+ forestry credits
    "forestry_plantation": 55000, # Plantation forestry
    "energy_renewable": 50000,    # Renewable energy credits
    "energy_efficiency": 45000,   # Energy efficiency
    "waste_management": 40000,    # Waste sector
    "default": 55000,             # Default average
}

# Forestry-specific: premium for high-integrity forest carbon
FORESTRY_PREMIUM_FACTORS = {
    "AUDIT_PASS": 1.0,       # Full price for validated
    "AUDIT_WARN": 0.7,       # Discounted for warning
    "AUDIT_FAIL": 0.0,       # No value for failed audit
}

# SNI 7724:2011: Carbon fraction for tropical forests
CARBON_FRACTION = 0.46

# CO2e conversion: C → CO2e = × 3.667 (molecular weight ratio)
C_TO_CO2E = 3.667


def calculate_carbon_value(
    estimated_carbon_mg_ha: float,
    area_ha: float,
    audit_status: str = "AUDIT_PASS",
    sector: str = "forestry_redd",
) -> dict:
    """
    Calculate carbon economic value under Indonesia's NEK framework.

    Args:
        estimated_carbon_mg_ha: Carbon stock (Mg C/ha) from satellite estimation
        area_ha: Area of the site in hectares
        audit_status: Audit result (AUDIT_PASS, AUDIT_WARN, AUDIT_FAIL)
        sector: Carbon market sector

    Returns:
        dict with carbon metrics and economic valuation
    """
    # Convert C → CO2e
    carbon_co2e_ha = estimated_carbon_mg_ha * C_TO_CO2E
    total_co2e = carbon_co2e_ha * area_ha

    # Base price from IDXCarbon market
    base_price = IDXCARBON_PRICES.get(sector, IDXCARBON_PRICES["default"])

    # Integrity premium (higher price for validated data)
    premium = FORESTRY_PREMIUM_FACTORS.get(audit_status, 0.0)
    effective_price = base_price * premium

    # Economic value
    total_value_idr = total_co2e * effective_price
    total_value_usd = total_value_idr / 15500  # Approximate IDR/USD

    return {
        "carbon_stock_mg_c_ha": round(estimated_carbon_mg_ha, 2),
        "carbon_stock_ton_co2e_ha": round(carbon_co2e_ha, 2),
        "total_co2e": round(total_co2e, 2),
        "area_ha": area_ha,
        "idxcarbon_price_rp_per_ton": base_price,
        "effective_price_rp_per_ton": round(effective_price),
        "integrity_premium": premium,
        "total_value_rp": round(total_value_idr),
        "total_value_usd": round(total_value_usd, 2),
        "sector": sector,
        "framework": "Perpres 98/2021 (NEK) + IDXCarbon",
        "references": [
            "Wijayanto et al. (2026) — AI-Driven Carbon Pricing Indonesia",
            "Perpres 98/2021 — Nilai Ekonomi Karbon",
            "IDXCarbon — Indonesia Carbon Exchange",
        ],
    }


def format_carbon_report(carbon_value: dict, site_id: str) -> str:
    """Format carbon value as readable Markdown."""
    cv = carbon_value
    status_emoji = "✅" if cv["integrity_premium"] == 1.0 else "⚠️" if cv["integrity_premium"] > 0 else "❌"

    return f"""
### 💰 Valuasi Ekonomi Karbon (Indonesia NEK)
| Parameter | Nilai |
|-----------|-------|
| **Stok Karbon** | {cv['carbon_stock_mg_c_ha']} Mg C/ha |
| **Setara CO₂** | {cv['carbon_stock_ton_co2e_ha']} ton CO₂e/ha |
| **Total CO₂e ({cv['area_ha']} ha)** | {cv['total_co2e']:,.0f} ton CO₂e |
| **Harga IDXCarbon** | Rp {cv['idxcarbon_price_rp_per_ton']:,}/ton CO₂e |
| **Premium Integritas** | {status_emoji} ×{cv['integrity_premium']} |
| **Harga Efektif** | Rp {cv['effective_price_rp_per_ton']:,}/ton CO₂e |
| **Total Nilai Karbon** | **Rp {cv['total_value_rp']:,.0f}** (~USD {cv['total_value_usd']:,.0f}) |
| **Kerangka** | {cv['framework']} |
"""