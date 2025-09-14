## Overview

Graduate research project (EPFL, Fall 2023) modeling the heating & electricity system of the EPFL campus and evaluating decarbonization pathways under technology, price, and policy scenarios. The workflow couples building thermal demand modeling, data clustering, NLP heat-recovery optimization, two-stage heat-pump modeling with data reconciliation, and campus-scale MILP for multi-objective and sensitivity analysis. 

## Core question: 

How can EPFL minimize cost (TOTEX) and CO₂ emissions while ensuring comfort and feasibility across realistic operating conditions?

## Key Results 

* Baseline Pareto (cost–emissions): a representative trade-off is ~2,000 tCO₂/yr at ~11.5 MCHF/yr TOTEX.
* High gas price (0.29 vs 0.03 CHF/kWh): optimal trade-off ~1,500 tCO₂/yr at ~14 MCHF/yr (+22% TOTEX, –25% CO₂).
* High electricity price (0.49 vs 0.0916 CHF/kWh): optimal ~2,600 tCO₂/yr at ~18.1 MCHF/yr (+57% TOTEX, +30% CO₂).
* Lower grid EF (30 vs 75.3 gCO₂/kWh): optimal ~1,000 tCO₂/yr at ~8.5 MCHF/yr (–26% TOTEX, –50% CO₂).
* No-boiler (gas ban): ~–38% emissions vs baseline, but +62% TOTEX (~18.6 MCHF/yr).

Electrification becomes far more attractive as the grid’s emissions factor drops; gas price spikes reduce optimal emissions at moderate cost increases; electricity price spikes push the system back toward gas and raise both cost and emissions.


## Methods (pipeline)

### 1) Building thermal demand modeling  
Hourly heating demand per building was computed from steady-state energy balances with the following contributions:  
- Envelope and ventilation losses (k_th)  
- Solar gains (k_sun)  
- Internal gains from people and appliances (assumed 80% of appliance electricity converts to heat)  
- Occupancy schedules and heating cut-on logic  

Unknown parameters were solved via **Newton–Raphson calibration** to annual demands. A comfort band and cut-off temperatures were applied to determine heating needs.

---

### 2) Typical-periods (clustering)  
To reduce temporal complexity, **k-means clustering** was applied on external temperature and irradiance.  
- Only hours with heating demand were retained.  
- Cluster quality was assessed with elbow and silhouette methods, yielding **n = 6 clusters**.  
- An additional **extreme cold hour** was included to capture peak demand sizing.

---

### 3) Heat recovery (NLP, AMPL)  
Nonlinear programs in AMPL evaluated three scenarios in addition to a heat pump baseline:  
1. **Data center recovery**: use waste heat to preheat lake water (improves HP efficiency and cools datacenter).  
2. **Ventilation recovery**: recover heat from exhaust air to preheat building inlet air.  
3. **Ventilation + auxiliary HP**: combine ventilation recovery with an external air–air HP.  

For each case, **heat exchanger areas** were sized using ΔT_LM, **CAPEX/OPEX** were computed, and scenarios were compared by **TOTEX**.

---

### 4) Two-stage heat pump modeling & data reconciliation  
- Built and validated two-stage HP models for **R-290 (propane)** and **propylene**, using **degree-of-freedom analysis**.  
- Applied **Belsim VALI data reconciliation** to reduce measurement errors and ensure thermodynamic consistency.  
- Fitted a quadratic regression for the **Carnot factor vs ambient temperature**.  
- Estimated unit costs by sizing heat exchangers and compressors at worst-case design points.

---

### 5) Campus MILP & multi-objective analysis (AMPL)  
The entire EPFL energy system was modeled as a MILP including:  
- Two heating loops (65 °C medium-T, 50 °C low-T)  
- Technologies: gas boilers, single- and two-stage HPs, PV, solar thermal collectors, SOFC/CHP, and electricity imports  

The model optimized across five objectives:  
- **CAPEX**  
- **OPEX**  
- **TOTEX**  
- **Emissions**  
- **TOTEX + CO₂ tax**  

Pareto frontiers were generated for cost–emissions trade-offs, and **sensitivity analyses** were performed for gas price, electricity price, and grid emission factor.

