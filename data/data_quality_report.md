# Data Quality & Preprocessing Audit Report: Indian Food Delivery Logistics Dataset

**Project:** DebugLeaf — QML-01 (Delay-Risk-Aware Fleet Dispatch)  
**Dataset Source:** Indian Food Delivery Dataset ([Kaggle: `gauravmalik26/food-delivery-dataset`](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset))  
**Target Application:** Newton Second-Order GBDT & Decision-Focused Variational Quantum Circuit (VQC)  
**Date:** October 2026  
**Auditor:** Senior ML Data Engineer (Indian Logistics & Hybrid Quantum ML Specialist)  

---

## 1. Executive Summary

This report establishes the data engineering foundation, quality audit, feature engineering, and leakage-safe partitioning for the Indian Food Delivery Dataset (Gaurav Malik, Kaggle). The labeled `train.csv` contains **45,593 rows** across **20 raw columns**, capturing real-world urban delivery operations across **22 Indian metropolitan and Tier-1/Tier-2 logistics hubs** (including Jaipur, Bangalore, Hyderabad, Mumbai, Chennai, Pune, Delhi, Kolkata, Kochi, and Vadodara).

Our pipeline transforms this raw operational feed into two model-ready feature views:
1. **Newton GBDT View (`indian_delivery_gbdt.parquet`):** A 63-column tabular feature matrix capturing spatial, temporal, courier, vehicle, traffic, and meteorological dimensions with zero-leakage training imputations.
2. **Variational Quantum Circuit View (`indian_delivery_vqc.parquet`):** A compact, numerically conditioned 2-qubit (and 4-qubit extended) representation scaled strictly to $[0, 1]$ for parameterized $R_y(\pi x)$ angle rotation circuits.

All preprocessing parameters (imputation medians, modes, and MinMax scaling bounds) are fitted **strictly on the training partition**. Partitioning is executed via a courier-isolated **GroupShuffleSplit** on `Delivery_person_ID` (70% Train, 15% Validation, 15% Test), guaranteeing **zero courier identity leakage** while maintaining full 22-city geographical representation across all three partitions.

---

## 2. Raw Dataset Profiling & Initial Audit

### 2.1 Structural Overview
- **Raw Dimensions:** 45,593 rows $\times$ 20 columns
- **Exact Duplicate Rows:** 0
- **Unique Delivery Order IDs:** 45,593 (100% unique primary keys)
- **Target Variable:** `Time_taken(min)` formatted as strings (e.g., `(min) 24`)
- **Target Distribution (Extracted Numeric Minutes):**
  - Minimum: **10.0 minutes**
  - 25th Percentile: **19.0 minutes**
  - Median: **26.0 minutes**
  - Mean: **26.29 minutes** (Std: 9.38 minutes)
  - 75th Percentile: **32.0 minutes**
  - Maximum: **54.0 minutes**
  - Nulls: **0** (100% valid target labels)

### 2.2 Whitespace & Missingness Markers
Raw data inspection revealed extensive string pollution:
- Every text column contained leading and trailing whitespace (e.g., `'0x4607 '`, `'High '`, `'conditions Sunny '`, `'(min) 24'`).
- Missing values were logged as diverse text markers: `'NaN'`, `'nan'`, `'conditions NaN'`, `''`, and `'?'`.
- **Systematic Missingness Audit:**
  | Column | Missing Count | Missing Percentage | Imputation Strategy (Fitted on Train) |
  | :--- | :---: | :---: | :--- |
  | `Delivery_person_Age` | 1,854 | 4.07% | Median (30.0 years) |
  | `Delivery_person_Ratings` | 1,908 | 4.18% | Median (4.7 / 5.0) |
  | `Time_Orderd` | 1,731 | 3.80% | Imputed prep time (10.0 min) + Missing Flag |
  | `City` (Urbanization Level) | 1,200 | 2.63% | Mode (`Metropolitian`) |
  | `multiple_deliveries` | 993 | 2.18% | Median (1.0 concurrent order) |
  | `Weatherconditions` | 616 | 1.35% | Mode (`Fog`) |
  | `Road_traffic_density` | 601 | 1.32% | Mode (`Low` / 0.0) |
  | `Festival` | 228 | 0.50% | Mode (`No` / 0) |

---

## 3. Critical Data Anomalies & Engineering Resolutions

### 3.1 Geographic Coordinate Corruption: Null Island & Inversions
India lies between approximately $6.0^\circ\text{N} - 38.0^\circ\text{N}$ Latitude and $68.0^\circ\text{E} - 98.0^\circ\text{E}$ Longitude. Audit of the 4 coordinate columns revealed severe corruption:
1. **The "Null Island" Artifact (3,640 rows, 7.98%):**
   - In 3,640 records, `Restaurant_latitude` and `Restaurant_longitude` were recorded as `(0.0000, 0.0000)`.
   - The corresponding `Delivery_location_latitude` and `longitude` were small offsets (e.g., $0.063^\circ, 0.063^\circ$), located in the Gulf of Guinea off the coast of West Africa.
   - These represent uncalibrated relative coordinates from testing devices or telemetry initialization failures in cities such as Vadodara (1,590 rows) and Ranchi (665 rows).
2. **Negative Latitude/Longitude Inversion (431 rows, 0.95%):**
   - 431 records recorded negative latitudes (e.g., $-22.7450^\circ$ instead of $+22.7450^\circ$ in Indore) or negative longitudes ($-88.3662^\circ$ instead of $+88.3662^\circ$ in Kolkata).
   - If evaluated uncorrected, the Haversine distance between $-22.7^\circ$ and $+22.7^\circ$ evaluates to over **19,000 kilometers**, creating catastrophic gradient blowups in regression and VQC optimizers.
   - If `abs()` is applied, the coordinates match the true city cluster ($9.74\text{ km}$ mean distance). However, **per operating constraints, we do not silently alter corrupted telemetry**.

**Engineering Resolution:**
- In `indian_delivery_clean.parquet`, all 45,593 records are preserved with explicit diagnostic flags: `is_zero_coords`, `is_negative_coords`, `is_suspicious_coords`, and `is_model_eligible`.
- The **4,071 anomalous coordinate rows (8.93%)** are cleanly isolated and excluded from the model training views (`indian_delivery_gbdt.parquet` and `indian_delivery_vqc.parquet`).
- Across the **41,522 model-eligible rows**, Haversine distance is strictly physical for Indian urban logistics:
  - Minimum: **1.47 km**
  - 25th Percentile: **4.66 km**
  - Median: **9.19 km**
  - Mean: **9.72 km**
  - 75th Percentile: **13.68 km**
  - Maximum: **20.97 km** (strictly within legitimate hyper-local delivery radiuses)

### 3.2 Midnight Crossing Resolution in Prep Times
`Time_Orderd` and `Time_Order_picked` record times in `HH:MM:SS` format.
- In **831 orders**, the order was placed just before midnight (e.g., `23:55:00`) and the courier picked it up just after midnight (e.g., `00:10:00`).
- A naive subtraction ($\text{pickup} - \text{order}$) yields negative durations (e.g., $-1,425\text{ minutes}$).
- **Resolution:** By applying a modulo day offset:
  $$\Delta t = (\text{pickup\_min} - \text{order\_min}) \pmod{1440}$$
  every single one of the 831 midnight crossings resolved to exact discrete operational kitchen prep times of **5, 10, or 15 minutes**, perfectly matching the daytime prep time distribution!

### 3.3 Courier Age & Rating Sentinel Values
- 38 rows contained `Delivery_person_Age = 15` and `Delivery_person_Ratings = 1.0`.
- 53 rows contained `Delivery_person_Age = 50` and `Delivery_person_Ratings = 6.0` (on a 1–5 scale).
- These represent QA test probes / system sentinels rather than genuine courier demographics. They are preserved in the clean dataset and imputed via training medians (Age: 30, Rating: 4.7) in the model view.

---

## 4. Indian Logistics Feature Engineering

Features were engineered strictly under an **anti-leakage constraint**: only information available at the moment of delivery dispatch (after restaurant pickup) is permitted as inputs.

| Feature Name | Type | Source & Engineering Logic | Logistics Rationale |
| :--- | :---: | :--- | :--- |
| `haversine_distance_km` | Continuous | Great-circle distance between restaurant and customer ($R=6371.0\text{ km}$). | Primary physical driver of travel duration. |
| `departure_hour` | Continuous | $H + M / 60.0$ extracted from `Time_Order_picked` ($0.0 - 23.92$). | Captures diurnal traffic and dispatch cycle. |
| `is_lunch_rush` | Binary | 1 if $12.0 \le \text{departure\_hour} < 15.0$ else 0. | Peak lunch ordering in Indian commercial districts. |
| `is_dinner_rush` | Binary | 1 if $18.0 \le \text{departure\_hour} < 22.0$ else 0. | Major dinner peak coinciding with evening traffic congestion. |
| `is_rush_hour` | Binary | `is_lunch_rush` $\lor$ `is_dinner_rush`. | General urban peak travel indicator. |
| `day_of_week` | Integer | $0\text{ (Monday)} - 6\text{ (Sunday)}$ from `Order_Date`. | Day-level delivery demand variations. |
| `is_weekend` | Binary | 1 if Saturday or Sunday else 0. | Weekend vs weekday traffic patterns. |
| `traffic_density_ordinal_imp` | Ordinal | Clean mapping: Low (0), Medium (1), High (2), Jam (3). | Primary real-time road condition constraint. |
| `is_adverse_weather` | Binary | 1 for Fog, Stormy, Sandstorms, Windy; 0 for Sunny, Cloudy. | Monsoons, storms, and winter fog severely slow two-wheelers. |
| `vehicle_condition` | Ordinal | $0\text{ (Poor)}, 1\text{ (Fair)}, 2\text{ (Good)}$. | Influences vehicle reliability and cruising speed. |
| `multiple_deliveries_imp` | Integer | Clean integer count of concurrent orders ($0, 1, 2, 3$). | Multi-drop workload increases dispatch detour times. |
| `is_festival` | Binary | 1 if `Festival == 'Yes'` else 0. | Indian festivals (Diwali, Holi, Eid) drastically alter road density. |
| `city_code` | Categorical | Regex `^([A-Z]+?)RES` from `Delivery_person_ID` (22 cities). | Captures city-specific infrastructure and geography. |
| `city_type_clean_imp` | Categorical | Metropolitian, Urban, Semi-Urban. | Urban density classification. |
| `order_to_pickup_min_imp` | Continuous | Kitchen prep duration ($5, 10, 15\text{ min}$). | In-store kitchen bottleneck before rider departure. |
| `order_to_pickup_missing_flag`| Binary | 1 if `Time_Orderd` was missing else 0. | Preserves missingness pattern without data loss. |

---

## 5. Model Feature Views Specification

### 5.1 View A: Newton GBDT (`indian_delivery_gbdt.parquet`)
- **Sample Count:** 41,522 rows
- **Feature Dimension:** 63 columns
- **Composition:**
  - 10 continuous / ordinal numeric features
  - 7 binary indicators
  - 46 one-hot encoded categorical columns (weather, order type, vehicle type, urbanization type, and 22 city indicators)
- **Zero-Leakage Assurance:** All missing values are imputed strictly using training partition medians and modes.

### 5.2 View B: Variational Quantum Circuit (`indian_delivery_vqc.parquet`)
Designed for parameterized quantum circuits with $R_y(\pi x)$ angle embedding on 2 to 4 qubits:

| Qubit Wire | Feature Name | Formula / Scaling (Fitted on Train) | Physical Interpretation |
| :---: | :--- | :--- | :--- |
| **$q_0$** | `vqc_x0_distance` | $\frac{d - d_{\min}}{d_{\max} - d_{\min}} \in [0.0, 1.0]$ | Normalized route travel distance. |
| **$q_1$** | `vqc_x1_congestion` | $0.60 \cdot \frac{\text{traffic}}{3} + 0.25 \cdot \text{rush} + 0.15 \cdot \text{adverse} \in [0.0, 1.0]$ | Urban congestion & environmental friction. |
| **$q_2$** *(4Q)* | `vqc_x2_multidelivery` | $\frac{\text{multiple\_deliveries}}{3.0} \in [0.0, 1.0]$ | Delivery partner multi-drop workload. |
| **$q_3$** *(4Q)* | `vqc_x3_preptime` | $\frac{\text{prep\_min} - 5.0}{10.0} \in [0.0, 1.0]$ | Restaurant kitchen dispatch delay. |

- **Target Variables Provided:**
  1. `delivery_duration_min`: Ground-truth duration in minutes ($10.0 - 54.0$).
  2. `target_normalized`: Continuous $[0, 1]$ normalized target for regression VQC:
     $$y_{\text{norm}} = \frac{y - 10.0}{44.0}$$
  3. `is_sla_breach_synthetic`: Defined against a standard 30-minute delivery commitment ($y > 30.0$). **Explicitly labeled as a synthetic scenario benchmark.**

---

## 6. Leakage-Safe Courier-Group Partitioning Strategy

### 6.1 Evaluation of Partitioning Candidates
1. **Random Row Split (Rejected):** Couriers and locations leak across splits; model memorizes individual driver speeds rather than learning generalized logistics dynamics.
2. **Pure Temporal Cutoff (Rejected):** 10 out of 22 cities (Agra, Ludhiana, Bhopal, Dehradun, etc.) were active only in specific calendar months. A temporal split leaves test sets with zero representation for half the country.
3. **Courier-Group Shuffle Split (Adopted):** `GroupShuffleSplit` on `Delivery_person_ID` (Random seed `42`).

### 6.2 Partitioning Verification & Audit Proof
- **Train Partition:** **29,592 rows (71.27%)** across **819 couriers** (Mean target: $26.31\text{ min}$)
- **Validation Partition:** **5,989 rows (14.42%)** across **175 couriers** (Mean target: $26.15\text{ min}$)
- **Test Partition:** **5,941 rows (14.31%)** across **176 couriers** (Mean target: $26.50\text{ min}$)
- **Excluded Telemetry:** **4,071 rows (8.93%)**

```
Courier Overlap Verification:
  Train ∩ Validation: 0 couriers
  Train ∩ Test:       0 couriers
  Validation ∩ Test:  0 couriers

Geographical Integrity:
  Train Cities:       22 / 22 cities represented
  Validation Cities:  22 / 22 cities represented
  Test Cities:        22 / 22 cities represented
```

---

## 7. Deliverables & Cryptographic Manifest

All artifacts are persisted under `data/`:

| Artifact File | Format | Row Count | Column Count | File Size (Bytes) | SHA-256 Digest |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `indian_delivery_clean.parquet` | Parquet | 45,593 | 64 | 1,939,424 | `8dfd38a0a9d94fe969ee73ae2857053e1a0df331002df35978a3cba6a7f347eb` |
| `indian_delivery_gbdt.parquet` | Parquet | 41,522 | 63 | 928,787 | `47c20c0839e948c26da2db2c96c567c9c2a68894451000f074d440ad81515949` |
| `indian_delivery_vqc.parquet` | Parquet | 41,522 | 12 | 618,330 | `fa262e3d36005c87a2a0c441b1d7d6f5f3e589a19c7a26ad0b0ef854fb606145` |
| `preprocessing_artifacts.json` | JSON | N/A | N/A | 892 | Preprocessing medians, modes, MinMax boundaries |
| `split_manifest.json` | JSON | N/A | N/A | 1,845 | Split parameters, courier counts, and hashes |
| `feature_schema.json` | JSON | N/A | N/A | 3,124 | Complete feature dictionary and VQC mapping |
| `tools/preprocess_indian_delivery.py` | Python | 538 lines | N/A | 26,278 | Reproducible end-to-end preprocessing script |

---

## 8. Summary Conclusion & Next Steps

The Indian Food Delivery Dataset is now fully audited, cleansed of unphysical telemetry, engineered with domain-specific logistics features, and partitioned without leakage. It provides two strictly calibrated feature views:
1. A 63-column tabular view ready for training our **Second-Order Newton GBDT**.
2. A bounded $[0, 1]^2$ and $[0, 1]^4$ feature view ready for angle encoding in our **Decision-Focused Variational Quantum Circuit**.

The pipeline is 100% reproducible via `python tools/preprocess_indian_delivery.py`.
