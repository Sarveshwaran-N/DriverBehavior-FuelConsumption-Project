# **Driver Behavior Classification & Fuel Consumption Prediction**

*DA204o – Data Science in Practice*  
*Indian Institute of Science, Bengaluru*

---

## **👥 Team Members**

**1. Riddhi Ranjan DE** – Roll Number: 23604
* Responsible for data analysis, feature engineering, and developing the prediction model for fuel consumption
* Model performance evaluation and key insights interpretation

**2. Sarveshwaran N** – Roll Number: 23585
* Handles data analysis, preprocessing, and building the classification model for detecting anomalous driving behavior
* Model validation and result visualization

---

## **📌 Project Title**

**Driver Behavior Classification & Fuel Consumption Prediction**

---

## **🧩 Problem Statement**

Modern vehicles generate rich telemetry data—speed, acceleration, braking, steering, RPM, route deviations, and environmental conditions—yet most of this information remains underutilized. This project addresses two critical challenges in telematics analytics:

1. **Detect aggressive or unsafe driving patterns** from unlabeled telemetry data (the dataset has *no ground-truth behavior labels*)
2. **Predict per-trip fuel consumption** based on driver behavior, trip characteristics, and environmental conditions

These insights enable improved fleet safety, reduced accidents, optimized fuel efficiency, and support for intelligent transportation systems.

---

## **📂 Dataset Description**

**Dataset:** Driver Behavior & Route Anomaly Detection (DBRA24)  
**Source:** Kaggle  
**Format:** CSV  
**Size:** **120,000 trips × 26 features**  
**Download Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/datasetengineer/driver-behavior-and-route-anomaly-dataset-dbra24)

### **Feature Categories**

* **Telemetry:** `speed`, `acceleration`, `rpm`
* **Behavior Indicators:** `brake_usage`, `steering_angle`, `lane_deviation`, `acceleration_variation`
* **Environmental:** `weather`, `road_type`, `traffic_condition`
* **Trip Metadata:** `distance`, `duration`, `fuel_consumption`
* **Route Anomalies:** `anomalous_event`, `route_anomaly`
* **Derived Features:** `route_deviation_score`, `behavioral_consistency_index`

**Data Quality:** Zero missing values, confirmed trip-level granularity (1 row = 1 complete trip).

---

## **🛠️ High-Level Approach & Methods Used**

### **1️⃣ Exploratory Data Analysis (EDA)**

* Distribution analysis and outlier detection
* Driver-level profiling across behavioral metrics
* Environmental impact analysis on driving patterns
* Feature correlations and percentile analysis

### **2️⃣ Aggressive Driving Classification (Cluster-Then-Classify Pipeline)**

**Challenge:** No ground-truth labels for safe vs. aggressive driving behavior.

**Stage 1 — Unsupervised Learning**
* **K-Means clustering** (k = 4) on 6 behavioral features
* Custom domain-inspired centroids for initialization
* Composite scoring to identify the "aggressive" cluster
* Pseudo-label generation for supervised training

**Stage 2 — Supervised Learning**
* **XGBoost classifier** trained on pseudo-labels
* Enables real-time, deployment-ready inference (<1 ms latency)
* Comprehensive evaluation using accuracy, precision, recall, and F1-score

### **3️⃣ Fuel Consumption Prediction (Regression Pipeline)**

**Feature Engineering:**
* Physics-based features: kinetic energy, power demand
* Efficiency metrics: RPM/speed ratios
* Driving smoothness indices: acceleration/braking harshness
* Quadratic and interaction features for non-linear relationships

**Modeling Approach:**
* **XGBoost Regressor**
* **HistGradientBoosting Regressor**
* **ExtraTrees Regressor**
* Mutual Information–based feature selection
* RobustScaler + Median imputation preprocessing pipeline

---

## **📈 Summary of Results**

### **Aggressive Driving Classification**

* **Pipeline:** K-Means → Pseudo-labels → XGBoost
* **Accuracy:** **99.74%**
* **Precision:** **98.90%**
* **Recall:** **99.17%**
* **F1-Score:** **99.03%**
* Strong cluster separation verified via PCA visualization

### **Fuel Consumption Prediction**

* **Best Models:** XGBoost, HistGradientBoosting
* **Mean Absolute Error (MAE):** **~3.4–3.6 L**
* **R² Score:** Moderate (due to inherent high variability in real-world fuel consumption)
* **Key Predictors:** Distance, duration, acceleration patterns, RPM efficiency, environmental conditions

---

## **📁 Repository Contents**

**1. Driving Behavior Classification and Fuel Consumption Prediction.ipynb**
* End-to-end Jupyter Notebook containing EDA, clustering-based pseudo-labeling, aggressive driving classification, feature engineering, and fuel consumption regression

**2. driver_behavior_route_anomaly_dataset_with_derived_features.csv**
* Trip-level telematics dataset (120,000 rows × 26 features) with speed, acceleration, braking, steering, RPM, environmental conditions, and route anomaly indicators

**3. README.md**
* Project overview and comprehensive documentation

**4. LICENSE**
* Project license information

---

## **🔧 Technologies & Libraries Used**

* **Python 3.x**
* **Data Analysis:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Machine Learning:** scikit-learn, XGBoost
* **Clustering:** K-Means
* **Preprocessing:** RobustScaler, StandardScaler
* **Model Evaluation:** classification_report, confusion_matrix, mean_absolute_error

---

## **📌 How to Use This Repository**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sarveshwaran-N/DriverBehavior-FuelConsumption-Project.git
   cd DriverBehavior-FuelConsumption-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_minimal.txt
   ```

3. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook "Driving Behavior Classification and Fuel Consumption Prediction.ipynb"
   ```

4. **Run the notebook** to reproduce:
   * Exploratory Data Analysis
   * K-Means clustering and pseudo-labeling
   * XGBoost classification model
   * Feature engineering for fuel prediction
   * Regression models and evaluation

---

## **🎯 Key Insights & Contributions**

* Successfully developed a **self-supervised learning pipeline** for aggressive driving detection without labeled data
* Achieved **near-perfect classification accuracy** (99.74%) using cluster-based pseudo-labeling
* Created **physics-informed features** that improved fuel consumption prediction
* Demonstrated the practical application of **unsupervised learning** followed by **supervised fine-tuning**
* Provided actionable insights for **fleet management**, **driver safety programs**, and **fuel optimization**

---

## **📜 License**

Educational use only — part of the **DA204o: Data Science in Practice** course at IISc.

---



