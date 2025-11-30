# **Driver Behavior Classification & Fuel Consumption Prediction**

This repository contains the complete project materials for a two-part machine learning study on **driver behavior analysis** and **fuel consumption prediction** using trip-level telematics data. The work integrates exploratory data analysis, behavioral clustering, classification, and regression modeling, supported by formal reports and presentations.

---

## 📁 **Repository Contents**

| File                                                                      | Description                                                                                                                                                            |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Driving Behavior Classification and Fuel Consumption Prediction.ipynb** | End-to-end Jupyter Notebook containing EDA, clustering-based pseudo-labeling, aggressive driving classification, feature engineering, and fuel consumption regression. |
| **driver_behavior_route_anomaly_dataset_with_derived_features.csv**       | Trip-level telematics dataset (120,000 rows × 26 features) with speed, acceleration, braking, steering, RPM, environmental conditions, and route anomaly indicators.   |
| **Course_Project_Report.pdf**                                             | Full academic report combining aggressive driving detection and fuel prediction pipelines.                                                                             |
| **Course project proposal.pptx**                                          | Initial project proposal slides.                                                                                                                                       |
| **Course_project_presentation.pptx**                                      | Final project presentation slides summarizing methodology and results.                                                                                                 |
| **README.md**                                                             | Project overview and documentation.                                                                                                                                    |

---

## 🚀 **Project Overview**

This project addresses two core telematics analytics problems:

### **1️⃣ Aggressive Driving Classification (Unsupervised → Supervised)**

* Dataset contains **no ground-truth labels** for safe/aggressive driving.
* A **cluster-then-classify** pipeline is used:

  * K-Means (k=4) discovers natural behavioral patterns.
  * Cluster interpretation identifies the aggressive cluster.
  * An **XGBoost classifier** is trained using pseudo-labels for deployment.
* Achieved: **99.74% accuracy**, **99.03% F1-score**.

---

### **2️⃣ Fuel Consumption Prediction (Regression)**

* Extensive feature engineering including:

  * Kinetic energy, power demand
  * RPM/speed ratios
  * Smoothness, harshness, braking intensity
* Non-linear models evaluated:

  * **XGBoost**, **HistGradientBoosting**, **ExtraTrees**
* Achieved: **MAE ≈ 3.4–3.6 L**, reflecting inherently high variability in real-world fuel use.

---

## 📊 **Dataset Summary**

* **120,000 trips**, each row = 1 complete trip
* **26 features**, grouped into:

  * Telemetry (speed, acceleration, RPM)
  * Behavioral indicators (brake_usage, acceleration_variation)
  * Context (weather, road_type, traffic)
  * Route anomalies
* Zero missing values; validated for trip-level consistency.

Dataset used: *Driver Behavior and Route Anomaly Detection (DBRA24)*.

---

## 🧠 **Key Techniques Used**

* Exploratory Data Analysis (EDA)
* Unsupervised clustering (K-Means with custom centroids)
* Pseudo-labeling (self-supervised learning)
* Tree-based classification (XGBoost)
* Regression with gradient boosting models
* Physics-based and interaction-driven feature engineering
* Mutual Information–based feature selection

---

## 🎯 **Outcomes**

* A scalable ML pipeline for detecting unsafe driving from unlabeled data
* A regression model capable of predicting fuel consumption from trip conditions
* Visual and statistical insights captured in the notebook and presentation files
* A publication-ready project report consolidating all findings

---

## 📌 **How to Use This Repository**

1. Download the repository
2. Install dependencies:

   ```bash
   pip install -r requirements_minimal.txt
   ```
3. Open the Jupyter notebook:

   ```
   Driving Behavior Classification and Fuel Consumption Prediction.ipynb
   ```
4. Place the dataset (`driver_behavior_route_anomaly_dataset_with_derived_features.csv`) in the same folder
5. Run the notebook to reproduce:

   * EDA
   * Clustering
   * Classification model
   * Fuel regression models
6. Refer to the PDF report and PPTX slides for structured documentation and presentation material.

---

## 📜 **License**

Educational use only — part of the **DA204o: Data Science in Practice** course at IISc.

---

If you want, I can also generate:
✅ A shorter README
✅ A more academic README
✅ A banner/ASCII title for GitHub
Just say the word!
