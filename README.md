
# 🛍️ Customer Segmentation using K-Means Clustering
#"This project was completed as part of my training under @Arnita-Infotech".
 arnitainfotech@gmail.com

---

## 📌 Project Overview

This project implements an advanced **K-Means Clustering** algorithm to segment retail customers based on:

- Annual Income
- Spending Score

The system automatically detects the optimal number of clusters using **Silhouette Analysis**, allowing businesses to identify meaningful customer groups and develop targeted marketing strategies.

---

## ⚙️ System Architecture & Workflow

The project follows a structured machine learning pipeline:

---

### 📋 Technical Execution Table

| Step | 🛠️ Phase | 📝 Description | 📤 Output |
|------|-----------|----------------|------------|
| 1 | Data Ingestion | Load `Mall_Customers.csv` using Pandas | Raw DataFrame |
| 2 | Feature Scaling | Apply StandardScaler (Z-score normalization) | Scaled Matrix |
| 3 | Optimization | Automatic best K detection using Silhouette Score | Optimal K |
| 4 | Model Training | Execute K-Means clustering | Trained Model |
| 5 | Analysis | Generate multiple visualizations | Graphical Reports |

---

## 🚀 Logical Flow (Pseudocode)

START

Load Dataset

Select Features:
Annual Income
Spending Score

Scale Data:
Mean = 0
Standard Deviation = 1

FOR k = 2 to 10:
Train KMeans
Calculate Silhouette Score

Select Best k

Train Final Model

Generate Visualizations:
- Scatter Plot
- Pairplot
- Boxplots
- Heatmap

END

---

## 🛠️ Installation & Usage

### 1️⃣ Prerequisites

Install required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

2️⃣ Running the Project

Place dataset here:
data/Mall_Customers.csv
Run the main script:
python customer_segmentation.py
📊 Key Visualizations Included
✅ Optimal Cluster Detection

Line graph showing Silhouette Score vs number of clusters.

✅ Customer Segmentation Map

Scatter plot with cluster centroids.

✅ Distribution Analysis

Boxplots comparing income and spending across clusters.

✅ Cluster Size Analysis

Customer count per cluster.

✅ Correlation Heatmap

Feature relationship visualization.

🧠 Model Optimization Details

Instead of manually selecting clusters using the elbow method, this project uses the Silhouette Coefficient:

s = (b - a) / max(a, b)


Where:

a = average distance within the cluster

b = average distance to nearest neighboring cluster

This ensures:

High intra-cluster similarity

Clear inter-cluster separation

📂 Project Structure
project/
│
├── data/
│   └── Mall_Customers.csv
│
├── customer_segmentation.py
│
└── README.md
🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you find this project useful, consider giving it a ⭐.

"This project was completed as part of my training under @Arnita-Infotech".
 arnitainfotech@gmail.com


