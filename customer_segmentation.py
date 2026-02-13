# Customer Segmentation using KMeans (Advanced Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. Load Dataset
# -----------------------------

def load_data(path):
    data = pd.read_csv(path)
    return data

# -----------------------------
# 2. Prepare Features
# -----------------------------

def prepare_features(data):
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# -----------------------------
# 3. Automatic Best Cluster Detection
# -----------------------------

def detect_best_k(X_scaled):

    scores = []
    K = range(2, 11)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    best_k = K[scores.index(max(scores))]

    plt.figure(figsize=(8,5))
    plt.plot(K, scores, marker='o')
    plt.title("Automatic Cluster Detection (Silhouette Score)")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

    return best_k

# -----------------------------
# 4. Train Model
# -----------------------------

def train_model(X_scaled, best_k):
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

# -----------------------------
# 5. Main Cluster Visualization
# -----------------------------

def plot_clusters(data, model, scaler):

    centers = scaler.inverse_transform(model.cluster_centers_)

    plt.figure(figsize=(10,7))

    sns.scatterplot(
        data=data,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set2',
        s=100
    )

    plt.scatter(
        centers[:,0],
        centers[:,1],
        s=300,
        marker='X',
        color='black',
        label='Cluster Centers'
    )

    plt.title("Customer Segmentation Analysis")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.show()

# -----------------------------
# 6. Pairplot
# -----------------------------

def plot_pairplot(data):

    sns.pairplot(
        data,
        vars=['Annual Income (k$)', 'Spending Score (1-100)'],
        hue='Cluster',
        palette='Set2'
    )

    plt.show()

# -----------------------------
# 7. Cluster Size Graph
# -----------------------------

def plot_cluster_counts(data):

    plt.figure(figsize=(8,5))

    sns.countplot(x='Cluster', data=data, palette='Set2')

    plt.title("Customer Count per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.show()

# -----------------------------
# 8. Boxplots
# -----------------------------

def plot_boxplots(data):

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    sns.boxplot(x='Cluster', y='Annual Income (k$)', data=data)
    plt.title("Income Distribution by Cluster")

    plt.subplot(1,2,2)
    sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=data)
    plt.title("Spending Distribution by Cluster")

    plt.show()

# -----------------------------
# 9. Heatmap
# -----------------------------

def plot_heatmap(data):

    plt.figure(figsize=(6,5))

    sns.heatmap(
        data[['Annual Income (k$)', 'Spending Score (1-100)']].corr(),
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Feature Correlation Heatmap")
    plt.show()

# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":

    data = load_data("data/Mall_Customers.csv")

    print("Dataset Preview:")
    print(data.head())

    X_scaled, scaler = prepare_features(data)

    best_k = detect_best_k(X_scaled)

    print("Best cluster detected:", best_k)

    model, labels = train_model(X_scaled, best_k)

    data["Cluster"] = labels

    # Generate Graphs
    plot_clusters(data, model, scaler)
    plot_pairplot(data)
    plot_cluster_counts(data)
    plot_boxplots(data)
    plot_heatmap(data)
