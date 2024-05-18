import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Description: Load the dataset
data = pd.read_csv('/Users/frankoswanepoel/Desktop/Assingments/Assingment 1/COS783AssingmentFinal/cs448b_ipasn.csv')

# Description: Convert date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Description: Check for missing values and fill if necessary
data = data.fillna(0)

# Description: Create new features such as day of the week and month
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month

# Description: Aggregate data to capture trends (e.g., average flows over the past week)
data['avg_flows_7d'] = data.groupby('l_ipn')['f'].rolling(window=7).mean().reset_index(0, drop=True)
data['avg_flows_7d'] = data['avg_flows_7d'].fillna(0)

# Description: Define features for anomaly detection
features = ['r_asn', 'f', 'day_of_week', 'month', 'avg_flows_7d']
X = data[features]

# Description: Train Isolation Forest model to detect anomalies
iso_forest = IsolationForest(contamination=0.1, random_state=42)
data['anomaly'] = iso_forest.fit_predict(X)

# Description: Anomalies are labeled as -1
anomalies = data[data['anomaly'] == -1]
print(f"Number of anomalies detected: {len(anomalies)}")

# Description: Statistical Summary of the Data
print("\nStatistical Summary of the Data:")
print(data.describe())

# Description: Percentage of Anomalies
percentage_anomalies = (len(anomalies) / len(data)) * 100
print(f"\nPercentage of anomalies: {percentage_anomalies:.2f}%")

# Description: Top Anomalous IPs
top_anomalous_ips = anomalies['l_ipn'].value_counts().head(5)
print("\nTop Anomalous IPs:")
print(top_anomalous_ips)

# Detailed Summary for Each Anomalous IP
print("\nDetailed Summary for Each Anomalous IP:")
for ip in top_anomalous_ips.index:
    ip_data = anomalies[anomalies['l_ipn'] == ip]
    print(f"\nIP {ip} Summary:")
    print(ip_data.describe())

# Description: Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Description: Add PCA components to the data
data['pca1'] = X_pca[:, 0]
data['pca2'] = X_pca[:, 1]

# Description: Plot the PCA components and highlight anomalies
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='pca1', y='pca2', hue='anomaly', palette=['red', 'blue'])
plt.title('PCA of Network Traffic Data (Anomalies Highlighted)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Description: Plot the flows over time for the top anomalous IPs
plt.figure(figsize=(12, 8))
for ip in top_anomalous_ips.index:
    ip_data = data[data['l_ipn'] == ip]
    plt.plot(ip_data['date'], ip_data['f'], label=f'IP {ip}')
plt.legend()
plt.title('Flows Over Time for Top Anomalous IPs')
plt.xlabel('Date')
plt.ylabel('Flows')
plt.show()
