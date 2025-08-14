import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 300
data = {
    'CustomerID': range(1, num_customers + 1),
    'Recency': np.random.randint(1, 365, size=num_customers), # Days since last purchase
    'Frequency': np.random.poisson(lam=5, size=num_customers), # Number of purchases
    'MonetaryValue': np.random.exponential(scale=100, size=num_customers) # Average purchase value
}
df = pd.DataFrame(data)
# --- 2. Feature Engineering: Customer Lifetime Value (CLV) ---
# A simplified CLV calculation (more sophisticated models exist)
df['CLV'] = df['Frequency'] * df['MonetaryValue']
# --- 3. Data Scaling ---
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'MonetaryValue', 'CLV']])
df_scaled = pd.DataFrame(df_scaled, columns=['Recency', 'Frequency', 'MonetaryValue', 'CLV'])
# --- 4. Customer Segmentation using K-Means Clustering ---
# Determine optimal number of clusters (e.g., using the Elbow method -  simplified here)
kmeans = KMeans(n_clusters=3, random_state=42) # Choosing 3 clusters as an example.  A more robust method would be needed in a real-world scenario.
kmeans.fit(df_scaled)
df['Segment'] = kmeans.labels_
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CLV', y='Frequency', hue='Segment', data=df, palette='viridis')
plt.title('Customer Segments based on CLV and Frequency')
plt.xlabel('Customer Lifetime Value (CLV)')
plt.ylabel('Purchase Frequency')
plt.savefig('customer_segments.png')
print("Plot saved to customer_segments.png")
plt.figure(figsize=(10,6))
sns.boxplot(x='Segment', y='CLV', data=df, palette='viridis')
plt.title('CLV Distribution Across Segments')
plt.xlabel('Customer Segment')
plt.ylabel('Customer Lifetime Value (CLV)')
plt.savefig('clv_distribution.png')
print("Plot saved to clv_distribution.png")
# --- 6. Analysis and Interpretation ---
# Analyze the characteristics of each segment (e.g., average CLV, frequency, recency)
print("\nSegment Characteristics:")
print(df.groupby('Segment')[['Recency', 'Frequency', 'MonetaryValue', 'CLV']].mean())
#Further analysis and targeted marketing strategies would be developed based on these segments.  This is a simplified example.