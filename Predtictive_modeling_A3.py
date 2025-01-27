import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Download the dataset (time_series_covid19_confirmed_global.csv) from the provided URL
# Load the dataset
data = pd.read_csv("C:\\time_series_covid19_confirmed_global.csv")

total_cases = data.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1)
total_cases = total_cases.apply(pd.to_numeric, errors='coerce')
# Find the top three infected countries
top_countries = total_cases.sum(axis=1).nlargest(3).index
total_cases = total_cases.apply(pd.to_numeric, errors='coerce')
# Filter the dataset for the top three countries
top_countries_data = data[data['Country/Region'].isin(top_countries)]
#Function to convert dates to week numbers
def convert_to_week_number(date):
    start_date = pd.Timestamp('2020-01-22')  # Start date of record
    return (date - start_date).days // 7 + 1
# Visualize the data for the top three infected countries
for country in top_countries:
    country_data = top_countries_data[top_countries_data['Country/Region'] == country]
    country_data.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1, inplace=True)
    country_data = country_data.transpose()
    country_data.columns = ['Confirmed Cases']
    country_data.plot(title=f'Confirmed Cases Over Time - {country}')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.show()
# Step 3: Predictive Modeling

selected_country = top_countries[0]


# Fit linear regression model for the selected country
# Perform model analysis to identify the model with the highest variance

# Example: Fitting linear regression model for the selected country
selected_country_data = top_countries_data[top_countries_data['Country/Region'] == selected_country]
selected_country_data.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1, inplace=True)
selected_country_data = selected_country_data.transpose()
selected_country_data.columns = ['Confirmed Cases']

# Fit linear regression model for the selected country
selected_country_data.index = pd.to_datetime(selected_country_data.index)
X = selected_country_data.index.strftime('%U').astype(int).values.reshape(-1, 1)  # Extracting week of the year
y = selected_country_data['Confirmed Cases'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

# Predict using the model
y_pred = model.predict(X)

# Compute evaluation metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")

# Step 4: K-Means Clustering
scaler = StandardScaler()
normalized_data = scaler.fit_transform(selected_country_data)

# Using the elbow method to find the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_data)
    sse.append(kmeans.inertia_)

# Plotting the elbow curve
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Perform K-Means clustering with the chosen K
optimal_k = 3  # For demonstration, assuming optimal K is 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

# Analyze Clusters
# Add cluster labels to the dataset
selected_country_data['Cluster'] = clusters

# Plotting clusters over time
for cluster_id in range(optimal_k):
    cluster_data = selected_country_data[selected_country_data['Cluster'] == cluster_id].drop('Cluster', axis=1)
    plt.plot(cluster_data.index, cluster_data.values, label=f'Cluster {cluster_id}')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('Clusters Over Time')
plt.legend()
plt.show()

