import pandas as pd  # Open source library for data manipulation and analysis
import numpy as np  # Open source library for numerical computations
from sklearn.ensemble import IsolationForest  # Open source machine learning library
from sklearn.decomposition import PCA  # Open source machine learning library
import matplotlib.pyplot as plt  # Open source library for plotting
import seaborn as sns  # Open source library for data visualization
from statsmodels.tsa.arima.model import ARIMA  # Open source library for time series analysis
from pmdarima import auto_arima  # Open source library for automatic ARIMA model selection
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Open source library for plotting ACF and PACF
from statsmodels.tsa.stattools import adfuller  # Open source library for statistical tests
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Open source library for time series forecasting


def load_data(filepath):
    print(f"Loading dataset from {filepath}...")
    data = pd.read_csv(filepath)
    print("Initial data loaded:")
    print(data.head())

    data['date'] = pd.to_datetime(data['date'])
    print("\nDate column converted to datetime.")

    missing_values_count = data.isnull().sum().sum()
    print(f"\nNumber of missing values before filling: {missing_values_count}")

    data = data.fillna(0)
    missing_values_count_after = data.isnull().sum().sum()
    print(f"Number of missing values after filling: {missing_values_count_after}")

    print("Dataset loaded and preprocessed successfully.")
    return data


def anomaly_detection(data):
    print("Performing anomaly detection on local IPs...")
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['avg_flows_7d'] = data.groupby('l_ipn')['f'].rolling(window=7).mean().reset_index(0, drop=True)
    data['avg_flows_7d'] = data['avg_flows_7d'].fillna(0)

    features = ['r_asn', 'f', 'day_of_week', 'month', 'avg_flows_7d']
    X = data[features]

    iso_forest = IsolationForest(contamination=0.5, random_state=42)
    data['anomaly'] = iso_forest.fit_predict(X)

    anomalies = data[data['anomaly'] == -1]
    num_anomalies = len(anomalies)
    print(f"Number of anomalies detected: {num_anomalies}")

    print("\nTop Anomalous IPs:")
    top_anomalous_ips = anomalies['l_ipn'].value_counts().head(5)
    print(top_anomalous_ips)

    print("\nDetailed Summary for Each Anomalous IP:")
    for ip in top_anomalous_ips.index:
        ip_data = anomalies[anomalies['l_ipn'] == ip]
        print(f"\nIP {ip} Summary:")
        print(ip_data.describe())

    # Plot the PCA components and highlight anomalies
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    data['pca1'] = X_pca[:, 0]
    data['pca2'] = X_pca[:, 1]
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x='pca1', y='pca2', hue='anomaly', palette=['red', 'blue'])
    plt.title('PCA of Network Traffic Data (Anomalies Highlighted)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('pca_anomalies.png')
    plt.close()

    # Plot the flows over time for the top anomalous IPs
    plt.figure(figsize=(12, 8))
    for ip in top_anomalous_ips.index:
        ip_data = data[data['l_ipn'] == ip]
        plt.plot(ip_data['date'], ip_data['f'], label=f'IP {ip}')
    plt.legend()
    plt.title('Flows Over Time for Top Anomalous IPs')
    plt.xlabel('Date')
    plt.ylabel('Flows')
    plt.savefig('flows_over_time.png')
    plt.close()

    print("Anomaly detection plots saved as 'pca_anomalies.png' and 'flows_over_time.png'.")
    print(
        f"\nSummary:\nDetected {num_anomalies} anomalies across multiple IPs. The top anomalous IPs are {', '.join(map(str, top_anomalous_ips.index))}.")
    return data


def remote_isp_analysis(data):
    print("Performing remote ISP anomaly analysis...")
    if 'anomaly' not in data.columns:
        data = anomaly_detection(data)

    anomalies = data[data['anomaly'] == -1]
    anomalous_isps = anomalies['r_asn'].value_counts()
    normal_isps = data[data['anomaly'] == 1]['r_asn'].value_counts()

    print("\nTop Remote ISPs Associated with Anomalies:")
    print(anomalous_isps.head(10))

    print("\nTop Remote ISPs Associated with Normal Data:")
    print(normal_isps.head(10))

    print("\nSummary:\nThe top remote ISPs associated with anomalies are:")
    for isp, count in anomalous_isps.head(10).items():
        print(f"ISP {isp}: {count} anomalies")
    print("\nThe top remote ISPs associated with normal data are:")
    for isp, count in normal_isps.head(10).items():
        print(f"ISP {isp}: {count} normal instances")


def forecasting(data):
    print("Performing time-series forecasting for all IPs...")
    unique_ips = data['l_ipn'].unique()

    for ip in unique_ips:
        ip_data = data[data['l_ipn'] == ip].set_index('date')
        ip_data = ip_data['f'].resample('D').sum()

        result = adfuller(ip_data.dropna())
        print(f"\nADF Statistic for IP {ip}: {result[0]}")
        print(f"p-value: {result[1]}")
        if result[1] > 0.05:
            ip_data_diff = ip_data.diff().dropna()
        else:
            ip_data_diff = ip_data

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        plot_acf(ip_data_diff, ax=axes[0])
        plot_pacf(ip_data_diff, ax=axes[1])
        plt.suptitle(f'ACF and PACF for IP {ip}')
        plt.savefig(f'acf_pacf_ip_{ip}.png')
        plt.close()

        arima_model = auto_arima(ip_data_diff, seasonal=True, m=7, stepwise=True, trace=True, error_action='ignore',
                                 suppress_warnings=True)
        model_arima = ARIMA(ip_data, order=arima_model.order, seasonal_order=arima_model.seasonal_order)
        model_fit_arima = model_arima.fit()

        model_ets = ExponentialSmoothing(ip_data, seasonal='add', seasonal_periods=7).fit()

        forecast_arima = model_fit_arima.forecast(steps=30)
        forecast_index = pd.date_range(start=ip_data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        forecast_series_arima = pd.Series(forecast_arima, index=forecast_index)

        forecast_ets = model_ets.forecast(steps=30)
        forecast_series_ets = pd.Series(forecast_ets, index=forecast_index)

        combined_arima = pd.concat([ip_data, forecast_series_arima])
        combined_ets = pd.concat([ip_data, forecast_series_ets])

        plt.figure(figsize=(12, 8))
        plt.plot(combined_arima.index, combined_arima.values, label='Historical and Forecast (ARIMA)', color='blue')
        plt.plot(forecast_series_arima.index, forecast_series_arima.values, label='Forecast (ARIMA)', color='red')
        plt.axvline(x=ip_data.index[-1], color='grey', linestyle='--', label='Forecast Start')
        plt.legend()
        plt.title(f'Flow Forecasting for IP {ip} (ARIMA)')
        plt.xlabel('Date')
        plt.ylabel('Flows')
        plt.savefig(f'forecast_arima_ip_{ip}.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(combined_ets.index, combined_ets.values, label='Historical and Forecast (ETS)', color='blue')
        plt.plot(forecast_series_ets.index, forecast_series_ets.values, label='Forecast (ETS)', color='red')
        plt.axvline(x=ip_data.index[-1], color='grey', linestyle='--', label='Forecast Start')
        plt.legend()
        plt.title(f'Flow Forecasting for IP {ip} (ETS)')
        plt.xlabel('Date')
        plt.ylabel('Flows')
        plt.savefig(f'forecast_ets_ip_{ip}.png')
        plt.close()

        print(f"Forecast plots for IP {ip} saved as 'forecast_arima_ip_{ip}.png' and 'forecast_ets_ip_{ip}.png'.")

    print("\nSummary:\nTime-series forecasting completed for all IPs. The forecast plots have been saved for each IP.")


def main():
    dataset_path = '/Users/frankoswanepoel/Desktop/Assingments/Assingment 1/COS783AssingmentFinal/cs448b_ipasn.csv'
    data = load_data(dataset_path)

    print("\nSelect an analysis to perform:")
    print("1. Local IP Anomaly Detection")
    print("2. Remote ISP Anomaly Analysis")
    print("3. Time-Series Forecasting")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        data = anomaly_detection(data)
    elif choice == '2':
        remote_isp_analysis(data)
    elif choice == '3':
        forecasting(data)
    else:
        print("Invalid choice. Please restart the program and enter a valid option.")


if __name__ == "__main__":
    main()
