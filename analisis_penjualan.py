import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Pengumpulan Data
data_penjualan = pd.read_csv('data_penjualan.csv')
print("Data Penjualan:")
print(data_penjualan.head())

# Data Cleaning
print("\nMemeriksa nilai yang hilang:")
print(data_penjualan.isnull().sum())
data_penjualan.fillna(method='ffill', inplace=True)

# Data Transformation
data_penjualan['Total_Harga'] = data_penjualan['Jumlah'] * data_penjualan['Harga Satuan']
print("\nData setelah transformasi:")
print(data_penjualan.head())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
sns.countplot(x='Jenis Kelamin', data=data_penjualan)
plt.title('Jumlah Penjualan berdasarkan Jenis Kelamin')
plt.show()

plt.figure(figsize=(10, 5))
data_penjualan.groupby('Jenis Barang')['Total_Harga'].sum().plot(kind='bar')
plt.title('Total Penjualan berdasarkan Jenis Barang')
plt.show()

# Scatter Plot: Korelasi antara Jumlah dan Harga Satuan
plt.figure(figsize=(10, 6))
plt.scatter(data_penjualan['Jumlah'], data_penjualan['Harga Satuan'], alpha=0.5)
plt.xlabel('Jumlah')
plt.ylabel('Harga Satuan')
plt.title('Scatter Plot: Korelasi antara Jumlah dan Harga Satuan')
plt.show()

# Hexbin Plot: Korelasi antara Jumlah dan Harga Satuan dengan diagram lambang
plt.figure(figsize=(10, 6))
plt.hexbin(data_penjualan['Jumlah'], data_penjualan['Harga Satuan'], gridsize=20, cmap='YlGnBu')
plt.colorbar(label='Jumlah Penjualan')
plt.xlabel('Jumlah')
plt.ylabel('Harga Satuan')
plt.title('Hexbin Plot: Korelasi antara Jumlah dan Harga Satuan')
plt.show()

# Modelling Data
X = data_penjualan[['Jumlah', 'Harga Satuan']]
y = data_penjualan['Total_Harga']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Validasi dan Tuning Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')

# Interpretasi dan Penyajian Hasil
print('\nKoefisien Model:', model.coef_)
print('Intercept Model:', model.intercept_)

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred)
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.title('Hasil Prediksi vs Aktual')
plt.show()

# Deploy dan Monitoring
joblib.dump(model, 'model_penjualan.pkl')
print("\nModel telah disimpan sebagai 'model_penjualan.pkl'")

# Contoh penggunaan model yang disimpan
model = joblib.load('model_penjualan.pkl')
data_baru = [[3, 200000]]
prediksi_baru = model.predict(data_baru)
print(f'\nPrediksi Penjualan untuk data baru {data_baru}: {prediksi_baru}')
