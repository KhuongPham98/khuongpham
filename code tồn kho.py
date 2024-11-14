import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Giả sử dữ liệu bao gồm các thông tin về khách hàng, sản phẩm, và doanh thu
data = {
    'Customer_ID': [1, 2, 3, 4, 5],
    'Recency': [30, 60, 90, 10, 50],  # Số ngày từ lần mua hàng gần nhất
    'Frequency': [10, 5, 2, 15, 8],   # Số lần mua hàng
    'Monetary': [1000, 500, 200, 1500, 800],  # Tổng chi tiêu
    'Product_Category': ['A', 'B', 'C', 'A', 'D'],
    'Inventory': [50, 30, 20, 60, 40]  # Số lượng tồn kho
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Chuẩn hóa dữ liệu cho K-means clustering
scaler = StandardScaler()
X = df[['Recency', 'Frequency', 'Monetary']]
X_scaled = scaler.fit_transform(X)

# Phân cụm khách hàng thành 3 nhóm sử dụng K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Customer_Segment'] = kmeans.fit_predict(X_scaled)

# Hiển thị phân đoạn khách hàng
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Recency'], y=df['Monetary'], hue=df['Customer_Segment'], palette='viridis')
plt.title('Phân đoạn khách hàng dựa trên Recency và Monetary')
plt.xlabel('Recency (Ngày)')
plt.ylabel('Monetary (Tổng chi tiêu)')
plt.show()

# Dự báo nhu cầu sản phẩm bằng hồi quy tuyến tính
X_inventory = df[['Inventory']].values
y_monetary = df['Monetary'].values

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_inventory, y_monetary)

# Dự đoán doanh thu dựa trên số lượng hàng tồn kho
inventory_range = np.linspace(X_inventory.min(), X_inventory.max(), 100).reshape(-1, 1)
monetary_predictions = model.predict(inventory_range)

# Hiển thị biểu đồ dự báo nhu cầu sản phẩm
plt.figure(figsize=(10, 6))
plt.plot(inventory_range, monetary_predictions, color='red', label='Dự báo doanh thu')
plt.scatter(df['Inventory'], df['Monetary'], alpha=0.6, label='Dữ liệu thực tế')
plt.xlabel('Số lượng tồn kho')
plt.ylabel('Doanh thu (Monetary)')
plt.title('Dự báo doanh thu theo số lượng hàng tồn kho')
plt.legend()
plt.show()

# Hiển thị phân đoạn khách hàng
print("Phân đoạn khách hàng:")
print(df[['Customer_ID', 'Recency', 'Frequency', 'Monetary', 'Customer_Segment']])

# Hiển thị hệ số hồi quy (ảnh hưởng của tồn kho đến doanh thu)
print(f"Hệ số hồi quy (ảnh hưởng của tồn kho đến doanh thu): {model.coef_[0]}")