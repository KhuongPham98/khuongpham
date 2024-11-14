import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Đọc và xử lý dữ liệu
# Giả sử dữ liệu từ file CSV có các cột: 'last_purchase_date', 'purchase_count', 'total_spent', 'price', 'sales'
data = pd.read_csv('sales_data.csv')

# Xử lý dữ liệu khách hàng để phân tích hành vi
data['Recency'] = (pd.to_datetime('today') - pd.to_datetime(data['last_purchase_date'])).dt.days
data['Frequency'] = data['purchase_count']
data['Monetary'] = data['total_spent']

# 2. Phân tích hành vi khách hàng bằng K-means
X = data[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Áp dụng K-means để phân đoạn khách hàng
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Hiển thị kết quả phân cụm
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=data, palette='viridis')
plt.title('Phân đoạn khách hàng')
plt.xlabel('Recency (Ngày từ lần mua hàng cuối)')
plt.ylabel('Monetary (Tổng chi tiêu)')
plt.show()

# 3. Tối ưu hóa giá bằng hồi quy tuyến tính
# Chọn các cột 'price' và 'sales' để phân tích
X_price = data[['price']]
y_sales = data['sales']

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_price, y_sales)

# Dự đoán doanh số với các mức giá khác nhau
price_range = np.linspace(X_price.min(), X_price.max(), 100).reshape(-1, 1)
sales_predictions = model.predict(price_range)

# Vẽ đồ thị phân tích độ co giãn của giá
plt.figure(figsize=(10, 6))
plt.plot(price_range, sales_predictions, color='red', label='Dự đoán doanh số')
plt.scatter(data['price'], data['sales'], alpha=0.5, label='Dữ liệu thực tế')
plt.xlabel('Giá')
plt.ylabel('Doanh số')
plt.title('Phân tích độ co giãn của giá')
plt.legend()
plt.show()

# 4. Kết luận và gợi ý
print("Phân tích hành vi khách hàng:")
print(data[['Recency', 'Frequency', 'Monetary', 'Cluster']].head())

print("\nPhân tích độ co giãn của giá:")
print(f"Hệ số hồi quy (ảnh hưởng của giá đến doanh số): {model.coef_[0]}")

# Tạo biểu đồ cột cho số lượng khách hàng trong mỗi phân khúc
cluster_counts = data['Cluster'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.title('Số lượng khách hàng trong mỗi phân khúc')
plt.xlabel('Phân khúc khách hàng')
plt.ylabel('Số lượng khách hàng')
plt.xticks(rotation=0)
plt.show()
# Tạo biểu đồ cột cho doanh số thực tế và dự đoán
price_sales = data[['price', 'sales']].copy()
price_sales['sales_predicted'] = model.predict(X_price)

# Tạo biểu đồ cột cho doanh số thực tế và dự đoán tại mỗi mức giá
plt.figure(figsize=(10, 6))
plt.bar(price_sales['price'], price_sales['sales'], alpha=0.6, label='Doanh số thực tế', color='blue')
plt.bar(price_sales['price'], price_sales['sales_predicted'], alpha=0.6, label='Doanh số dự đoán', color='red')
plt.xlabel('Giá')
plt.ylabel('Doanh số')
plt.title('Doanh số thực tế và dự đoán theo giá')
plt.legend()
plt.show()
