import numpy as np  # Matematiksel işlemler ve diziler için
import matplotlib.pyplot as plt  # Grafik çizimi için
from sklearn.linear_model import LinearRegression  # Doğrusal regresyon modeli için
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için
from sklearn.datasets import load_diabetes  # Örnek diabetes veri setini yüklemek için
from sklearn.metrics import mean_squared_error  # Model performansını ölçmek için

diabetes = load_diabetes()  # Diabetes veri setini yüklüyoruz (10 özellik ile kan şekeri seviyesini tahmin etme)
X = diabetes.data  # Bağımsız değişkenler (özellikler) - yaş, cinsiyet, vücut kitle indeksi vb.
y = diabetes.target  # Bağımlı değişken (hedef) - kan şekeri seviyesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Veriyi %80 eğitim, %20 test olarak rastgele ayırıyoruz
linear_reg = LinearRegression()  # Doğrusal regresyon modelini oluşturuyoruz
linear_reg.fit(X_train, y_train)  # Modeli eğitim verisi ile eğitiyoruz (katsayıları ve bias'ı öğreniyor)
y_pred = linear_reg.predict(X_test)  # Eğitilen model ile test verisi üzerinde tahmin yapıyoruz
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error (Ortalama Kare Hata) hesaplıyoruz
rmse = np.sqrt(mse)  # Root Mean Squared Error (Kök Ortalama Kare Hata) hesaplıyoruz
print("rmse", rmse)  # RMSE değerini ekrana yazdırıyoruz (düşük değer daha iyi performans demektir)



# ALTERNATIF ÖRNEK - 2 değişkenli sentetik veri ile çok değişkenli regresyon görselleştirmesi
# X = np.random.rand(100, 2)  # 100 satır, 2 sütunluk rastgele veri (0-1 arası)
# coef = np.array([3, 5])  # Katsayılar dizisi [3, 5]
# y = 0 + np.dot(X, coef)  # y = 0 + (X1*3 + X2*5) formülü ile hedef değişken oluşturma
# fig = plt.figure()  # Yeni figür oluşturuyoruz
# ax = fig.add_subplot(111, projection="3d")  # 3D alt grafik ekliyoruz
# ax.scatter(X[:, 0], X[:, 1], y)  # X1, X2, y noktalarını 3D uzayda gösteriyoruz
# ax.set_xlabel("X1")  # X ekseni etiketi
# ax.set_ylabel("X2")  # Y ekseni etiketi  
# ax.set_zlabel("y")  # Z ekseni etiketi
# ax.set_title("Multi-Variable Regression")  # Grafik başlığı
# plt.show()  # Grafiği gösteriyoruz
# linear_reg = LinearRegression()  # Yeni model oluşturuyoruz
# linear_reg.fit(X, y)  # Sentetik veri ile modeli eğitiyoruz
# fig = plt.figure()  # Yeni figür
# ax = fig.add_subplot(111, projection="3d")  # 3D alt grafik
# ax.scatter(X[:, 0], X[:, 1], y)  # Veri noktalarını gösteriyoruz
# ax.set_xlabel("X1")  # X ekseni etiketi
# ax.set_ylabel("X2")  # Y ekseni etiketi
# ax.set_zlabel("y")  # Z ekseni etiketi
# ax.set_title("Multi-Variable Regression")  # Grafik başlığı
# x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))  # 10x10 grid oluşturuyoruz
# y_pred = linear_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)  # Grid noktaları için tahmin yapıyoruz
# ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), color="red", alpha=0.3)  # Regresyon düzlemini kırmızı renkte, şeffaf olarak çiziyoruz
# plt.title("Multi-Variable Regression")  # Grafik başlığı
# print("katsayılar", linear_reg.coef_)  # Öğrenilen katsayıları yazdırıyoruz
# print("bias", linear_reg.intercept_)  # Bias (kesim noktası) değerini yazdırıyoruz
# plt.show()  # Grafiği gösteriyoruz

# ==================================================================================
# ÇOK DEĞİŞKENLİ DOĞRUSAL REGRESYON (MULTI-VARIABLE LINEAR REGRESSION) KONU AÇIKLAMASI
# ==================================================================================
# 
# Çok değişkenli doğrusal regresyon, birden fazla bağımsız değişken (özellik) kullanarak
# sürekli bir hedef değişkeni tahmin etmeye yarayan makine öğrenmesi algoritmasıdır.
#
# FORMÜL: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
# - y: Hedef değişken (tahmin edilmek istenen)
# - x₁, x₂, ..., xₙ: Bağımsız değişkenler (özellikler)
# - β₀: Bias (kesim noktası)
# - β₁, β₂, ..., βₙ: Katsayılar (her özelliğin ağırlığı)
# - ε: Hata terimi
#
# AVANTAJLARI:
# - Basit ve anlaşılır algoritma
# - Hızlı eğitim ve tahmin
# - Özellik önemini katsayılardan anlayabilme
# - Overfitting riski düşük
#
# DEZAVANTAJLARI:
# - Sadece doğrusal ilişkileri yakalayabilir
# - Outlier'lara hassas
# - Özellikler arası korelasyon (multicollinearity) problemi
#
# KULLANIM ALANLARI:
# - Ev fiyat tahmini (alan, oda sayısı, lokasyon vb.)
# - Satış tahmini (reklam bütçesi, mevsim, fiyat vb.)
# - Tıbbi tanı (yaş, cinsiyet, test sonuçları vb.)
#
# DEĞERLENDİRME METRİKLERİ:
# - RMSE (Root Mean Square Error): Düşük değer daha iyi
# - R² (R-squared): 0-1 arası, 1'e yakın daha iyi
# - MAE (Mean Absolute Error): Ortalama mutlak hata
#
# Bu örnekte diabetes veri seti kullanılarak 10 farklı özellik ile
# kan şekeri seviyesi tahmin edilmektedir.
# ==================================================================================