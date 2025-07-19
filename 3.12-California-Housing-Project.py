from sklearn.datasets import fetch_california_housing  # Scikit-learn'den California Housing veri setini yüklemek için gerekli fonksiyonu içe aktarıyoruz
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test olarak bölmek için train_test_split fonksiyonunu içe aktarıyoruz
from sklearn.preprocessing import PolynomialFeatures  # Polinom özellik dönüşümü için PolynomialFeatures sınıfını içe aktarıyoruz
from sklearn.linear_model import LinearRegression  # Lineer regresyon modeli için LinearRegression sınıfını içe aktarıyoruz
from sklearn.metrics import mean_squared_error  # Modelin performansını ölçmek için ortalama kare hata (MSE) fonksiyonunu içe aktarıyoruz
import numpy as np  # Matematik işlemleri için NumPy kütüphanesini içe aktarıyoruz

housing = fetch_california_housing()  # California Housing veri setini yükleyip housing değişkenine atıyoruz
X = housing.data  # Bağımsız değişkenleri (özellikler) X değişkenine atıyoruz
y = housing.target  # Bağımlı değişkeni (hedef değer - ev fiyatları) y değişkenine atıyoruz

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Veri setini %80 eğitim, %20 test olmak üzere rastgele böl

poly_features = PolynomialFeatures(degree=2)  # 2. derece polinom özellik dönüştürücü oluşturuyoruz (x₁, x₂, x₁², x₁x₂, x₂² gibi)
X_train_poly = poly_features.fit_transform(X_train)  # Eğitim verilerine polinom dönüşümü uygulayıp yeni özellikler oluşturuyoruz
X_test_poly = poly_features.fit_transform(X_test)  # Test verilerine de aynı polinom dönüşümü uyguluyoruz (fit_transform yerine transform kullanılmalı)

poly_reg = LinearRegression()  # Polinom regresyon için lineer regresyon modeli oluşturuyoruz
poly_reg.fit(X_train_poly, y_train)  # Polinom dönüştürülmüş eğitim verileri ile modeli eğitiyoruz
y_pred = poly_reg.predict(X_test_poly)  # Polinom dönüştürülmüş test verileri ile tahmin yapıyoruz

mse = mean_squared_error(y_test, y_pred)  # Gerçek değerler ile tahmin değerleri arasındaki ortalama kare hatayı hesaplıyoruz
rmse = np.sqrt(mse)  # MSE'nin karekökünü alarak RMSE (Root Mean Square Error) değerini hesaplıyoruz
print(rmse)  # Polinom regresyon RMSE değerini ekrana yazdırıyoruz

linear_reg = LinearRegression()  # Karşılaştırma için basit lineer regresyon modeli oluşturuyoruz
linear_reg.fit(X_train, y_train)  # Orijinal eğitim verileri ile lineer regresyon modelini eğitiyoruz
y_pred = linear_reg.predict(X_test)  # Orijinal test verileri ile lineer regresyon ile tahmin yapıyoruz

mse = mean_squared_error(y_test, y_pred)  # Lineer regresyon için MSE hesaplıyoruz
rmse = np.sqrt(mse)  # Lineer regresyon için RMSE hesaplıyoruz
print("Multi-Variable Polynomial Regression RMSE:", rmse)  # Karşılaştırma için lineer regresyon RMSE değerini yazdırıyoruz

"""
CALIFORNIA HOUSING PROJESİ AÇIKLAMASI:

Bu proje, California eyaletindeki ev fiyatlarını tahmin etmek için makine öğrenmesi algoritmaları kullanmaktadır.
California Housing veri seti 1990 sayımından alınmış gerçek verilerden oluşur ve şu özellikleri içerir:

ÖZELLIKLER:
- MedInc: Medyan gelir
- HouseAge: Ev yaşı
- AveRooms: Ortalama oda sayısı
- AveBedrms: Ortalama yatak odası sayısı
- Population: Nüfus
- AveOccup: Ortalama kişi sayısı
- Latitude: Enlem
- Longitude: Boylam

HEDEF DEĞİŞKEN:
- MedHouseVal: Medyan ev değeri (yüz binlerce dolar cinsinden)

MODEL KARŞILAŞTIRMASI:
1. POLİNOM REGRESYON: Özelliklerin ikinci dereceden etkileşimlerini de hesaba katarak daha karmaşık ilişkileri yakalayabilir
2. LİNEER REGRESYON: Özellikler arasında doğrusal ilişki varsayar, daha basit model

RMSE (Root Mean Square Error): Modelin tahmin hatalarının karekök ortalamasıdır. Düşük RMSE değeri daha iyi performans anlamına gelir.

Polinom regresyon genellikle daha düşük RMSE değeri verir çünkü özellikler arasındaki karmaşık ilişkileri de yakalayabilir.
Ancak aşırı öğrenme (overfitting) riski taşır.
"""






