import numpy as np  # Sayısal hesaplamalar için numpy kütüphanesini içe aktarıyoruz
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib kütüphanesini içe aktarıyoruz
from sklearn.linear_model import LinearRegression  # Lineer regresyon modelini içe aktarıyoruz
from sklearn.preprocessing import PolynomialFeatures  # Polinom özellik dönüşümü için gerekli sınıfı içe aktarıyoruz

X = 4 * np.random.rand(100, 1)  # 0-4 arasında 100 adet rastgele sayı üretiyoruz (bağımsız değişken)
y = 2 + 3 * X**2  # X'in karesi ile doğrusal olmayan bir ilişki oluşturuyoruz (bağımlı değişken)

#plt.scatter(X,y)  # Veri noktalarını dağılım grafiği olarak çizmek için (şu anda kapalı)

poly_features = PolynomialFeatures(degree=1)  # 1. derece polinom özellik dönüştürücüsü oluşturuyoruz
X_poly = poly_features.fit_transform(X)  # X verilerini polinom özelliklerine dönüştürüyoruz

poly_reg = LinearRegression()  # Lineer regresyon modeli oluşturuyoruz
poly_reg.fit(X_poly, y)  # Polinom özelliklerle dönüştürülmüş X ve y ile modeli eğitiyoruz

y_pred = poly_reg.predict(X_poly)  # Eğitilmiş model ile tahmin yapıyoruz

plt.scatter(X, y, color="blue")  # Orijinal veri noktalarını mavi renkte dağılım grafiği olarak çiziyoruz
X_test = np.linspace(0, 4, 100)  # 0-4 arasında eşit aralıklarla 100 test noktası oluşturuyoruz
X_test_poly = poly_features.transform(X_test.reshape(-1, 1))  # Test noktalarını polinom özelliklerine dönüştürüyoruz
y_pred = poly_reg.predict(X_test_poly)  # Test noktaları için tahmin yapıyoruz
plt.plot(X_test, y_pred, color="red")  # Tahmin edilen değerleri kırmızı çizgi olarak çiziyoruz
plt.xlabel("X")  # X ekseni etiketini ekliyoruz
plt.ylabel("y")  # Y ekseni etiketini ekliyoruz
plt.title("Polynomial Regression")  # Grafik başlığını ekliyoruz
plt.show()  # Grafiği ekranda gösteriyoruz

"""
POLİNOM REGRESYON KONUSU AÇIKLAMASI:

Polinom Regresyon, lineer regresyonun genişletilmiş bir versiyonudur. Değişkenler arasındaki 
doğrusal olmayan ilişkileri modellemek için kullanılır.

TEMEL KAVRAMLAR:
- Lineer regresyon sadece doğrusal ilişkileri modelleyebilir (y = a + bx)
- Polinom regresyon eğrisel ilişkileri modelleyebilir (y = a + bx + cx² + dx³ + ...)
- Polinom derecesi arttıkça model daha karmaşık hale gelir

BU KODDAKİ DURUM:
- Orijinal veri y = 2 + 3*X² şeklinde kuadratik bir ilişkiye sahip
- Ancak kod degree=1 kullanıyor, yani sadece lineer özellik kullanıyor
- Bu yüzden kuadratik ilişkiyi tam olarak yakalayamayacak
- Daha iyi sonuç için degree=2 kullanılmalı

POLİNOM REGRESYONUN AVANTAJLARI:
1. Doğrusal olmayan ilişkileri modelleyebilir
2. Mevcut lineer regresyon algoritmalarını kullanır
3. Esnek ve güçlü bir yöntemdir

POLİNOM REGRESYONUN DİKKAT EDİLECEK NOKTALARI:
1. Yüksek derece overfitting'e neden olabilir
2. Veri aralığı dışında tahminler güvenilir olmayabilir
3. Özellik sayısı hızla artar (curse of dimensionality)

UYGULAMA ALANLARI:
- Fizik ve mühendislik problemleri
- Ekonomi ve finans modellemesi
- Biyoloji ve tıp araştırmaları
- Trend analizi ve tahmin
"""
