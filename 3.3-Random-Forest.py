"""
# Random Forest (Rastgele Orman) Algoritması
#
# Random Forest, birden fazla karar ağacının bir araya gelerek oluşturduğu bir topluluk öğrenme algoritmasıdır.
# Temel özellikleri:
# 1. Birden fazla karar ağacı kullanır (ensemble learning)
# 2. Her ağaç için veri setinden rastgele örnekler seçer (bootstrap sampling)
# 3. Her düğümde rastgele özellik alt kümeleri kullanır
# 4. Sınıflandırma için çoğunluk oylaması, regresyon için ortalama değer kullanır
#
# Bu örnekte hem sınıflandırma (yüz tanıma) hem de regresyon (ev fiyatı tahmini) problemleri gösterilmektedir.
"""

# Gerekli kütüphanelerin import edilmesi
from sklearn.datasets import fetch_olivetti_faces, fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Olivetti yüz veri setinin yüklenmesi
olivetti = fetch_olivetti_faces()
"""
# Veri seti hakkında bilgi:
# Her bir yüz görüntüsü 64x64 piksel boyutundadır
# Bu 2D görüntüler, 4096 boyutlu 1D vektörlere dönüştürülmüştür
"""

# Örnek yüz görüntülerinin görselleştirilmesi
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i + 1)  # 1 satır 2 sütunluk bir plot oluşturur
    plt.imshow(olivetti.images[i + 10], cmap=plt.cm.gray)  # Gri tonlamalı görüntü gösterimi
    plt.title(f"Face {olivetti.target[i+10]}")  # Görüntünün sınıf etiketini başlık olarak ekler
    plt.axis("off")  # Eksenleri gizler
plt.show()

# Veri setinin özellikler (X) ve hedef değişken (y) olarak ayrılması
X = olivetti.data  # features - her bir pikselin değeri bir özellik olarak kullanılır
y = olivetti.target  # target - her bir yüzün ait olduğu kişinin indeksi

# Veri setinin eğitim ve test olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # %80 eğitim, %20 test verisi
)

# Sınıflandırma için Random Forest modelinin oluşturulması ve eğitilmesi
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 adet karar ağacı kullanılacak
rf_clf.fit(X_train, y_train)  # Modelin eğitim verileriyle eğitilmesi

# Test verileri üzerinde tahmin yapılması ve doğruluk oranının hesaplanması
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")

# California ev fiyatları veri setinin yüklenmesi (Regresyon örneği için)
california_housing = fetch_california_housing()
X = california_housing.data  # Ev özellikleri (gelir, oda sayısı, vb.)
y = california_housing.target  # Ev fiyatları

# Veri setinin eğitim ve test olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Regresyon için Random Forest modelinin oluşturulması ve eğitilmesi
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapılması ve hata metriklerinin hesaplanması
y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # Ortalama kare hata
rmse = np.sqrt(mse)  # Kök ortalama kare hata
print(f"Random Forest RMSE: {rmse}")
