# -*- coding: utf-8 -*-
#####################################################################
# KARAR AĞAÇLARI (DECISION TREES) UYGULAMA ÖRNEKLERİ
#####################################################################
# Bu kod, karar ağaçları algoritmasının hem sınıflandırma hem de regresyon 
# problemlerinde nasıl kullanıldığını göstermektedir.
#
# Karar Ağaçları Nedir?
# - Veriyi sürekli olarak bölerek karar verme yapısı oluşturan bir algoritma
# - Hem sınıflandırma hem de regresyon problemlerinde kullanılabilir
# - Kolay yorumlanabilir ve görselleştirilebilir
# - Aşırı öğrenmeye (overfitting) eğilimli olabilir
#
# Bu Kodda Yapılanlar:
# 1. Iris veri seti ile çiçek türü sınıflandırma
# 2. Diabetes veri seti ile hastalık ilerlemesi tahmini (regresyon)
# 3. Yapay veri ile farklı model derinliklerinin etkisini inceleme
#
# Kullanılan Metrikler:
# - Sınıflandırma için: Doğruluk oranı (accuracy) ve karmaşıklık matrisi
# - Regresyon için: MSE (Ortalama Kare Hata) ve RMSE (Kök Ortalama Kare Hata)
#####################################################################

# Gerekli kütüphanelerin import edilmesi
from sklearn.datasets import load_iris,load_diabetes # Hazır veri setleri için
from sklearn.model_selection import train_test_split # Veri seti bölme işlemi için
from sklearn.tree import DecisionTreeClassifier,plot_tree,DecisionTreeRegressor # Karar ağacı modelleri için
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error # Metrikler için
from sklearn.inspection import DecisionBoundaryDisplay # Karar sınırlarını görselleştirmek için
import matplotlib.pyplot as plt # Görselleştirme için
import numpy as np # Sayısal işlemler için

#####################################################################
# BÖLÜM 1: IRIS VERİ SETİ İLE SINIFLANDIRMA MODELİ
#####################################################################
# Iris veri seti: 3 farklı çiçek türünü 4 özellik kullanarak sınıflandırma
iris = load_iris() # Veri setini yükle

# Veri setinin hazırlanması
X = iris.data     # Öznitelikler: sepal uzunluk, sepal genişlik, petal uzunluk, petal genişlik
y = iris.target   # Hedef değişken: çiçek türleri (0: setosa, 1: versicolor, 2: virginica)

# Veri setinin eğitim ve test olarak bölünmesi
# test_size=0.2: verilerin %20'si test, %80'i eğitim için
# random_state=42: tekrar üretilebilirlik için sabit sayı üreteci
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelinin oluşturulması ve eğitilmesi
# criterion="gini": bölünme kriteri olarak gini indeksi kullanılır
# max_depth=15: ağacın maksimum derinliği (aşırı öğrenmeyi önlemek için)
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=42)
tree_clf.fit(X_train, y_train)  # Modeli eğitim verileriyle eğit

# Model performansının değerlendirilmesi
y_pred = tree_clf.predict(X_test)  # Test verileri için tahmin yap
accuracy = accuracy_score(y_test, y_pred)  # Doğruluk oranını hesapla
confusion_matrix = confusion_matrix(y_test, y_pred)  # Karmaşıklık matrisini hesapla

# Öznitelik önemliliklerinin hesaplanması
feature_importances = tree_clf.feature_importances_  # Her özniteliğin önem derecesi
feature_names = iris.feature_names  # Öznitelik isimleri

# Görselleştirme için hazırlık
n_classes = 3  # Sınıf sayısı
plot_colors = "ryb"  # Grafiklerde kullanılacak renkler (red, yellow, blue)

#####################################################################
# BÖLÜM 2: DIABETES VERİ SETİ İLE REGRESYON MODELİ
#####################################################################
# Diabetes veri seti: Hastaların çeşitli özelliklerine göre hastalık ilerlemesini tahmin
diabetes = load_diabetes()  # Veri setini yükle
X = diabetes.data  # Hasta özellikleri
y = diabetes.target  # Hastalık ilerleme değeri

# Veri setinin bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regresyon modeli oluşturma ve eğitme
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Model performansının değerlendirilmesi
y_pred = tree_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # Ortalama kare hata
print(f"Diabetes Veri Seti ile eğitilen DT regresyon modeli MSE : {mse}")
rmse = np.sqrt(mse)  # Kök ortalama kare hata
print(f"Diabetes Veri Seti ile eğitilen DT regresyon modeli RMSE : {rmse}")

#####################################################################
# BÖLÜM 3: YAPAY VERİ SETİ İLE REGRESYON ÖRNEĞİ
#####################################################################
# Yapay veri oluşturma
X = np.sort(5 * np.random.rand(80, 1), axis=0)  # 0-5 arasında 80 adet rastgele sayı
y = np.sin(X).ravel()  # Sinüs fonksiyonu ile hedef değişken oluşturma
y[::5] += 0.5 * (0.5 - np.random.rand(16))  # Her 5. noktaya gürültü ekleme

# İki farklı derinlikte model oluşturma ve eğitme
regr_1 = DecisionTreeRegressor(max_depth=2)  # Basit model (az detaylı)
regr_2 = DecisionTreeRegressor(max_depth=5)  # Karmaşık model (çok detaylı)
regr_1.fit(X, y)  # Modelleri eğit
regr_2.fit(X, y)

# Test verisi oluşturma ve tahmin yapma
X_test = np.arange(0.0, 5.0, 0.05)[:, np.newaxis]  # 0-5 arası düzenli aralıklı test verileri
y_1 = regr_1.predict(X_test)  # Basit model tahminleri
y_2 = regr_2.predict(X_test)  # Karmaşık model tahminleri

# Sonuçların görselleştirilmesi
plt.figure()
plt.scatter(X, y, label="Gerçek Veri", c="red")  # Gerçek veri noktaları
plt.plot(X_test, y_1, label="Derinlik=2", color="green")  # Basit model tahminleri
plt.plot(X_test, y_2, label="Derinlik=5", color="blue")  # Karmaşık model tahminleri
plt.xlabel("Girdi Değeri")
plt.ylabel("Hedef Değeri")
plt.title("Karar Ağacı Regresyon Modeli Karşılaştırması")
plt.legend()
plt.show()
    
    
    