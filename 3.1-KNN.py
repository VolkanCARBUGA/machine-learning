"""
K-EN YAKIN KOMŞU (KNN) ALGORİTMASI DETAYLI KONU ANLATIMI

1. KNN NEDİR?
---------------
K-En Yakın Komşu (K-Nearest Neighbors), denetimli öğrenme algoritmalarından biridir.
Temel mantığı: "Bana arkadaşını söyle, sana kim olduğunu söyleyeyim" atasözüne benzer.
Yeni bir veri noktası geldiğinde, bu verinin en yakın K komşusuna bakarak tahmin yapar.

2. NASIL ÇALIŞIR?
-----------------
a) Sınıflandırma için:
   - Yeni bir veri noktası gelir
   - Bu noktaya en yakın K tane komşu bulunur
   - Komşuların çoğunluğunun sınıfı, yeni verinin sınıfı olur
   Örnek: K=3 için
   - 2 komşu "kötü huylu", 1 komşu "iyi huylu" ise
   - Yeni veri "kötü huylu" olarak sınıflandırılır

b) Regresyon için:
   - Yeni bir veri noktası gelir
   - En yakın K komşu bulunur
   - Komşuların hedef değerlerinin ortalaması alınır
   - Bu ortalama, yeni verinin tahmini değeri olur

3. ÖNEMLİ PARAMETRELER
----------------------
a) K Değeri:
   - En önemli parametredir
   - K küçük (1,2,3): Model karmaşık, aşırı öğrenme riski
   - K büyük (15,20+): Model basit, yetersiz öğrenme riski
   - Genelde tek sayı seçilir (oy eşitliğini önlemek için)

b) Uzaklık Metriği:
   - Öklid Uzaklığı (en yaygın)
   - Manhattan Uzaklığı
   - Minkowski Uzaklığı
   - Cosine Benzerliği

4. ÖN İŞLEME GEREKSİNİMLERİ
---------------------------
a) Özellik Ölçeklendirme:
   - StandardScaler veya MinMaxScaler kullanılmalı
   - Farklı ölçekteki özellikler uzaklık hesabını bozar
   
b) Boyut İndirgeme:
   - Çok boyutlu verilerde performans düşer
   - PCA veya feature selection kullanılabilir

5. AVANTAJLARI VE DEZAVANTAJLARI
--------------------------------
Avantajlar:
- Basit ve anlaşılır
- Eğitim süresi yok (lazy learning)
- Yeni veri eklemek kolay
- Parametrik olmayan model

Dezavantajlar:
- Tahmin süresi uzun
- Bellek kullanımı yüksek
- Gürültülü veriye duyarlı
- Dengesiz veri setlerinde sorun yaşayabilir

6. KULLANIM ALANLARI
--------------------
- Kredi risk değerlendirmesi
- Görüntü sınıflandırma
- Öneri sistemleri
- Tıbbi teşhis
- Doküman sınıflandırma

7. MODEL OPTİMİZASYONU
----------------------
- K-fold Cross Validation ile K seçimi
- Grid Search ile parametre optimizasyonu
- Feature Selection/Extraction
- Veri ön işleme teknikleri
"""

# BÖLÜM 1: KNN SINIFLANDIRMA
# Gerekli kütüphanelerin import edilmesi
from sklearn.datasets import load_breast_cancer  # Meme kanseri veri seti
from sklearn.neighbors import KNeighborsClassifier  # KNN sınıflandırıcı
from sklearn.metrics import accuracy_score, confusion_matrix  # Model değerlendirme metrikleri
from sklearn.model_selection import train_test_split  # Veri seti bölme
from sklearn.preprocessing import StandardScaler  # Veri ölçeklendirme
import pandas as pd  # Veri manipülasyonu
import matplotlib.pyplot as plt  # Görselleştirme

# (1) Veri Seti İncelemesi
"""
load_breast_cancer(): Sklearn'in içinde gelen meme kanseri veri seti
- 569 örnek
- 30 özellik (hücre özellikleri)
- 2 sınıf (iyi huylu/kötü huylu)
"""
cancer = load_breast_cancer()  # Veri setini yükle
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)  # DataFrame oluştur
df["target"] = cancer.target  # Hedef değişkeni ekle

# (2) Veri Hazırlama ve Model Eğitimi
"""
Veri Hazırlama Adımları:
1. Veriyi özellikler (X) ve hedef (y) olarak ayırma
2. Veriyi eğitim ve test seti olarak bölme (70% eğitim, 30% test)
3. Verileri ölçeklendirme (StandardScaler ile)
   - Özellikler farklı ölçeklerde olduğunda gereklidir
   - Her özelliği ortalama=0, standart sapma=1 olacak şekilde dönüştürür
"""
X = cancer.data  # Özellikler
y = cancer.target  # Hedef değişken

# Veri setini eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veri ölçeklendirme
scaler = StandardScaler()  # Ölçeklendirici oluştur
X_train = scaler.fit_transform(X_train)  # Eğitim verisini ölçeklendir
X_test = scaler.transform(X_test)  # Test verisini ölçeklendir

# KNN modelini oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=3)  # 3 komşulu KNN modeli
knn.fit(X_train, y_train)  # Modeli eğit

# (3) Model Değerlendirme
"""
Model Performans Metrikleri:
1. Doğruluk (Accuracy): Doğru tahmin edilen örneklerin tüm örneklere oranı
2. Karmaşıklık Matrisi (Confusion Matrix):
   - True Positive (TP): Doğru pozitif tahminler
   - True Negative (TN): Doğru negatif tahminler
   - False Positive (FP): Yanlış pozitif tahminler
   - False Negative (FN): Yanlış negatif tahminler
"""
y_pred = knn.predict(X_test)  # Test verisi üzerinde tahmin yap
accuracy = accuracy_score(y_test, y_pred)  # Doğruluk hesapla
print("Dogruluk : ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)  # Karmaşıklık matrisi hesapla
print("Confusion Matrix:\n", conf_matrix)

# (4) K Değeri Optimizasyonu
"""
K Değerinin Model Performansına Etkisi:
- K çok küçük: Model aşırı öğrenme (overfitting) yapabilir
- K çok büyük: Model yetersiz öğrenme (underfitting) yapabilir
- Optimum K değeri veri setine göre değişir
- K'yı belirlemek için farklı değerler deneyip en iyi performansı veren seçilir
"""
accuracy_values = []  # Doğruluk değerlerini tutacak liste
k_values = []  # K değerlerini tutacak liste

# Farklı K değerleri için model performansını test et
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)  # Yeni K değeri ile model oluştur
    knn.fit(X_train, y_train)  # Modeli eğit
    y_pred = knn.predict(X_test)  # Tahmin yap
    accuracy = accuracy_score(y_test, y_pred)  # Doğruluk hesapla
    accuracy_values.append(accuracy)  # Doğruluğu listeye ekle
    k_values.append(k)  # K değerini listeye ekle

    # Sonuçları görselleştir
    plt.figure()
    plt.plot(k_values, accuracy_values, marker="o", linestyle="-")
    plt.title("K Değerine Göre Doğruluk")
    plt.xlabel("K Değeri")
    plt.ylabel("Doğruluk")
    plt.xticks(k_values)
    plt.grid()
    plt.show()

# BÖLÜM 2: KNN REGRESYON
"""
KNN Regresyon Örneği:
- Rastgele veri oluşturulur
- Sinüs fonksiyonu ile hedef değerler belirlenir
- Gürültü eklenir
- İki farklı ağırlık stratejisi ile model oluşturulur:
  1. uniform: Tüm komşulara eşit ağırlık
  2. distance: Uzaklığa bağlı ağırlık
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Rastgele veri oluştur
X = np.sort(5 * np.random.rand(40, 1), axis=0)  # Özellikler
y = np.sin(X).ravel()  # Hedef değerler (sinüs fonksiyonu)

# Gürültü ekle
y[::5] += 1 * (0.5 - np.random.rand(8))

# Tahmin için düzenli aralıklı noktalar oluştur
T = np.linspace(0, 5, 500)[:, np.newaxis]

# İki farklı ağırlık stratejisi için modelleri oluştur ve görselleştir
for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)  # Model oluştur
    y_pred = knn.fit(X, y).predict(T)  # Model eğit ve tahmin yap
    
    # Sonuçları görselleştir
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="green", label="data")  # Gerçek veri noktaları
    plt.plot(T, y_pred, color="blue", label="prediction")  # Tahminler
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights= {}".format(weight))

plt.tight_layout()
plt.show()
