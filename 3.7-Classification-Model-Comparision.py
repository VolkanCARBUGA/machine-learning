# ================================================================================
# KONU: Makine Öğrenmesi Sınıflandırma Algoritmalarının Karşılaştırmalı Analizi
# 
# AMAÇ:
# Bu kod, yaygın olarak kullanılan 5 farklı sınıflandırma algoritmasının performansını
# 3 farklı yapay veri seti üzerinde test eder ve karşılaştırır.
#
# KULLANILAN ALGORİTMALAR:
# 1. K-En Yakın Komşu (KNN) - Mesafe bazlı sınıflandırma
# 2. Doğrusal SVM - Doğrusal ayırıcı düzlem ile sınıflandırma
# 3. Karar Ağacı - Kural bazlı sınıflandırma
# 4. Rastgele Orman - Topluluk öğrenmesi ile sınıflandırma
# 5. Naive Bayes - Olasılık bazlı sınıflandırma
#
# VERİ SETLERİ:
# 1. make_classification: Doğrusal olarak ayrılabilir sınıflar
# 2. make_moons: Ay şeklinde, doğrusal olmayan sınıflar
# 3. make_circles: İç içe daireler şeklinde, dairesel sınıflar
# ================================================================================

# ==== KÜTÜPHANE İMPORTLARI ====
# Yapay veri setleri oluşturmak için sklearn.datasets modülünden gerekli fonksiyonlar
from sklearn.datasets import make_classification,make_moons,make_circles

# Sınıflandırma algoritmaları için gerekli sınıflar
from sklearn.neighbors import KNeighborsClassifier      # K-En Yakın Komşu algoritması
from sklearn.model_selection import train_test_split    # Veri setini eğitim/test olarak bölme
from sklearn.svm import SVC                            # Destek Vektör Makinesi
from sklearn.tree import DecisionTreeClassifier        # Karar Ağacı
from sklearn.ensemble import RandomForestClassifier    # Rastgele Orman
from sklearn.naive_bayes import GaussianNB             # Gaussian Naive Bayes

# Veri ön işleme ve model değerlendirme araçları
from sklearn.preprocessing import StandardScaler        # Özellik ölçeklendirme
from sklearn.pipeline import make_pipeline             # İşlem zinciri oluşturma
from sklearn.inspection import DecisionBoundaryDisplay # Karar sınırlarını görselleştirme

# Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt                        # Temel görselleştirme aracı
from matplotlib.colors import ListedColormap           # Özel renk haritaları
import numpy as np                                     # Sayısal işlemler için

# ==== VERİ SETİ 1: DOĞRUSAL SINIFLAR ====
# make_classification ile yapay veri seti oluşturma
X, y = make_classification(
    n_samples=1000,                # Toplam örnek sayısı
    n_features=2,                  # Özellik sayısı (2D görselleştirme için)
    n_informative=2,               # Sınıflandırma için kullanılacak özellik sayısı
    n_redundant=0,                 # Gereksiz (tekrarlı) özellik sayısı
    random_state=42,               # Tekrarlanabilirlik için sabit tohum değeri
    n_clusters_per_class=1,        # Her sınıf için küme sayısı
)

# Veri setine gerçekçilik katmak için rastgele gürültü ekleme
# 1.2 çarpanı gürültü miktarını belirler
X += 1.2 * np.random.uniform(size=X.shape)
Xy = (X, y)  # İlk veri setini daha sonra kullanmak üzere saklama

# Oluşturulan ilk veri setinin görselleştirilmesi
plt.scatter(X[:,0], X[:,1],       # X'in ilk iki sütunu için scatter plot
           c=y,                    # Nokta renklerini sınıf etiketlerine göre belirleme
           cmap="viridis")         # Renk haritası seçimi
plt.show()

# ==== VERİ SETİ 2: AY ŞEKLİNDE SINIFLAR ====
# make_moons ile ay şeklinde veri seti oluşturma
X, y = make_moons(
    n_samples=1000,                # Toplam örnek sayısı
    noise=0.2,                     # Eklenecek gürültü miktarı
    random_state=42                # Tekrarlanabilirlik için sabit tohum
)
# Veri setinin görselleştirilmesi
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# ==== VERİ SETİ 3: DAİRESEL SINIFLAR ====
# make_circles ile iç içe daireler şeklinde veri seti oluşturma
X, y = make_circles(
    n_samples=1000,                # Toplam örnek sayısı
    noise=0.1,                     # Eklenecek gürültü miktarı
    random_state=42,               # Tekrarlanabilirlik için sabit tohum
    factor=0.3                     # İç ve dış daire arasındaki oran
)
# Veri setinin görselleştirilmesi
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# ==== TÜM VERİ SETLERİNİN BİR ARADA TUTULMASI ====
# Veri setlerini tek bir liste içinde toplama
datasets = [
    Xy,                                                             # Doğrusal veri seti
    make_moons(n_samples=1000, noise=0.2, random_state=42),        # Ay şeklinde veri
    make_circles(n_samples=1000, noise=0.1, random_state=42, factor=0.3)  # Dairesel veri
]

# ==== VERİ SETLERİNİN KARŞILAŞTIRMALI GÖRSELLEŞTİRİLMESİ ====
# Alt alta üç veri setinin gösterimi
fig = plt.figure(figsize=(6,9))    # 6x9 boyutunda bir figür oluşturma
i = 1  # Alt grafik sayacı

# Her veri seti için döngü
for ds_count, ds in enumerate(datasets):
    X, y = ds  # Veri setinden X ve y değerlerini ayırma
    
    # Her veri seti için farklı renk belirleme
    if ds_count == 0:
        colors = "darkred"         # İlk veri seti için kırmızı
    elif ds_count == 1:
        colors = "darkblue"        # İkinci veri seti için mavi
    else:
        colors = "darkgreen"       # Üçüncü veri seti için yeşil
    
    # Alt grafik oluşturma ve veriyi çizme
    ax = plt.subplot(len(datasets), 1, i)
    ax.scatter(X[:,0], X[:,1],     # Veri noktalarını çizme
              c=y,                  # Sınıf renklerini belirleme
              cmap=plt.cm.coolwarm, # Renk haritası
              edgecolors="black")   # Nokta kenarları
    
    # Grafik başlıkları ve etiketleri
    ax.set_title(f"Dataset {ds_count+1}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    i += 1
plt.show()

# ==== SINIFLANDIRICI MODELLERİN TANIMLANMASI ====
# Model isimleri ve sınıflandırıcıların listeleri
names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(),        # Varsayılan parametrelerle KNN
    SVC(kernel="linear"),          # Doğrusal kernel ile SVM
    DecisionTreeClassifier(),      # Varsayılan parametrelerle Karar Ağacı
    RandomForestClassifier(),      # Varsayılan parametrelerle Rastgele Orman
    GaussianNB(),                  # Gaussian Naive Bayes
]

# ==== KARŞILAŞTIRMALI MODEL ANALİZİ VE GÖRSELLEŞTİRME ====
# Tüm modeller ve veri setleri için sonuçların gösterimi
fig = plt.figure(figsize=(10,10))  # 10x10 boyutunda bir figür
i = 1  # Alt grafik sayacı

# Her veri seti için döngü
for ds_count, ds in enumerate(datasets):
    X, y = ds  # Veri setinden X ve y değerlerini ayırma
    
    # Veri setini eğitim ve test olarak %80-%20 oranında bölme
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,             # Test seti oranı
        random_state=42            # Tekrarlanabilirlik için sabit tohum
    )
    
    # Test verileri için özel renk haritası (kırmızı-yeşil)
    cm_bright = ListedColormap(["#FF0000", "#00FF00"])
    
    # Veri setinin orijinal halini gösterme
    ax = plt.subplot(len(datasets), len(classifiers), i)
    if ds_count == 0:
        ax.set_title("Input Data")
    
    # Eğitim ve test verilerini farklı renklerle gösterme
    ax.scatter(X_train[:,0], X_train[:,1],    # Eğitim verileri
              c=y_train,
              cmap=plt.cm.coolwarm,
              edgecolors="black")
    ax.scatter(X_test[:,0], X_test[:,1],      # Test verileri
              c=y_test,
              cmap=cm_bright,
              edgecolors="black",
              alpha=0.6)           # Yarı saydam gösterim
    i += 1
    
    # Her sınıflandırıcı için döngü
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers)+1, i)
        
        # Veri standardizasyonu ve model eğitimi pipeline'ı
        clf = make_pipeline(
            StandardScaler(),       # Özellikleri standardize etme
            clf                     # Sınıflandırıcı model
        )
        
        # Modeli eğitme ve test seti üzerinde değerlendirme
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)  # Doğruluk skoru hesaplama
        
        # Karar sınırlarını görselleştirme
        DecisionBoundaryDisplay.from_estimator(
            clf, X,
            cmap=plt.cm.RdBu,      # Kırmızı-Mavi renk haritası
            alpha=0.8,             # Yarı saydamlık
            ax=ax,
            eps=0.5                # Izgara hassasiyeti
        )
        
        # Eğitim ve test verilerini gösterme
        ax.scatter(X_train[:,0], X_train[:,1],
                  c=y_train,
                  cmap=plt.cm.coolwarm,
                  edgecolors="black")
        ax.scatter(X_test[:,0], X_test[:,1],
                  c=y_test,
                  cmap=cm_bright,
                  edgecolors="black",
                  alpha=0.6)
        
        # Model ismini ve doğruluk skorunu gösterme
        if ds_count == 0:
            ax.set_title(name)
        ax.text(
            X[:,0].max()-0.25,     # X koordinatı
            X[:,1].min()-0.45,     # Y koordinatı
            str(score)             # Doğruluk skoru
        )
        i += 1

# Son görselleştirmeyi gösterme
plt.show()
    

