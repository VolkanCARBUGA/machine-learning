# Gerekli kütüphaneleri içe aktarıyoruz
from sklearn.datasets import fetch_openml,load_iris  # Veri setlerini yüklemek için
from sklearn.decomposition import PCA  # Temel Bileşen Analizi için
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Doğrusal Diskriminant Analizi için
import matplotlib.pyplot as plt  # Görselleştirme için

# MNIST veri seti ile LDA örneği (yorum satırında)
# mnist=fetch_openml("mnist_784",version=1,as_frame=False)  # MNIST veri setini yükle

# X=mnist.data  # Özellik matrisi (28x28 piksel değerleri)
# y=mnist.target.astype(int)  # Hedef değişken (0-9 rakamları), string'den int'e çevir
# lda=LinearDiscriminantAnalysis(n_components=2)  # 2 bileşenli LDA modeli oluştur
# X_lda=lda.fit_transform(X,y)  # LDA'yı eğit ve veriyi dönüştür
# plt.figure()  # Yeni bir figür oluştur
# plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap="tab10")  # LDA bileşenlerini scatter plot ile çiz
# plt.xlabel("LDA 1")  # X ekseni etiketi
# plt.ylabel("LDA 2")  # Y ekseni etiketi
# plt.title("LDA of MNIST Dataset")  # Grafik başlığı
# plt.colorbar(label="Digit")  # Renk çubuğu ekle
# plt.show()  # Grafiği göster

# Iris veri seti ile PCA ve LDA karşılaştırması
iris=load_iris()  # Iris veri setini yükle (150 örnek, 4 özellik, 3 sınıf)
X=iris.data  # Özellik matrisi (sepal/petal uzunluk ve genişlik)
y=iris.target  # Hedef değişken (0: setosa, 1: versicolor, 2: virginica)
target_names=iris.target_names  # Sınıf isimleri ['setosa', 'versicolor', 'virginica']

# PCA (Temel Bileşen Analizi) uygulaması
pca=PCA(n_components=2)  # 2 bileşenli PCA modeli oluştur
X_pca=pca.fit_transform(X)  # PCA'yı eğit ve veriyi 2 boyuta indir

# LDA (Doğrusal Diskriminant Analizi) uygulaması  
lda=LinearDiscriminantAnalysis(n_components=2)  # 2 bileşenli LDA modeli oluştur
X_lda=lda.fit_transform(X,y)  # LDA'yı eğit ve veriyi dönüştür (y parametresi gerekli)

# Görselleştirme için renkler tanımla
colors=["navy","turquoise","darkorange"]  # Her sınıf için farklı renk

# PCA sonuçlarını görselleştir
plt.figure(figsize=(12,5))  # Figür boyutunu ayarla

# İlk subplot: PCA sonuçları
plt.subplot(1,2,1)  # 1 satır, 2 sütun, 1. grafik
for color,i,target_name in zip(colors,[0,1,2],target_names):  # Her sınıf için döngü
    plt.scatter(X_pca[y==i,0],X_pca[y==i,1],color=color,alpha=0.8,label=target_name)  # Sınıfa ait noktaları çiz
plt.legend(loc="best",shadow=False,scatterpoints=1)  # Lejand ekle
plt.xlabel("İlk Temel Bileşen")  # X ekseni etiketi
plt.ylabel("İkinci Temel Bileşen")  # Y ekseni etiketi
plt.title("Iris Veri Seti - PCA")  # Grafik başlığı

# İkinci subplot: LDA sonuçları
plt.subplot(1,2,2)  # 1 satır, 2 sütun, 2. grafik
for color,i,target_name in zip(colors,[0,1,2],target_names):  # Her sınıf için döngü
    plt.scatter(X_lda[y==i,0],X_lda[y==i,1],color=color,alpha=0.8,label=target_name)  # Sınıfa ait noktaları çiz
plt.legend(loc="best",shadow=False,scatterpoints=1)  # Lejand ekle
plt.xlabel("İlk LDA Bileşeni")  # X ekseni etiketi
plt.ylabel("İkinci LDA Bileşeni")  # Y ekseni etiketi
plt.title("Iris Veri Seti - LDA")  # Grafik başlığı

plt.tight_layout()  # Alt grafikleri düzenle
plt.show()  # Grafikleri göster

# Açıklanan varyans oranlarını yazdır
print(f"PCA ile açıklanan varyans oranı: {pca.explained_variance_ratio_}")  # PCA'nın açıkladığı varyans
print(f"PCA ile toplam açıklanan varyans: {sum(pca.explained_variance_ratio_):.3f}")  # Toplam açıklanan varyans

"""
=== LİNEAR DİSCRİMİNANT ANALİZİ (LDA) HAKKINDA DETAYLI AÇIKLAMA ===

LDA (Linear Discriminant Analysis), hem boyut azaltma hem de sınıflandırma için kullanılan 
denetimli bir öğrenme algoritmasıdır.

TEMEL PRENSİPLER:
1. LDA, sınıflar arası varyansı maksimize ederken sınıf içi varyansı minimize etmeye çalışır
2. PCA'dan farklı olarak, LDA sınıf bilgisini (y değişkenini) kullanır
3. Maksimum (n_classes - 1) boyuta indirgenebilir (Iris için max 2 boyut)

PCA vs LDA FARKLARI:
- PCA: Denetimsiz, maksimum varyansı korur, sınıf bilgisi kullanmaz
- LDA: Denetimli, sınıflar arası ayrımı maksimize eder, sınıf bilgisi gerekir

AVANTAJLARI:
1. Sınıflandırma performansını artırır
2. Boyut azaltma sağlar
3. Hesaplama maliyetini düşürür
4. Overfitting'i azaltır

DEZAVANTAJLARI:
1. Doğrusal ilişki varsayımı yapar
2. Normal dağılım varsayımı gerekir
3. Sınıf sayısından az boyut üretebilir
4. Outlier'lara hassastır

KULLANIM ALANLARI:
- Yüz tanıma sistemleri
- Metin sınıflandırma
- Tıbbi tanı sistemleri
- Görüntü işleme uygulamaları
- Finansal risk analizi

Bu örnekte Iris veri setinde LDA, sınıfları PCA'ya göre daha net ayrıştırmaktadır.
"""






