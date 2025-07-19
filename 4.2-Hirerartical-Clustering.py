# Gerekli kütüphaneleri import ediyoruz
from sklearn.datasets import make_blobs  # Yapay kümeleme verisi oluşturmak için
from sklearn.cluster import AgglomerativeClustering  # Hierarchical clustering algoritması için
from scipy.cluster.hierarchy import dendrogram,linkage  # Dendrogram çizmek ve linkage hesaplamak için
import matplotlib.pyplot as plt  # Grafik çizmek için

# 300 adet veri noktası ile 4 adet merkez etrafında yapay veri seti oluşturuyoruz
X,_=make_blobs(n_samples=300,centers=4,random_state=42,cluster_std=0.6)

# İlk grafiği çiziyoruz - orijinal veri noktalarını görmek için
plt.figure()
plt.scatter(X[:,0],X[:,1],c="blue",alpha=0.5)  # X'in tüm satırları, 0. ve 1. sütunları (2D koordinatlar)
plt.xlabel("Feature 1")  # X ekseni etiketi
plt.ylabel("Feature 2")  # Y ekseni etiketi
plt.title("K-Means Clustering")  # Başlık (yanlış - Hierarchical Clustering olmalı)

# Farklı linkage metodlarını tanımlıyoruz
linkage_methods=["ward","complete","average","single"]

# 2x4'lük subplot düzeni için yeni figure oluşturuyoruz
plt.figure()

# Her linkage metodu için döngü başlatıyoruz
for i,method in enumerate(linkage_methods):  # i: index, method: mevcut linkage metodu
    # Agglomerative Clustering modelini oluşturuyoruz
    model=AgglomerativeClustering(n_clusters=4,linkage=method)  # 4 küme, belirtilen linkage metoduyla
    cluster_labels=model.fit_predict(X)  # Modeli eğitip tahmin sonuçlarını alıyoruz
   
    # Üst satırda dendrogram çiziyoruz (1x4 alt grafik)
    plt.subplot(2,4,i+1)  # 2 satır, 4 sütun, i+1. pozisyon
    plt.title(f"{method.capitalize()} Linkage: Dendogram")  # Dendrogram başlığı
    dendrogram(linkage(X,method=method),no_labels=True)  # Dendrogram çizimi, etiket yok
    plt.xlabel("Veri Noktaları")  # X ekseni etiketi
    plt.ylabel("Uzaklık")  # Y ekseni etiketi - kümeleme uzaklığını gösterir
    
    # Alt satırda kümeleme sonuçlarını çiziyoruz (2x4 alt grafik)
    plt.subplot(2,4,i+5)  # 2 satır, 4 sütun, alt satırın i+1. pozisyonu (i+5)
    plt.scatter(X[:,0],X[:,1],c=cluster_labels,cmap="viridis")  # Küme etiketlerine göre renklendirilmiş noktalar
    plt.title(f"{method.capitalize()} Linkage: Kümeleme")  # Kümeleme sonucu başlığı
    plt.xlabel("X")  # X ekseni etiketi
    plt.ylabel("Y")  # Y ekseni etiketi

plt.show()  # Tüm grafikleri ekranda göster

"""
HİYERARŞİK KÜMELEME (HIERARCHICAL CLUSTERING) KONUSU AÇIKLAMASI:

Hiyerarşik kümeleme, veri noktalarını ağaç benzeri bir yapıda organize eden bir kümeleme yöntemidir.
İki ana türü vardır:

1. AGGLOMERATİVE (Birleştirici) - Aşağıdan yukarıya:
   - Her veri noktasını ayrı bir küme olarak başlatır
   - En yakın kümeleri adım adım birleştirir
   - Bu kodda kullanılan yöntem budur

2. DIVISIVE (Bölücü) - Yukarıdan aşağıya:
   - Tüm veriyi tek kümede başlatır
   - Adım adım böler

LINKAGE METODLARİ (Küme Mesafe Hesaplama Yöntemleri):

1. WARD: Küme içi varyansı minimize eder (en popüler)
2. COMPLETE: İki kümedeki en uzak noktalar arası mesafe
3. AVERAGE: İki kümedeki tüm noktaların ortalama mesafesi
4. SINGLE: İki kümedeki en yakın noktalar arası mesafe

DENDROGRAM:
- Kümeleme sürecini ağaç diyagramı olarak gösterir
- Y ekseni: Birleştirme mesafesini gösterir
- Yatay çizgilerin yüksekliği, o seviyedeki birleştirme maliyetini belirtir
- Optimal küme sayısını belirlemek için kullanılabilir

AVANTAJLARI:
- Önceden küme sayısı belirtmek zorunda değilsiniz
- Dendrogram görsel analiz imkanı sağlar
- Farklı şekillerdeki kümeleri bulabilir

DEZAVANTAJLARI:
- Büyük veri setleri için yavaş (O(n³) karmaşıklık)
- Gürültüye karşı hassas
- Yerel optimuma takılabilir
"""
