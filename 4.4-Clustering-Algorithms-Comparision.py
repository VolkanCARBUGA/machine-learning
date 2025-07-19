# -*- coding: utf-8 -*-
"""
CLUSTERING ALGORİTMALARI KARŞILAŞTIRMA PROJESİ
==============================================
Bu proje, farklı kümeleme algoritmalarının performansını
çeşitli veri setleri üzerinde görsel olarak karşılaştırır.

Kullanılan Algoritmalar:
- MiniBatchKMeans: Hızlı K-Means varyantı
- SpectralClustering: Spektral kümeleme
- Ward: Hiyerarşik kümeleme (Ward linkage)
- AgglomerativeClustering: Birleştirici kümeleme
- DBSCAN: Yoğunluk tabanlı kümeleme
- Birch: Büyük veri setleri için hiyerarşik kümeleme

Kullanılan Veri Setleri:
- Noisy Circles: İç içe geçmiş daireler
- Noisy Moons: Ay şeklinde veriler
- Blobs: Küresel gruplar
- No Structure: Rastgele dağılmış veriler
"""

# Gerekli kütüphaneleri import et
from sklearn import datasets, cluster  # Scikit-learn'den veri setleri ve kümeleme algoritmaları
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib
from sklearn.preprocessing import StandardScaler  # Veriyi standartlaştırmak için
import numpy as np  # Matematiksel işlemler için numpy

# VERİ SETLERİNİ OLUŞTUR
# ======================

# 1. İç içe geçmiş daireler: n_samples=örnek sayısı, factor=iç dairenin büyüklüğü, noise=gürültü miktarı
noisy_circles = datasets.make_circles(n_samples=1500, factor=0.5, noise=0.05)

# 2. Ay şeklinde veriler: n_samples=örnek sayısı, noise=gürültü miktarı
noisy_moons = datasets.make_moons(n_samples=1500, noise=0.05)

# 3. Küresel gruplar: n_samples=örnek sayısı, centers=küme sayısı, n_features=özellik sayısı
blobs = datasets.make_blobs(n_samples=1500, centers=3, n_features=2, random_state=8)

# 4. Yapısız rastgele veriler: uniform dağılımdan 1500 nokta, 2 boyutlu
no_structure = np.random.rand(1500, 2), None

# KÜMELEME ALGORİTMALARININ İSİMLERİ
clustering_names = ["MiniBatchKMeans", "SpectralClustering", "Ward", "AgglomerativeClustering", "DBSCAN", "Birch"]

# HER KÜME İÇİN RENK PALETİ
colors = np.array(["red", "blue", "green", "yellow", "purple", "orange"])

# TÜM VERİ SETLERİNİ BİR LİSTEDE TOPLA
datasets = [noisy_circles, noisy_moons, blobs, no_structure]

# GÖRSELLEŞTİRME AYARLARI
# Şekil boyutu: (algoritma sayısı * 3) x (veri seti sayısı * 3)
plt.figure(figsize=(len(clustering_names)*3, len(datasets)*3))
i = 1  # Subplot indeksi için sayaç

# HER VERİ SETİ İÇİN DÖNGÜ
for i_dataset, dataset in enumerate(datasets):
    # Veri setini X (özellikler) ve y (etiketler) olarak ayır
    X, y = dataset
    
    # Veriyi standartlaştır: ortalama=0, standart sapma=1
    X = StandardScaler().fit_transform(X)
    
    # KÜMELEME ALGORİTMALARINI TANIMLA
    # ================================
    
    # 1. MiniBatchKMeans: Hızlı K-Means, 2 küme
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    
    # 2. Ward Linkage: Minimum varyans ile hiyerarşik kümeleme
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward")
    
    # 3. Spektral Kümeleme: Graf teorisi tabanlı kümeleme
    spectral = cluster.SpectralClustering(n_clusters=2)
    
    # 4. DBSCAN: Yoğunluk tabanlı kümeleme, eps=komşuluk yarıçapı
    dbscan = cluster.DBSCAN(eps=0.3)
    
    # 5. Average Linkage: Ortalama mesafe ile hiyerarşik kümeleme
    average_linkage = cluster.AgglomerativeClustering(n_clusters=2, linkage="average")
    
    # 6. Birch: Büyük veri setleri için hiyerarşik kümeleme
    birch = cluster.Birch(n_clusters=2)
    
    # Tüm algoritmaları bir listede topla
    clusterings_algorithms = [
        two_means, ward, spectral, dbscan, average_linkage, birch
    ]
    
    # HER ALGORİTMA İÇİN DÖNGÜ
    for name, algo in zip(clustering_names, clusterings_algorithms):
        # Algoritmayı veri üzerinde eğit
        algo.fit(X)
        
        # Algoritmanın tahmin ettiği küme etiketlerini al
        if hasattr(algo, "labels_"):  # DBSCAN gibi algoritmalar labels_ özelliğine sahip
            y_pred = algo.labels_.astype(int)
        else:  # Diğer algoritmalar predict() metoduna sahip
            y_pred = algo.predict(X)
        
        # DBSCAN'de -1 değeri gürültü noktalarını temsil eder
        # Bu noktaları son renge (turuncu) ata, diğerlerini de renk dizisine göre döngüye sok
        y_pred_corrected = np.where(y_pred == -1, len(colors)-1, y_pred % len(colors))
        
        # Alt grafik oluştur: (satır sayısı, sütun sayısı, indeks)
        plt.subplot(len(datasets), len(clustering_names), i)
        
        # Sadece ilk satırda algoritma isimlerini başlık olarak yaz
        if i_dataset == 0:
            plt.title(name, size=10)
        
        # Her noktanın rengini tahmin edilen küme etiketine göre belirle
        point_colors = colors[y_pred_corrected]
        
        # Noktaları scatter plot ile çiz: x koordinatı, y koordinatı, renk, nokta boyutu
        plt.scatter(X[:, 0], X[:, 1], c=point_colors, s=10)
        
        # Eksen etiketlerini gizle (daha temiz görünüm için)
        plt.xticks([])
        plt.yticks([])
        
        # Bir sonraki subplot için sayacı artır
        i += 1

# Subplotlar arasındaki boşlukları optimize et
plt.tight_layout()

# Grafikleri ekranda göster
plt.show()

"""
SONUÇ YORUMU:
=============
Bu görselleştirme, farklı clustering algoritmalarının farklı veri yapılarında
nasıl performans gösterdiğini ortaya koyar:

1. Circles (Daireler): Spektral kümeleme en iyi sonucu verir
2. Moons (Aylar): Spektral kümeleme ve DBSCAN başarılı
3. Blobs (Kümeler): Çoğu algoritma başarılı
4. Random (Rastgele): Hiçbir algoritma anlamlı küme bulamaz

Bu analiz, algoritma seçiminde veri yapısının önemini gösterir.
"""
    
    








