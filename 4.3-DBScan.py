# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Algoritması Uygulaması

# Gerekli kütüphaneleri içe aktarıyoruz
from sklearn.datasets import make_circles  # Çember şeklinde veri seti oluşturmak için
import matplotlib.pyplot as plt            # Grafik çizimi için
from sklearn.cluster import DBSCAN         # DBSCAN kümeleme algoritması için

# Çember şeklinde veri seti oluşturuyoruz
# n_samples=1000: 1000 veri noktası oluştur
# factor=0.5: İç ve dış çemberin oranı (0.5 = iç çember dış çemberin yarısı kadar)
# noise=0.08: Verilere eklenecek gürültü miktarı
# random_state=42: Rastgelelik için sabit tohum değeri (tekrarlanabilir sonuçlar için)
X, _ = make_circles(n_samples=1000, factor=0.5, noise=0.08, random_state=42)

# İlk grafik penceresini oluştur
plt.figure()
# Orijinal veriyi mavi nokta bulutları halinde göster
# X[:,0]: Birinci özellik (x koordinatı)
# X[:,1]: İkinci özellik (y koordinatı)  
# c="blue": Renk mavi
# alpha=0.5: Şeffaflık değeri (0-1 arası, 0.5 = yarı şeffaf)
plt.scatter(X[:,0], X[:,1], c="blue", alpha=0.5)

# DBSCAN modelini oluştur
# eps=0.1: Epsilon parametresi - bir noktanın komşuluğunun yarıçapı
# min_samples=5: Bir küme oluşturmak için gereken minimum nokta sayısı
model = DBSCAN(eps=0.1, min_samples=5)

# Modeli veriye uygula ve küme etiketlerini al
# fit_predict: Hem modeli eğitir hem de tahmin yapar
# cluster_labels: Her veri noktası için küme etiketi (-1 = gürültü)
cluster_labels = model.fit_predict(X)

# İkinci grafik penceresini oluştur
plt.figure()
# Kümelenmiş veriyi farklı renklerle göster
# c=cluster_labels: Her küme farklı renkte olacak
# cmap="viridis": Renk paleti (sarıdan mora geçiş)
plt.scatter(X[:,0], X[:,1], c=cluster_labels, cmap="viridis")
# X ekseni etiketi
plt.xlabel("Feature 1")
# Y ekseni etiketi  
plt.ylabel("Feature 2")
# Grafik başlığı
plt.title("DBScan Clustering")
# Grafikleri ekranda göster
plt.show()

"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) ALGORITMASI AÇIKLAMASI:

DBSCAN, yoğunluk tabanlı bir kümeleme algoritmasıdır. Ana özellikleri:

1. TEMEL KAVRAMLAR:
   - Eps (ε): Bir noktanın komşuluğunun yarıçapı
   - MinPts: Bir küme oluşturmak için gereken minimum nokta sayısı
   - Core Point (Çekirdek Nokta): Eps yarıçapı içinde en az MinPts kadar komşusu olan nokta
   - Border Point (Sınır Nokta): Çekirdek nokta değil ama çekirdek noktanın komşuluğunda
   - Noise Point (Gürültü Noktası): Ne çekirdek ne de sınır noktası (-1 etiketi)

2. ALGORİTMA ÇALIŞMA PRENSİBİ:
   - Her veri noktası için eps yarıçapındaki komşuları bulur
   - Yeterli komşusu olan noktalar çekirdek nokta olur
   - Çekirdek noktalardan başlayarak kümeleri genişletir
   - Yoğunluk bağlantılı noktalar aynı kümeye dahil edilir

3. AVANTAJLARI:
   - Küme sayısını önceden belirtmeye gerek yok
   - Farklı şekillerde kümeler bulabilir (dairesel olmayan)
   - Gürültü noktalarını tespit edebilir
   - Yoğunluk farklılıklarını iyi handle eder

4. DEZAVANTAJLARI:
   - Eps ve MinPts parametrelerinin doğru seçimi kritik
   - Farklı yoğunluklardaki kümelerde zorluk yaşayabilir
   - Yüksek boyutlu verilerde performans düşebilir

5. KULLANIM ALANLARI:
   - Anomali tespiti
   - Görüntü işleme
   - Müşteri segmentasyonu  
   - Jeolokasyon analizi
   - Sosyal ağ analizi

Bu örnekte çember şeklindeki veriler üzerinde DBSCAN uygulandı ve algoritma
iç ve dış çemberleri başarıyla farklı kümeler olarak ayırt etti.
"""
