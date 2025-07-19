from sklearn.datasets import make_blobs  # Yapay veri kümesi oluşturmak için sklearn'den make_blobs fonksiyonu import ediliyor
from sklearn.cluster import KMeans  # K-Means kümeleme algoritması için KMeans sınıfı import ediliyor
import matplotlib.pyplot as plt  # Görselleştirme için matplotlib kütüphanesi import ediliyor

# 300 örnek, 4 merkez, rastgele durum 42, küme standart sapması 0.1 ile yapay veri kümesi oluşturuluyor
X,_=make_blobs(n_samples=300,centers=4,random_state=42,cluster_std=0.1)

plt.figure()  # Yeni bir grafik figürü oluşturuluyor
plt.scatter(X[:,0],X[:,1],c="blue",alpha=0.5)  # Veri noktalarını mavi renkte, %50 şeffaflık ile scatter plot olarak çiziliyor
plt.xlabel("Feature 1")  # X ekseni etiketi belirleniyor
plt.ylabel("Feature 2")  # Y ekseni etiketi belirleniyor
plt.title("K-Means Clustering")  # Grafik başlığı belirleniyor

kmeans=KMeans(n_clusters=4)  # 4 küme için K-Means algoritması nesnesi oluşturuluyor
kmeans.fit(X)  # K-Means algoritması veri üzerinde eğitiliyor ve kümeler belirleniyor

plt.figure()  # Kümelenmiş veri için yeni bir grafik figürü oluşturuluyor
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap="viridis")  # Veri noktaları küme etiketlerine göre renklendirilip çiziliyor
centers=kmeans.cluster_centers_  # Küme merkezleri alınıyor
plt.scatter(centers[:,0],centers[:,1],c="red",marker="x")  # Küme merkezleri kırmızı X işareti ile işaretleniyor
plt.title("K-Means Clustering")  # Grafik başlığı belirleniyor
plt.show()  # Tüm grafikler ekranda gösteriliyor

"""
K-MEANS KÜMELEMESİ HAKKINDA GENEL BİLGİ:

K-Means, gözetimsiz öğrenme algoritmalarından biridir ve veri noktalarını k sayıda kümeye ayırmaya yarar.
Bu algoritma şu adımları takip eder:

1. BAŞLANGIÇ: K sayıda küme merkezi rastgele belirlenir
2. ATAMA: Her veri noktası en yakın küme merkezine atanır
3. GÜNCELLEME: Küme merkezleri, o kümedeki noktaların ortalaması alınarak güncellenir
4. TEKRAR: 2. ve 3. adımlar küme merkezleri değişmeyene kadar tekrarlanır

AVANTAJLARI:
- Basit ve hızlı bir algoritmadır
- Büyük veri setlerinde etkili çalışır
- Küre şeklindeki kümeler için idealdir

DEZAVANTAJLARI:
- K değerinin önceden bilinmesi gerekir
- Başlangıç merkezlerine duyarlıdır
- Farklı boyutlardaki kümelerde başarısız olabilir
- Gürültüye ve aykırı değerlere duyarlıdır

BU ÖRNEKTE:
- 300 veri noktası 4 farklı merkezde üretildi
- K-Means algoritması bu 4 kümeyi başarıyla tanımladı
- İlk grafik orijinal veriyi, ikinci grafik kümelenmiş veriyi gösteriyor
- Kırmızı X işaretleri algoritmanın bulduğu küme merkezlerini gösteriyor
"""

# K-Means algoritması, veri madenciliği ve makine öğrenmesi alanında yaygın olarak kullanılır.