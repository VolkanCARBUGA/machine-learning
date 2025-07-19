# Gerekli kütüphaneleri içe aktarıyoruz
from sklearn.datasets import load_iris  # Iris veri setini yüklemek için
from sklearn.decomposition import PCA   # Principal Component Analysis (Temel Bileşen Analizi) için
import matplotlib.pyplot as plt        # Grafik çizimi için

# Iris veri setini yüklüyoruz
iris = load_iris()  # Sklearn'den hazır iris veri setini getiriyoruz

# Veri setinden özellik matrisi (X) ve hedef değişkeni (y) ayırıyoruz
X = iris.data      # Özellik matrisi (4 boyutlu: sepal length, sepal width, petal length, petal width)
y = iris.target    # Hedef değişken (0: setosa, 1: versicolor, 2: virginica)

# PCA nesnesini oluşturuyoruz - 2 bileşene indirgeyelim
pca = PCA(n_components=2)  # 4 boyutlu veriyi 2 boyuta indirgemek için PCA tanımlıyoruz

# PCA'yı veriye uyguluyoruz ve dönüştürülmüş veriyi alıyoruz
X_pca = pca.fit_transform(X)  # Veriyi 2 boyuta indirgiyoruz

# 2D PCA sonuçlarını görselleştiriyoruz
plt.figure()  # Yeni bir figür oluşturuyoruz

# Her iris türü için ayrı ayrı scatter plot çiziyoruz
for i in range(len(iris.target_names)):  # 3 iris türü için döngü (setosa, versicolor, virginica)
    # Her tür için farklı renkte nokta çizimi yapıyoruz
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=iris.target_names[i])

# Grafik etiketlerini ve başlığını ekliyoruz
plt.xlabel("PCA 1")                    # X ekseni etiketi - Birinci temel bileşen
plt.ylabel("PCA 2")                    # Y ekseni etiketi - İkinci temel bileşen
plt.title("PCA of Iris Dataset")       # Grafik başlığı
plt.legend()                           # Renk açıklaması (legend) ekliyoruz
#plt.show()                            # Grafiği göster (yorum satırı yapılmış)

#%% Jupyter notebook hücre ayırıcısı

# Şimdi 3D PCA görselleştirmesi yapıyoruz
pca = PCA(n_components=3)  # Bu sefer 3 bileşene indirgemek için yeni PCA tanımlıyoruz
X_pca = pca.fit_transform(X)  # Veriyi 3 boyuta indirgiyoruz

# 3D grafik için figür oluşturuyoruz
fig = plt.figure(1, figsize=(8, 6))  # 8x6 boyutunda figür oluşturuyoruz

# 3D subplot ekliyoruz ve görüş açısını ayarlıyoruz
ax = fig.add_subplot(111, projection="3d", elev=150, azim=120)  # 3D ekseni, yükseklik 150°, azimut 120°

# 3D scatter plot çiziyoruz
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, s=40)  # X, Y, Z koordinatları, renk hedef değişkene göre

# 3D grafik için eksen etiketlerini ekliyoruz
ax.set_xlabel("PCA 1")   # X ekseni - Birinci temel bileşen
ax.set_ylabel("PCA 2")   # Y ekseni - İkinci temel bileşen  
ax.set_zlabel("PCA 3")   # Z ekseni - Üçüncü temel bileşen
ax.set_title("PCA of Iris Dataset")  # Grafik başlığı
ax.legend()              # Renk açıklaması ekliyoruz
plt.show()               # Grafiği ekranda gösteriyoruz

"""
PCA (Principal Component Analysis - Temel Bileşen Analizi) Nedir?

PCA, çok boyutlu veri setlerinin boyutunu azaltmak için kullanılan bir makine öğrenmesi tekniğidir.
Ana amaçları:

1. BOYUT AZALTMA: Yüksek boyutlu veriyi daha az boyutta temsil etmek
   - Bu örnekte 4 boyutlu iris verisini 2D ve 3D'ye indirgedik

2. VERİ GÖRSELLEŞTİRME: Yüksek boyutlu veriyi görselleştirilebilir hale getirmek
   - 4D veriyi görselleştiremeyiz ama 2D/3D'yi görebiliriz

3. GÜRÜLTÜyÜ AZALTMA: Önemli bilgiyi koruyarak gereksiz değişkenleri elimine etmek

4. HESAPLAMA HIZI: Daha az boyutla daha hızlı işlem yapabilmek

NASIL ÇALIŞIR?
- Verideki en fazla varyansa sahip yönleri (temel bileşenleri) bulur
- Bu yönler boyunca veriyi yeniden organize eder
- En önemli bileşenleri seçerek boyutu azaltır

Bu örnekte iris çiçeğinin 4 özelliğini (sepal/petal uzunluk/genişlik) 
2-3 temel bileşene indirgedik ve türleri başarıyla ayırt edebildik.
"""




