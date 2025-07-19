# Scikit-learn kütüphanesinden veri seti indirmek için fetch_openml fonksiyonunu içe aktarıyoruz
from sklearn.datasets import fetch_openml
# Boyut indirgeme için TSNE (t-Distributed Stochastic Neighbor Embedding) algoritmasını içe aktarıyoruz
from sklearn.manifold import TSNE
# Grafik çizimi için matplotlib kütüphanesini içe aktarıyoruz
import matplotlib.pyplot as plt

# MNIST el yazısı rakam veri setini yüklüyoruz (28x28 piksel = 784 özellik)
# version=1: belirli bir versiyonu kullan, as_frame=False: pandas DataFrame yerine numpy array döndür
mnist=fetch_openml("mnist_784",version=1,as_frame=False)

# Veri setindeki özellik verilerini (piksel değerleri) X değişkenine atıyoruz
X=mnist.data

# Veri setindeki hedef etiketleri (rakam sınıfları 0-9) y değişkenine atıyoruz
y=mnist.target

# TSNE modelini oluşturuyoruz
# n_components=2: 2 boyutlu düzleme indirgemek istiyoruz (görselleştirme için)
# random_state=42: tekrarlanabilir sonuçlar için rastgele sayı üretecini sabitleiyoruz
tsne=TSNE(n_components=2,random_state=42)

# Yüksek boyutlu veriyi (784 boyut) 2 boyutlu uzaya dönüştürüyoruz
# fit_transform: modeli eğitir ve aynı anda dönüştürme işlemini yapar
X_tsne=tsne.fit_transform(X)

# 10x10 inç boyutunda bir grafik figürü oluşturuyoruz
plt.figure(figsize=(10,10))

# Scatter plot (dağılım grafiği) çiziyoruz
# X_tsne[:,0]: TSNE'nin 1. boyutu (x ekseni)
# X_tsne[:,1]: TSNE'nin 2. boyutu (y ekseni)  
# c=y: noktaların renklerini hedef etiketlere göre belirliyoruz
# cmap="tab10": 10 farklı renk paleti kullanıyoruz (0-9 rakamları için)
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y,cmap="tab10")

# Renk çubuğu ekliyoruz ve hangi rengin hangi rakamı temsil ettiğini gösteriyoruz
plt.colorbar(label="Digit")

# X ekseni etiketi
plt.xlabel("TSNE 1")

# Y ekseni etiketi
plt.ylabel("TSNE 2")

# Grafik başlığı
plt.title("TSNE of MNIST Dataset")

# Grafiği ekranda gösteriyoruz
plt.show()

"""
TSNE (t-Distributed Stochastic Neighbor Embedding) HAKKINDA:

TSNE, yüksek boyutlu verileri düşük boyutlu uzaya (genellikle 2D veya 3D) 
dönüştürmek için kullanılan bir boyut indirgeme ve görselleştirme tekniğidir.

TEMEL ÖZELLİKLERİ:
- Yüksek boyutlu verideki benzer noktaların düşük boyutlu uzayda da yakın kalmasını sağlar
- Özellikle veri görselleştirme için çok etkilidir
- Non-linear (doğrusal olmayan) boyut indirgeme yapar
- Kümeleme (clustering) yapısını korumaya odaklanır

NASIL ÇALIŞIR:
1. Yüksek boyutlu uzayda noktalar arası benzerlikleri hesaplar
2. Düşük boyutlu uzayda rastgele yerleştirme yapar  
3. İki uzaydaki benzerlik dağılımlarını mümkün olduğunca benzer hale getirmeye çalışır
4. Gradient descent ile optimizasyon yapar

AVANTAJLARI:
- Kompleks veri yapılarını görselleştirmede çok başarılı
- Kümeleri açık şekilde ayırabilir
- Non-linear ilişkileri yakalayabilir

DİKKAT EDİLMESİ GEREKENLER:
- Hesaplama açısından pahalı (büyük veri setlerinde yavaş)
- Hiperparametrelere duyarlı (perplexity, learning_rate vb.)
- Farklı çalıştırmalarda farklı sonuçlar verebilir
- Uzaklık bilgisini her zaman koruamayabilir

BU ÖRNEKTE:
MNIST veri setindeki 784 boyutlu el yazısı rakam görüntülerini 2 boyutlu
düzleme indirgedik. Sonuç olarak benzer rakamların birbirine yakın 
kümelerde toplandığını görebiliriz.
"""