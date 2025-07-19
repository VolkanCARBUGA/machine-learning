# Lineer Regresyon (Doğrusal Regresyon) Uygulaması
# Bu dosya, sklearn kütüphanesi kullanarak lineer regresyon algoritmasını gösterir

from sklearn.linear_model import LinearRegression  # Lineer regresyon algoritması
from sklearn.metrics import mean_squared_error,r2_score  # Model performans metrikleri
import numpy as np  # Sayısal işlemler için
import matplotlib.pyplot as plt  # Grafik çizimi için
from sklearn.datasets import load_diabetes  # Diabetes veri seti

#veri oluştur
# Aşağıdaki kod bloğu rastgele veri üretme örneğidir (şu anda kapalı)
# X = np.random.rand(100,1)  # 100 adet rastgele X değeri üret
# y = 3+4*X+np.random.randn(100,1) #y=3+4x+random  # Doğrusal ilişkili Y değerleri üret


# Bu blok rastgele veri ile basit lineer regresyon örneğidir (şu anda kapalı)
# linear_reg = LinearRegression()  # Lineer regresyon modeli oluştur
# linear_reg.fit(X,y)  # Modeli eğit
# plt.figure()  # Yeni grafik penceresi
# plt.scatter(X,y)  # Veri noktalarını scatter plot olarak çiz
# plt.plot(X,linear_reg.predict(X),color="red",alpha=0.7)  # Regresyon çizgisini çiz
# plt.xlabel("X")  # X ekseni etiketi
# plt.ylabel("y")  # Y ekseni etiketi
# plt.title("Linear Regression")  # Grafik başlığı
# a1=linear_reg.coef_[0][0]  # Eğim katsayısını al
# a0=linear_reg.intercept_[0]  # Y kesim noktasını al
# for i in range(100):  # Her veri noktası için
#     y_=a0+a1*X  # Lineer fonksiyonu hesapla
#     plt.plot(X,y_,color="blue",alpha=0.3)  # Mavi çizgi çiz
# plt.show()  # Grafiği göster




# DIABETES VERİ SETİ İLE LİNEER REGRESYON UYGULAMASI
diabetes_X,diabetes_y=load_diabetes(return_X_y=True)  # Diabetes veri setini yükle (X: özellikler, y: hedef değer)
diabetes_X=diabetes_X[:,np.newaxis,2]  # Sadece 3. özelliği al (BMI - Vücut Kitle İndeksi)
diabetes_X_train=diabetes_X[:-20]  # İlk veriyi eğitim için ayır (son 20 hariç)
diabetes_X_test=diabetes_X[-20:]  # Son 20 veriyi test için ayır
diabetes_y_train=diabetes_y[:-20]  # Eğitim için hedef değerler (son 20 hariç)
diabetes_y_test=diabetes_y[-20:]  # Test için hedef değerler (son 20)

linear_reg=LinearRegression()  # Lineer regresyon modeli oluştur
linear_reg.fit(diabetes_X_train,diabetes_y_train)  # Modeli eğitim verisi ile eğit
diabetes_y_pred=linear_reg.predict(diabetes_X_test)  # Test verisi için tahmin yap
mse=mean_squared_error(diabetes_y_test,diabetes_y_pred)  # Ortalama Kare Hata hesapla
r2=r2_score(diabetes_y_test,diabetes_y_pred)  # R² skoru hesapla (modelin açıklama gücü)
print("mse",mse)  # MSE değerini yazdır (düşük olması daha iyi)
print("r2",r2)  # R² değerini yazdır (1'e yakın olması daha iyi)
plt.scatter(diabetes_X_test,diabetes_y_test,color="green")  # Gerçek test verilerini yeşil noktalarla göster
plt.plot(diabetes_X_test,diabetes_y_pred,color="red")  # Tahmin edilen değerleri kırmızı çizgi ile göster
plt.xlabel("X")  # X ekseni etiketi (BMI değerleri)
plt.ylabel("y")  # Y ekseni etiketi (Diabetes ilerlemesi)
plt.title("Linear Regression")  # Grafik başlığı
plt.show()  # Grafiği ekranda göster


"""
LİNEER REGRESYON (DOĞRUSAL REGRESYON) AÇIKLAMASI:
================================================

Bu kod, makine öğrenmesinde en temel algoritmalardan biri olan Lineer Regresyon'u göstermektedir.

LİNEER REGRESYON NEDİR?
- Bağımsız değişken(ler) ile bağımlı değişken arasındaki doğrusal ilişkiyi modellemek için kullanılır
- Formül: y = a*x + b (a: eğim, b: y-kesim noktası)
- Sürekli sayısal değerler tahmin etmek için kullanılır

VERİ SETİ:
- Diabetes (Şeker Hastalığı) veri seti kullanılmıştır
- BMI (Vücut Kitle İndeksi) özelliği ile hastalığın ilerlemesi arasındaki ilişki incelenmiştir

MODEL PERFORMANSI:
- MSE (Mean Squared Error): Ortalama kare hata - ne kadar düşük o kadar iyi
- R² Score: Modelin veriyi açıklama gücü - 1'e yakın olması daha iyi (0-1 arası)

GRAFİK:
- Yeşil noktalar: Gerçek test verileri
- Kırmızı çizgi: Modelin tahminleri
- İyi bir model için noktalar çizgiye yakın olmalı

KULLANIM ALANLARI:
- Ev fiyat tahmini
- Satış tahminleri
- Maaş tahmini
- Hisse senedi fiyat tahminleri
"""






