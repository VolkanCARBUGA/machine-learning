# ============= LOJİSTİK REGRESYON =============
# Lojistik Regresyon, sınıflandırma problemleri için kullanılan bir makine öğrenmesi algoritmasıdır.
# Bu algoritma, verilen özelliklere göre bir olayın gerçekleşme olasılığını tahmin eder.
# Örneğin: Kalp hastalığı tahmini, spam e-posta tespiti, kredi risk değerlendirmesi gibi.
#
# Çalışma Prensibi:
# 1. Doğrusal bir model oluşturur (wx + b)
# 2. Sigmoid fonksiyonu kullanarak çıktıyı 0-1 arasına sıkıştırır
# 3. 0.5'ten büyük değerler için 1, küçük değerler için 0 tahminini yapar
#
# Avantajları:
# - Basit ve yorumlanabilir bir modeldir
# - Hızlı eğitim ve tahmin süresi
# - Olasılık değerleri üretir
# - Az bellek kullanır
#
# Dezavantajları:
# - Doğrusal olmayan ilişkileri modelleyemez
# - Çok sayıda özellik varsa performansı düşebilir
# - Dengesiz veri setlerinde zorlanabilir
#
# Bu örnekte:
# - UCI veri tabanından kalp hastalığı veri seti kullanılmıştır
# - Hastanın özelliklerine bakarak kalp hastalığı riski tahmin edilmektedir
# - L2 regularizasyon kullanılarak aşırı öğrenme (overfitting) önlenmektedir
# - Veri seti %90 eğitim, %10 test olarak bölünmüştür

from ucimlrepo import fetch_ucirepo  # UCI Machine Learning Repository'den veri setlerini çekmek için kullanılan kütüphane
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test olarak bölmek için kullanılan fonksiyon
from sklearn.linear_model import LogisticRegression  # Lojistik Regresyon modelini import ediyoruz

import pandas as pd  # Veri manipülasyonu için pandas kütüphanesi
heart_disease=fetch_ucirepo(name="Heart Disease")  # UCI'dan kalp hastalığı veri setini çekiyoruz
df=pd.DataFrame(heart_disease.data.features)  # Veri setinin özelliklerini DataFrame'e dönüştürüyoruz
df["target"]=heart_disease.data.targets  # Hedef değişkeni (hasta/sağlıklı) DataFrame'e ekliyoruz

#drop missing values
if df.isna().any().any():  # Veri setinde eksik değer kontrolü yapıyoruz
    df.dropna(inplace=True)  # Eksik değerleri içeren satırları kaldırıyoruz
    print("NaN değerleri kaldırıldı")

X=df.drop(columns=["target"],axis=1).values  # Bağımsız değişkenleri (özellikleri) X matrisine atıyoruz
y=df.target.values  # Hedef değişkeni y vektörüne atıyoruz

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)  # Veriyi %90 eğitim, %10 test olarak bölüyoruz

# Lojistik Regresyon modelini oluşturuyoruz:
# penalty="l2": L2 regularizasyon kullanarak overfitting'i önlüyoruz
# C=1.0: Regularizasyon şiddeti (düşük değerler daha güçlü regularizasyon)
# solver="lbfgs": Optimizasyon algoritması
# max_iter=100: Maksimum iterasyon sayısı
log_reg=LogisticRegression(penalty="l2",C=1.0,solver="lbfgs",max_iter=100)

accuracy=log_reg.fit(X_train,y_train).score(X_test,y_test)  # Modeli eğitip test verisi üzerinde doğruluk oranını hesaplıyoruz
print(f"Logistic Regression Accuracy: {accuracy}")  # Modelin doğruluk oranını yazdırıyoruz



