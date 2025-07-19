# REGULARİZASYON DURUM ÇALIŞMASI
# Bu çalışmada kalp hastalığı veri seti üzerinde Logistic Regression modelini kullanarak
# L1 ve L2 regularization tekniklerini karşılaştıracağız
# Regularization, modelin overfitting (aşırı öğrenme) yapmasını önlemek için kullanılır

# GEREKLİ KÜTÜPHANELERİN İMPORT EDİLMESİ
import numpy as np                              # Sayısal hesaplamalar için numpy kütüphanesi
import pandas as pd                             # Veri işleme ve CSV dosyası okuma için pandas
import matplotlib.pyplot as plt                 # Grafik çizimi için matplotlib
import seaborn as sns                           # Gelişmiş görselleştirme için seaborn

# MAKİNE ÖĞRENMESİ KÜTÜPHANELERİ
from sklearn.linear_model import LogisticRegression    # Logistic Regression modeli
from sklearn.preprocessing import StandardScaler       # Verileri standardize etmek için
from sklearn.model_selection import train_test_split, GridSearchCV  # Veriyi bölme ve hiperparametre optimizasyonu
from sklearn.metrics import accuracy_score, roc_curve             # Model performans metrikleri

# VERİ SETİNİN OKUNMASI
df = pd.read_csv("8_3_RegularizationDurumCalismasi_dataset.csv")  # Kalp hastalığı veri setini okuyoruz

# VERİ SETİNİN İLK 5 SATIRINI GÖRÜNTÜLEME
df.head()                                       # Verinin nasıl göründüğünü kontrol ediyoruz

# VERİNİN TEMEL İSTATİSTİKSEL BİLGİLERİNİ GÖRÜNTÜLEME
df.describe()                                   # Sayısal sütunların ortalama, standart sapma, min, max değerleri

# VERİ ÇERÇEVESİ HAKKINDA GENEL BİLGİ
df.info()                                       # Sütun isimleri, veri tipleri ve null olmayan değer sayıları

# EKSİK DEĞER KONTROLÜ
df.isnull().sum()                              # Her sütundaki eksik değer sayısını kontrol ediyoruz

# HER SÜTUNDAKI BENZERSİZ DEĞER SAYISINI KONTROL ETME
for i in list(df.columns):                     # Tüm sütunlar için döngü
    print("{} -- {}".format(i, df[i].value_counts().shape[0]))  # Sütun adı ve benzersiz değer sayısı

# KATEGORİK DEĞİŞKENLERİN LİSTESİ
categorical_list = ["sex", "cp","fbs","restecg","exng","slp","caa","thall","output"]

# KATEGORİK DEĞİŞKENLERİ AYIRMA VE GÖRSELLEŞTİRME
df_categoric = df.loc[:, categorical_list]     # Sadece kategorik sütunları seçiyoruz
for i in categorical_list:                     # Her kategorik değişken için
    plt.figure()                               # Yeni figür oluştur
    sns.countplot(x = i, data = df_categoric, hue = "output")  # Hedef değişkene göre sayım grafiği
    plt.title(i)                               # Grafik başlığını ayarla
    
# SAYISAL DEĞİŞKENLERİN LİSTESİ
numeric_list = ["age", "trtbps","chol","thalachh","oldpeak","output"]

# SAYISAL DEĞİŞKENLERİ AYIRMA VE PAIRPLOT GÖRSELLEŞTİRMESİ
df_numeric = df.loc[:, numeric_list]           # Sadece sayısal sütunları seçiyoruz
sns.pairplot(df_numeric, hue = "output", diag_kind = "kde")  # Değişkenler arası ilişkileri görselleştir
plt.show()                                     # Grafiği göster

# VERİLERİ STANDARDİZE ETME HAZIRLIĞI
scaler = StandardScaler()                      # Standardizer objesi oluştur
scaler                                         # Objeyi görüntüle

# SAYISAL DEĞİŞKENLERİ STANDARDİZE ETME (output hariç)
scaled_array = scaler.fit_transform(df[numeric_list[:-1]])  # Son eleman (output) hariç standardize et

scaled_array                                   # Standardize edilmiş array'i görüntüle

# STANDARDİZE EDİLMİŞ VERİLERİ DATAFRAME'E ÇEVİRME
df_dummy = pd.DataFrame(scaled_array, columns = numeric_list[:-1])  # Yeni DataFrame oluştur
df_dummy.head()                                # İlk 5 satırı kontrol et

# OUTPUT SÜTUNUNU GERİ EKLEME
df_dummy = pd.concat([df_dummy, df.loc[:, "output"]], axis = 1)  # Output sütununu birleştir
df_dummy.head()                                # Sonucu kontrol et

# VERİYİ MELT İŞLEMİYLE UZUN FORMATA ÇEVİRME
data_melted = pd.melt(df_dummy, id_vars = "output", var_name = "features", value_name = "value")
data_melted.head(20)                           # İlk 20 satırı görüntüle

# BOX PLOT GÖRSELLEŞTİRMESİ (Aykırı değerleri görmek için)
plt.figure()                                   # Yeni figür
sns.boxplot(x = "features", y = "value", hue = "output", data= data_melted)  # Kutu grafiği
plt.show()                                     # Grafiği göster

# SWARM PLOT GÖRSELLEŞTİRMESİ (Veri noktalarının dağılımını görmek için)
plt.figure()                                   # Yeni figür
sns.swarmplot(x = "features", y = "value", hue = "output", data= data_melted)  # Swarm grafiği
plt.show()                                     # Grafiği göster

# KATEGORİK PLOT (Cinsiyet ve egzersizin yaşla ilişkisi)
sns.catplot(x = "exng", y = "age", hue = "output", col = "sex", kind = "swarm", data = df)
plt.show()                                     # Grafiği göster

# KORELASYON MATRİSİ HEATMAP'İ
plt.figure(figsize = (14,10))                  # Büyük figür boyutu
sns.heatmap(df.corr(), annot = True, fmt = ".1f", linewidths = .7)  # Korelasyon ısı haritası
plt.show()                                     # Grafiği göster

# SAYISAL DEĞİŞKENLERİ YENİDEN TANIMLAMA (output olmadan)
numeric_list = ["age", "trtbps","chol","thalachh","oldpeak"]
df_numeric = df.loc[:, numeric_list]           # Sadece sayısal sütunları seç
df_numeric.head()                              # İlk 5 satırı kontrol et

df.describe()                                  # Temel istatistikleri tekrar görüntüle

# AYKIRI DEĞER TESPİTİ VE TEMİZLEME
for i in numeric_list:                         # Her sayısal değişken için
    
    # IQR (Interquartile Range) HESAPLAMA
    Q1 = np.percentile(df.loc[:, i],25)        # 1. çeyrek (25. yüzdelik)
    Q3 = np.percentile(df.loc[:, i],75)        # 3. çeyrek (75. yüzdelik)
    
    IQR = Q3 - Q1                              # Çeyrekler arası mesafe
    
    print("Old shape: ", df.loc[:, i].shape)   # Eski boyutu yazdır
    
    # ÜST SINIR AYKIRI DEĞER TESPİTİ
    upper = np.where(df.loc[:, i] >= (Q3 +2.5*IQR))  # Üst sınırı aşan değerler
    
    # ALT SINIR AYKIRI DEĞER TESPİTİ
    lower = np.where(df.loc[:, i] <= (Q1 - 2.5*IQR))  # Alt sınırın altında kalan değerler
    
    print("{} -- {}".format(upper, lower))     # Aykırı değer indekslerini yazdır
    
    # ÜST SINIR AYKIRI DEĞERLERİ SİLME
    try:
        df.drop(upper[0], inplace = True)      # Üst aykırı değerleri sil
    except: print("KeyError: {} not found in axis".format(upper[0]))  # Hata durumunda mesaj
    
    # ALT SINIR AYKIRI DEĞERLERİ SİLME
    try:
        df.drop(lower[0], inplace = True)      # Alt aykırı değerleri sil
    except:  print("KeyError: {} not found in axis".format(lower[0]))  # Hata durumunda mesaj
        
    print("New shape: ", df.shape)             # Yeni boyutu yazdır

# TEMİZLENMİŞ VERİNİN KOPYASINI ALMA
df1 = df.copy()                                # Orijinal veriyi korumak için kopya oluştur

# KATEGORİK DEĞİŞKENLERİ DUMMY DEĞİŞKENLERE ÇEVİRME
df1 = pd.get_dummies(df1, columns = categorical_list[:-1], drop_first = True)  # One-hot encoding (ilk kategori çıkarılarak)
df1.head()                                     # Sonucu kontrol et

# BAĞIMSIZ VE BAĞIMLI DEĞİŞKENLERİ AYIRMA
X = df1.drop(["output"], axis = 1)            # Bağımsız değişkenler (özellikler)
y = df1[["output"]]                           # Bağımlı değişken (hedef)

# YENİ BİR STANDARDIZER OLUŞTURMA
scaler = StandardScaler()                      # Yeni standardizer objesi
scaler                                         # Objeyi görüntüle

# SADECE SAYISAL ÖZELLİKLERİ STANDARDİZE ETME
X[numeric_list[:-1]] = scaler.fit_transform(X[numeric_list[:-1]])  # Sayısal sütunları standardize et
X.head()                                       # Sonucu kontrol et

# VERİYİ EĞİTİM VE TEST SETLERİNE BÖLME
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
print("X_train: {}".format(X_train.shape))    # Eğitim seti özellik boyutu
print("X_test: {}".format(X_test.shape))      # Test seti özellik boyutu
print("y_train: {}".format(y_train.shape))    # Eğitim seti hedef boyutu
print("y_test: {}".format(y_test.shape))      # Test seti hedef boyutu

# TEMEL LOGİSTİK REGRESYON MODELİ OLUŞTURMA
logreg = LogisticRegression()                  # Varsayılan parametrelerle logistic regression
logreg                                         # Model objesini görüntüle

# MODELİ EĞİTME
logreg.fit(X_train, y_train)                  # Modeli eğitim verisiyle eğit

# OLASILIK TAHMİNLERİ HESAPLAMA
y_pred_prob = logreg.predict_proba(X_test)    # Her sınıf için olasılık tahminleri
y_pred_prob                                    # Olasılıkları görüntüle

# SINIF TAHMİNLERİ YAPMA
y_pred = np.argmax(y_pred_prob, axis = 1)     # En yüksek olasılıklı sınıfı seç
y_pred                                         # Tahminleri görüntüle

# TEST DOĞRULUĞUNU HESAPLAMA
print("Test accuracy: {}".format(accuracy_score(y_pred, y_test)))  # Doğruluk skorunu yazdır

# ROC EĞRİSİ HESAPLAMA
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])  # False Positive Rate ve True Positive Rate

# ROC EĞRİSİNİ ÇİZME
plt.plot([0,1],[0,1],"k--")                   # Rastgele tahmin çizgisi (diagonal)
plt.plot(fpr, tpr, label = "Logistic Regression")  # Logistic regression ROC eğrisi
plt.xlabel("False Positive Rate")              # X ekseni etiketi
plt.ylabel("True Positive Rate")               # Y ekseni etiketi
plt.title("Logistic Regression ROC Curve")    # Grafik başlığı
plt.show()                                     # Grafiği göster

# REGULARİZASYON İÇİN YENİ LOGİSTİK REGRESYON MODELİ
lr = LogisticRegression()                      # Yeni logistic regression modeli
lr                                             # Model objesini görüntüle

# REGULARİZASYON TİPLERİNİ TANIMLAMA
penalty = ["l1", "l2"]                        # L1 (Lasso) ve L2 (Ridge) regularization

# HİPERPARAMETRE ARAŞTIRMASI İÇİN PARAMETRE SÖZLÜĞÜ
parameters = {"penalty":penalty}               # Grid search için parametre sözlüğü

# GRID SEARCH CROSS VALİDATİON OLUŞTURMA
lr_searcher = GridSearchCV(lr, parameters)    # En iyi hiperparametreleri bulmak için grid search

# GRID SEARCH'İ EĞİTME
lr_searcher.fit(X_train, y_train)             # Tüm parametre kombinasyonlarını dene

# EN İYİ PARAMETRELERİ YAZDIRMA
print("Best parameters: ",lr_searcher.best_params_)  # En iyi regularization tipini yazdır

# EN İYİ MODELLE TAHMİN YAPMA
y_pred = lr_searcher.predict(X_test)          # En iyi modelle test seti tahminleri

# FINAL TEST DOĞRULUĞUNU HESAPLAMA
print("Test accuracy: {}".format(accuracy_score(y_pred, y_test)))  # Regularization sonrası doğruluk

