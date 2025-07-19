import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib kütüphanesi
import plotly.express as px  # İnteraktif grafikler için plotly express
import numpy as np  # Sayısal hesaplamalar için numpy
import missingno as msno  # Eksik veri analizi için missingno
import pandas as pd  # Veri manipülasyonu için pandas
import seaborn as sns  # İstatistiksel görselleştirme için seaborn

# ML
from sklearn.tree import DecisionTreeClassifier  # Karar ağacı algoritması
from sklearn.ensemble import RandomForestClassifier  # Rastgele orman algoritması
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split  # Model seçimi ve veri bölme araçları
from sklearn.metrics import precision_score, confusion_matrix  # Model değerlendirme metrikleri
from sklearn import tree  # Karar ağacı görselleştirmesi için


df = pd.read_csv("7_4_ModelSecimiveDegerlendirmeDurumCalismasi_dataset.csv")  # CSV dosyasını pandas DataFrame olarak yükle

df.head()  # İlk 5 satırı görüntüle

df.describe()  # Sayısal sütunların istatistiksel özetini göster

df.info()  # DataFrame hakkında genel bilgileri (veri tipleri, null değerler) göster

d = pd.DataFrame(df["Potability"].value_counts())  # Potability sütununun değer sayılarını DataFrame olarak oluştur
fig = px.pie(d, values = "count", names = ["Not Potable", "Potable"], hole = 0.35, opacity = 0.8,  # Pasta grafik oluştur
            labels = {"label" :"Potability","count":"Number of Samples"})  # Grafik etiketlerini ayarla
fig.update_layout(title = dict(text = "Pie Chart of Potability Feature"))  # Grafik başlığını ayarla
fig.update_traces(textposition = "outside", textinfo = "percent+label")  # Metin konumunu ve bilgilerini ayarla
fig.show()  # Grafiği göster

df.corr()  # Özellikler arası korelasyon matrisini hesapla

sns.clustermap(df.corr(), cmap = "vlag", dendrogram_ratio = (0.1, 0.2), annot = True, linewidths = .8, figsize = (9,10))  # Korelasyon ısı haritasını kümeleme ile göster
plt.show()  # Grafiği göster

non_potable = df.query("Potability == 0")  # İçilebilir olmayan su örneklerini filtrele
potable = df.query("Potability == 1")  # İçilebilir su örneklerini filtrele

plt.figure(figsize = (15,15))  # 15x15 boyutunda figür oluştur
for ax, col in enumerate(df.columns[:9]):  # İlk 9 sütun için döngü
    plt.subplot(3,3, ax + 1)  # 3x3 alt grafik düzeninde pozisyon belirle
    plt.title(col)  # Alt grafik başlığını sütun adı olarak ayarla
    sns.kdeplot(x = non_potable[col], label = "Non Potable")  # İçilebilir olmayan veriler için kernel density plot
    sns.kdeplot(x = potable[col], label = "Potable")  # İçilebilir veriler için kernel density plot
    plt.legend()  # Efsane ekle
plt.tight_layout()  # Alt grafikleri düzenle

msno.matrix(df)  # Eksik veri matrisini görselleştir
plt.show()  # Grafiği göster

df.isnull().sum()  # Her sütundaki eksik değer sayısını hesapla

# handle missing value with average of features
df["ph"].fillna(value = df["ph"].mean(), inplace = True)  # pH sütunundaki eksik değerleri ortalama ile doldur
df["Sulfate"].fillna(value = df["Sulfate"].mean(), inplace = True)  # Sülfat sütunundaki eksik değerleri ortalama ile doldur
df["Trihalomethanes"].fillna(value = df["Trihalomethanes"].mean(), inplace = True)  # Trihalometan sütunundaki eksik değerleri ortalama ile doldur

X = df.drop("Potability", axis = 1).values  # Hedef değişken dışındaki tüm sütunları özellik matrisi olarak al
y = df["Potability"].values  # Hedef değişkeni (içilebilirlik) al

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)  # Veriyi %70 eğitim %30 test olarak böl
print("X_train",X_train.shape)  # Eğitim özellik matrisinin boyutunu yazdır
print("X_test",X_test.shape)  # Test özellik matrisinin boyutunu yazdır
print("y_train",y_train.shape)  # Eğitim hedef değişkeninin boyutunu yazdır
print("y_test",y_test.shape)  # Test hedef değişkeninin boyutunu yazdır

# min-max normalization
x_train_max = np.max(X_train)  # Eğitim verisindeki maksimum değeri bul
x_train_min = np.min(X_train)  # Eğitim verisindeki minimum değeri bul
X_train = (X_train - x_train_min)/(x_train_max-x_train_min)  # Eğitim verisini 0-1 arasında normalize et
X_test = (X_test - x_train_min)/(x_train_max-x_train_min)  # Test verisini aynı parametrelerle normalize et

models = [("DTC", DecisionTreeClassifier(max_depth = 3)),  # Karar ağacı modeli (maksimum derinlik 3)
          ("RF",RandomForestClassifier())]  # Rastgele orman modeli

finalResults = []  # Sonuçları tutacak liste
cmList = []  # Karmaşıklık matrislerini tutacak liste
for name, model in models:  # Her model için döngü
    model.fit(X_train, y_train)  # Modeli eğit
    model_result = model.predict(X_test)  # Test verisi üzerinde tahmin yap
    score = precision_score(y_test, model_result)  # Precision skoru hesapla
    cm = confusion_matrix(y_test, model_result)  # Karmaşıklık matrisi oluştur
    
    finalResults.append((name, score))  # Model adı ve skorunu listeye ekle
    cmList.append((name, cm))  # Model adı ve karmaşıklık matrisini listeye ekle
finalResults  # Final sonuçları göster


for name, i in cmList:  # Her karmaşıklık matrisi için döngü
    plt.figure()  # Yeni figür oluştur
    sns.heatmap(i, annot = True, linewidths = 0.8, fmt = ".1f")  # Isı haritası olarak göster
    plt.title(name)  # Grafik başlığını model adı olarak ayarla
    plt.show()  # Grafiği göster

dt_clf = models[0][1]  # Karar ağacı modelini al
dt_clf  # Modeli göster


plt.figure(figsize = (25,20))  # 25x20 boyutunda figür oluştur
tree.plot_tree(dt_clf,  # Karar ağacını görselleştir
               feature_names =  df.columns.tolist()[:-1],  # Özellik adlarını ver (son sütun hariç)
               class_names = ["0", "1"],  # Sınıf adlarını ver
               filled = True,  # Düğümleri renklendir
               precision = 5)  # Hassasiyet değeri
plt.show()  # Grafiği göster

model_params = {  # Hiperparametre arama için model parametreleri sözlüğü
    "Random Forest":  # Rastgele orman için parametreler
    {
        "model":RandomForestClassifier(),  # Model objesi
        "params":  # Aranacak parametreler
        {
            "n_estimators":[10, 50, 100],  # Ağaç sayısı seçenekleri
            "max_features":["auto","sqrt","log2"],  # Maksimum özellik sayısı seçenekleri
            "max_depth":list(range(1,21,3))  # Maksimum derinlik seçenekleri (1'den 21'e 3'er artarak)
        }
    }
    
}
model_params  # Parametreleri göster

cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 2)  # 5 katlı çapraz doğrulama, 2 tekrar
scores = []  # Skorları tutacak liste
for model_name, params in model_params.items():  # Her model için döngü
    rs = RandomizedSearchCV(params["model"], params["params"], cv = cv, n_iter = 10)  # Rastgele arama objesi oluştur (10 iterasyon)
    rs.fit(X,y)  # Tüm veri üzerinde hiperparametre araması yap
    scores.append([model_name, dict(rs.best_params_),rs.best_score_])  # En iyi parametreler ve skoru listeye ekle
scores  # Sonuçları göster


"""
KONU AÇIKLAMASI: MODEL SEÇİMİ VE DEĞERLENDİRME

Bu proje, su içilebilirlik tahmini için makine öğrenmesi modelleri geliştirme ve değerlendirme sürecini kapsamaktadır.

PROJE AŞAMALARI:

1. VERİ YÜKLEME VE KEŞFETME:
   - Su kalitesi veri seti yüklenir
   - Temel istatistikler incelenir (head, describe, info)
   - Hedef değişken (Potability) dağılımı pasta grafiği ile görselleştirilir

2. VERİ ANALİZİ:
   - Özellikler arası korelasyon analizi yapılır
   - Cluster map ile korelasyon görselleştirilir
   - İçilebilir ve içilebilir olmayan sular için özellik dağılımları karşılaştırılır

3. VERİ TEMİZLEME:
   - Eksik değerler tespit edilir (missingno matrix ile)
   - pH, Sulfate ve Trihalomethanes sütunlarındaki eksik değerler ortalama ile doldurulur

4. VERİ HAZIRLIĞI:
   - Özellik matrisi (X) ve hedef değişken (y) ayrılır
   - Veri %70 eğitim, %30 test olarak bölünür
   - Min-Max normalizasyon uygulanır (0-1 arasında ölçekleme)

5. MODEL EĞİTİMİ VE KARŞILAŞTIRMA:
   - İki farklı algoritma test edilir:
     * Karar Ağacı (Decision Tree) - max_depth=3
     * Rastgele Orman (Random Forest)
   - Her model için precision skoru hesaplanır
   - Confusion matrix ile detaylı performans analizi yapılır

6. MODEL GÖRSELLEŞTİRME:
   - Confusion matrix'ler ısı haritası olarak gösterilir
   - Karar ağacı yapısı detaylı olarak çizilir

7. HİPERPARAMETRE OPTİMİZASYONU:
   - RandomizedSearchCV ile en iyi parametreler aranır
   - Random Forest için:
     * n_estimators: [10, 50, 100]
     * max_features: ["auto", "sqrt", "log2"]
     * max_depth: [1, 4, 7, 10, 13, 16, 19]
   - 5-katlı çapraz doğrulama ile 2 tekrar yapılır
   - En iyi parametreler ve skorlar kaydedilir

KULLANILAN TEKNİKLER:
- Veri görselleştirme (matplotlib, seaborn, plotly)
- Eksik veri analizi (missingno)
- Veri normalizasyonu (Min-Max scaling)
- Sınıflandırma algoritmaları (Decision Tree, Random Forest)
- Model değerlendirme (precision score, confusion matrix)
- Hiperparametre optimizasyonu (RandomizedSearchCV)
- Çapraz doğrulama (RepeatedStratifiedKFold)

Bu proje, makine öğrenmesi iş akışının tüm adımlarını kapsamlı bir şekilde göstermektedir:
veri keşfi, temizleme, model geliştirme, değerlendirme ve optimizasyon.
"""


