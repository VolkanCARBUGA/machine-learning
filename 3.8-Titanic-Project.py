# Sayısal işlemler için kullanılan temel bilimsel kütüphane (dizi işlemleri, matematiksel hesaplamalar)
import numpy as np
# Veri işleme ve analiz için kullanılan kütüphane (DataFrame yapısı ile çalışır)
import pandas as pd
# Grafik çizimi için temel kütüphane (çizgi grafiği, histogram vs.)
import matplotlib.pyplot as plt
# Gelişmiş ve estetik veri görselleştirme için kullanılır (ısı haritası, kutu grafiği vs.)
import seaborn as sns
# Veri setini eğitim ve test olarak böler
# StratifiedKFold: sınıf dağılımını koruyarak K-katlı çapraz doğrulama yapar
# GridSearchCV: hiperparametre optimizasyonu için kullanılır (en iyi ayarları bulur)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# Lojistik regresyon modeli: özellikle ikili sınıflandırma problemleri için yaygın olarak kullanılır
from sklearn.linear_model import LogisticRegression
# Support Vector Machine sınıflandırıcısı: sınıflar arası sınırı en iyi çizen düzlemi bulur
from sklearn.svm import SVC
# Random Forest: birden fazla karar ağacının ortalamasını alarak çalışan güçlü bir sınıflandırma modeli
# Voting Classifier: birden fazla modelin tahminlerini birleştirerek son kararı verir (ensemble learning)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# K-En Yakın Komşu algoritması: komşu verilere bakarak sınıflandırma yapar
from sklearn.neighbors import KNeighborsClassifier
# Karar ağacı sınıflandırıcısı: veriyi dallara ayırarak sınıflandırma yapan görsel olarak da anlaşılır model
from sklearn.tree import DecisionTreeClassifier
# Modelin doğruluk oranını hesaplamak için kullanılır (% olarak başarı ölçümü)
from sklearn.metrics import accuracy_score
sns.set_style("whitegrid")  # Seaborn'un kendi stil ayarını kullanıyoruz
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
import os

# CSV (virgül ile ayrılmış veri dosyası) dosyasını okur ve bir DataFrame olarak içe aktarır
train_df = pd.read_csv("3_8_GozetimliOgrenmeDurumCalismasi1_dataset.csv")
# DataFrame'deki tüm sütun isimlerini alır
train_df.columns
# Sütun isimlerini ekrana yazdırır
print(train_df.columns)
# DataFrame’in ilk 5 satırını görüntüler (görsel çıktı verir, print gerekmez ama istersek kullanılabilir)
train_df.head()
# DataFrame’in ilk 10 satırını yazdırır
print(train_df.head(10))
# Sayısal sütunlar için istatistiksel özet bilgileri verir (ortalama, std, min, max, çeyrekler vs.)
print(train_df.describe())
# DataFrame hakkında genel bilgi verir: kaç satır-sütun var, her sütunun tipi (int, float, object), boş değer var mı vs.
print(train_df.info())


def bar_plot(variable):
    """
        input: variable örnek: "Sex"
        output: bar plot (çubuk grafik) ve value count (frekans tablosu)
    """
    # Seçilen sütunu (değişkeni) alıyoruz
    var = train_df[variable]
    # Her benzersiz değerin kaç kez geçtiğini sayıyoruz (örneğin: Erkek 577, Kadın 314 gibi)
    varValue = var.value_counts()
    # Grafik boyutunu ayarlıyoruz (9 birim genişlik, 3 birim yükseklik)
    plt.figure(figsize = (9, 3))
    # Çubuk grafiği çiziyoruz
    plt.bar(varValue.index, varValue)
    # X ekseni etiketlerini ayarlıyoruz (kategoriler: örn. Kadın, Erkek)
    plt.xticks(varValue.index, varValue.index.values)
    # Y ekseni etiketini ayarlıyoruz
    plt.ylabel("Frequency")  # Frekans yani kaç kez tekrar ettiği
    # Grafiğe başlık ekliyoruz (değişken adı)
    plt.title(variable)
    # Grafiği ekrana yazdırıyoruz
    plt.show()
    # Ayrıca frekansları yazdırıyoruz (print olarak)
    print("{}: \n {}".format(variable, varValue))


category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]  # Grafikle görselleştirilecek sınırlı kategorik değişkenler
for c in category1:                         # Her bir değişken için döngü başlat
    bar_plot(c)                             # bar_plot fonksiyonunu çağırarak grafik ve frekansları göster
category2 = ["Cabin", "Name", "Ticket"]     # Sınıfları çok fazla olan kategorik değişkenler (grafik çizilmez, sadece yazdırılır)
for c in category2:                         
    print("{} \n".format(train_df[c].value_counts()))  # Her bir sütunun değer tekrarlarını yazdır (örnek: her biletten kaçar tane var)
def plot_hist(variable):
    plt.figure(figsize = (9,3))                          # Grafik boyutunu genişlik 9, yükseklik 3 olarak ayarla
    plt.hist(train_df[variable], bins = 50)              # Seçilen değişkenin histogramını 50 aralığa bölecek şekilde çiz
    plt.xlabel(variable)                                 # X eksenine değişken adını yaz
    plt.ylabel("Frequency")                              # Y eksenine "Frequency" (Frekans) etiketi ekle
    plt.title("{} distribution with hist".format(variable))  # Başlıkta değişkenin dağılımı olduğunu belirt
    plt.show()                                          # Grafiği göster


numericVar = ["Fare", "Age", "PassengerId"]                    # Sayısal değişkenlerin listesi
for n in numericVar:                                           # Her sayısal değişken için döngü başlat
    plot_hist(n)                                              # histogram çizdir (plot_hist fonksiyonunu kullanarak)
# Pclass (bilet sınıfı) ile Survived (hayatta kalma) arasındaki ilişkiyi incele
train_df[["Pclass", "Survived"]].groupby("Pclass", as_index=False).mean().sort_values(by="Survived", ascending=False)
# Sex (cinsiyet) ile Survived arasındaki ilişkiyi incele
train_df[["Sex", "Survived"]].groupby("Sex", as_index=False).mean().sort_values(by="Survived", ascending=False)
# SibSp (kardeş/eş sayısı) ile Survived arasındaki ilişkiyi incele
train_df[["SibSp", "Survived"]].groupby("SibSp", as_index=False).mean().sort_values(by="Survived", ascending=False)
# Parch (ebeveyn/çocuk sayısı) ile Survived arasındaki ilişkiyi incele
train_df[["Parch", "Survived"]].groupby("Parch", as_index=False).mean().sort_values(by="Survived", ascending=False)
def detect_outliers(df, features):
    outlier_indices = []  # Tüm özellikler için saptanan aykırı değerlerin indekslerini tutacak liste
    for c in features:  # Her bir özellik (sütun) için döngü
        Q1 = np.percentile(df[c], 25)           # 1. çeyrek (25. yüzdelik)
        Q3 = np.percentile(df[c], 75)           # 3. çeyrek (75. yüzdelik)
        IQR = Q3 - Q1                           # IQR: Interquartile Range (orta %50'lik veri aralığı)
        outlier_step = IQR * 1.5                # IQR'nin 1.5 katı dışına çıkan değerler aykırı kabul edilir
        # Aykırı değerlerin bulunduğu satırların indeksleri (Q1-1.5*IQR'den küçük veya Q3+1.5*IQR'den büyük)
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)  # Bulunan aykırı indeksleri listeye ekle
    outlier_indices = Counter(outlier_indices)  # Aynı indeksten kaç kez aykırı bulunduğunu sayar
    # Birden fazla özelliğe ait aykırı değer olan indeksleri al (yani 2'den fazla kez aykırı bulunanlar)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multiple_outliers   # Bu indeks listesini döndürür (çoklu aykırı değer içeren satırlar)


# detect_outliers fonksiyonu ile tespit edilen aykırı değerlerin satırlarını train_df'den seçip göster (burada küçük bir yazım hatası var, 'rain_df' değil 'train_df' olmalı)
train_df.loc[detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]
# Tespit edilen aykırı değerleri train_df'den sil, satır (axis=0) bazında drop işlemi yap ve indeksleri yeniden düzenle
train_df = train_df.drop(detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis=0).reset_index(drop=True)
# train_df'nin güncel uzunluğunu (satır sayısını) değişkene ata
train_df_len = len(train_df)
# train_df'de eksik (null) değere sahip olan sütunların isimlerini al
train_df.columns[train_df.isnull().any()]
# Her sütundaki eksik değer sayısını göster
train_df.isnull().sum()
# "Embarked" sütununda eksik olan satırları görüntüle
train_df[train_df["Embarked"].isnull()]
# "Fare" değerlerinin "Embarked" değişkenine göre kutu grafiğini çiz
train_df.boxplot(column="Fare", by="Embarked")
plt.show()
# "Embarked" sütunundaki eksik değerleri "C" ile doldur
train_df["Embarked"] = train_df["Embarked"].fillna("C")
# Eksik "Embarked" değerlerinin dolduğunu kontrol et (boş kaldı mı diye)
train_df[train_df["Embarked"].isnull()]
# "Fare" sütunundaki eksik değerleri kontrol et
train_df[train_df["Fare"].isnull()]
# Eksik "Fare" değerlerini, 3. sınıf yolcuların ortalama bilet fiyatı ile doldur
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
# "Fare" sütunundaki eksik değerlerin tamamen dolup dolmadığını kontrol et
train_df[train_df["Fare"].isnull()]

# Korelasyonları görmek için kullanılacak sütunlar listesi
list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
# Bu değişkenler arasındaki korelasyon matrisini ısı haritası (heatmap) olarak çiz
sns.heatmap(train_df[list1].corr(), annot=True, fmt=".2f")  # annot=True → hücrelere sayısal değerleri yaz, fmt=".2f" → virgülden sonra 2 basamak göster
plt.show()  # Grafiği göster


# Kardeş/eş sayısı (SibSp) ile hayatta kalma oranı (Survived) arasındaki ilişkiyi bar grafiği ile göster
g = sns.catplot(x="SibSp", y="Survived", data=train_df, kind="bar")  # x: kategori, y: ortalama hayatta kalma
g.set_ylabels("Survived Probability")  # Y eksen etiketi
plt.show()  # Grafiği göster

# Ebeveyn/çocuk sayısı (Parch) ile hayatta kalma oranı arasındaki ilişkiyi bar grafiği ile göster
g = sns.catplot(x="Parch", y="Survived", kind="bar", data=train_df)
g.set_ylabels("Survived Probability")
plt.show()


# Yolcu sınıfı (Pclass) ile hayatta kalma oranı arasındaki ilişkiyi bar grafiği ile göster
g = sns.catplot(x="Pclass", y="Survived", data=train_df, kind="bar")
g.set_ylabels("Survived Probability")
plt.show()


g = sns.FacetGrid(train_df, col="Survived")                  # Survived (0/1) değerlerine göre iki ayrı grafik oluştur
g.map(sns.histplot, "Age", bins=25)                          # Her grupta yaş dağılımını histogram olarak çiz (25 aralıkta)
plt.show()                                                   # Grafiği göster


g = sns.FacetGrid(train_df, col="Survived", row="Pclass")    # Satırlarda Pclass, sütunlarda Survived → her sınıf ve durum için ayrı yaş dağılımı grafiği
g.map(plt.hist, "Age", bins=25)                              # Yaş histogramlarını çiz
g.add_legend()                                               # Grafiklere açıklama (legend) ekle
plt.show()


g = sns.FacetGrid(train_df, row="Embarked")                 # Her limandan binen yolcular için ayrı grafik çiz
g.map(sns.pointplot, "Pclass", "Survived", "Sex")           # Pclass ve cinsiyete göre hayatta kalma oranlarını nokta grafikle göster
g.add_legend()
plt.show()


g = sns.FacetGrid(train_df, row="Embarked", col="Survived")  # Satırlarda liman, sütunlarda hayatta kalma durumu
g.map(sns.barplot, "Sex", "Fare")                             # Cinsiyete göre ortalama bilet fiyatlarını bar grafikle göster
g.add_legend()
plt.show()


train_df[train_df["Age"].isnull()]                        # Yaş (Age) bilgisi eksik (NaN) olan satırları görüntüler


sns.catplot(x="Sex", y="Age", data=train_df, kind="box")  # Cinsiyete göre yaş dağılımını kutu grafiği ile göster
plt.show()                                                # Grafiği ekrana bas

sns.catplot(x="Sex", y="Age", hue="Pclass", data=train_df, kind="box")  # Cinsiyet + sınıfa göre yaş kutu grafiği
plt.show()


sns.catplot(x="Parch", y="Age", data=train_df, kind="box")  # Ailedeki ebeveyn/çocuk sayısına göre yaş kutu grafiği
sns.catplot(x="SibSp", y="Age", data=train_df, kind="box")  # Kardeş/eş sayısına göre yaş kutu grafiği
plt.show()


sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot=True)  # Sayısal değişkenler arasında korelasyon matrisi
plt.show()


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index) # Yaş değeri eksik olan satırların indekslerini liste olarak al
for i in index_nan_age: # Eksik yaş verisi olan her bir satır için döngü başlat
    age_pred = train_df["Age"][
        ((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & 
         (train_df["Parch"] == train_df.iloc[i]["Parch"]) & 
         (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))
    ].median() # Aynı kardeş sayısı, ebeveyn sayısı ve sınıfa (Pclass) sahip kişilerin yaşlarının medyanını al (tahmini yaş)
    age_med = train_df["Age"].median() # Veri yetersizse genel medyan yaş yedek olarak alınır
    if not np.isnan(age_pred):  
        train_df["Age"].iloc[i] = age_pred # Eğer gruba özel yaş medyanı hesaplanabiliyorsa bunu ata
    else:
        train_df["Age"].iloc[i] = age_med # Hesaplanamıyorsa genel yaş medyanını ata
  
train_df[train_df["Age"].isnull()] # Yaş verisi hâlâ eksik olan satır kaldı mı, kontrol et (normalde kalmamalı)
train_df["Name"].head(10)  # İlk 10 satırdaki isimleri göster (ünvanları görmek için)
name = train_df["Name"]  # İsim sütununu geçici olarak değişkene al
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]  # İsimlerden ünvanları çıkart (örn: Mr, Miss, Dr...)
train_df["Title"].head(10)  # İlk 10 kişinin ünvanını görüntüle
sns.countplot(x="Title", data=train_df)  # Ünvanların dağılımını çubuk grafik olarak çiz
plt.xticks(rotation=60)  # X eksenindeki etiketleri eğik göster
plt.show()  # Grafiği göster
train_df["Title"] = train_df["Title"].replace(  # Seyrek (nadir) görülen ünvanları "other" kategorisinde birleştir
    ["Lady", "the Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], 
    "other"
)
train_df["Title"] = [  # Ünvanları sayısal kategorilere dönüştür
    0 if i == "Master" else                     # 0: Genç erkek (Master)
    1 if i in ["Miss", "Ms", "Mlle", "Mrs"] else  # 1: Kadın ünvanları
    2 if i == "Mr" else                         # 2: Yetişkin erkek
    3                                           # 3: Diğer tüm ünvanlar
    for i in train_df["Title"]
]
train_df["Title"].head(20)  # Sayıya çevrilmiş ilk 20 ünvanı göster
sns.countplot(x="Title", data=train_df)  # Yeni sayısal ünvanların dağılımını çubuk grafikle göster
plt.xticks(rotation=60)  # X ekseni etiketlerini eğik göster
plt.show()  # Grafiği göster


g = sns.catplot(x="Title", y="Survived", data=train_df, kind="bar")  # Ünvanlara göre hayatta kalma oranlarını göster
g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])  # X eksen etiketlerini belirle
g.set_ylabels("Survival Probability")  # Y ekseni etiketini belirle
plt.show()  # Grafiği göster

train_df.drop(labels=["Name"], axis=1, inplace=True)  # Artık kullanılmayacak Name sütununu sil
train_df.head()  # İlk 5 satırı göster

train_df = pd.get_dummies(train_df, columns=["Title"])  # Title sütununu one-hot encoding ile kategorik sütunlara dönüştür
train_df.head()  # Yeni haliyle ilk 5 satırı göster

train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1  # Aile büyüklüğünü hesapla
train_df.head()  # Kontrol için ilk 5 satırı göster

g = sns.catplot(x="Fsize", y="Survived", data=train_df, kind="bar")  # Aile büyüklüğüne göre hayatta kalma oranlarını göster
g.set_ylabels("Survival")  # Y ekseni etiketini belirle
plt.show()  # Grafiği göster

train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]  # Küçük aile = 1, büyük aile = 0 olarak kategorize et
train_df.head(10)  # İlk 10 satırı kontrol et

sns.countplot(x="family_size", data=train_df)  # Küçük/büyük aile sayısını görselleştir
plt.show()

g = sns.catplot(x="family_size", y="Survived", data=train_df, kind="bar")  # Aile boyutuna göre hayatta kalma oranı
g.set_ylabels("Survival")  # Y ekseni etiketini belirle
plt.show()

train_df = pd.get_dummies(train_df, columns=["family_size"])  # family_size sütununu one-hot encoding ile dönüştür
train_df.head()  # Yeni haliyle veri setine göz at

train_df["Embarked"].head()  # Embarked sütununu kontrol et

sns.countplot(x="Embarked", data=train_df)  # Limanlara göre yolcu sayısını göster
plt.show()

train_df = pd.get_dummies(train_df, columns=["Embarked"])  # Embarked sütununu one-hot encoding ile dönüştür
train_df.head()

train_df["Ticket"].head(20)  # İlk 20 ticket verisini göster

a = "A/5. 2151"  
a.replace(".", "").replace("/", "").strip().split(" ")[0]  # Ticket numarasından ön eki ayıkla

tickets = []  
for i in list(train_df.Ticket):  # Ticket verilerinde sayısal olmayanları düzenle
    if not i.isdigit():
        tickets.append(i.replace(".", "").replace("/", "").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets  # Düzenlenmiş ticket verisini at

train_df["Ticket"].head(20)  # İlk 20 ticket değerini tekrar kontrol et

train_df.head()  # Veri setini kontrol et

train_df = pd.get_dummies(train_df, columns=["Ticket"], prefix="T")  # Ticket sütununu one-hot encoding ile dönüştür
train_df.head(10)

sns.countplot(x="Pclass", data=train_df)  # Yolcu sınıf dağılımını görselleştir
plt.show()

train_df["Pclass"] = train_df["Pclass"].astype("category")  # Pclass’ı kategorik yap
train_df = pd.get_dummies(train_df, columns=["Pclass"])  # One-hot encoding uygula
train_df.head()

train_df["Sex"] = train_df["Sex"].astype("category")  # Cinsiyeti kategorik yap
train_df = pd.get_dummies(train_df, columns=["Sex"])  # One-hot encoding uygula
train_df.head()

train_df.drop(labels=["PassengerId", "Cabin"], axis=1, inplace=True)  # ID ve Cabin sütunlarını kaldır (anlamsız/eksik)

train_df.columns  # Tüm sütunları göster

train = train_df[:train_df_len]  # Eğitim setini oluştur

X_train = train.drop(labels="Survived", axis=1)  # Özellikler (bağımsız değişkenler)
y_train = train["Survived"]  # Etiket (bağımlı değişken)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)  # Eğitim/test seti böl

print("X_train", len(X_train))  # Eğitim veri sayısı
print("X_test", len(X_test))  # Test veri sayısı
print("y_train", len(y_train))  # Eğitim etiket sayısı
print("y_test", len(y_test))  # Test etiket sayısı

logreg = LogisticRegression()  # Lojistik regresyon modeli tanımla
logreg.fit(X_train, y_train)  # Modeli eğit

acc_log_train = round(logreg.score(X_train, y_train)*100, 2)  # Eğitim doğruluğu
acc_log_test = round(logreg.score(X_test, y_test)*100, 2)  # Test doğruluğu

print("Training Accuracy: % {}".format(acc_log_train))  # Eğitim doğruluğunu yazdır
print("Testing Accuracy: % {}".format(acc_log_test))  # Test doğruluğunu yazdır

# Modeller ve parametre grid’leri
random_state = 42
classifier = [
    DecisionTreeClassifier(random_state=random_state),
    SVC(random_state=random_state),
    RandomForestClassifier(random_state=random_state),
    LogisticRegression(random_state=random_state),
    KNeighborsClassifier()
]
dt_param_grid = {"min_samples_split": range(10, 500, 20), "max_depth": range(1, 20, 2)}
svc_param_grid = {"kernel": ["rbf"], "gamma": [0.001, 0.01, 0.1, 1], "C": [1, 10, 50, 100, 200, 300, 1000]}
rf_param_grid = {"max_features": [1, 3, 10], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], "bootstrap": [False], "n_estimators": [100, 300], "criterion": ["gini"]}
logreg_param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
knn_param_grid = {"n_neighbors": np.linspace(1, 19, 10, dtype=int).tolist(), "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
classifier_param = [dt_param_grid, svc_param_grid, rf_param_grid, logreg_param_grid, knn_param_grid]

# Grid Search ile en iyi modelleri bulma
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10), scoring="accuracy", n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])  # Her model için en iyi doğruluk

cv_results = pd.DataFrame({
    "Cross Validation Accuracy Means": cv_result,
    "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier"]
})  # Tüm sonuçları DataFrame olarak sakla

s = sns.barplot(cv_results, x="ML Models", y="Cross Validation Accuracy Means")  # Modellerin doğruluklarını karşılaştır

# Voting Classifier ile toplu model
votingC = VotingClassifier(
    estimators=[
        ("dt", best_estimators[0]),
        ("rfc", best_estimators[2]),
        ("lr", best_estimators[3])
    ],
    voting="soft", n_jobs=-1
)
votingC = votingC.fit(X_train, y_train)  # Voting classifier'ı eğit

print(accuracy_score(votingC.predict(X_test), y_test))  # Ensemble modelin test doğruluğunu yazdır
