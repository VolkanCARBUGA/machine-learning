# Gerekli kütüphaneleri içe aktarıyoruz
from sklearn.datasets import load_iris  # Iris veri setini yüklemek için
from sklearn.model_selection import train_test_split,GridSearchCV  # Veriyi ayırmak ve parametre araması için
from sklearn.tree import DecisionTreeClassifier  # Karar ağacı algoritması için
from sklearn.svm import SVC  # Support Vector Machine algoritması için (kullanılmamış)
import numpy as np  # Sayısal işlemler için

# Iris veri setini yüklüyoruz
iris=load_iris()  # Iris çiçeği veri setini yükler (150 örnek, 4 özellik, 3 sınıf)
X=iris.data  # Özellik verilerini (sepal/petal uzunluk/genişlik) X'e atıyoruz
y=iris.target  # Hedef değişkeni (çiçek türleri: 0,1,2) y'ye atıyoruz

# Veriyi eğitim ve test olmak üzere ikiye ayırıyoruz
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# %80 eğitim, %20 test verisi olacak şekilde ayırır, random_state=42 ile sonuçları tekrarlanabilir yapar

# Karar ağacı modelini oluşturuyoruz
tree=DecisionTreeClassifier()  # Varsayılan parametrelerle karar ağacı classifier'ı oluşturur

# Hiperparametre arama alanını tanımlıyoruz
tree_params_grid={"max_depth":[3,5,7],  # Ağacın maksimum derinliği: 3, 5 veya 7 seviye
                  "max_leaf_nodes":[None,5,10,20,30,50],  # Maksimum yaprak düğüm sayısı seçenekleri
                 }

# Cross-validation için fold sayısını belirliyoruz
nb_cv=3  # 3-fold cross validation kullanacağız (veriyi 3 parçaya böler)

# GridSearchCV ile hiperparametre araması ve cross-validation yapıyoruz
tree_params_search=GridSearchCV(tree,tree_params_grid,cv=nb_cv,scoring="accuracy")
# tree: kullanılacak model, tree_params_grid: aranacak parametreler, cv=3: 3-fold cross validation, scoring: doğruluk metriği

# Modeli eğitim verisi üzerinde eğitiyoruz
tree_params_search.fit(X_train,y_train)  # Tüm parametre kombinasyonlarını dener ve en iyisini bulur

# En iyi parametreleri ve skorunu yazdırıyoruz
print("Decision Tree Params Search Best Params: ",tree_params_search.best_params_)  # En iyi parametre kombinasyonu
print("Decision Tree Params Search Best Score: ",tree_params_search.best_score_)   # En iyi cross-validation skoru

# Tüm parametre kombinasyonları ve ortalama skorlarını yazdırıyoruz (1. yöntem)
for mean_score,params in zip(tree_params_search.cv_results_["mean_test_score"],tree_params_search.cv_results_["params"]):
    print(f"Params: {params}, Mean Score: {mean_score}")  # Her parametre kombinasyonu için ortalama CV skoru

# Cross-validation sonuçlarını detaylı olarak inceliyoruz (2. yöntem)
cv_results=tree_params_search.cv_results_  # Tüm CV sonuçlarını alıyoruz
for i,params in enumerate(cv_results["params"]):  # Her parametre kombinasyonu için döngü
    print(f"Params: {params}, Mean Score: {cv_results['mean_test_score'][i]}")  # Parametre ve ortalama skor
    for j in range(nb_cv):  # Her fold için döngü (3 fold)
        accuracy=cv_results[f'split{j}_test_score'][i]  # j. fold'daki doğruluk skoru
        print(f"\tFold {j+1} Score: {accuracy}")  # Her fold'un skorunu ayrı ayrı yazdırır

"""
CROSS-VALIDATION (ÇAPRAz DOĞRULAMA) KONUSU HAKKINDA GENEL AÇIKLAMA:

Cross-Validation, makine öğrenmesinde modelin performansını daha güvenilir şekilde değerlendirmek 
için kullanılan bir tekniktir. Temel amacı, modelin farklı veri alt kümelerinde nasıl performans 
gösterdiğini anlamaktır.

K-FOLD CROSS VALIDATION NASIL ÇALIŞIR:
1. Veri seti K eşit parçaya bölünür (burada K=3)
2. Her iterasyonda 1 parça test, geri kalan K-1 parça eğitim için kullanılır
3. Bu işlem K kez tekrarlanır, her seferinde farklı bir parça test olarak kullanılır
4. Sonuçların ortalaması alınarak final performans hesaplanır

GRIDSEARCHCV İLE HIPERPARAMETRE OPTIMİZASYONU:
- GridSearchCV, farklı hiperparametre kombinasyonlarını sistematik olarak dener
- Her kombinasyon için cross-validation yapar
- En yüksek performansı veren parametre setini bulur
- Bu sayede hem model seçimi hem de değerlendirme işlemini birlikte yapar

BU KODDA YAPILAN İŞLEMLER:
1. Iris veri seti yüklendi ve eğitim/test olarak ayrıldı
2. Karar ağacı için farklı max_depth ve max_leaf_nodes değerleri tanımlandı
3. 3-fold cross validation ile her parametre kombinasyonu test edildi
4. En iyi parametre kombinasyonu ve performansı bulundu
5. Tüm sonuçlar detaylı olarak görüntülendi

AVANTAJLARI:
- Modelin genelleme yeteneğini daha iyi değerlendirir
- Overfitting (aşırı öğrenme) problemini tespit etmeye yardımcı olur
- Sınırlı veri ile daha güvenilir sonuçlar elde edilir
- Hiperparametre optimizasyonu için ideal bir yöntemdir

SONUÇ:
Cross-validation, makine öğrenmesi projelerinde model performansını doğru şekilde 
değerlendirmek için vazgeçilmez bir tekniktir. Özellikle küçük veri setlerinde 
ve hiperparametre optimizasyonunda çok değerlidir.
"""







