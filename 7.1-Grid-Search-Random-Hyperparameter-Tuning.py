# Gerekli kütüphaneleri import ediyoruz
from sklearn.datasets import load_iris  # Iris veri setini yüklemek için
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV  # Veri ayırma, grid search ve random search için
from sklearn.neighbors import KNeighborsClassifier  # K-En Yakın Komşu algoritması için
from sklearn.tree import DecisionTreeClassifier  # Karar ağacı algoritması için
from sklearn.svm import SVC  # Destek Vektör Makinesi algoritması için
import numpy as np  # Sayısal işlemler için numpy kütüphanesi

# Iris veri setini yüklüyoruz
iris=load_iris()  # Scikit-learn'deki hazır iris veri setini yüklüyoruz
X=iris.data  # Özellik verilerini (4 özellik: sepal length, sepal width, petal length, petal width) X'e atıyoruz
y=iris.target  # Hedef değişkeni (3 sınıf: setosa, versicolor, virginica) y'ye atıyoruz

# Veriyi eğitim ve test setlerine ayırıyoruz
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)  # %80 eğitim, %20 test, sabit rastgelelik için seed=42

# ======================= K-EN YAKIN KOMŞU (KNN) ALGORİTMASI =======================

# KNN modelini oluşturuyoruz
knn=KNeighborsClassifier()  # Varsayılan parametrelerle KNN sınıflandırıcısı oluşturuyoruz

# KNN için hiperparametre ızgarasını tanımlıyoruz
knn_params_grid={"n_neighbors":np.arange(2,31),}  # Komşu sayısını 2'den 30'a kadar test edeceğiz

# Grid Search ile en iyi hiperparametreleri buluyoruz
knn_grid_search=GridSearchCV(knn,knn_params_grid,cv=5,scoring="accuracy")  # 5-fold cross validation ile doğruluk skoruna göre en iyi parametreyi buluyoruz
knn_grid_search.fit(X_train,y_train)  # Eğitim verisi üzerinde grid search'ü çalıştırıyoruz
print("KNN Grid Search Best Params: ",knn_grid_search.best_params_)  # En iyi parametreleri yazdırıyoruz
print("KNN Grid Search Best Score: ",knn_grid_search.best_score_)  # En iyi cross-validation skorunu yazdırıyoruz

# Random Search ile en iyi hiperparametreleri buluyoruz
knn_random_search=RandomizedSearchCV(knn,knn_params_grid)  # Rastgele arama ile hiperparametre optimizasyonu
knn_random_search.fit(X_train,y_train)  # Eğitim verisi üzerinde random search'ü çalıştırıyoruz
print("KNN Random Search Best Params: ",knn_random_search.best_params_)  # En iyi parametreleri yazdırıyoruz
print("KNN Random Search Best Score: ",knn_random_search.best_score_)  # En iyi skorunu yazdırıyoruz

# ======================= KARAR AĞACI (DECISION TREE) ALGORİTMASI =======================

# Karar ağacı modelini oluşturuyoruz
tree=DecisionTreeClassifier()  # Varsayılan parametrelerle karar ağacı sınıflandırıcısı oluşturuyoruz

# Karar ağacı için hiperparametre ızgarasını tanımlıyoruz
tree_params_grid={"max_depth":[3,5,7],  # Ağacın maksimum derinliğini test ediyoruz (overfitting'i önlemek için)
                  "max_leaf_nodes":[None,5,10,20,30,50],  # Maksimum yaprak düğüm sayısını test ediyoruz
                 }

# Grid Search ile karar ağacı hiperparametrelerini optimize ediyoruz
tree_params_search=RandomizedSearchCV(tree,tree_params_grid)  # Not: Burada GridSearchCV yerine RandomizedSearchCV kullanılmış
tree_params_search.fit(X_train,y_train)  # Eğitim verisi üzerinde arama yapıyoruz
print("Decision Tree Params Search Best Params: ",tree_params_search.best_params_)  # En iyi parametreleri yazdırıyoruz
print("Decision Tree Params Search Best Score: ",tree_params_search.best_score_)  # En iyi skorunu yazdırıyoruz

# Random Search ile karar ağacı hiperparametrelerini optimize ediyoruz
tree_random_search=RandomizedSearchCV(tree,tree_params_grid)  # Rastgele arama ile hiperparametre optimizasyonu
tree_random_search.fit(X_train,y_train)  # Eğitim verisi üzerinde random search'ü çalıştırıyoruz
print("Decision Tree Random Search Best Params: ",tree_random_search.best_params_)  # En iyi parametreleri yazdırıyoruz
print("Decision Tree Random Search Best Score: ",tree_random_search.best_score_)  # En iyi skorunu yazdırıyoruz

# ======================= DESTEK VEKTÖR MAKİNESİ (SVM) ALGORİTMASI =======================

# SVM modelini oluşturuyoruz
svm=SVC()  # Varsayılan parametrelerle SVM sınıflandırıcısı oluşturuyoruz

# SVM için hiperparametre ızgarasını tanımlıyoruz
svm_params_grid={"C":[0.1,1,10,100],  # Regularization parametresi C'yi test ediyoruz (overfitting kontrolü)
                 "gamma":[0.1,0.01,0.001,0.0001],  # RBF kernel'ın gamma parametresini test ediyoruz (karar sınırının şekli için)
                 }

# Grid Search ile SVM hiperparametrelerini optimize ediyoruz
svm_params_search=RandomizedSearchCV(svm,svm_params_grid)  # Not: Burada da GridSearchCV yerine RandomizedSearchCV kullanılmış
svm_params_search.fit(X_train,y_train)  # Eğitim verisi üzerinde arama yapıyoruz
print("SVM Params Search Best Params: ",svm_params_search.best_params_)  # En iyi parametreleri yazdırıyoruz
print("SVM Params Search Best Score: ",svm_params_search.best_score_)  # En iyi skorunu yazdırıyoruz

# Random Search ile SVM hiperparametrelerini optimize ediyoruz
svm_random_search=RandomizedSearchCV(svm,svm_params_grid)  # Rastgele arama ile hiperparametre optimizasyonu
svm_random_search.fit(X_train,y_train)  # Eğitim verisi üzerinde random search'ü çalıştırıyoruz
print("SVM Random Search Best Params: ",svm_random_search.best_params_)  # En iyi parametreleri yazdırıyoruz
print("SVM Random Search Best Score: ",svm_random_search.best_score_)  # En iyi skorunu yazdırıyoruz

"""
======================= HİPERPARAMETRE OPTİMİZASYONU HAKKINDA DETAYLI AÇIKLAMA =======================

Bu kod, makine öğrenmesi modellerinin performansını artırmak için HİPERPARAMETRE OPTİMİZASYONU tekniklerini göstermektedir.

1. HİPERPARAMETRE NEDİR?
   - Hiperparametreler, modelin öğrenme sürecini kontrol eden ve eğitim öncesinde belirlenmesi gereken parametrelerdir
   - Model parametrelerinden farklıdır (model parametreleri eğitim sırasında öğrenilir)
   - Örnek: KNN'de komşu sayısı (n_neighbors), Karar Ağacında maksimum derinlik (max_depth)

2. GRID SEARCH (IZGARA ARAMASI):
   - Belirtilen hiperparametre değerlerinin tüm kombinasyonlarını sistematik olarak dener
   - Kapsamlı ama hesaplama açısından pahalı bir yöntemdir
   - Her kombinasyon için cross-validation yaparak en iyi sonucu verir
   - Küçük hiperparametre uzayları için idealdir

3. RANDOM SEARCH (RASTGELE ARAMA):
   - Hiperparametre uzayından rastgele örnekler alarak arama yapar
   - Grid Search'ten daha hızlıdır ve büyük hiperparametre uzayları için daha etkilidir
   - Genellikle Grid Search kadar iyi sonuçlar verir ama çok daha az hesaplama gerektirir
   - Yüksek boyutlu hiperparametre uzayları için tercih edilir

4. CROSS-VALIDATION:
   - Model performansını değerlendirmek için veriyi k parçaya böler
   - Her seferinde 1 parçayı test, geri kalanını eğitim için kullanır
   - Overfitting'i önler ve modelin genelleme yeteneğini test eder
   - Bu kodda 5-fold cross-validation kullanılmıştır

5. KULLANILAN ALGORİTMALAR:
   
   a) K-En Yakın Komşu (KNN):
      - n_neighbors: Karar verirken bakılacak komşu sayısı
      - Çok az komşu = overfitting, çok fazla komşu = underfitting

   b) Karar Ağacı (Decision Tree):
      - max_depth: Ağacın maksimum derinliği (overfitting kontrolü)
      - max_leaf_nodes: Maksimum yaprak düğüm sayısı (komplekslik kontrolü)

   c) Destek Vektör Makinesi (SVM):
      - C: Regularization parametresi (overfitting-underfitting dengesi)
      - gamma: RBF kernel parametresi (karar sınırının esnekliği)

6. AMAÇ:
   - Her algoritma için en iyi hiperparametre kombinasyonunu bulmak
   - Model performansını maksimize etmek
   - Overfitting ve underfitting'i önlemek
   - Genelleme yeteneği yüksek modeller elde etmek

Bu teknikler, makine öğrenmesi projelerinde model performansını artırmak için kritik öneme sahiptir.
Doğru hiperparametre seçimi, aynı algoritmayla çok farklı performans sonuçları elde edilmesini sağlar.
"""





