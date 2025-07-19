# Cross Validation Yöntemleri: KFold vs Leave-One-Out (LOO) Karşılaştırması
# Bu kod, makine öğrenmesinde model doğrulama için kullanılan iki farklı cross validation yöntemini karşılaştırır
# KFold: Veriyi k parçaya böler, k-1 parça ile eğitir, 1 parça ile test eder
# Leave-One-Out: Her seferinde sadece 1 örneği test için ayırır, geriye kalan tüm örneklerle eğitir

# Gerekli kütüphaneleri içe aktarıyoruz
from sklearn.datasets import load_iris  # Iris veri setini yüklemek için
from sklearn.model_selection import KFold,LeaveOneOut,train_test_split,GridSearchCV  # Cross validation ve hiperparametre optimizasyonu için
from sklearn.tree import DecisionTreeClassifier  # Karar ağacı sınıflandırıcısı için

# Iris veri setini yüklüyoruz (150 örnek, 4 özellik, 3 sınıf içeren klasik sınıflandırma veri seti)
iris=load_iris()
X=iris.data  # Özellik matrisi (150x4) - sepal/petal length/width değerleri
y=iris.target  # Hedef değişken (150,) - çiçek türleri: setosa, versicolor, virginica

# Veriyi eğitim (%80) ve test (%20) setlerine ayırıyoruz
# random_state=42 ile sonuçların tekrarlanabilir olmasını sağlıyoruz
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Karar ağacı sınıflandırıcısı modelini oluşturuyoruz
tree=DecisionTreeClassifier()

# Hiperparametre arama uzayını tanımlıyoruz
# max_depth: Ağacın maksimum derinliği (overfitting'i önlemek için)
tree_params_dist={
    "max_depth":[3,5,7],  # 3 farklı derinlik değeri deneyeceğiz
}

# ========== KFOLD CROSS VALIDATION ==========
# KFold: Eğitim verisini 10 eşit parçaya böler
# Her iterasyonda 9 parça eğitim, 1 parça doğrulama için kullanılır
# shuffle=True: Veriyi karıştırır (bias'ı azaltır)
# random_state=42: Sonuçların tekrarlanabilir olması için
kf=KFold(n_splits=10,shuffle=True,random_state=42)

# GridSearchCV ile hiperparametre optimizasyonu yapıyoruz
# KFold cross validation kullanarak en iyi max_depth değerini bulur
# scoring="accuracy": Doğruluk (accuracy) metriğini optimize eder
tree_grid_search=GridSearchCV(tree,tree_params_dist,cv=kf,scoring="accuracy")

# Modeli eğitim verisi üzerinde eğitir ve en iyi hiperparametreleri bulur
tree_grid_search.fit(X_train,y_train)

# KFold ile bulunan en iyi hiperparametreleri yazdırıyoruz
print("Decision Tree Grid KF Best Params: ",tree_grid_search.best_params_)
# KFold ile elde edilen en iyi cross validation skorunu yazdırıyoruz
print("Decision Tree Grid KF Best Score: ",tree_grid_search.best_score_)

# ========== LEAVE-ONE-OUT CROSS VALIDATION ==========
# LeaveOneOut: Her iterasyonda sadece 1 örneği test için ayırır
# Eğer eğitim setinde n örnek varsa, n iterasyon yapar
# Bu yöntem daha kapsamlı fakat hesaplama açısından daha pahalıdır
loo=LeaveOneOut()

# GridSearchCV ile hiperparametre optimizasyonu yapıyoruz
# Leave-One-Out cross validation kullanarak en iyi max_depth değerini bulur
tree_grid_search_loo=GridSearchCV(tree,tree_params_dist,cv=loo,scoring="accuracy")

# Modeli eğitim verisi üzerinde eğitir ve en iyi hiperparametreleri bulur
tree_grid_search_loo.fit(X_train,y_train)

# Leave-One-Out ile bulunan en iyi hiperparametreleri yazdırıyoruz
print("Decision Tree Grid Search LOO Best Params: ",tree_grid_search_loo.best_params_)
# Leave-One-Out ile elde edilen en iyi cross validation skorunu yazdırıyoruz
print("Decision Tree Grid Search LOO Best Score: ",tree_grid_search_loo.best_score_)

# ========== SONUÇ VE DEĞERLENDİRME ==========
# KFold: Daha hızlı, daha az hesaplama maliyeti, genelde yeterli doğruluk
# Leave-One-Out: Daha yavaş, yüksek hesaplama maliyeti, maksimum veri kullanımı
# Küçük veri setlerinde LOO, büyük veri setlerinde KFold tercih edilir






