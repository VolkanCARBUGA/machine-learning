from sklearn.datasets import load_diabetes  # Scikit-learn'den diabetes dataset'ini yüklemek için gerekli fonksiyon
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için
from sklearn.linear_model import Ridge,Lasso  # Ridge (L2) ve Lasso (L1) regularizasyon modellerini import ediyoruz
from sklearn.model_selection import GridSearchCV  # En iyi hiperparametreleri bulmak için grid search
from sklearn.metrics import mean_squared_error  # Model performansını ölçmek için ortalama karesel hata
import matplotlib.pyplot as plt  # Grafik çizmek için (şu an kullanılmıyor ama ileride kullanılabilir)
import numpy as np  # Sayısal işlemler için numpy kütüphanesi

diabetes=load_diabetes()  # Diabetes dataset'ini yüklüyoruz (diyabet hastaları üzerinde regresyon problemi)
X=diabetes.data  # Bağımsız değişkenler (özellikler) - hasta bilgileri
y=diabetes.target  # Bağımlı değişken (hedef) - diyabet ilerlemesi sayısal değeri

# Veriyi %80 eğitim, %20 test olacak şekilde rastgele ayırıyoruz (random_state=42 ile tekrarlanabilir sonuçlar)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Ridge Regression (L2 Regularizasyon)
# Ridge regresyon, katsayıların karelerinin toplamını cezalandırarak overfitting'i engeller

ridge=Ridge()  # Ridge regresyon modelini oluşturuyoruz
ridge_params_grid={"alpha":[0.1,1,10,100]}  # Alpha parametresi için test edilecek değerler (regularizasyon gücü)
# GridSearchCV ile farklı alpha değerlerini deneyip en iyisini buluyoruz (5-fold cross validation ile)
ridge_params_search=GridSearchCV(ridge,ridge_params_grid,cv=5,scoring="neg_mean_squared_error")
ridge_params_search.fit(X_train,y_train)  # Modeli eğitim verisi üzerinde eğitiyoruz
print("Ridge Regression Best Params: ",ridge_params_search.best_params_)  # En iyi alpha değerini yazdırıyoruz
print("Ridge Regression Best Score: ",ridge_params_search.best_score_)  # En iyi cross-validation skorunu yazdırıyoruz
best_ridge_model=ridge_params_search.best_estimator_  # En iyi parametrelerle eğitilmiş modeli alıyoruz
y_pred_ridge=best_ridge_model.predict(X_test)  # Test verisi üzerinde tahmin yapıyoruz
ridge_mse=mean_squared_error(y_test,y_pred_ridge)  # Gerçek ve tahmin edilen değerler arasındaki MSE'yi hesaplıyoruz
print("Ridge Regression MSE: ",ridge_mse)  # Ridge regresyon için test MSE'sini yazdırıyoruz


# Lasso Regression (L1 Regularizasyon)
# Lasso regresyon, katsayıların mutlak değerlerinin toplamını cezalandırır ve özellik seçimi yapar

lasso=Lasso()  # Lasso regresyon modelini oluşturuyoruz
lasso_params_grid={"alpha":[0.1,1,10,100]}  # Alpha parametresi için test edilecek değerler
# GridSearchCV ile farklı alpha değerlerini deneyip en iyisini buluyoruz
lasso_params_search=GridSearchCV(lasso,lasso_params_grid,cv=5,scoring="neg_mean_squared_error")
lasso_params_search.fit(X_train,y_train)  # Modeli eğitim verisi üzerinde eğitiyoruz
print("Lasso Regression Best Params: ",lasso_params_search.best_params_)  # En iyi alpha değerini yazdırıyoruz
print("Lasso Regression Best Score: ",lasso_params_search.best_score_)  # En iyi cross-validation skorunu yazdırıyoruz
best_lasso_model=lasso_params_search.best_estimator_  # En iyi parametrelerle eğitilmiş modeli alıyoruz
y_pred_lasso=best_lasso_model.predict(X_test)  # Test verisi üzerinde tahmin yapıyoruz
lasso_mse=mean_squared_error(y_test,y_pred_lasso)  # Gerçek ve tahmin edilen değerler arasındaki MSE'yi hesaplıyoruz
print("Lasso Regression MSE: ",lasso_mse)  # Lasso regresyon için test MSE'sini yazdırıyoruz

"""
REGULARİZASYON TEKNİKLERİ HAKKINDA DETAYLI AÇIKLAMA:

1. REGULARİZASYON NEDİR?
   - Regularizasyon, makine öğrenmesi modellerinde overfitting (aşırı öğrenme) problemini çözmek için kullanılan tekniktir
   - Model çok karmaşık hale geldiğinde eğitim verisini ezberler ama yeni verilerde başarısız olur
   - Regularizasyon, modelin karmaşıklığını sınırlayarak genelleme yetisini artırır

2. L1 REGULARİZASYON (LASSO):
   - Lasso: "Least Absolute Shrinkage and Selection Operator"
   - Maliyet fonksiyonuna katsayıların mutlak değerlerinin toplamını ekler
   - Formül: MSE + α * Σ|βi|
   - Özellik seçimi yapar: Önemsiz özelliklerin katsayılarını sıfıra indirger
   - Sparse (seyrek) modeller oluşturur
   - Yüksek boyutlu verilerde özellik sayısını azaltmak için idealdir

3. L2 REGULARİZASYON (RIDGE):
   - Ridge regresyon, katsayıların karelerinin toplamını cezalandırır
   - Formül: MSE + α * Σβi²
   - Katsayıları küçültür ama sıfıra indirmez
   - Tüm özellikleri modelde tutar
   - Multicollinearity (çoklu doğrusal bağlantı) problemine karşı daha dayanıklıdır

4. ALPHA PARAMETRESİ:
   - α (alpha): Regularizasyon gücünü kontrol eder
   - α = 0: Regularizasyon yok (normal linear regression)
   - Yüksek α: Güçlü regularizasyon, basit model
   - Düşük α: Zayıf regularizasyon, karmaşık model
   - GridSearchCV ile optimal α değeri bulunur

5. KARŞILAŞTIRMA:
   - Lasso: Özellik seçimi + regularizasyon
   - Ridge: Sadece regularizasyon
   - Elastic Net: L1 + L2 kombinasyonu
   
Bu kodda diabetes dataset'i kullanılarak hem Ridge hem de Lasso regresyon modelleri eğitilmiş,
en iyi hiperparametreler bulunmuş ve test performansları karşılaştırılmıştır.
"""
