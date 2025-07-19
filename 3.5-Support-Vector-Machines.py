# Support Vector Machines (SVM) Sınıflandırma Algoritması
#
# SVM, veri noktalarını en iyi şekilde ayıran bir hiper düzlem bularak sınıflandırma yapan güçlü bir 
# makine öğrenmesi algoritmasıdır. Doğrusal ve doğrusal olmayan sınıflandırma problemlerinde kullanılabilir.
# Bu örnekte, el yazısı rakamları tanıma problemi üzerinde SVM kullanılmaktadır.
#
# Temel özellikleri:
# - Margin maksimizasyonu ile sınıfları ayırır
# - Kernel trick sayesinde doğrusal olmayan sınıflandırma yapabilir
# - Yüksek boyutlu verilerde etkilidir
# - Aşırı öğrenmeye karşı dirençlidir

# Gerekli kütüphanelerin import edilmesi
from sklearn.datasets import load_digits  # El yazısı rakam veriseti için
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test olarak bölmek için
from sklearn.metrics import classification_report  # Model performans metrikleri için
from sklearn.svm import SVC  # Support Vector Classification sınıfı
import matplotlib.pyplot as plt  # Görselleştirme için

# Digits (el yazısı rakamlar) veri setinin yüklenmesi
digits = load_digits()

# Örnek rakam görüntülerinin görselleştirilmesi
# 2x5'lik bir grid oluşturarak ilk 10 rakamı gösterir
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,5), subplot_kw={"xticks":[], "yticks":[]})
for i, ax in enumerate(axes.flat):
    # Her bir rakam görüntüsünü binary renk haritası ile göster
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
    # Görüntünün üzerine rakamın değerini yaz
    ax.set_title(f"Digit: {digits.target[i]}")
plt.show()

# Veri setinin özellikler (X) ve hedef değişken (y) olarak ayrılması
X = digits.data  # Özellikler: her piksel bir özellik
y = digits.target  # Hedef değişken: rakamın gerçek değeri (0-9)

# Veri setinin eğitim ve test olarak bölünmesi
# test_size=0.2: verinin %20'si test için ayrılır
# random_state=42: tekrar üretilebilirlik için sabit sayı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM sınıflandırıcısının oluşturulması ve eğitilmesi
# kernel="linear": doğrusal kernel kullanılır
# C=1.0: düzenlileştirme parametresi
svm_clf = SVC(kernel="linear", C=1.0, random_state=42)
svm_clf.fit(X_train, y_train)  # Modelin eğitim verileriyle eğitilmesi

# Test verileri üzerinde tahmin yapılması ve performans değerlendirmesi
y_pred = svm_clf.predict(X_test)  # Test verileri üzerinde tahmin
# Sınıflandırma raporunun yazdırılması (precision, recall, f1-score)
print(classification_report(y_test, y_pred))





