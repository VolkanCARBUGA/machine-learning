# Naive Bayes Sınıflandırma Algoritması
#
# Naive Bayes, olasılık teorisine dayanan bir sınıflandırma algoritmasıdır.
# Bayes teoremini kullanarak özniteliklerin birbirinden bağımsız olduğunu varsayar.
# Bu nedenle "naive" (saf) olarak adlandırılır.
# Özellikle metin sınıflandırma ve spam tespiti gibi alanlarda yaygın kullanılır.

# Gerekli kütüphanelerin içe aktarılması
from sklearn.datasets import load_iris  # Iris veri setini yüklemek için
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak bölmek için
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes sınıflandırıcısı
from sklearn.metrics import classification_report  # Model performans metrikleri için

# Iris veri setinin yüklenmesi
iris = load_iris()  # Iris veri setini yükle
X = iris.data  # Öznitelik verileri (çiçeğin ölçümleri)
y = iris.target  # Hedef değişken (çiçek türleri)

# Veri setinin eğitim ve test olarak bölünmesi
# test_size=0.2: Verinin %20'si test için kullanılacak
# random_state=42: Tekrarlanabilirlik için sabit rastgele sayı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gaussian Naive Bayes modelinin oluşturulması ve eğitilmesi
nb_clf = GaussianNB()  # Gaussian Naive Bayes sınıflandırıcı nesnesi oluştur
nb_clf.fit(X_train, y_train)  # Modeli eğitim verileriyle eğit

# Test verileri üzerinde tahmin yapma
y_pred = nb_clf.predict(X_test)  # Test verileri üzerinde tahminler yap

# Model performansının değerlendirilmesi
# Classification report, precision, recall, f1-score gibi metrikleri gösterir
print(classification_report(y_test, y_pred))  # Model performans raporunu yazdır





