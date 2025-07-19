import gymnasium as gym  # Gymnasium kütüphanesi - OpenAI'ın gym environment'larını kullanmak için
from tqdm import tqdm  # İlerleme çubuğu göstermek için
import matplotlib.pyplot as plt  # Grafik çizmek için
import random  # Rastgele sayı üretmek için (kullanılmıyor ama import edilmiş)
import numpy as np  # Sayısal hesaplamalar ve matris işlemleri için

# FrozenLake environment'ını oluştur - buzlu göl ortamı simülasyonu
environment = gym.make("FrozenLake-v1")
state, info = environment.reset()  # Environment'ı başlangıç durumuna döndür ve ilk durumu al
nb_states = environment.observation_space.n  # Toplam durum sayısını al (FrozenLake'de 16 durum var)
nb_actions = environment.action_space.n  # Toplam aksiyon sayısını al (4 yön: yukarı, aşağı, sol, sağ)

# Q-tablosunu sıfırlarla başlat - her durum-aksiyon çifti için Q değeri
q_table = np.zeros([nb_states, nb_actions])
print("Q-table boyutu:", q_table.shape)  # Q-tablosunun boyutunu yazdır (16x4)
print("Başlangıç durumu:", state)  # Başlangıç durumunu yazdır

# Rastgele bir aksiyon seç ve test et
action = environment.action_space.sample()  # Rastgele aksiyon seç
new_state, reward, terminated, truncated, info = environment.step(action)  # Aksiyonu uygula
done = terminated or truncated  # Oyun bitip bitmediğini kontrol et
print("Yeni durum:", new_state, "Ödül:", reward, "Bitti mi:", done, "Bilgi:", info)

# EĞİTİM AŞAMASI - Q-Learning algoritması
episodes = 10000  # Eğitim için toplam episode sayısı
alpha = 0.5  # Öğrenme oranı - yeni bilginin ne kadar hızlı öğrenileceği
gamma = 0.9  # İndirim faktörü - gelecekteki ödüllerin şimdiki değeri
outcomes = []  # Her episode'un sonucunu saklamak için liste

for _ in tqdm(range(episodes)):  # Her episode için döngü (ilerleme çubuğu ile)
    state, info = environment.reset()  # Her episode başında environment'ı sıfırla
    done = False  # Episode bitip bitmediğini takip eden değişken

    outcomes.append("Failure")  # Başlangıçta başarısız olarak işaretle

    while not done:  # Episode bitene kadar devam et
        # Epsilon-greedy stratejisi: eğer o durum için öğrenilmiş bir değer varsa en iyi aksiyonu seç
        if np.max(q_table[state]) > 0:
            action = np.argmax(q_table[state])  # En yüksek Q değerine sahip aksiyonu seç
        else:
            action = environment.action_space.sample()  # Rastgele aksiyon seç (exploration)
        
        # Seçilen aksiyonu uygula ve sonuçları al
        new_state, reward, terminated, truncated, info = environment.step(action)
        
        # Q-Learning güncelleme formülü: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )
        
        state = new_state  # Yeni duruma geç
        done = terminated or truncated  # Episode bitip bitmediğini kontrol et
        
        if reward:  # Eğer ödül aldıysak (hedefe ulaştıysak)
            outcomes[-1] = "Success"  # Bu episode'u başarılı olarak işaretle

# Eğitim sonuçlarını göster
print("Başarı oranı:",outcomes.count("Success")/episodes)  # Başarılı episode'ların yüzdesini hesapla
plt.bar(range(episodes),outcomes)  # Sonuçları bar grafik olarak çiz
plt.xlabel("Eğitim")  # X ekseni etiketi
plt.ylabel("Başarı")  # Y ekseni etiketi
plt.title("Q-Learning")  # Grafik başlığı
#plt.show()  # Grafiği göster (yorum satırında)

#test

# TEST AŞAMASI - Eğitilmiş Q-tablosunu test et
episodes = 10000  # Test için episode sayısı
nb_success=0  # Başarılı episode sayacı

for _ in tqdm(range(episodes)):  # Her test episode'u için
    state, _ = environment.reset()  # Environment'ı sıfırla
    done = False  # Episode durumu

    outcomes.append("Failure")  # Bu satır yanlış yerde, test için outcomes kullanılmamalı

    while not done:  # Episode bitene kadar
        # Sadece öğrenilmiş politikayı kullan (exploration yok)
        if np.max(q_table[state]) > 0:
            action = np.argmax(q_table[state])  # En iyi aksiyonu seç
        else:
            action = environment.action_space.sample()  # Eğer hiç öğrenilmemişse rastgele
        
        new_state, reward, terminated, truncated, info = environment.step(action)  # Aksiyonu uygula
        
        state = new_state  # Yeni duruma geç
        nb_success+=reward  # Ödül aldıysak başarı sayacını artır

# Test sonuçlarını yazdır
print("Başarı oranı:",100*nb_success/episodes)  # Test başarı oranını yüzde olarak hesapla

"""
Q-LEARNING KONUSU AÇIKLAMASI:

Q-Learning, Reinforcement Learning (Pekiştirmeli Öğrenme) algoritmasının en popüler türlerinden biridir.
Bu algoritma, bir ajanın çevre ile etkileşime girerek optimal politikayı öğrenmesini sağlar.

TEMEL KAVRAMLAR:
1. STATE (Durum): Ajanın bulunduğu konum/durum
2. ACTION (Aksiyon): Ajanın yapabileceği hareketler
3. REWARD (Ödül): Aksiyonun sonucunda alınan geri bildirim
4. Q-TABLE: Her durum-aksiyon çifti için öğrenilen değerler tablosu

FROZEN LAKE PROBLEMİ:
- 4x4'lük buzlu göl üzerinde hareket eden ajan
- Başlangıç noktasından (S) hedefe (G) ulaşmaya çalışır
- Deliklerden (H) kaçınmalı, buzlu zeminde (F) kayabilir
- Amac: En kısa ve güvenli yoldan hedefe ulaşmak

Q-LEARNING FORMÜLÜ:
Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]

Burada:
- α (alpha): Öğrenme oranı (0.5) - Ne kadar hızlı öğreneceği
- γ (gamma): İndirim faktörü (0.9) - Gelecek ödüllerin değeri
- r: Anlık ödül
- s: Mevcut durum, s': Yeni durum
- a: Mevcut aksiyon, a': Yeni durumdaki en iyi aksiyon

ALGORİTMA AŞAMALARI:
1. Q-tablosunu sıfırla
2. Her episode için:
   - Başlangıç durumuna dön
   - Episode bitene kadar:
     * En iyi aksiyonu seç (exploitation) veya rastgele seç (exploration)
     * Aksiyonu uygula
     * Q-tablosunu güncelle
     * Yeni duruma geç
3. Öğrenilen politikayı test et

Bu kod 10.000 episode eğitim yapıp, sonra 10.000 episode test ederek
Q-Learning algoritmasının FrozenLake problemindeki performansını ölçer.
"""
        