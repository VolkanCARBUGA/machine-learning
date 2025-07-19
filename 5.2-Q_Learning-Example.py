import gymnasium as gym  # OpenAI Gym kütüphanesi - pekiştirmeli öğrenme ortamları
import numpy as np  # Sayısal hesaplamalar için NumPy kütüphanesi
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib
import random  # Rastgele sayı üretimi için
from tqdm import tqdm  # İlerleme çubuğu göstermek için

# Taxi-v3 ortamını oluştur (taksi oyunu simülasyonu)
env = gym.make("Taxi-v3")

# Ortamı başlangıç durumuna getir
env.reset()

"""
Hareket kodları:
0: güney (aşağı git)
1: kuzey (yukarı git)  
2: doğu (sağa git)
3: batı (sola git)
4: yolcu almak
5: yolcu bırakmak
"""

# Mümkün olan toplam hareket sayısını al
action_space = env.action_space.n

# Mümkün olan toplam durum sayısını al  
state_space = env.observation_space.n

# Q-tablosunu sıfırlarla başlat (durum x hareket matrisi)
q_table = np.zeros([state_space, action_space])

# Hiperparametreler
alpha = 0.1    # Öğrenme oranı (learning rate) - ne kadar hızlı öğrensin
gamma = 0.9    # İndirim faktörü (discount factor) - gelecekteki ödüllerin önemi
epsilon = 0.1  # Keşif oranı (exploration rate) - rastgele hareket yapma olasılığı

# Eğitim döngüsü - 100.000 episode çalıştır
for i in range(1, 100001):
    
    # Her episode'da ortamı sıfırla ve başlangıç durumunu al
    state, _ = env.reset()
    
    # Episode istatistikleri
    epochs, penalties, reward = 0, 0, 0
    done = False  # Episode bitmiş mi kontrolü
    
    # Episode bitene kadar devam et
    while not done:
        # Epsilon-greedy stratejisi: %10 ihtimalle rastgele hareket (exploration)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Rastgele hareket seç
        else:  # %90 ihtimalle en iyi hareketi seç (exploitation)
            action = np.argmax(q_table[state])  # Q-tablosundan en yüksek değerli hareketi seç
        
        # Seçilen hareketi gerçekleştir ve sonuçları al
        next_state, reward, done, info, _ = env.step(action)
        
        # Q-Learning güncellemesi (Bellman denklemi)
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        # Bir sonraki duruma geç
        state = next_state

print("Eğitim tamamlandı!")

# Test aşaması - eğitilen modeli değerlendir
total_epoch, total_penalties = 0, 0
episodes = 100  # 100 test episode'u çalıştır

print("Test aşaması başlıyor...")
for i in tqdm(range(episodes)):
    # Her test episode'unda ortamı sıfırla
    state, _ = env.reset()
    epochs, penalties = 0, 0  # Episode istatistiklerini sıfırla
    done = False
    
    # Episode bitene kadar devam et
    while not done:
        # Test aşamasında sadece exploitation (en iyi hareketi seç)
        action = np.argmax(q_table[state])
        
        # Hareketi gerçekleştir
        next_state, reward, done, info, _ = env.step(action)
        state = next_state
        
        # Ceza aldıysa say (yanlış hareket)
        if reward == -10:
            penalties += 1
        
        epochs += 1  # Adım sayısını arttır
    
    # Toplam istatistikleri güncelle
    total_epoch += epochs
    total_penalties += penalties

# Test sonuçlarını yazdır
print(f"Sonuçlar {episodes} episode sonrası:")
print(f"Episode başına ortalama adım sayısı: {total_epoch/episodes}")
print(f"Episode başına ortalama ceza sayısı: {total_penalties/episodes}")

"""
Q-LEARNING NEDİR?

Q-Learning, pekiştirmeli öğrenmenin (reinforcement learning) temel algoritmalarından biridir.
Bir ajanın (agent) çevreyle etkileşime girerek optimal politikayı öğrenmesini sağlar.

TEMEL KAVRAMLAR:
- State (Durum): Ajanın bulunduğu mevcut konum/durum
- Action (Hareket): Ajanın yapabileceği eylemler
- Reward (Ödül): Bir hareketten sonra alınan geri bildirim
- Q-Table: Her durum-hareket çifti için beklenen toplam ödülü tutan tablo

ÇALIŞMA PRENSİBİ:
1. Ajan bir durumda başlar
2. Epsilon-greedy stratejisiyle hareket seçer (exploration vs exploitation)
3. Hareketi gerçekleştirir ve ödül alır
4. Q-tablosunu Bellman denklemiyle günceller
5. Yeni duruma geçer ve işlemi tekrarlar

BELLMAN DENKLEMİ:
Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
- α: Öğrenme oranı (ne kadar hızlı öğrensin)
- γ: İndirim faktörü (gelecekteki ödüllerin önemi)
- r: Anlık ödül
- max(Q(s',a')): Bir sonraki durumda alınabilecek maksimum Q değeri

TAXI PROBLEMI:
Bu örnekte ajan bir taksi şoförüdür. Görevi:
- Yolcuyu belirli bir noktadan alıp
- Hedef noktaya güvenli şekilde bırakmak
- Minimum adımda ve minimum ceza ile

SONUÇ:
Eğitim sonrası ajan, rastgele hareket etmek yerine öğrendiği Q-tablosunu
kullanarak en optimal yolu bulup görevi başarıyla tamamlar.
"""
        
        
        
        