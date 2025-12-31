# -*- coding: utf-8 -*-
"""
En İyi Yol Bulma İçin Parçacık Sürü Optimizasyonu (PSO)
Dosya: pso_optimization.py
"""

import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import math

# Tekrarlanabilirlik için tohum ayarla
random.seed(42)
np.random.seed(42)

# --- AĞ VERİSİNİN CSV'DEN YÜKLENMESİ FONKSİYONU (ACO'dan korunmuştur) ---

def load_network_data():
    """
    Node ve link verilerini CSV dosyalarından yükler ve NetworkX Grafiğini oluşturur.
    Göreceli yollar, root klasöründen çalıştırıldığında sağlam olacak şekilde ayarlanmıştır.
    """
    
    # 1. Dosya Yollarını Belirle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    node_file_path = os.path.join(base_dir, 'data', 'node_properties.csv')
    link_file_path = os.path.join(base_dir, 'data', 'link_properties.csv')
    
    # Okumaya çalışmadan önce dosyaların var olup olmadığını kontrol et
    if not os.path.exists(node_file_path) or not os.path.exists(link_file_path):
        # Betiğin kök dizinden çalıştırıldığı varsayımıyla tekrar dene
        node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'node_properties.csv')
        link_file_path_fallback = os.path.join(os.getcwd(), 'data', 'link_properties.csv')

        if os.path.exists(node_file_path_fallback) and os.path.exists(link_file_path_fallback):
             node_file_path = node_file_path_fallback
             link_file_path = link_file_path_fallback
        else:
             print(f"\n[HATA] Dosya denenen hiçbir yolda bulunamadı.")
             print(f"Lütfen CSV dosyalarının proje kökündeki 'data' klasöründe olduğundan emin olun.")
             return None, None, None
    
    print(f"Veri şu konumlardan yüklenmeye çalışılıyor:\nNode: {node_file_path}\nLink: {link_file_path}")
    
    try:
        # 2. Veriyi Yükle
        node_df = pd.read_csv(node_file_path)
        link_df = pd.read_csv(link_file_path)
        
        # 3. Ağ Grafiğini Oluştur
        G = nx.Graph()

        # Node'ları ve özelliklerini ekle
        for index, row in node_df.iterrows():
            node_id = row['NodeID']
            G.add_node(
                node_id, 
                ProcessingDelay=row['ProcessingDelay'], 
                NodeReliability=row['NodeReliability']
            )

        # Link'leri ve özelliklerini ekle
        for index, row in link_df.iterrows():
            source = row['Source']
            destination = row['Destination']
            G.add_edge(
                source, 
                destination, 
                Bandwidth=row['Bandwidth'], 
                LinkDelay=row['LinkDelay'], 
                LinkReliability=row['LinkReliability']
            )
            
        print(f"Grafik, {G.number_of_nodes()} node ve {G.number_of_edges()} link ile başarıyla oluşturuldu.")
        
        # Kaynak ve Hedef node'ları varsay
        source_node = node_df['NodeID'].min()
        destination_node = node_df['NodeID'].max()
        
        return G, source_node, destination_node

    except Exception as e:
        print(f"\n[HATA] CSV verisi işlenirken bir hata oluştu: {e}")
        return None, None, None


# --- METRİK VE UYGUNLUK (FITNESS) FONKSİYONLARI (ACO'dan korunmuştur) ---

def calculate_path_metrics(graph, path):
    """
    Yol metriklerini (Güvenilirlik, Gecikme, Bant Genişliği) hesaplar.
    Güvenilirlik: Tüm link ve node güvenilirliklerinin çarpımı.
    Gecikme: Tüm link gecikmeleri ve işleme gecikmelerinin toplamı.
    Bant Genişliği: Tüm link bant genişliklerinin minimum değeri.
    """
    if not path:
        return 0.0, float('inf'), 0.0

    total_reliability = 1.0
    total_delay = 0.0
    min_bandwidth = float('inf')

    # Node metriklerini hesapla
    for node in path:
        node_data = graph.nodes[node]
        total_delay += node_data.get('ProcessingDelay', 0.0)
        total_reliability *= node_data.get('NodeReliability', 1.0)

    # Link metriklerini hesapla
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = graph.edges.get((u, v), {}) 

        total_delay += edge_data.get('LinkDelay', 0.0)
        total_reliability *= edge_data.get('LinkReliability', 1.0)
        min_bandwidth = min(min_bandwidth, edge_data.get('Bandwidth', float('inf')))

    return total_reliability, total_delay, min_bandwidth if min_bandwidth != float('inf') else 0.0

def fitness_function(path, graph, source, destination):
    """
    Çok Amaçlı Uygunluk (Fitness) Fonksiyonu: Fitness = (Güvenilirlik * Bant Genişliği) / Gecikme
    Amaç: Fitness'ı Maksimize Etmek.
    """
    if not path or path[-1] != destination:
        return 0.0

    reliability, delay, bandwidth = calculate_path_metrics(graph, path)

    # Sıfıra bölmeyi önle
    if delay <= 1e-6:
        return 0.0
    
    fitness = (reliability * bandwidth) / delay
    return fitness

def calculate_reliability_cost(graph, path):
    """
    Güvenilirlik Maliyetini (Reliability Cost) şu formüle göre hesaplar:
    ReliabilityCost(P) = Toplam[-log(LinkReliability)] + Toplam[-log(NodeReliability)]
    """
    if not path:
        return float('inf')

    total_cost = 0.0

    # 1. Node Güvenilirlik Maliyeti
    for node in path:
        node_data = graph.nodes[node]
        reliability = node_data.get('NodeReliability', 1.0)
        if reliability > 0:
            total_cost += -np.log(reliability)
        else:
            total_cost += float('inf') 

    # 2. Link Güvenilirlik Maliyeti
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = graph.edges.get((u, v), {})
        reliability = edge_data.get('LinkReliability', 1.0)
        
        if reliability > 0:
            total_cost += -np.log(reliability)
        else:
            total_cost += float('inf')
            
    return total_cost

def calculate_resource_cost(graph, path, max_bandwidth=1.0):
    """
    Kaynak Kullanım Maliyetini (Resource Cost) şu formüle göre hesaplar:
    ResourceCost(P) = Toplam[ (1 / Bant Genişliği) ]
    """
    if not path:
        return float('inf')

    total_cost = 0.0

    # Sadece Link metriklerini hesapla
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = graph.edges.get((u, v), {})
        bandwidth = edge_data.get('Bandwidth', 0.0) 

        if bandwidth > 0:
            total_cost += (1.0 / bandwidth)
        else:
            return float('inf') 
            
    return total_cost

def calculate_all_metrics(graph, path, source, destination):
    """
    Tüm 4 metriği ve fitness'ı hesaplamak için sarmalayıcı (wrapper) fonksiyon.
    """
    if not path:
         # Eğer yol boşsa, kötü varsayılan değerleri döndür
         return 0.0, float('inf'), 0.0, float('inf'), float('inf'), 0.0

    reliability, delay, bandwidth = calculate_path_metrics(graph, path)
    
    reliability_cost = calculate_reliability_cost(graph, path)
    resource_cost = calculate_resource_cost(graph, path)
    
    fitness = fitness_function(path, graph, source, destination)
    
    return reliability, delay, bandwidth, reliability_cost, resource_cost, fitness


# --- PSO TEMEL BİLEŞEN FONKSİYONLARI ---

def initialize_pso_metadata(graph):
    """
    PSO vektörü için NodeID'den indekse eşlemesini başlatır.
    """
    node_list = list(graph.nodes())
    node_to_index = {node_id: i for i, node_id in enumerate(node_list)}
    index_to_node = {i: node_id for i, node_id in enumerate(node_list)}
    num_nodes = len(node_list)
    return num_nodes, node_to_index, index_to_node

def construct_pso_path(graph, source, destination, position, node_to_index, max_path_length_multiplier=2):
    """
    Parçacık pozisyonuna (tercihine) göre yolu oluşturur.
    Komşu bir node'a geçme olasılığı, o node'un pozisyon vektöründeki 
    tercih değerinin üstel değeriyle orantılıdır.
    """
    path = [source]
    current_node = source
    visited_nodes = {source}
    
    # Yol uzunluğu sınırı için sezgisel yöntem
    try:
        shortest_len = nx.shortest_path_length(graph, source=source, target=destination)
        max_path_length = int(shortest_len * max_path_length_multiplier)
    except nx.NetworkXNoPath:
        max_path_length = graph.number_of_nodes() * 2 

    while current_node != destination:
        if len(path) > max_path_length: 
            return [] # Uzunluk sınırı aşıldı

        unvisited_neighbors = [neighbor for neighbor in graph.neighbors(current_node) if neighbor not in visited_nodes]

        if not unvisited_neighbors:
            return [] # Sıkışıp kaldı

        probabilities = []
        for neighbor in unvisited_neighbors:
            # Node tercihini pozisyon vektöründen al
            neighbor_index = node_to_index[neighbor]
            preference = position[neighbor_index]
            
            # Pozitif değerler sağlamak ve tercih farklılıklarını güçlendirmek için 
            # üstel değeri (burada üstel) kullan
            prob_numerator = math.exp(preference) 
            probabilities.append(prob_numerator)

        total_probability = sum(probabilities)
        if total_probability == 0:
            # Tüm olasılıklar sıfırsa geri dönüş
            next_node = random.choice(unvisited_neighbors)
        else:
            normalized_probabilities = [p / total_probability for p in probabilities]
            next_node = random.choices(unvisited_neighbors, weights=normalized_probabilities, k=1)[0]

        path.append(next_node)
        visited_nodes.add(next_node)
        current_node = next_node

    return path

def particle_swarm_optimization(graph, source, destination, num_particles, num_iterations, w, c1, c2, max_v):
    """
    Parçacık Sürü Optimizasyonu sürecini yönetir.
    """
    num_nodes, node_to_index, _ = initialize_pso_metadata(graph)
    print(f"\n--- Kaynak={source}, Hedef={destination} İçin Parçacık Sürü Optimizasyonu Başlatılıyor ---")
    print(f"PSO Parametreleri: Parçacık={num_particles}, İterasyon={num_iterations}, w={w}, c1={c1}, c2={c2}")

    # Sürüyü başlat
    swarm = []
    gbest_path = []
    gbest_fitness = -float('inf')
    gbest_pos = np.zeros(num_nodes) 

    # Parçacıkları başlat
    for i in range(num_particles):
        # Pozisyon (node tercihleri) rastgele olarak [-5, 5] aralığında başlatılır
        position = np.random.uniform(-5.0, 5.0, num_nodes)
        velocity = np.zeros(num_nodes)
        
        # Yol başlatma
        initial_path = construct_pso_path(graph, source, destination, position, node_to_index)
        initial_fitness = fitness_function(initial_path, graph, source, destination)

        # pbest'i başlat
        pbest_pos = position.copy()
        pbest_path = initial_path
        pbest_fitness = initial_fitness
        
        # gbest'i güncelle
        if initial_fitness > gbest_fitness:
            gbest_fitness = initial_fitness
            gbest_path = initial_path
            gbest_pos = position.copy()

        swarm.append({
            'position': position,
            'velocity': velocity,
            'pbest_pos': pbest_pos,
            'pbest_path': pbest_path,
            'pbest_fitness': pbest_fitness
        })
        
    print(f"Sürü başlatıldı. Başlangıç En İyi Uygunluk = {gbest_fitness:.4f}")

    # PSO İterasyonu
    for iteration in range(num_iterations):
        for particle in swarm:
            # 1. Hızı Güncelle (Standart PSO Formülü)
            r1 = np.random.rand(num_nodes)
            r2 = np.random.rand(num_nodes)
            
            cognitive_component = c1 * r1 * (particle['pbest_pos'] - particle['position'])
            social_component = c2 * r2 * (gbest_pos - particle['position'])
            
            particle['velocity'] = w * particle['velocity'] + cognitive_component + social_component
            
            # Hızı Sınırla (isteğe bağlı ama önerilir)
            particle['velocity'] = np.clip(particle['velocity'], -max_v, max_v)
            
            # 2. Pozisyonu Güncelle
            particle['position'] += particle['velocity']
            
            # 3. Yol Oluşturma ve Uygunluk Değerlendirmesi
            current_path = construct_pso_path(graph, source, destination, particle['position'], node_to_index)
            current_fitness = fitness_function(current_path, graph, source, destination)
            
            if current_fitness > 0.0: # Yalnızca geçerli yolları dikkate al (> 0 uygunluğa sahip olanlar)
                
                # 4. pbest'i Güncelle
                if current_fitness > particle['pbest_fitness']:
                    particle['pbest_fitness'] = current_fitness
                    particle['pbest_path'] = current_path
                    particle['pbest_pos'] = particle['position'].copy()
                    
                    # 5. gbest'i Güncelle
                    if current_fitness > gbest_fitness:
                        gbest_fitness = current_fitness
                        gbest_path = current_path
                        gbest_pos = particle['position'].copy()


        if iteration % (num_iterations // 10 if num_iterations > 10 else 1) == 0 or iteration == num_iterations - 1:
            print(f"İterasyon {iteration+1}/{num_iterations}: Genel En İyi Uygunluk = {gbest_fitness:.4f}")

    print(f"\n--- Parçacık Sürü Optimizasyonu Tamamlandı ---")

    # En iyi yolu ve uygunluğu döndür
    return gbest_path, gbest_fitness

# --- YÜRÜTME BÖLÜMÜ ---

if __name__ == "__main__":
    
    # 1. Veri Yükleme
    Network_Graph, source_node, destination_node = load_network_data()

    if Network_Graph is None:
        print("\nVeri yükleme hatası nedeniyle devam edilemiyor.")
    else:
        print("\nAğ Verisi başarıyla yüklendi.")
        
        # --- PSO Parametreleri ---
        NUM_PARTICLES = 50 # Parçacık Sayısı
        NUM_ITERATIONS = 100 # İterasyon Sayısı
        W = 0.7  # Atalet ağırlığı (Inertia weight)
        C1 = 2.0 # Bilişsel ağırlık (pbest) (Cognitive weight)
        C2 = 2.0 # Sosyal ağırlık (gbest) (Social weight)
        MAX_V = 3.0 # Maksimum hız sıkıştırması (Maximum velocity clamping)

        # 2. PSO'yu Çalıştır
        # PSO, en iyi yolu ve en iyi uygunluğu döndürecektir
        pso_best_path, pso_best_fitness_from_run = particle_swarm_optimization(
            Network_Graph, source_node, destination_node,
            NUM_PARTICLES, NUM_ITERATIONS, W, C1, C2, MAX_V
        )

        # 3. En İyi PSO Yolu İçin Tam Metrikleri Hesapla
        print("\n--- Yol Metriği Analizi ---")
        
        if pso_best_path:
            pso_reliability, pso_delay, pso_bandwidth, pso_rel_cost, pso_res_cost, pso_fitness_recalc = \
                calculate_all_metrics(Network_Graph, pso_best_path, source_node, destination_node)

            print("\n           Parçacık Sürü Optimizasyonu En İyi Yolu")
            print("----------------------------------------------------")
            print(f"  Yol: {pso_best_path}")
            print(f"  Toplam Güvenilirlik (Maksimize Et): {pso_reliability:.6f}")
            print(f"  Toplam Gecikme (Minimize Et): {pso_delay:.2f} ms")
            print(f"  Güvenilirlik Maliyeti (Minimize Et): {pso_rel_cost:.4f}")
            print(f"  Kaynak Maliyeti (Minimize Et - Bant Genişliği Ters Orantılı): {pso_res_cost:.4f}")
            print(f"  Minimum Bant Genişliği: {pso_bandwidth:.2f} Mbps")
            # Doğrulama için uygunluk skorunu tekrar göster
            print(f"  Birleşik Uygunluk Skoru (Maksimize Et): {pso_fitness_recalc:.4f}")
        else:
             print("PSO geçerli bir yol bulamadı.")
             
        # Not: NetworkX'in varsayılan algoritmasını (nx.shortest_path) kullanan 
        # En Kısa Yol (Hop Sayısı) ile karşılaştırma kısmı 
        # talep üzerine kaldırılmıştır.