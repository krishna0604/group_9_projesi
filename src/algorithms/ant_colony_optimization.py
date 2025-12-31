# -*- coding: utf-8 -*-
"""
En İyi Yol Bulma İçin Karınca Kolonisi Optimizasyonu (Ant Colony Optimization - ACO)
Dosya: drb_routing_optimization-main/src/algorithms/ant_colony_optimization.py
"""

import networkx as nx
import random
import numpy as np
import pandas as pd
import os

# Tekrarlanabilirlik için tohum (seed) ayarla
random.seed(42)

# --- CSV'DEN AĞ VERİSİ YÜKLEME FONKSİYONU ---

def load_network_data():
    """
    Düğüm ve bağlantı verilerini CSV dosyalarından yükler ve NetworkX Grafiği oluşturur.
    Göreceli yolların proje kökünden çalıştırıldığında sağlam olması için ayarlanmıştır.
    """
    
    # 1. Dosya Yollarını Belirle (ÖNEMLİ: Komut dosyasının proje kökünden çalıştırıldığı varsayılır: drb_routing_optimization-main)
    # Erişilen yollar 'data/node_properties.csv' ve 'data/link_properties.csv'dir.
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Proje köküne ulaşmak için iki dizin seviyesi yukarı çık, ardından 'data'ya gir
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    node_file_path = os.path.join(base_dir, 'data', 'node_properties.csv')
    link_file_path = os.path.join(base_dir, 'data', 'link_properties.csv')
    
    # Okumaya çalışmadan önce dosyaların mevcut olup olmadığını kontrol et
    if not os.path.exists(node_file_path) or not os.path.exists(link_file_path):
        # Komut dosyasının kök dizinden (drb_routing_optimization-main) çalıştırıldığı varsayımıyla tekrar dene
        node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'node_properties.csv')
        link_file_path_fallback = os.path.join(os.getcwd(), 'data', 'link_properties.csv')

        if os.path.exists(node_file_path_fallback) and os.path.exists(link_file_path_fallback):
             node_file_path = node_file_path_fallback
             link_file_path = link_file_path_fallback
        else:
             print(f"\n[HATA] Dosyalar denenen hiçbir yolda bulunamadı.")
             print(f"Lütfen CSV dosyalarının proje kökündeki 'data' klasöründe olduğundan emin olun.")
             return None, None, None
    
    # Aynı CSV yükleme koduna devam et
    print(f"Verileri şu yollardan yüklemeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {link_file_path}")
    
    try:
        # 2. Veriyi Yükle
        node_df = pd.read_csv(node_file_path)
        link_df = pd.read_csv(link_file_path)
        
        # 2. Ağ Grafiğini Oluştur
        G = nx.Graph()

        # Düğümleri ve özelliklerini ekle
        for index, row in node_df.iterrows():
            node_id = row['NodeID']
            G.add_node(
                node_id, 
                ProcessingDelay=row['ProcessingDelay'], 
                NodeReliability=row['NodeReliability']
            )

        # Bağlantıları ve özelliklerini ekle
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
            
        print(f"Grafik başarıyla {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile oluşturuldu.")
        
        # Başlangıç ve Hedef düğümleri varsay
        source_node = node_df['NodeID'].min()
        destination_node = node_df['NodeID'].max()
        
        return G, source_node, destination_node

    except Exception as e:
        print(f"\n[HATA] CSV verileri işlenirken bir hata oluştu: {e}")
        return None, None, None


# --- METRİK VE UYGUNLUK FONKSİYONLARI ---

def calculate_path_metrics(graph, path):
    """
    Yol metriklerini (Güvenilirlik, Gecikme, Bant Genişliği) hesaplar.
    Güvenilirlik: Tüm bağlantı ve düğüm güvenilirliklerinin çarpımı.
    Gecikme: Tüm bağlantı gecikmeleri ve işlem gecikmelerinin toplamı.
    Bant Genişliği: Tüm bağlantı bant genişliklerinin minimum değeri.
    """
    if not path:
        return 0.0, float('inf'), 0.0

    total_reliability = 1.0
    total_delay = 0.0
    min_bandwidth = float('inf')

    # Düğüm metriklerini hesapla
    for node in path:
        node_data = graph.nodes[node]
        total_delay += node_data.get('ProcessingDelay', 0.0)
        total_reliability *= node_data.get('NodeReliability', 1.0)

    # Bağlantı metriklerini hesapla
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        # Kenar verilerine erişmek için .get((u, v)) kullan
        edge_data = graph.edges.get((u, v), {}) 

        total_delay += edge_data.get('LinkDelay', 0.0)
        total_reliability *= edge_data.get('LinkReliability', 1.0)
        min_bandwidth = min(min_bandwidth, edge_data.get('Bandwidth', float('inf')))

    return total_reliability, total_delay, min_bandwidth if min_bandwidth != float('inf') else 0.0

def fitness_function(path, graph, source, destination):
    """
    Çok Amaçlı Uygunluk Fonksiyonu: Fitness = (Güvenilirlik * Bant Genişliği) / Gecikme
    """
    if not path or path[-1] != destination:
        return 0.0

    reliability, delay, bandwidth = calculate_path_metrics(graph, path)

    if delay <= 0:
        return 0.0
    
    fitness = (reliability * bandwidth) / delay
    return fitness


# --- ACO ÇEKİRDEK BİLEŞEN FONKSİYONLARI ---

def initialize_pheromones(graph, initial_pheromone=1.0):
    """
    Her bir kenar üzerindeki feromon seviyelerini başlatır.
    """
    for u, v in graph.edges():
        graph.edges[u, v]['pheromone'] = initial_pheromone
    print(f"Tüm kenarlar üzerindeki feromonlar başlangıç değeri olan {initial_pheromone} ile başlatıldı.")

def calculate_heuristic_info(graph):
    """
    Her bir kenar için sezgisel değeri hesaplar.
    Formül: (Bant Genişliği * Bağlantı Güvenilirliği) / (Bağlantı Gecikmesi + 1e-6)
    """
    for u, v, data in graph.edges(data=True):
        bandwidth = data.get('Bandwidth', 1.0)  
        link_reliability = data.get('LinkReliability', 1.0)
        link_delay = data.get('LinkDelay', 1e-6) 

        if link_delay <= 0:
            link_delay = 1e-6

        heuristic = (bandwidth * link_reliability) / (link_delay)
        graph.edges[u, v]['heuristic'] = heuristic
    print("Sezgisel bilgi hesaplandı ve tüm kenarlar için saklandı.")

def select_next_node(graph, current_node, visited_nodes, alpha=1.0, beta=2.0):
    """
    Feromon seviyelerine ve sezgisel bilgiye dayanarak karınca için bir sonraki düğümü seçer.
    """
    unvisited_neighbors = [neighbor for neighbor in graph.neighbors(current_node) if neighbor not in visited_nodes]

    if not unvisited_neighbors:
        return None

    probabilities = []
    for neighbor in unvisited_neighbors:
        edge_data = graph.edges.get((current_node, neighbor), {})
        pheromone = edge_data.get('pheromone', 1.0)
        heuristic = edge_data.get('heuristic', 1e-6) 

        probability_numerator = (pheromone ** alpha) * (heuristic ** beta)
        probabilities.append(probability_numerator)

    total_probability = sum(probabilities)
    if total_probability == 0:
        # Tüm olasılıklar sıfırsa geri dönüş (sezgisel > 0 ise nadiren olur)
        return random.choice(unvisited_neighbors)

    normalized_probabilities = [p / total_probability for p in probabilities]

    # random.choices ağırlıklı seçim için daha verimli ve moderndir
    next_node = random.choices(unvisited_neighbors, weights=normalized_probabilities, k=1)[0]
    return next_node

def construct_ant_path(graph, source, destination, alpha, beta, max_path_length_multiplier=2):
    """
    Kaynak'tan hedefe giden yolu oluşturan bir karıncayı simüle eder.
    """
    path = [source]
    current_node = source
    visited_nodes = {source}

    # Sezgisel maksimum yol uzunluğu belirleme
    try:
        # Başlangıç uzunluk rehberi almak için shortest_path_length kullan
        shortest_len = nx.shortest_path_length(graph, source=source, target=destination)
        max_path_length = int(shortest_len * max_path_length_multiplier)
        if max_path_length < 5: 
             max_path_length = 5
    except nx.NetworkXNoPath:
        # Yol bulunamazsa (grafik bağlantılı değil),
        max_path_length = graph.number_of_nodes() * 2 # Keşif için daha fazla alan bırak

    while current_node != destination:
        if len(path) > max_path_length: 
            return [] # Uzunluk sınırı aşıldı

        next_node = select_next_node(graph, current_node, visited_nodes, alpha, beta)

        if next_node is None: 
            return [] # Sıkışıldı, ziyaret edilmemiş komşu yok

        path.append(next_node)
        visited_nodes.add(next_node)
        current_node = next_node

    return path

def update_pheromones(graph, ant_paths_with_fitness, evaporation_rate, pheromone_deposition_weight):
    """
    Feromon seviyelerini günceller (Buharlaşma ve Birikim).
    """
    # 1. Feromon Buharlaşması
    for u, v in graph.edges():
        current_pheromone = graph.edges[u, v].get('pheromone', 0.0)
        graph.edges[u, v]['pheromone'] = current_pheromone * (1 - evaporation_rate)

    # 2. Feromon Birikimi
    for path, fitness in ant_paths_with_fitness:
        if path: 
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Kenarın var olduğundan emin ol (her zaman olmalı)
                if graph.has_edge(u, v): 
                    current_pheromone = graph.edges[u, v].get('pheromone', 0.0)
                    graph.edges[u, v]['pheromone'] = current_pheromone + (pheromone_deposition_weight * fitness)


# --- ANA ACO ALGORİTMASI ---

def ant_colony_optimization(graph, source, destination, num_ants, num_iterations, evaporation_rate, pheromone_deposition_weight, alpha, beta):
    """
    Karınca Kolonisi Optimizasyonu sürecini yönetir.
    """
    print(f"\n--- Kaynak={source}, Hedef={destination} İçin Karınca Kolonisi Optimizasyonu Başlatılıyor ---")
    print(f"ACO Parametreleri: Karınca={num_ants}, İterasyon={num_iterations}, Buharlaşma={evaporation_rate}, Birikim Ağırlığı={pheromone_deposition_weight}, Alfa={alpha}, Beta={beta}")

    # 1. Feromonları başlat ve sezgisel bilgiyi hesapla
    initialize_pheromones(graph, initial_pheromone=1.0)
    calculate_heuristic_info(graph)

    best_aco_path = []
    best_aco_fitness = 0.0

    for iteration in range(num_iterations):
        ant_paths_with_fitness = []
        current_iteration_best_fitness = 0.0
        
        for ant in range(num_ants):
            path = construct_ant_path(graph, source, destination, alpha, beta)
            if path:
                fitness = fitness_function(path, graph, source, destination)
                if fitness > 0.0: 
                    ant_paths_with_fitness.append((path, fitness))

                    if fitness > current_iteration_best_fitness:
                        current_iteration_best_fitness = fitness
                    
                    if fitness > best_aco_fitness:
                        best_aco_fitness = fitness
                        best_aco_path = path

        # 3. Tüm karıncalar yollarını tamamladıktan sonra feromonları güncelle
        if ant_paths_with_fitness: 
            update_pheromones(graph, ant_paths_with_fitness, evaporation_rate, pheromone_deposition_weight)

        if iteration % (num_iterations // 10 if num_iterations > 10 else 1) == 0 or iteration == num_iterations - 1:
            print(f"İterasyon {iteration+1}/{num_iterations}: Genel En İyi Uygunluk = {best_aco_fitness:.4f}")

    print(f"\n--- Karınca Kolonisi Optimizasyonu Tamamlandı ---")

    # En iyi yolu ve uygunluğu döndür
    return best_aco_path, best_aco_fitness

# --- BELGE FORMÜLLERİNE DAYALI EK METRİK FONKSİYONLARI ---

def calculate_reliability_cost(graph, path):
    """
    Formüle dayalı Güvenilirlik Maliyetini (Reliability Cost) hesaplar:
    ReliabilityCost(P) = Sum[-log(LinkReliability)] + Sum[-log(NodeReliability)]
    Bu değeri minimize etmek, Toplam Güvenilirliği maksimize etmeye eşdeğerdir.
    """
    if not path:
        return float('inf')

    total_cost = 0.0

    # 1. Düğüm Güvenilirlik Maliyeti
    for node in path:
        node_data = graph.nodes[node]
        reliability = node_data.get('NodeReliability', 1.0)
        # log(0)'ı ele al
        if reliability > 0:
            total_cost += -np.log(reliability)
        else:
            total_cost += float('inf') 

    # 2. Bağlantı Güvenilirlik Maliyeti
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
    Formüle dayalı Kaynak Kullanım Maliyetini (Resource Cost) hesaplar:
    ResourceCost(P) = Sum[ (1 / Bandwidth) ]
    """
    if not path:
        return float('inf')

    total_cost = 0.0

    # Yalnızca Bağlantı metriklerini hesapla
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = graph.edges.get((u, v), {})
        bandwidth = edge_data.get('Bandwidth', 0.0) # Birim: Mbps

        # Ters değere dönüştür
        if bandwidth > 0:
            # Daha düşük Bant Genişliği, daha yüksek Maliyet ile sonuçlanacaktır
            total_cost += (1.0 / bandwidth)
        else:
            return float('inf') # Bant genişliği sıfırsa maliyet sonsuz
            
    return total_cost

def calculate_all_metrics(graph, path, source, destination):
    """
    Tüm 4 metriği ve uygunluğu hesaplamak için sarmalayıcı fonksiyon.
    """
    if not path:
         # Yol boşsa, kötü varsayılan değerleri döndür
         return 0.0, float('inf'), 0.0, float('inf'), float('inf'), 0.0

    # Mevcut fonksiyonları kullan: TotalReliability, TotalDelay, MinBandwidth
    reliability, delay, bandwidth = calculate_path_metrics(graph, path)
    
    # Yeni metrikleri hesapla
    reliability_cost = calculate_reliability_cost(graph, path)
    resource_cost = calculate_resource_cost(graph, path)
    
    # Uygunluğu tekrar hesapla
    fitness = fitness_function(path, graph, source, destination)
    
    return reliability, delay, bandwidth, reliability_cost, resource_cost, fitness


# --- ÇALIŞTIRMA BÖLÜMÜ ---

if __name__ == "__main__":
    
    # 1. Veri Yükle
    Network_Graph, source_node, destination_node = load_network_data()

    if Network_Graph is None:
        print("\nVeri yükleme hatası nedeniyle devam edilemiyor.")
    else:
        print("\nAğ Verisi başarıyla yüklendi.")
        
        # --- ACO Parametreleri ---
        NUM_ANTS = 50
        NUM_ITERATIONS = 100
        EVAPORATION_RATE = 0.1
        PHEROMONE_DEPOSITION_WEIGHT = 1.0
        ALPHA = 1.0
        BETA = 2.0

        # 2. ACO'yu Çalıştır
        # ACO, en iyi yolu ve en iyi uygunluğu döndürecektir
        aco_best_path, aco_best_fitness_from_run = ant_colony_optimization(
            Network_Graph, source_node, destination_node,
            NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE, PHEROMONE_DEPOSITION_WEIGHT,
            ALPHA, BETA
        )

        # 3. En İyi ACO Yolu İçin Tam Metrikleri Hesapla
        print("\n--- Yol Metrik Analizi ---")
        
        if aco_best_path:
            aco_reliability, aco_delay, aco_bandwidth, aco_rel_cost, aco_res_cost, aco_fitness_recalc = \
                calculate_all_metrics(Network_Graph, aco_best_path, source_node, destination_node)

            print("\n           Karınca Kolonisi Optimizasyonu En İyi Yolu")
            print("----------------------------------------------------")
            print(f"  Yol: {aco_best_path}")
            print(f"  Toplam Güvenilirlik (Maksimize Et): {aco_reliability:.6f}")
            print(f"  Toplam Gecikme (Minimize Et): {aco_delay:.2f} ms")
            print(f"  Güvenilirlik Maliyeti (Minimize Et): {aco_rel_cost:.4f}")
            print(f"  Kaynak Maliyeti (Minimize Et - Bant Genişliği Ters Değeri): {aco_res_cost:.4f}")
            print(f"  Minimum Bant Genişliği: {aco_bandwidth:.2f} Mbps")
            # Yeniden hesaplanan uygunluk değerini doğrulama için tekrar göster, aco_best_fitness_from_run ile aynı olmalıdır
            print(f"  Kombine Uygunluk Skoru (Maksimize Et): {aco_fitness_recalc:.4f}")
        else:
             print("ACO geçerli bir yol bulamadı.")
             
        # Not: NetworkX'in varsayılan algoritmasını (nx.shortest_path) kullanan
        # En Kısa Yol (Sıçrama Sayısı) ile karşılaştırma bölümü
        # istek üzerine kaldırılmıştır.
        
        print("\nNetworkX Varsayılan En Kısa Yol (Sıçrama Sayısı) ile karşılaştırma istek üzerine kaldırılmıştır.")