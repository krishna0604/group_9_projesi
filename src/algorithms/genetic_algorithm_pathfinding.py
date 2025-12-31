# -*- coding: utf-8 -*-
"""
Genetik Algoritma (GA) ile Optimal Yol Bulma
Dosya: genetic_algorithm_pathfinding.py
"""

import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import copy

# Tekrarlanabilirlik için tohum ayarla
random.seed(42)
np.random.seed(42)

# ==============================================================================
# --- CSV'DEN AĞ VERİSİNİ YÜKLEME FONKSİYONU (ACO KOMUT DOSYASINDAN KORUNMUŞTUR) ---
# ==============================================================================

def load_network_data():
    """
    Düğüm (node) ve bağlantı (link) verilerini CSV dosyalarından yükler ve bir NetworkX Grafiği oluşturur.
    ÖNEMLİ: Çalışması için proje kök dizininde 'data/node_properties.csv' ve 'data/link_properties.csv' 
    dosyalarını gerektirir.
    """
    
    # 1. Dosya Yollarını Belirle (ÖNEMLİ: Bu kod, verinin konumunu tahmin etmeye çalışır)
    
    # Komut dosyasının konumundan göreceli yolu al (proje yapısı varsayımıyla)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Proje kök dizinine ulaşmak için iki dizin seviyesi yukarı çık, ardından 'data' klasörüne gir
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    node_file_path = os.path.join(base_dir, 'data', 'node_properties.csv')
    link_file_path = os.path.join(base_dir, 'data', 'link_properties.csv')
    
    # Komut dosyası başka bir konumdan çalıştırılırsa geri dönüş kontrolü
    if not os.path.exists(node_file_path) or not os.path.exists(link_file_path):
        node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'node_properties.csv')
        link_file_path_fallback = os.path.join(os.getcwd(), 'data', 'link_properties.csv')

        if os.path.exists(node_file_path_fallback) and os.path.exists(link_file_path_fallback):
             node_file_path = node_file_path_fallback
             link_file_path = link_file_path_fallback
        else:
             print(f"\n[HATA] Dosyalar denenen hiçbir yolda bulunamadı.")
             print(f"CSV dosyalarının proje kök dizinindeki 'data' klasöründe olduğundan emin olun.")
             return None, None, None
    
    print(f"Verileri şuradan yüklemeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {link_file_path}")
    
    try:
        # 2. Verileri Yükle
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
            
        print(f"Grafik, {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile başarıyla oluşturuldu.")
        
        # Başlangıç (Source) ve Hedef (Destination) düğümlerini varsay
        source_node = node_df['NodeID'].min()
        destination_node = node_df['NodeID'].max()
        
        return G, source_node, destination_node

    except Exception as e:
        print(f"\n[HATA] CSV verileri işlenirken bir hata oluştu: {e}")
        return None, None, None


# ==============================================================================
# --- METRİK VE UYGUNLUK (FITNESS) FONKSİYONLARI (ACO KOMUT DOSYASINDAN KORUNMUŞTUR) ---
# ==============================================================================

def calculate_path_metrics(graph, path):
    """
    Yol metriklerini (Güvenilirlik, Gecikme, Bant Genişliği) hesaplar.
    Güvenilirlik (Reliability): Tüm bağlantı ve düğüm güvenilirliklerinin çarpımı.
    Gecikme (Delay): Tüm bağlantı gecikmeleri ve işlem gecikmelerinin toplamı.
    Bant Genişliği (Bandwidth): Tüm bağlantı bant genişliklerinin minimum değeri.
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
        edge_data = graph.edges.get((u, v), {}) 

        total_delay += edge_data.get('LinkDelay', 0.0)
        total_reliability *= edge_data.get('LinkReliability', 1.0)
        min_bandwidth = min(min_bandwidth, edge_data.get('Bandwidth', float('inf')))

    return total_reliability, total_delay, min_bandwidth if min_bandwidth != float('inf') else 0.0

def fitness_function(path, graph, source, destination):
    """
    Çok Amaçlı Uygunluk (Fitness) Fonksiyonu: Fitness = (Güvenilirlik * Bant Genişliği) / Gecikme
    """
    if not path or path[-1] != destination:
        return 0.0

    reliability, delay, bandwidth = calculate_path_metrics(graph, path)

    if delay <= 0:
        return 0.0
    
    # Daha yüksek fitness değeri daha iyidir
    fitness = (reliability * bandwidth) / delay
    return fitness

def calculate_reliability_cost(graph, path):
    """
    Güvenilirlik Maliyetini (Reliability Cost) şu formüle göre hesaplar:
    ReliabilityCost(P) = Sum[-log(LinkReliability)] + Sum[-log(NodeReliability)]
    """
    if not path:
        return float('inf')

    total_cost = 0.0

    # 1. Düğüm Güvenilirlik Maliyeti
    for node in path:
        node_data = graph.nodes[node]
        reliability = node_data.get('NodeReliability', 1.0)
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

def calculate_resource_cost(graph, path):
    """
    Kaynak Kullanım Maliyetini (Resource Cost) şu formüle göre hesaplar:
    ResourceCost(P) = Sum[ (1 / Bandwidth) ]
    """
    if not path:
        return float('inf')

    total_cost = 0.0

    # Sadece Bağlantı metriklerini hesapla
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
    Tüm 4 metriği ve uygunluğu hesaplamak için sarmalayıcı fonksiyon.
    """
    if not path:
         # Eğer yol boşsa, kötü varsayılan değerleri döndür
         return 0.0, float('inf'), 0.0, float('inf'), float('inf'), 0.0

    reliability, delay, bandwidth = calculate_path_metrics(graph, path)
    
    reliability_cost = calculate_reliability_cost(graph, path)
    resource_cost = calculate_resource_cost(graph, path)
    
    fitness = fitness_function(path, graph, source, destination)
    
    return reliability, delay, bandwidth, reliability_cost, resource_cost, fitness

# ==============================================================================
# --- GENETİK ALGORİTMA (GA) TEMEL BİLEŞEN FONKSİYONLARI ---
# ==============================================================================

def create_random_path(graph, source, destination, max_path_length_multiplier=2):
    """
    Kısıtlanmış rastgele yürüyüş (random walk) kullanarak kaynaktan hedefe geçerli bir yol oluşturur.
    """
    try:
        # NetworkX en kısa yoluna göre yol uzunluğu sınırlaması
        shortest_len = nx.shortest_path_length(graph, source=source, target=destination)
        max_len = int(shortest_len * max_path_length_multiplier)
    except nx.NetworkXNoPath:
        max_len = graph.number_of_nodes() * 2 # Geri dönüş
        
    for _ in range(graph.number_of_nodes() * 2): # Birkaç kez dene
        path = [source]
        current_node = source
        visited_nodes = {source}

        while current_node != destination:
            if len(path) > max_len:
                break
                
            # Sonsuz döngüleri önlemek için henüz ziyaret edilmemiş düğümleri tercih et
            neighbors = list(graph.neighbors(current_node))
            unvisited_neighbors = [n for n in neighbors if n not in visited_nodes]
            
            if not unvisited_neighbors:
                # Sıkışma: Geri dön veya zaten ziyaret edilmiş diğer komşuları dene (döngü riski)
                if len(path) > 1:
                     # Doğrudan önceki düğüm olmayan komşuları seç
                    neighbors_to_try = [n for n in neighbors if n != path[-2]]
                    if neighbors_to_try:
                         next_node = random.choice(neighbors_to_try)
                    else:
                         break # Tamamen sıkıştı
                else:
                    break # Kaynakta sıkıştı

            else:
                next_node = random.choice(unvisited_neighbors)

            if next_node == destination:
                path.append(next_node)
                return path

            path.append(next_node)
            visited_nodes.add(next_node)
            current_node = next_node
    
    return []

def initialize_population(graph, source, destination, pop_size):
    """Geçerli yollardan oluşan başlangıç popülasyonunu oluşturur."""
    population = []
    # Geçerli ve benzersiz yollarla popülasyon dolana kadar döngü
    while len(population) < pop_size:
        path = create_random_path(graph, source, destination)
        # Yolun döngü içermediğinden ve popülasyonda henüz bulunmadığından emin ol
        if path and len(path) == len(set(path)) and path not in population:
            population.append(path)
    return population

def select_parents_tournament(population_with_fitness, tournament_size=5):
    """Turnuva Seçimi: Rastgele bir örneklemden en iyi fitness'a sahip bireyi seçer."""
    # Rastgele bir örneklem al
    parents = random.sample(population_with_fitness, min(tournament_size, len(population_with_fitness)))
    # Maksimum fitness'a sahip bireyi seç
    winner = max(parents, key=lambda item: item[1])[0] 
    return winner

def validate_and_repair_path(path, destination, graph):
    """Yolu döngülerden onarır ve hedefe ulaştığından emin olur."""
    if not path or path[-1] != destination:
        return []

    # 1. Döngüleri Kaldır
    unique_nodes = []
    seen = set()
    is_valid = True
    for node in path:
        if node in seen:
            is_valid = False # Döngü bulundu
            break 
        unique_nodes.append(node)
        seen.add(node)
    
    # 2. Döngü bulunursa VE yol hedeften önce kesilirse yeniden bağlamayı dene
    if not is_valid and unique_nodes[-1] != destination:
        try:
            # NetworkX en kısa yolunu kullanarak yeniden bağla
            reconnect_path = nx.shortest_path(graph, unique_nodes[-1], destination)
            return unique_nodes + reconnect_path[1:]
        except nx.NetworkXNoPath:
            return [] # Bağlanamadı
    
    # 3. Yol geçerliyse veya döngü bulundu ama son düğüm hedefse
    if unique_nodes[-1] == destination:
         return unique_nodes
         
    return [] # Geçersiz olan diğer durumlar

def crossover_paths(parent1, parent2, graph):
    """
    Ortak Düğüm tabanlı Çaprazlama (Crossover): Ortak düğüm bulur ve yol segmentlerini değiştirir.
    """
    children = []
    
    # Ebeveyn 1 ve Ebeveyn 2 arasındaki ortak düğümleri belirle (kaynak/hedef hariç)
    common_nodes = [node for node in set(parent1[1:-1]) & set(parent2[1:-1])]
    
    if not common_nodes:
        # Çaprazlama başarısız, ebeveynleri geri döndür
        return parent1, parent2

    # Çaprazlama noktası olarak ortak düğümü seç
    crossover_node = random.choice(common_nodes)
    
    idx1 = parent1.index(crossover_node)
    idx2 = parent2.index(crossover_node)
    
    # Çocuk 1'i Oluştur
    child1 = parent1[:idx1 + 1] + parent2[idx2 + 1:]
    # Çocuk 2'yi Oluştur
    child2 = parent2[:idx2 + 1] + parent1[idx1 + 1:]

    # Doğrula ve Onar
    valid_child1 = validate_and_repair_path(child1, parent1[-1], graph)
    valid_child2 = validate_and_repair_path(child2, parent1[-1], graph)

    return valid_child1, valid_child2

def mutate_path(path, graph, destination):
    """
    Parçalı Mutasyon (Segmental Mutation): Rastgele bir yol parçasını NetworkX en kısa yolu ile değiştirir.
    """
    if len(path) < 3:
        return path 
        
    # Başlangıç/bitiş hariç iki dizin (düğüm) seç
    start_idx = random.randint(0, len(path) - 2) 
    end_idx = random.randint(start_idx + 1, len(path) - 1)
    
    # Çok kısaysa tüm yolu seçmediğinden emin ol
    if len(path) < 5 and start_idx == 0 and end_idx == len(path) - 1:
        if end_idx > 1:
            end_idx -= 1
        else:
             return path

    start_node = path[start_idx]
    end_node = path[end_idx]
    
    try:
        # Döngü içermeyen yeni en kısa parçayı bul
        new_segment = nx.shortest_path(graph, start_node, end_node)
        
        # Eski parçayı yeni parçayla değiştir
        new_path = path[:start_idx] + new_segment + path[end_idx+1:]
        
        # Döngü ve hedef için son doğrulama
        if new_path[-1] == destination and len(new_path) == len(set(new_path)):
            return new_path
        
    except nx.NetworkXNoPath:
        # Seçilen iki düğüm arasında yol yoksa
        pass 
        
    return path 

def genetic_algorithm(graph, source, destination, pop_size, num_generations, crossover_rate, mutation_rate, tournament_size):
    """
    Genetik Algoritma (GA) sürecini organize eder.
    """
    print(f"\n--- Kaynak={source}, Hedef={destination} için Genetik Algoritma Başlatılıyor ---")
    print(f"GA Parametreleri: Popülasyon Boyutu={pop_size}, Nesil Sayısı={num_generations}, Çaprazlama Oranı={crossover_rate}, Mutasyon Oranı={mutation_rate}")

    # 1. Popülasyonu Başlat
    population = initialize_population(graph, source, destination, pop_size)
    print(f"Başlangıç popülasyonu boyutu: {len(population)}")

    best_ga_path = []
    best_ga_fitness = 0.0

    for generation in range(num_generations):
        
        # 2. Uygunluğu (Fitness) Değerlendir
        population_with_fitness = []
        for path in population:
            fitness = fitness_function(path, graph, source, destination)
            population_with_fitness.append((path, fitness))
            
            if fitness > best_ga_fitness:
                best_ga_fitness = fitness
                best_ga_path = path

        # Elitizm: En iyi bireyi bir sonraki nesle taşı (1 birey)
        new_population = [best_ga_path] 
        
        while len(new_population) < pop_size:
            
            # 3. Seçilim (Selection)
            parent1 = select_parents_tournament(population_with_fitness, tournament_size)
            parent2 = select_parents_tournament(population_with_fitness, tournament_size)

            child1 = []
            child2 = []

            # 4. Çaprazlama (Crossover)
            if random.random() < crossover_rate:
                child1, child2 = crossover_paths(parent1, parent2, graph)
            else:
                # Çaprazlama olmazsa, çocuklar ebeveynlerdir
                child1 = parent1
                child2 = parent2

            # 5. Mutasyon (Mutation)
            if child1 and random.random() < mutation_rate:
                child1 = mutate_path(child1, graph, destination)
            if child2 and random.random() < mutation_rate and len(new_population) + 1 < pop_size:
                 child2 = mutate_path(child2, graph, destination)

            # 6. Yeni popülasyona ekle (yalnızca geçerli ve benzersiz ise)
            if child1 and child1 not in new_population:
                new_population.append(child1)
            
            if child2 and child2 not in new_population and len(new_population) < pop_size:
                 new_population.append(child2)

        population = new_population[:pop_size] # Popülasyon boyutunu koru

        if generation % (num_generations // 10 if num_generations > 10 else 1) == 0 or generation == num_generations - 1:
            print(f"Nesil {generation+1}/{num_generations}: Genel En İyi Uygunluk = {best_ga_fitness:.4f}")
            
    print(f"\n--- Genetik Algoritma Tamamlandı ---")

    # En iyi yolu ve uygunluğu döndür
    return best_ga_path, best_ga_fitness


# ==============================================================================
# --- ÇALIŞTIRMA KISMI ---
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Veriyi Yükle
    Network_Graph, source_node, destination_node = load_network_data()

    if Network_Graph is None:
        print("\nVeri yükleme hatası nedeniyle devam edilemiyor.")
    else:
        print("\nAğ Verisi başarıyla yüklendi.")
        
        # --- GA Parametreleri ---
        POPULATION_SIZE = 100
        NUM_GENERATIONS = 200
        CROSSOVER_RATE = 0.8
        MUTATION_RATE = 0.3
        TOURNAMENT_SIZE = 5

        # 2. GA'yı Çalıştır
        # GA, en iyi yolu ve en iyi uygunluğu döndürecektir
        ga_best_path, ga_best_fitness_from_run = genetic_algorithm(
            Network_Graph, source_node, destination_node,
            POPULATION_SIZE, NUM_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE,
            TOURNAMENT_SIZE
        )

        # 3. En İyi GA Yolu için Tam Metrikleri Hesapla
        print("\n--- Yol Metrik Analizi ---")
        
        if ga_best_path:
            ga_reliability, ga_delay, ga_bandwidth, ga_rel_cost, ga_res_cost, ga_fitness_recalc = \
                calculate_all_metrics(Network_Graph, ga_best_path, source_node, destination_node)

            print("\n           Genetik Algoritma En İyi Yolu")
            print("----------------------------------------------------")
            print(f"  Yol: {ga_best_path}")
            print(f"  Toplam Güvenilirlik (Maksimize Et): {ga_reliability:.6f}")
            print(f"  Toplam Gecikme (Minimize Et): {ga_delay:.2f} ms")
            print(f"  Güvenilirlik Maliyeti (Minimize Et): {ga_rel_cost:.4f}")
            print(f"  Kaynak Maliyeti (Minimize Et - Bant Genişliği Tersi): {ga_res_cost:.4f}")
            print(f"  Minimum Bant Genişliği: {ga_bandwidth:.2f} Mbps")
            # Fitness değerini tekrar göster
            print(f"  Kombine Uygunluk Skoru (Maksimize Et): {ga_fitness_recalc:.4f}")
        else:
             print("GA geçerli bir yol bulamadı.")
             
        print("\nGenetik Algoritma Kodu Tamamlandı.")