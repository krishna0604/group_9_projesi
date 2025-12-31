import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import copy
import math
import streamlit as st
import itertools
import matplotlib.pyplot as plt
import warnings

# Optional UI/visualization libraries (use if installed)
try:
    import streamlit_antd_components as antd
except Exception:
    antd = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_echarts import st_echarts
except Exception:
    st_echarts = None


def set_global_seed(seed=42):
    """Set seeds for common RNG sources for reproducible runs.

    Call this right before running a stochastic algorithm to make results
    deterministic for a given seed. This seeds Python's `random` and
    NumPy's RNG, and sets `PYTHONHASHSEED` for hash determinism.
    """
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        os.environ['PYTHONHASHSEED'] = str(int(seed))
    except Exception:
        pass


def ag_modeli():
    """
    Ağ verilerini NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından yükler
    ve bir NetworkX Grafiği oluşturur.
    """
    
    # 1. Dosya Yollarını Belirle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
    edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')
    demand_file_path = os.path.join(base_dir, 'data', 'DemandData.csv')
    
    # Yedek yollar (geçerli çalışma dizininden)
    if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
        node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
        edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')
        demand_file_path_fallback = os.path.join(os.getcwd(), 'data', 'DemandData.csv')
        
        if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
            node_file_path = node_file_path_fallback
            edge_file_path = edge_file_path_fallback
            demand_file_path = demand_file_path_fallback
    
    print(f"Ağ verileriniz yükleniyor:\nDüğümler: {node_file_path}\nBağlantılar: {edge_file_path}")
    
    try:
        # 2. CSV Dosyalarını Yükle
        # NodeData.csv: node_id, s_ms (işlem gecikme), r_node (düğüm güvenilirliği)
        node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
        # EdgeData.csv: src, dst, capacity_mbps (bant genişliği), delay_ms, r_link (bağlantı güvenilirliği)
        edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')
        # DemandData.csv: src, dst, demand_mbps (talep)
        demand_df = pd.read_csv(demand_file_path, delimiter=';', decimal=',') if os.path.exists(demand_file_path) else None
        
        print(f"Dosyalar başarıyla yüklendi.")
        print(f"Düğüm sayısı: {len(node_df)}, Bağlantı sayısı: {len(edge_df)}")
        
        # 3. Ağ Grafiğini Oluştur
        Network_Graph = nx.Graph()
        
        # Düğümleri ve özelliklerini ekle
        for _, row in node_df.iterrows():
            node_id = int(row['node_id'])
            Network_Graph.add_node(
                node_id,
                ProcessingDelay=float(row['s_ms']),  # s_ms -> ProcessingDelay
                NodeReliability=float(row['r_node'])  # r_node -> NodeReliability
            )
        
        # Bağlantıları ve özelliklerini ekle
        for _, row in edge_df.iterrows():
            src = int(row['src'])
            dst = int(row['dst'])
            Network_Graph.add_edge(
                src,
                dst,
                Bandwidth=float(row['capacity_mbps']),  # capacity_mbps -> Bandwidth
                LinkDelay=float(row['delay_ms']),       # delay_ms -> LinkDelay
                LinkReliability=float(row['r_link'])    # r_link -> LinkReliability
            )
        
        print(f"Grafik {Network_Graph.number_of_nodes()} düğüm ve {Network_Graph.number_of_edges()} bağlantı ile başarıyla oluşturuldu.")
        
        # --- 5. Doğrulama ve Örnek Veri Erişimi (Validation and Example Data Access) ---
        print("\n--- Özellik Doğrulama ve Örnekler ---")
        
        # İlk 5 düğüm için örnek özellikler
        print("\n5 Örnek Düğüm Özelliği:")
        for i, node_id in enumerate(list(Network_Graph.nodes())[:5]):
            node_data = Network_Graph.nodes[node_id]
            print(f"Düğüm {node_id}: İşlem Süresi={node_data['ProcessingDelay']:.3f} ms, Güvenilirliği={node_data['NodeReliability']:.4f}")
        
        # İlk 5 bağlantı için örnek özellikler
        print("\n5 Örnek Bağlantı Özelliği:")
        for i, (u, v) in enumerate(list(Network_Graph.edges())[:5]):
            edge_data = Network_Graph.edges[u, v]
            print(f"Bağlantı ({u}-{v}): Bant Genişliği={edge_data['Bandwidth']:.1f} Mbps, Gecikme={edge_data['LinkDelay']:.2f} ms, Güvenilirliği={edge_data['LinkReliability']:.4f}")
        
        # Demand verilerini göster (varsa)
        if demand_df is not None:
            print(f"\nTalep (Demand) Sayısı: {len(demand_df)}")
            print("\n5 Örnek Talep:")
            for i, (_, row) in enumerate(demand_df.head(5).iterrows()):
                print(f"Talep: {int(row['src'])} -> {int(row['dst'])}: {float(row['demand_mbps']):.2f} Mbps")
        
        return Network_Graph
        
    except Exception as e:
        print(f"\n[HATA] CSV verisi yüklenirken bir hata oluştu: {e}")
        return None

def ga(graph=None, source_node=None, destination_node=None):
    def load_network_data():
        """
        NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından ağ verilerini yükler
        ve bir NetworkX Grafiği oluşturur.
        """
        
        # 1. Dosya Yollarını Belirle
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
        edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')
        
        # Yedek yollar
        if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
            node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
            edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')
            
            if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
                node_file_path = node_file_path_fallback
                edge_file_path = edge_file_path_fallback
            else:
                print(f"\n[HATA] Dosyalar denenen hiçbir yolda bulunamadı.")
                print(f"CSV dosyalarının proje kök dizinindeki 'data' klasöründe olduğundan emin olun.")
                return None, None, None
        
        print(f"Verileri şuradan yüklemeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {edge_file_path}")
        
        try:
            # 2. CSV Dosyalarını Yükle
            node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
            edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')
            
            # 3. Ağ Grafiğini Oluştur
            G = nx.Graph()

            # Düğümleri ve özelliklerini ekle
            for index, row in node_df.iterrows():
                node_id = int(row['node_id'])
                G.add_node(
                    node_id, 
                    ProcessingDelay=float(row['s_ms']), 
                    NodeReliability=float(row['r_node'])
                )

            # Bağlantıları ve özelliklerini ekle
            for index, row in edge_df.iterrows():
                source = int(row['src'])
                destination = int(row['dst'])
                G.add_edge(
                    source, 
                    destination, 
                    Bandwidth=float(row['capacity_mbps']), 
                    LinkDelay=float(row['delay_ms']), 
                    LinkReliability=float(row['r_link'])
                )
                
            print(f"Grafik, {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile başarıyla oluşturuldu.")
            
            # Başlangıç ve Hedef düğümlerini varsay
            source_node = node_df['node_id'].min()
            destination_node = node_df['node_id'].max()
            
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

    # Eğer fonksiyon bir grafik ve düğümler ile çağrılırsa, GA'yı çalıştırıp sonucu döndür
    if graph is not None and source_node is not None and destination_node is not None:
        POPULATION_SIZE = 100
        NUM_GENERATIONS = 200
        CROSSOVER_RATE = 0.8
        MUTATION_RATE = 0.3
        TOURNAMENT_SIZE = 5

        ga_best_path, ga_best_fitness_from_run = genetic_algorithm(
            graph, source_node, destination_node,
            POPULATION_SIZE, NUM_GENERATIONS, CROSSOVER_RATE, MUTATION_RATE,
            TOURNAMENT_SIZE
        )
        return ga_best_path, ga_best_fitness_from_run


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

def pso(graph=None, source_node=None, destination_node=None):
    # --- AĞ VERİSİNİN CSV'DEN YÜKLENMESİ FONKSİYONU (ACO'dan korunmuştur) ---

    def load_network_data():
        """
        NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından ağ verilerini yükler
        ve bir NetworkX Grafiği oluşturur.
        """
        
        # 1. Dosya Yollarını Belirle
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
        edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')
        
        # Yedek yollar
        if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
            node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
            edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')
            
            if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
                node_file_path = node_file_path_fallback
                edge_file_path = edge_file_path_fallback
            else:
                print(f"\n[HATA] Dosyalar denenen hiçbir yolda bulunamadı.")
                print(f"CSV dosyalarının proje kök dizinindeki 'data' klasöründe olduğundan emin olun.")
                return None, None, None
        
        print(f"Veri şu konumlardan yüklenmeye çalışılıyor:\nNode: {node_file_path}\nLink: {edge_file_path}")
        
        try:
            # 2. CSV Dosyalarını Yükle
            node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
            edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')
            
            # 3. Ağ Grafiğini Oluştur
            G = nx.Graph()

            # Node'ları ve özelliklerini ekle
            for index, row in node_df.iterrows():
                node_id = int(row['node_id'])
                G.add_node(
                    node_id, 
                    ProcessingDelay=float(row['s_ms']), 
                    NodeReliability=float(row['r_node'])
                )

            # Link'leri ve özelliklerini ekle
            for index, row in edge_df.iterrows():
                source = int(row['src'])
                destination = int(row['dst'])
                G.add_edge(
                    source, 
                    destination, 
                    Bandwidth=float(row['capacity_mbps']), 
                    LinkDelay=float(row['delay_ms']), 
                    LinkReliability=float(row['r_link'])
                )
                
            print(f"Grafik, {G.number_of_nodes()} node ve {G.number_of_edges()} link ile başarıyla oluşturuldu.")
            
            # Kaynak ve Hedef node'ları varsay
            source_node = node_df['node_id'].min()
            destination_node = node_df['node_id'].max()
            
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

    # Eğer fonksiyon bir grafik ve düğümler ile çağrılırsa, PSO'yu çalıştırıp sonucu döndür
    if graph is not None and source_node is not None and destination_node is not None:
        NUM_PARTICLES = 50
        NUM_ITERATIONS = 100
        W = 0.7
        C1 = 2.0
        C2 = 2.0
        MAX_V = 3.0

        pso_best_path, pso_best_fitness_from_run = particle_swarm_optimization(
            graph, source_node, destination_node,
            NUM_PARTICLES, NUM_ITERATIONS, W, C1, C2, MAX_V
        )
        return pso_best_path, pso_best_fitness_from_run

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

    def pso():
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

def aco(graph=None, source_node=None, destination_node=None):
    # --- CSV'DEN AĞ VERİSİ YÜKLEME FONKSİYONU ---

    def load_network_data():
        """
        NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından ağ verilerini yükler
        ve bir NetworkX Grafiği oluşturur.
        """
        
        # 1. Dosya Yollarını Belirle
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
        edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')
        
        # Yedek yollar
        if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
            node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
            edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')
            
            if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
                node_file_path = node_file_path_fallback
                edge_file_path = edge_file_path_fallback
            else:
                print(f"\n[HATA] Dosyalar denenen hiçbir yolda bulunamadı.")
                print(f"CSV dosyalarının proje kök dizinindeki 'data' klasöründe olduğundan emin olun.")
                return None, None, None
        
        print(f"Verileri şu yollardan yüklemeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {edge_file_path}")
        
        try:
            # 2. CSV Dosyalarını Yükle
            node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
            edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')
            
            # 3. Ağ Grafiğini Oluştur
            G = nx.Graph()

            # Düğümleri ve özelliklerini ekle
            for index, row in node_df.iterrows():
                node_id = int(row['node_id'])
                G.add_node(
                    node_id, 
                    ProcessingDelay=float(row['s_ms']), 
                    NodeReliability=float(row['r_node'])
                )

            # Bağlantıları ve özelliklerini ekle
            for index, row in edge_df.iterrows():
                source = int(row['src'])
                destination = int(row['dst'])
                G.add_edge(
                    source, 
                    destination, 
                    Bandwidth=float(row['capacity_mbps']), 
                    LinkDelay=float(row['delay_ms']), 
                    LinkReliability=float(row['r_link'])
                )
                
            print(f"Grafik başarıyla {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile oluşturuldu.")
            
            # Başlangıç ve Hedef düğümleri varsay
            source_node = node_df['node_id'].min()
            destination_node = node_df['node_id'].max()
            
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

        # Eğer optimizasyon geçerli bir yol bulamadıysa, NetworkX'in en kısa yolunu dene (fallback)
        if not best_aco_path or best_aco_fitness <= 0:
            try:
                sp = nx.shortest_path(graph, source, destination)
                sp_fitness = fitness_function(sp, graph, source, destination)
                if sp:
                    return sp, sp_fitness
            except Exception:
                pass

        # En iyi yolu ve uygunluğu döndür (optimizasyon sonucu veya fallback yoksa boş döner)
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


    # Jika dipanggil dengan graf dan node, jalankan ACO dan kembalikan hasilnya
    if graph is not None and source_node is not None and destination_node is not None:
        NUM_ANTS = 50
        NUM_ITERATIONS = 100
        EVAPORATION_RATE = 0.1
        PHEROMONE_DEPOSITION_WEIGHT = 1.0
        ALPHA = 1.0
        BETA = 2.0

        return ant_colony_optimization(
            graph, source_node, destination_node,
            NUM_ANTS, NUM_ITERATIONS, EVAPORATION_RATE, PHEROMONE_DEPOSITION_WEIGHT,
            ALPHA, BETA
        )

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

def abc(graph=None, source_node=None, destination_node=None):
    # ==============================================================================
    # --- CSV'DEN AĞ VERİSİ YÜKLEME FONKSİYONU (YENİDEN GEREKLİ) ---
    # ==============================================================================

    def load_network_data():
        """
        NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından ağ verilerini yükler
        ve bir NetworkX Grafiği oluşturur.
        """
        
        # 1. Dosya Yollarını Belirle
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
        edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')
        
        # Yedek yollar
        if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
            node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
            edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')
            
            if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
                node_file_path = node_file_path_fallback
                edge_file_path = edge_file_path_fallback
            else:
                print(f"\n[HATA] Dosya, denenen hiçbir yolda bulunamadı.")
                print(f"Lütfen CSV dosyalarının proje kökündeki 'data' klasöründe olduğundan emin olun.")
                return None, None, None
        
        print(f"Veri yüklenmeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {edge_file_path}")
        
        try:
            # 2. CSV Dosyalarını Yükle
            node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
            edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')
            
            # 3. Ağ Grafiğini Oluştur
            G = nx.Graph()

            # Düğümleri ve özelliklerini ekle
            for index, row in node_df.iterrows():
                node_id = int(row['node_id'])
                G.add_node(
                    node_id, 
                    ProcessingDelay=float(row['s_ms']), 
                    NodeReliability=float(row['r_node'])
                )

            # Bağlantıları ve özelliklerini ekle
            for index, row in edge_df.iterrows():
                source = int(row['src'])
                destination = int(row['dst'])
                G.add_edge(
                    source, 
                    destination, 
                    Bandwidth=float(row['capacity_mbps']), 
                    LinkDelay=float(row['delay_ms']), 
                    LinkReliability=float(row['r_link'])
                )
                
            print(f"Grafik {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile başarıyla oluşturuldu.")
            
            # Kaynak (Source) ve Hedef (Destination) düğümlerini varsay
            source_node = node_df['node_id'].min()
            destination_node = node_df['node_id'].max()
            
            return G, source_node, destination_node

        except Exception as e:
            print(f"\n[HATA] CSV verisi işlenirken bir hata oluştu: {e}")
            return None, None, None

    # ==============================================================================
    # --- METRİK VE UYGUNLUK (FİTNESS) FONKSİYONLARI (YENİDEN GEREKLİ) ---
    # ==============================================================================

    def calculate_path_metrics(graph, path):
        """
        Yol metriklerini (Güvenilirlik, Gecikme, Bant Genişliği) hesaplar.
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
            # Kenar (edge) verisine erişmek için .get((u, v)) kullanılıyor
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
        
        # Fitness değeri her zaman pozitif olmalıdır
        fitness = (reliability * bandwidth) / delay
        return fitness

    # ==============================================================================
    # --- BELGE FORMÜLLERİNE DAYALI EK METRİK FONKSİYONLARI (YENİDEN GEREKLİ) ---
    # ==============================================================================

    def calculate_reliability_cost(graph, path):
        """
        Güvenilirlik Maliyetini (Reliability Cost) hesaplar.
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
        ResourceCost(P) = Toplam[ (1 / Bant Genişliği) ]
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
        Tüm 4 metriği ve uygunluğu hesaplamak için sarmalayıcı (wrapper) fonksiyon.
        """
        if not path:
            return 0.0, float('inf'), 0.0, float('inf'), float('inf'), 0.0

        reliability, delay, bandwidth = calculate_path_metrics(graph, path)
        reliability_cost = calculate_reliability_cost(graph, path)
        resource_cost = calculate_resource_cost(graph, path)
        fitness = fitness_function(path, graph, source, destination)
        
        return reliability, delay, bandwidth, reliability_cost, resource_cost, fitness


    # ==============================================================================
    # --- ABC ÇEKİRDEK BİLEŞEN FONKSİYONLARI ---
    # ==============================================================================

    def generate_initial_path(graph, source, destination, max_attempts=100):
        """
        Rastgele DFS kullanarak kaynaktan hedefe rastgele bir başlangıç yolu oluşturur.
        Fallback: NetworkX built-in shortest_path algoritması kullanır.
        """
        # Önce NetworkX built-in dijkstra_path kullanarak garantili yol bul
        try:
            shortest = nx.shortest_path(graph, source, destination)
            if shortest:
                return shortest
        except nx.NetworkXNoPath:
            pass
        
        # Eğer yol yoksa, rastgele DFS ile deneme yap
        for _ in range(max_attempts):
            path = [source]
            current_node = source
            visited = {source}
            
            while current_node != destination:
                neighbors = [n for n in graph.neighbors(current_node) if n not in visited]
                
                if not neighbors:
                    # Sıkışıldı, yol devam ettirilemiyor, tekrar dene
                    break 

                # Komşu rastgele seçiliyor
                next_node = random.choice(neighbors)
                path.append(next_node)
                visited.add(next_node)
                current_node = next_node
            
            if current_node == destination:
                return path
                
        return [] # max_attempts sonrası yol bulunamadı

    def generate_neighbor_path(graph, path, source, destination):
        """
        Mevcut yoldan bir komşu yolu (mutasyon) oluşturur.
        Strateji: Yolda 2 düğüm seç ve aralarındaki segmenti yeniden yönlendir (reroute).
        """
        if len(path) < 3:
            return generate_initial_path(graph, source, destination)

        # Kaynak ve hedef hariç, 2 dahili dizin (index) rastgele seçilir
        # Eğer yeterli iç düğüm yoksa, DirectPath işini döndür (kopyalama)
        internal_indices = list(range(1, len(path) - 1))
        if len(internal_indices) < 2:
            return list(path)  # Yolu kopyalayarak döndür
        
        idx1, idx2 = sorted(random.sample(internal_indices, 2))
        
        # Yeniden yönlendirme için başlangıç ve bitiş noktalarını al
        start_reroute = path[idx1]
        end_reroute = path[idx2]
        
        # Yolun dış kısımlarını sakla
        path_start_segment = path[:idx1]
        path_end_segment = path[idx2+1:]
        
        # Yeniden yönlendirme için maksimum deneme
        max_reroute_attempts = 10
        
        for _ in range(max_reroute_attempts):
            # start_reroute'dan end_reroute'a yeni bir rastgele yol oluşturmayı dene
            reroute_path = [start_reroute]
            current_node = start_reroute
            # Yeniden yönlendirme segmentinde döngüleri (cycle) önlemek için ziyaret edilen (visited) kümesi kullan
            visited_in_reroute = {start_reroute}

            while current_node != end_reroute:
                # Ziyaret edilmemiş (start_reroute'dan beri) tüm komşuları kullan
                neighbors = [n for n in graph.neighbors(current_node) if n not in visited_in_reroute]

                if not neighbors:
                    break # Sıkışıldı, aynı idx1/idx2 ile tekrar dene

                next_node = random.choice(neighbors)
                reroute_path.append(next_node)
                visited_in_reroute.add(next_node)
                current_node = next_node

            if current_node == end_reroute:
                # Yeniden yönlendirme başarılı, yeni yolu birleştir
                new_path = path_start_segment + reroute_path + path_end_segment
                # Hedefe bağlantıyı kontrol et
                if not new_path or new_path[-1] != destination:
                    continue # Hatalı yol, tekrar dene

                # Basit döngüleri kaldır (rerouting_path kendi içinde döngüsüz olsa bile)
                # path_start/path_end ile bağlanırken döngüler ortaya çıkabilir
                final_path_no_cycles = []
                seen = set()
                for node in new_path:
                    if node not in seen:
                        final_path_no_cycles.append(node)
                        seen.add(node)
                    elif node == destination and final_path_no_cycles[-1] != destination:
                        # Hedefin son düğüm olması koşuluyla izin ver
                        final_path_no_cycles.append(node)
                        
                if final_path_no_cycles[-1] == destination and final_path_no_cycles[0] == source:
                    return final_path_no_cycles

        # Tüm denemeler başarısız olursa, başlangıç yolunu geri döndür
        return path


    def artificial_bee_colony(graph, source, destination, num_food_sources, num_iterations, limit):
        """
        Yapay Arı Kolonisi sürecini düzenler.
        """
        print(f"\n--- Kaynak={source}, Hedef={destination} için Yapay Arı Kolonisi Başlatılıyor ---")
        print(f"ABC Parametreleri: Yiyecek Kaynakları={num_food_sources}, İterasyonlar={num_iterations}, Limit={limit}")

        # Başlatma: İlk Yiyecek Kaynakları (Employed Bee Phase Başlangıcı)
        food_sources = []
        for i in range(num_food_sources):
            path = generate_initial_path(graph, source, destination)
            if path and len(path) > 0:
                fitness = fitness_function(path, graph, source, destination)
                if fitness > 0:
                    food_sources.append({'path': path, 'fitness': fitness, 'trials': 0})
        
        if not food_sources:
            print("\n[HATA] Yiyecek kaynakları başlatılamadı (yol bulunamadı).")
            # Fallback: en azından shortest path bul
            try:
                fallback_path = nx.shortest_path(graph, source, destination)
                fallback_fitness = fitness_function(fallback_path, graph, source, destination)
                if fallback_fitness > 0:
                    food_sources.append({'path': fallback_path, 'fitness': fallback_fitness, 'trials': 0})
                else:
                    return [], 0.0
            except:
                return [], 0.0

        best_abc_path = max(food_sources, key=lambda x: x['fitness'])['path']
        best_abc_fitness = max(food_sources, key=lambda x: x['fitness'])['fitness']

        for iteration in range(num_iterations):

            # 1. Çalışan Arı (Employed Bee) Aşaması
            new_food_sources = []
            for i in range(len(food_sources)):
                current_source = food_sources[i]
                
                # Komşu yolunu oluştur
                neighbor_path = generate_neighbor_path(graph, current_source['path'], source, destination)
                neighbor_fitness = fitness_function(neighbor_path, graph, source, destination)
                
                # Açgözlü Seçim (Daha iyiyi seç)
                if neighbor_fitness > current_source['fitness'] and len(neighbor_path) > 0:
                    # Kabul edildi
                    new_food_sources.append({'path': neighbor_path, 'fitness': neighbor_fitness, 'trials': 0})
                else:
                    # Reddedildi, denemeleri artır
                    new_food_sources.append({'path': current_source['path'], 'fitness': current_source['fitness'], 'trials': current_source['trials'] + 1})

            food_sources = new_food_sources


            # 2. İzleyici Arı (Onlooker Bee) Aşaması
            # Seçim Olasılıklarını Hesapla (Uygunluk ile Orantılı)
            total_fitness = sum(fs['fitness'] for fs in food_sources)
            if total_fitness > 0:
                probabilities = [fs['fitness'] / total_fitness for fs in food_sources]
                
                # Her İzleyici Arı bir yiyecek kaynağı seçer
                # Onlooker sayısı aktif food_sources sayısıyla sınırlı
                num_onlookers = len(food_sources)
                
                for _ in range(num_onlookers):
                    # Olasılıklara göre yiyecek kaynağını seç
                    try:
                        chosen_index = random.choices(range(len(food_sources)), weights=probabilities, k=1)[0]
                    except (ValueError, IndexError):
                        # Eğer sampling hata verirse, deterministik seçim yap
                        chosen_index = np.argmax(probabilities)
                    
                    if chosen_index < len(food_sources):
                        chosen_source = food_sources[chosen_index]

                        # Komşu yolunu oluştur
                        neighbor_path = generate_neighbor_path(graph, chosen_source['path'], source, destination)
                        neighbor_fitness = fitness_function(neighbor_path, graph, source, destination)

                        # Açgözlü Seçim
                        if neighbor_fitness > chosen_source['fitness']:
                            # Kabul edildi
                            food_sources[chosen_index] = {'path': neighbor_path, 'fitness': neighbor_fitness, 'trials': 0}
                        else:
                            # Reddedildi, denemeleri artır
                            food_sources[chosen_index]['trials'] += 1


            # 3. Gözcü Arı (Scout Bee) Aşaması
            for i in range(len(food_sources)):
                if food_sources[i]['trials'] >= limit:
                    # Yiyecek kaynağı terk edildi, yeni bir kaynakla değiştir
                    new_path = generate_initial_path(graph, source, destination)
                    if new_path and len(new_path) > 0:
                        new_fitness = fitness_function(new_path, graph, source, destination)
                        if new_fitness > 0:
                            food_sources[i] = {'path': new_path, 'fitness': new_fitness, 'trials': 0}
                        else:
                            # Fitness 0 jika yol tidak valid
                            food_sources[i]['trials'] = 0
                    else:
                        # Yeni bir yol bulunamazsa, deneme sayısını sıfırla
                        food_sources[i]['trials'] = 0


            # 4. En İyi Yolu Güncelle
            current_best_source = max(food_sources, key=lambda x: x['fitness'])
            if current_best_source['fitness'] > best_abc_fitness:
                best_abc_fitness = current_best_source['fitness']
                best_abc_path = current_best_source['path']

            if iteration % (num_iterations // 10 if num_iterations > 10 else 1) == 0 or iteration == num_iterations - 1:
                print(f"İterasyon {iteration+1}/{num_iterations}: Genel En İyi Uygunluk = {best_abc_fitness:.4f}")

        print(f"\n--- Yapay Arı Kolonisi Tamamlandı ---")

        # Jika tidak ada hasil yang valid, coba fallback ke jalur terpendek NetworkX
        if not best_abc_path or best_abc_fitness <= 0:
            try:
                sp = nx.shortest_path(graph, source, destination)
                sp_fitness = fitness_function(sp, graph, source, destination)
                if sp:
                    return sp, sp_fitness
            except Exception:
                pass

        return best_abc_path, best_abc_fitness


    # Jika dipanggil dengan graf dan node, jalankan ABC dan kembalikan hasilnya
    if graph is not None and source_node is not None and destination_node is not None:
        NUM_FOOD_SOURCES = 25
        NUM_ITERATIONS = 100
        LIMIT = 10
        return artificial_bee_colony(
            graph, source_node, destination_node,
            NUM_FOOD_SOURCES, NUM_ITERATIONS, LIMIT
        )

    # ==============================================================================
    # --- YÜRÜTME BÖLÜMÜ ---
    # ==============================================================================

    if __name__ == "__main__":
        
        # 1. Veriyi Yükle
        Network_Graph, source_node, destination_node = load_network_data()

        if Network_Graph is None:
            print("\nVeri yükleme hatası nedeniyle devam edilemiyor.")
        else:
            print("\nAğ Verisi başarıyla yüklendi.")
            
            # --- ABC Parametreleri ---
            NUM_FOOD_SOURCES = 25 # Çalışan Arı sayısına eşittir
            NUM_ITERATIONS = 100
            LIMIT = 10 # Gözcü Arı (Scout Bee) olmadan önceki 'deneme' eşiği
            
            # 2. ABC'yi Çalıştır
            # ABC en iyi yolu ve en iyi uygunluğu döndürecektir
            abc_best_path, abc_best_fitness_from_run = artificial_bee_colony(
                Network_Graph, source_node, destination_node,
                NUM_FOOD_SOURCES, NUM_ITERATIONS, LIMIT
            )

            # 3. En İyi ABC Yolu İçin Tam Metrikleri Hesapla
            print("\n--- Yol Metrik Analizi ---")
            
            if abc_best_path:
                abc_reliability, abc_delay, abc_bandwidth, abc_rel_cost, abc_res_cost, abc_fitness_recalc = \
                    calculate_all_metrics(Network_Graph, abc_best_path, source_node, destination_node)

                print("\n        Yapay Arı Kolonisi En İyi Yolu")
                print("----------------------------------------------------")
                print(f"  Yol: {abc_best_path}")
                print(f"  Toplam Güvenilirlik (Maksimize Edilecek): {abc_reliability:.6f}")
                print(f"  Toplam Gecikme (Minimize Edilecek): {abc_delay:.2f} ms")
                print(f"  Güvenilirlik Maliyeti (Minimize Edilecek): {abc_rel_cost:.4f}")
                print(f"  Kaynak Maliyeti (Minimize Edilecek - Bant Genişliği Ters Oranı): {abc_res_cost:.4f}")
                print(f"  Minimum Bant Genişliği: {abc_bandwidth:.2f} Mbps")
                # Doğrulama için uygunluk (fitness) değerini tekrar göster
                print(f"  Birleşik Uygunluk Puanı (Maksimize Edilecek): {abc_fitness_recalc:.4f}")
            else:
                print("ABC geçerli bir yol bulamadı.")

def sa(graph=None, source_node=None, destination_node=None):
    # --- AĞ VERİLERİNİ CSV'DEN YÜKLEME FONKSİYONLARI ---
    # (ant_colony_optimization.py'dan kopyalandı - Daha taşınabilir olacak şekilde uyarlandı)

    def load_network_data():
        """
        NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından ağ verilerini yükler
        ve bir NetworkX Grafiği oluşturur.
        """
        
        # 1. Dosya Yollarını Belirle
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
        edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')
        
        # Yedek yollar
        if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
            node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
            edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')
            
            if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
                node_file_path = node_file_path_fallback
                edge_file_path = edge_file_path_fallback
            else:
                print(f"\n[HATA] Dosyalar denenen yolların hiçbirinde bulunamadı.")
                print(f"CSV dosyalarının proje kök dizinindeki 'data' klasöründe olduğundan emin olun.")
                return None, None, None

        print(f"Verileri şuradan yüklemeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {edge_file_path}")
        
        try:
            # 2. CSV Dosyalarını Yükle
            node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
            edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')
            
            # 3. Ağ Grafiğini Oluştur
            G = nx.Graph()

            # Düğümleri ve özelliklerini ekle
            for index, row in node_df.iterrows():
                node_id = int(row['node_id'])
                G.add_node(
                    node_id, 
                    ProcessingDelay=float(row['s_ms']), 
                    NodeReliability=float(row['r_node'])
                )

            # Bağlantıları ve özelliklerini ekle
            for index, row in edge_df.iterrows():
                source = int(row['src'])
                destination = int(row['dst'])
                G.add_edge(
                    source, 
                    destination, 
                    Bandwidth=float(row['capacity_mbps']), 
                    LinkDelay=float(row['delay_ms']), 
                    LinkReliability=float(row['r_link'])
                )
                
            print(f"Grafik, {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile başarıyla oluşturuldu.")
            
            # Kaynak (Source) ve Hedef (Destination) düğümleri varsay
            source_node = node_df['node_id'].min()
            destination_node = node_df['node_id'].max()
            
            return G, source_node, destination_node

        except Exception as e:
            print(f"\n[HATA] CSV verileri işlenirken bir hata oluştu: {e}")
            return None, None, None


    # --- METRİK VE UYGUNLUK (FITNESS) FONKSİYONLARI (ant_colony_optimization.py'dan kopyalandı) ---

    def calculate_path_metrics(graph, path):
        """
        Yol metriklerini (Güvenilirlik, Gecikme, Bant Genişliği) hesaplar.
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
        
        fitness = (reliability * bandwidth) / delay
        return fitness


    # --- DOKÜMAN FORMÜLLERİNE DAYALI EK METRİK FONKSİYONLARI (ant_colony_optimization.py'dan kopyalandı) ---

    def calculate_reliability_cost(graph, path):
        """
        Güvenilirlik Maliyetini (Reliability Cost) hesaplar.
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
        Kaynak Kullanım Maliyetini (Resource Cost) hesaplar.
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
        Tüm 4 metrik ve uygunluğu (fitness) hesaplamak için sarmalayıcı (wrapper) fonksiyon.
        """
        if not path:
            return 0.0, float('inf'), 0.0, float('inf'), float('inf'), 0.0

        reliability, delay, bandwidth = calculate_path_metrics(graph, path)
        reliability_cost = calculate_reliability_cost(graph, path)
        resource_cost = calculate_resource_cost(graph, path)
        fitness = fitness_function(path, graph, source, destination)
        
        return reliability, delay, bandwidth, reliability_cost, resource_cost, fitness


    # --- SİMÜLE EDİLMİŞ TAVLAMANIN (SIMULATED ANNEALING) ÇEKİRDEK FONKSİYONLARI ---

    def get_initial_path(graph, source, destination):
        """
        Basit bir başlangıç yolu (hop sayısına göre en kısa yol) bulur.
        """
        try:
            # Başlangıç çözümü olarak basit en kısa yolu kullan
            path = nx.shortest_path(graph, source=source, target=destination)
            return path
        except nx.NetworkXNoPath:
            return []

    def get_neighbor_path(graph, current_path, destination):
        """
        Mevcut yolun bir parçasını rastgele değiştirerek bir komşu yol (neighbor path) oluşturur.
        """
        path_len = len(current_path)
        if path_len < 3:
            return current_path

        # İki rastgele düğüm indeksi seç (başlangıç veya bitiş düğümleri olmasın)
        # Aralık [1, path_len - 2]
        try:
            i1, i2 = sorted(random.sample(range(1, path_len - 1), 2))
        except ValueError:
            # Arada sadece 1 düğüm kaldı veya örnekleme sorunu
            return current_path

        node_a = current_path[i1]
        node_b = current_path[i2]
        
        try:
            # node_a ve node_b arasındaki en kısa yolu (hop sayısına göre) bul
            replacement_segment = nx.shortest_path(graph, source=node_a, target=node_b)
            
            # Yolu birleştir: Başlangıç kısmı + Yeni Segment + Bitiş kısmı
            # Yeni segmentten node_a'yı hariç tut (çünkü başlangıç kısmında var)
            # Bitiş kısmından node_b'yi hariç tut (çünkü yeni segmentte var)
            new_path = current_path[:i1] + replacement_segment + current_path[i2+1:]
            
            return new_path

        except nx.NetworkXNoPath:
            # node_a ve node_b arasında yol bulunamazsa
            return current_path


    def acceptance_probability(current_fitness, new_fitness, temperature):
        """
        Daha kötü çözümler için (varsa) kabul olasılığını hesaplar.
        """
        if new_fitness > current_fitness:
            return 1.0 # Daha iyi bir çözüm her zaman kabul edilir
        
        if temperature <= 0.0:
            return 0.0 # 0 sıcaklıkta daha kötü çözümler kabul edilmez

        # Uygunluk Farkı (Delta Fitness) = Yeni Uygunluk - Mevcut Uygunluk (negatif veya sıfır olacaktır)
        delta_fitness = new_fitness - current_fitness 
        
        # Olasılık (Probability) = exp(Delta_Fitness / Sıcaklık)
        return math.exp(delta_fitness / temperature)

    def simulated_annealing_path_finding(graph, source, destination, initial_temp, cooling_rate, num_iterations):
        """
        Simüle Edilmiş Tavlama (Simulated Annealing) sürecini yönetir.
        """
        print(f"\n--- Kaynak={source}, Hedef={destination} için Simüle Edilmiş Tavlama Başlatılıyor ---")
        print(f"SA Parametreleri: Başlangıç Sıcaklığı={initial_temp}, Soğutma Hızı={cooling_rate}, İterasyon Sayısı={num_iterations}")

        # 1. Başlangıç Çözümünü Al
        current_path = get_initial_path(graph, source, destination)
        if not current_path:
            print("Başlangıç yolu bulunamadı.")
            return [], 0.0

        current_fitness = fitness_function(current_path, graph, source, destination)
        best_sa_path = list(current_path)
        best_sa_fitness = current_fitness
        temperature = initial_temp

        print(f"Başlangıç Yolu: {current_path}, Başlangıç Uygunluğu: {current_fitness:.4f}")
        
        for iteration in range(num_iterations):
            # 2. Sıcaklık çok düşükse dur
            if temperature < 1e-6:
                break
            
            # 3. Komşu Oluştur
            new_path = get_neighbor_path(graph, current_path, destination)
            
            if not new_path or new_path == current_path:
                continue 

            # 4. Komşu Uygunluğunu Hesapla
            new_fitness = fitness_function(new_path, graph, source, destination)

            # 5. Kabul Kontrolü
            prob = acceptance_probability(current_fitness, new_fitness, temperature)
            
            if random.random() < prob:
                # Yeni yolu kabul et
                current_path = new_path
                current_fitness = new_fitness

                # Genel en iyi çözümü güncelle
                if current_fitness > best_sa_fitness:
                    best_sa_fitness = current_fitness
                    best_sa_path = list(current_path)

            # 6. Soğutma Programı
            temperature *= cooling_rate
            
            if iteration % (num_iterations // 10 if num_iterations > 10 else 1) == 0 or iteration == num_iterations - 1:
                print(f"İterasyon {iteration+1}/{num_iterations}: Mevcut Uygunluk = {current_fitness:.4f}, Genel En İyi Uygunluk = {best_sa_fitness:.4f}, Sıcaklık = {temperature:.4f}")


        print(f"\n--- Simüle Edilmiş Tavlama Tamamlandı ---")

        return best_sa_path, best_sa_fitness


    # Jika fonksiyon bir grafik ve düğümler ile çağrılırsa, SA'yı çalıştırıp sonucu döndür
    if graph is not None and source_node is not None and destination_node is not None:
        INITIAL_TEMP = 1000.0
        COOLING_RATE = 0.99
        NUM_ITERATIONS = 1000
        return simulated_annealing_path_finding(
            graph, source_node, destination_node,
            INITIAL_TEMP, COOLING_RATE, NUM_ITERATIONS
        )


    # --- ÇALIŞTIRMA BÖLÜMÜ (EXECUTION) ---

    if __name__ == "__main__":
        
        # 1. Veri Yükleme
        Network_Graph, source_node, destination_node = load_network_data()

        if Network_Graph is None or source_node is None or destination_node is None:
            print("\nVeri yükleme hatası nedeniyle devam edilemiyor.")
        else:
            print("\nAğ Verileri başarıyla yüklendi.")
            
            # --- SA Parametreleri (Ayarlanabilir) ---
            INITIAL_TEMP = 1000.0  # Yüksek başlangıç sıcaklığı
            COOLING_RATE = 0.99   # Geometrik soğutma hızı (1.0'a yakın = yavaş soğutma)
            NUM_ITERATIONS = 1000 # İterasyon adım sayısı

            # 2. SA'yı Çalıştır
            sa_best_path, sa_best_fitness_from_run = simulated_annealing_path_finding(
                Network_Graph, source_node, destination_node,
                INITIAL_TEMP, COOLING_RATE, NUM_ITERATIONS
            )

            # 3. SA En İyi Yolu İçin Tam Metrikleri Hesapla
            print("\n--- Yol Metrik Analizi ---")
            
            if sa_best_path:
                sa_reliability, sa_delay, sa_bandwidth, sa_rel_cost, sa_res_cost, sa_fitness_recalc = \
                    calculate_all_metrics(Network_Graph, sa_best_path, source_node, destination_node)

                print("\n           Simüle Edilmiş Tavlama En İyi Yolu")
                print("----------------------------------------------------")
                print(f"  Yol: {sa_best_path}")
                print(f"  Toplam Güvenilirlik (Maksimize): {sa_reliability:.6f}")
                print(f"  Toplam Gecikme (Minimize): {sa_delay:.2f} ms")
                print(f"  Güvenilirlik Maliyeti (Minimize): {sa_rel_cost:.4f}")
                print(f"  Kaynak Maliyeti (Minimize - Bant Genişliği Tersi): {sa_res_cost:.4f}")
                print(f"  Minimum Bant Genişliği: {sa_bandwidth:.2f} Mbps")
                print(f"  Birleşik Uygunluk Puanı (Maksimize): {sa_fitness_recalc:.4f}")
            else:
                print("SA geçerli bir yol bulamadı.")
                
            print("\nNot: Varsayılan en kısa yol algoritması ile karşılaştırma kaldırılmıştır.")

def vns(graph=None, source_node=None, destination_node=None):
    # --- AĞ VERİSİNİ CSV'DEN YÜKLEME FONKSİYONLARI ---
    # (Bu fonksiyon doğrudan ant_colony_optimization.py dosyasından alınmıştır)

    def load_network_data():
        """
        NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından ağ verilerini yükler
        ve bir NetworkX Grafiği oluşturur.
        """

        # 1. Dosya Yollarını Belirle
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

        node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
        edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')

        # Dosyaların var olup olmadığını kontrol et
        if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
            # Komut dosyasının kök dizinden çalıştırıldığı varsayımıyla tekrar dene (yedek)
            node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
            edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')

            if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
                node_file_path = node_file_path_fallback
                edge_file_path = edge_file_path_fallback
            else:
                print(f"\n[HATA] Dosya, denenen yolların hiçbirinde bulunamadı.")
                print(f"CSV dosyalarının proje kökündeki 'data' klasöründe olduğundan emin olun.")
                return None, None, None

        print(f"Veriler şu konumlardan yüklenmeye çalışılıyor:\nDüğüm (Node): {node_file_path}\nBağlantı (Link): {edge_file_path}")

        try:
            # 2. CSV Dosyalarını Yükle
            node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
            edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')

            # 3. Ağ Grafiğini Oluştur
            G = nx.Graph()

            # Düğümleri ve özelliklerini ekle
            for index, row in node_df.iterrows():
                node_id = int(row['node_id'])
                G.add_node(
                    node_id,
                    ProcessingDelay=float(row['s_ms']),
                    NodeReliability=float(row['r_node'])
                )

            # Bağlantıları ve özelliklerini ekle
            for index, row in edge_df.iterrows():
                source = int(row['src'])
                destination = int(row['dst'])
                G.add_edge(
                    source,
                    destination,
                    Bandwidth=float(row['capacity_mbps']),
                    LinkDelay=float(row['delay_ms']),
                    LinkReliability=float(row['r_link'])
                )

            print(f"Grafik {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile başarıyla oluşturuldu.")

            # Kaynak (Source) ve Hedef (Destination) düğümlerini varsay
            source_node = node_df['node_id'].min()
            destination_node = node_df['node_id'].max()

            return G, source_node, destination_node

        except Exception as e:
            print(f"\n[HATA] CSV verisi işlenirken bir hata oluştu: {e}")
            return None, None, None


    # --- METRİK VE UYGUNLUK (FITNESS) FONKSİYONLARI (ant_colony_optimization.py dosyasından alınmıştır) ---

    def calculate_path_metrics(graph, path):
        """
        Yol metriklerini (Güvenilirlik, Gecikme, Bant Genişliği) hesaplar.
        """
        if not path:
            return 0.0, float('inf'), 0.0

        total_reliability = 1.0
        total_delay = 0.0
        min_bandwidth = float('inf')

        # Düğüm (Node) metriklerini hesapla
        for node in path:
            node_data = graph.nodes[node]
            total_delay += node_data.get('ProcessingDelay', 0.0)
            total_reliability *= node_data.get('NodeReliability', 1.0)

        # Bağlantı (Link) metriklerini hesapla
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
        # Yol bağlantısını (basit) doğrula ve yolun geçerli olduğundan emin ol
        if not path or path[0] != source or path[-1] != destination:
            return 0.0

        # VNS için daha sıkı bağlantı kontrolü:
        for i in range(len(path) - 1):
            if not graph.has_edge(path[i], path[i+1]):
                return 0.0 # Geçersiz yol

        reliability, delay, bandwidth = calculate_path_metrics(graph, path)

        if delay <= 0:
            return 0.0

        fitness = (reliability * bandwidth) / delay
        return fitness

    def calculate_reliability_cost(graph, path):
        """
        Güvenilirlik Maliyetini (Reliability Cost) hesaplar.
        """
        if not path:
            return float('inf')

        total_cost = 0.0

        # 1. Düğüm Güvenilirlik Maliyeti
        for node in path:
            reliability = graph.nodes[node].get('NodeReliability', 1.0)
            if reliability > 0:
                total_cost += -np.log(reliability)
            else:
                total_cost += float('inf')

        # 2. Bağlantı Güvenilirlik Maliyeti
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            reliability = graph.edges.get((u, v), {}).get('LinkReliability', 1.0)

            if reliability > 0:
                total_cost += -np.log(reliability)
            else:
                total_cost += float('inf')

        return total_cost

    def calculate_resource_cost(graph, path):
        """
        Kaynak Kullanım Maliyetini (Resource Cost) hesaplar.
        """
        if not path:
            return float('inf')

        total_cost = 0.0

        # Sadece Bağlantı metriklerini hesapla
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            bandwidth = graph.edges.get((u, v), {}).get('Bandwidth', 0.0)

            if bandwidth > 0:
                total_cost += (1.0 / bandwidth)
            else:
                return float('inf')

        return total_cost

    def calculate_all_metrics(graph, path, source, destination):
        """
        Tüm 4 metriği ve uygunluğu hesaplamak için sarmalayıcı (wrapper) fonksiyon.
        """
        if not path:
            return 0.0, float('inf'), 0.0, float('inf'), float('inf'), 0.0

        reliability, delay, bandwidth = calculate_path_metrics(graph, path)
        reliability_cost = calculate_reliability_cost(graph, path)
        resource_cost = calculate_resource_cost(graph, path)
        fitness = fitness_function(path, graph, source, destination)

        return reliability, delay, bandwidth, reliability_cost, resource_cost, fitness


    # --- VNS'NİN ÇEKİRDEK BİLEŞENLERİ ---

    def is_valid_path(graph, path, source, destination):
        """Yolun geçerli olup olmadığını kontrol eder (S/D'de başlar/biter ve bağlıdır)."""
        if not path or path[0] != source or path[-1] != destination:
            return False
        for i in range(len(path) - 1):
            if not graph.has_edge(path[i], path[i+1]):
                return False
        return True

    def generate_initial_solution(graph, source, destination):
        """
        Başlangıç çözümü (NetworkX en kısa yol) üretir.
        """
        try:
            # Başlangıç çözümü olarak en kısa yolu (hop sayısına göre) dene
            path = nx.shortest_path(graph, source=source, target=destination)
            if is_valid_path(graph, path, source, destination):
                print(f"Başlangıç Çözümü Bulundu: {path}")
                return path
            else:
                print("Uyarı: En kısa yol geçerli değil, rastgele yol deneniyor.")
        except nx.NetworkXNoPath:
            print("Hata: Kaynak ve hedef arasında yol yok.")
            return []

        # Yedek/Alternatif: Basit bir bağlantılı yolu dene (eğer grafik bağlıysa)
        try:
            # Basit rastgele yol
            path = [source]
            current = source
            visited = {source}
            # Sıkışmamak için uzunluğu sınırla
            max_len = graph.number_of_nodes() * 2

            while current != destination and len(path) < max_len:
                neighbors = list(graph.neighbors(current))
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                if destination in unvisited_neighbors:
                    path.append(destination)
                    break
                elif unvisited_neighbors:
                    next_node = random.choice(unvisited_neighbors)
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                else:
                    # Sıkışıldı, önceki düğüme dön ve tekrar dene
                    if len(path) > 1:
                        path.pop()
                        current = path[-1]
                    else:
                        return [] # Başlangıçta sıkışıldı

            if current == destination:
                print(f"Başlangıç Çözümü Bulundu (Rastgele): {path}")
                return path
            else:
                return []

        except Exception:
            return []


    def generate_neighbors(graph, current_path, neighborhood_size):
        """
        Modifiye edilmiş 2-opt operasyonu kullanarak komşular üretir.
        Neighborhood_size (Komşuluk boyutu), takas edilecek düğüm sayısını belirler.
        """
        n = len(current_path)
        neighbors = []
        source = current_path[0]
        destination = current_path[-1]

        # VNS: Farklı takas boyutlarını (k) dene
        for k in range(1, neighborhood_size + 1):
            # Kaynak ve hedef hariç k tane iç düğümü rastgele seç
            interior_nodes = current_path[1:-1]
            
            # Takas edilebilecek düğümleri sınırla
            if len(interior_nodes) < 2:
                break # Takas yapılamaz

            # Verimlilik için sadece küçük bir alt küme al
            nodes_to_swap = random.sample(interior_nodes, min(k, len(interior_nodes)))

            # Seçilen düğüm alt kümesinin permütasyonlarını oluştur
            for perm in itertools.permutations(nodes_to_swap):
                temp_path = list(current_path)
                
                # Seçilen düğümleri permütasyonla değiştir
                node_map = dict(zip(nodes_to_swap, perm))
                new_path = [node_map.get(node, node) for node in temp_path]
                
                # Yolu Onar: Bağlantıyı kontrol et ve gerekirse onar
                repaired_path = fix_path_connectivity(graph, new_path, source, destination)
                
                if repaired_path and repaired_path != current_path:
                    neighbors.append(repaired_path)

        # Basit komşular ekle: Rastgele 2 bitişik iç düğümü takas et
        if n > 3:
            for _ in range(3): # 3 kez basit takas dene
                idx1 = random.randint(1, n - 3)
                idx2 = idx1 + 1
                
                temp_path = list(current_path)
                temp_path[idx1], temp_path[idx2] = temp_path[idx2], temp_path[idx1]
                repaired_path = fix_path_connectivity(graph, temp_path, source, destination)
                
                if repaired_path and repaired_path != current_path:
                    neighbors.append(repaired_path)

        # Yinelenenleri kaldır
        return [list(x) for x in set(tuple(x) for x in neighbors)]


    def fix_path_connectivity(graph, path, source, destination):
        """
        Kopuk düğüm segmentleri arasında NetworkX en kısa yolu arayarak yolu onarır.
        """
        if not path or path[0] != source or path[-1] != destination:
            return None

        repaired_path = [source]
        current_node = source

        for next_segment_node in path[1:]:
            if next_segment_node == current_node:
                continue # Yinelenen düğümü atla

            # Düğümlerin doğrudan bağlı olup olmadığını kontrol et
            if graph.has_edge(current_node, next_segment_node):
                repaired_path.append(next_segment_node)
                current_node = next_segment_node
            else:
                # Segmentleri bağlamak için en kısa yolu ara
                try:
                    sub_path = nx.shortest_path(graph, source=current_node, target=next_segment_node)
                    # sub_path'i ekle (zaten mevcut olan başlangıç düğümü hariç)
                    repaired_path.extend(sub_path[1:])
                    current_node = next_segment_node
                except nx.NetworkXNoPath:
                    # Bağlanamazsa, yol bu operasyonda onarılamaz
                    return None

        # Son kontrol: hefede bittiğinden emin ol
        if repaired_path[-1] != destination:
            return None

        # Varsa döngüleri kaldır (isteğe bağlı, daha katı VNS için)
        return remove_loops(repaired_path)

    def remove_loops(path):
        """Yoldan döngüleri kaldırır (örn. A-B-C-B-D'yi A-B-D yapar)."""
        if not path:
            return []
            
        seen = {}
        new_path = []
        
        for node in path:
            if node in seen:
                # Döngü bulundu, döngü noktasından mevcut düğüme kadar tüm düğümleri kaldır
                start_index = seen[node]
                new_path = new_path[:start_index]
                seen = {n: i for i, n in enumerate(new_path)} # Görülenleri sıfırla
            
            seen[node] = len(new_path)
            new_path.append(node)
            
        return new_path

    def local_search(graph, path, source, destination):
        """
        Yerel Arama (Local Search): Uygunluğu (fitness) en üst düzeye çıkarmak için 
        basit 2-opt kullanarak yinelemeli yol iyileştirmesi.
        """
        current_path = path
        current_fitness = fitness_function(current_path, graph, source, destination)
        
        if current_fitness == 0.0:
            return current_path, current_fitness

        improved = True
        while improved:
            improved = False
            n = len(current_path)

            # 2-opt iyileştirmesi dene (yolun yeni bir segmentini dene)
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    # Segmenti (i, j) ters çevir
                    new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
                    
                    # Bağlantıyı Kontrol Et ve Onar
                    temp_path = fix_path_connectivity(graph, new_path, source, destination)
                    
                    if temp_path:
                        new_fitness = fitness_function(temp_path, graph, source, destination)

                        if new_fitness > current_fitness:
                            current_path = temp_path
                            current_fitness = new_fitness
                            improved = True
                            break
                if improved:
                    break
        
        return current_path, current_fitness


    def variable_neighborhood_search(graph, source, destination, max_iterations, max_neighborhood_size):
        """
        Değişken Komşuluk Araması (VNS) sürecini düzenler.
        """
        print(f"\n--- Kaynak={source}, Hedef={destination} için Değişken Komşuluk Araması Başlatılıyor ---")
        print(f"VNS Parametreleri: Maksimum İterasyon={max_iterations}, Maksimum Komşuluk Boyutu={max_neighborhood_size}")

        # 1. Başlangıç Çözümünü Başlat
        current_best_path = generate_initial_solution(graph, source, destination)
        if not current_best_path:
            print("VNS geçerli bir başlangıç çözümü bulamadı.")
            return [], 0.0

        current_best_fitness = fitness_function(current_best_path, graph, source, destination)
        print(f"Başlangıç En İyi Uygunluk = {current_best_fitness:.4f}")
        
        if current_best_fitness <= 0:
            return [], 0.0

        current_path = current_best_path

        for iteration in range(max_iterations):
            k = 1 # Başlangıç komşuluk boyutu

            while k <= max_neighborhood_size:
                # 2. Sallama (Shaking)
                # N_k(x) komşuluğundan rastgele bir x' çözümü üret
                neighborhood = generate_neighbors(graph, current_path, k)
                
                if not neighborhood:
                    k += 1
                    continue
                
                # N_k komşuluğundan rastgele bir çözüm seç
                x_prime = random.choice(neighborhood) 
                
                # 3. Yerel Arama (Local Search)
                # x'' elde etmek için x' üzerinde yerel arama uygula
                x_double_prime, x_double_prime_fitness = local_search(graph, x_prime, source, destination)
                
                # 4. Hareket (Move)
                if x_double_prime_fitness > current_best_fitness:
                    # Daha iyi çözüme geçiş
                    current_best_path = x_double_prime
                    current_best_fitness = x_double_prime_fitness
                    current_path = x_double_prime
                    k = 1  # En yakın komşuluğa geri dön (Saf VNS)
                    print(f"İterasyon {iteration+1}/{max_iterations}: YENİ EN İYİ BULUNDU. Uygunluk = {current_best_fitness:.4f} (k={k})")
                else:
                    # Daha geniş bir komşulukta ara
                    k += 1
                    
            if iteration % (max_iterations // 10 if max_iterations > 10 else 1) == 0:
                print(f"İterasyon {iteration+1}/{max_iterations}: Mevcut En İyi Uygunluk = {current_best_fitness:.4f}")


        print(f"\n--- Değişken Komşuluk Araması Tamamlandı ---")
        return current_best_path, current_best_fitness


    # Eğer fonksiyon bir grafik ve düğümler ile çağrılırsa, VNS'i çalıştırıp sonucu döndür
    if graph is not None and source_node is not None and destination_node is not None:
        MAX_ITERATIONS = 50
        MAX_NEIGHBORHOOD_SIZE = 3

        vns_best_path, vns_best_fitness_from_run = variable_neighborhood_search(
            graph, source_node, destination_node,
            MAX_ITERATIONS, MAX_NEIGHBORHOOD_SIZE
        )
        return vns_best_path, vns_best_fitness_from_run


    # --- ÇALIŞTIRMA BÖLÜMÜ ---

    if __name__ == "__main__":

        # 1. Veriyi Yükle
        Network_Graph, source_node, destination_node = load_network_data()

        if Network_Graph is None:
            print("\nVeri yükleme hatası nedeniyle devam edilemiyor.")
        else:
            print("\nAğ Verisi başarıyla yüklendi.")

            # --- VNS Parametreleri ---
            MAX_ITERATIONS = 50 
            MAX_NEIGHBORHOOD_SIZE = 3 # N_k komşuluğu için maksimum k sayısı

            # 2. VNS'i Çalıştır
            vns_best_path, vns_best_fitness_from_run = variable_neighborhood_search(
                Network_Graph, source_node, destination_node,
                MAX_ITERATIONS, MAX_NEIGHBORHOOD_SIZE
            )

            # 3. En İyi VNS Yolu İçin Tam Metrikleri Hesapla
            print("\n--- Yol Metrikleri Analizi ---")

            if vns_best_path:
                vns_reliability, vns_delay, vns_bandwidth, vns_rel_cost, vns_res_cost, vns_fitness_recalc = \
                    calculate_all_metrics(Network_Graph, vns_best_path, source_node, destination_node)

                print("\n        Değişken Komşuluk Araması En İyi Yolu")
                print("----------------------------------------------------")
                print(f"  Yol (Path): {vns_best_path}")
                print(f"  Toplam Güvenilirlik (En Üst Düzeye Çıkar): {vns_reliability:.6f}")
                print(f"  Toplam Gecikme (En Aza İndir): {vns_delay:.2f} ms")
                print(f"  Güvenilirlik Maliyeti (En Aza İndir): {vns_rel_cost:.4f}")
                print(f"  Kaynak Maliyeti (En Aza İndir - Bant Genişliği Ters Oranı): {vns_res_cost:.4f}")
                print(f"  Minimum Bant Genişliği: {vns_bandwidth:.2f} Mbps")
                # Doğrulama için uygunluk değerini tekrar göster
                print(f"  Birleşik Uygunluk Puanı (En Üst Düzeye Çıkar): {vns_fitness_recalc:.4f}")
            else:
                print("VNS geçerli bir yol bulamadı.")



# --- UI Streamlit ---

def run_streamlit_app():
    st.set_page_config(page_title="Ağ Yönlendirme Optimizasyonu", layout="wide")
    
    st.title("🌐 Ağ Rota Optimizasyonu")
    st.markdown("""
    Bu uygulama, Gecikme (Delay), Güvenilirlik (Reliability) ve Bant Genişliği (Bandwidth)
    ölçütlerine göre ağdaki en iyi yolu bulmak için optimizasyon algoritmalarını karşılaştırır.
    """)

    # Initialize session state for results (so they persist when sliders change)
    if 'last_result_path' not in st.session_state:
        st.session_state.last_result_path = None
    if 'last_result_metrics' not in st.session_state:
        st.session_state.last_result_metrics = None
    if 'last_algorithm' not in st.session_state:
        st.session_state.last_algorithm = None
    # 1. Ağ Verilerini Yükleme
    def load_network_data():
        """
        NodeData.csv, EdgeData.csv, ve DemandData.csv dosyalarından ağ verilerini yükler
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        
        node_file_path = os.path.join(base_dir, 'data', 'NodeData.csv')
        edge_file_path = os.path.join(base_dir, 'data', 'EdgeData.csv')
        
        # Yedek yollar
        if not os.path.exists(node_file_path) or not os.path.exists(edge_file_path):
            node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'NodeData.csv')
            edge_file_path_fallback = os.path.join(os.getcwd(), 'data', 'EdgeData.csv')
            
            if os.path.exists(node_file_path_fallback) and os.path.exists(edge_file_path_fallback):
                node_file_path = node_file_path_fallback
                edge_file_path = edge_file_path_fallback

        try:
            # CSV Dosyalarını Yükle
            node_df = pd.read_csv(node_file_path, delimiter=';', decimal=',')
            edge_df = pd.read_csv(edge_file_path, delimiter=';', decimal=',')
            
            # Ağ Grafiğini Oluştur
            G = nx.Graph()
            
            # Düğümleri ekle
            for _, row in node_df.iterrows():
                node_id = int(row['node_id'])
                G.add_node(
                    node_id,
                    ProcessingDelay=float(row['s_ms']),
                    NodeReliability=float(row['r_node'])
                )
            
            # Bağlantıları ekle
            for _, row in edge_df.iterrows():
                source = int(row['src'])
                destination = int(row['dst'])
                G.add_edge(
                    source,
                    destination,
                    Bandwidth=float(row['capacity_mbps']),
                    LinkDelay=float(row['delay_ms']),
                    LinkReliability=float(row['r_link'])
                )
            
            return G, node_df, edge_df
        except Exception as e:
            st.error(f"Veri yüklenirken hata oluştu: {e}")
            return None, None, None
    
    # Cache graph in session state to avoid rebuilds when sliders change
    if 'network_graph' not in st.session_state:
        with st.spinner("Ağ verileri yükleniyor..."):
            G, node_df, edge_df = load_network_data()
            if G is not None:
                st.session_state.network_graph = G
                st.session_state.node_df = node_df
                st.session_state.edge_df = edge_df
            else:
                st.error("Ağ verileri yüklenemedi!")
                return
    # use cached graph
    G = st.session_state.network_graph
    
    G = st.session_state.network_graph

    # Weight sliders must sum to 1.0; store in session_state and normalize automatically
    if 'w_delay' not in st.session_state:
        st.session_state.w_delay = 0.33
    if 'w_reliability' not in st.session_state:
        st.session_state.w_reliability = 0.33
    if 'w_resource' not in st.session_state:
        st.session_state.w_resource = 0.34

    # Comparison section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Karşılaştırma**")
    algo_A = st.sidebar.selectbox("Algorithm A", ("Genetic Algorithm (GA)", "Particle Swarm Optimization (PSO)", "Ant Colony Optimization (ACO)", "Artificial Bee Colony (ABC)", "Simulated Annealing (SA)", "Variable Neighborhood Search (VNS)"), index=0)
    algo_B = st.sidebar.selectbox("Algorithm B", ("Genetic Algorithm (GA)", "Particle Swarm Optimization (PSO)", "Ant Colony Optimization (ACO)", "Artificial Bee Colony (ABC)", "Simulated Annealing (SA)", "Variable Neighborhood Search (VNS)"), index=2)
    try:
        if antd is not None:
            compare_btn = antd.button("Compare Algorithms")
        else:
            compare_btn = st.sidebar.button("Compare Algorithms")
    except Exception:
        compare_btn = st.sidebar.button("Compare Algorithms")

    # 3. Sonuç Bölümü
    def compute_metrics(graph, path):
        if not path:
            return 0.0, float('inf'), 0.0, float('inf'), 0.0

        total_reliability = 1.0
        total_delay = 0.0
        min_bandwidth = float('inf')

        # Düğüm metrikleri
        for node in path:
            node_data = graph.nodes[node]
            total_delay += node_data.get('ProcessingDelay', 0.0)
            total_reliability *= node_data.get('NodeReliability', 1.0)

        # Kenar metrikleri
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = graph.edges.get((u, v), {})
            total_delay += edge.get('LinkDelay', 0.0)
            total_reliability *= edge.get('LinkReliability', 1.0)
            bw = edge.get('Bandwidth', 0.0)
            if bw and bw > 0:
                min_bandwidth = min(min_bandwidth, bw)

        # Güvenilirlik maliyeti
        reliability_cost = 0.0
        for node in path:
            r = graph.nodes[node].get('NodeReliability', 1.0)
            reliability_cost += -np.log(r) if r > 0 else float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            r = graph.edges.get((u, v), {}).get('LinkReliability', 1.0)
            reliability_cost += -np.log(r) if r > 0 else float('inf')

        # Kaynak maliyeti (1 / bandwidth toplamı)
        resource_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            bw = graph.edges.get((u, v), {}).get('Bandwidth', 0.0)
            if bw and bw > 0:
                resource_cost += (1.0 / bw)
            else:
                resource_cost = float('inf')
                break

        min_bw = min_bandwidth if min_bandwidth != float('inf') else 0.0
        return total_reliability, reliability_cost, resource_cost, min_bw, total_delay

    def weighted_score_from_metrics(reliability, delay, bandwidth, w_delay, w_reliability, w_resource, max_bw):
        # Normalize bandwidth using max observed bandwidth
        bw_score = (bandwidth / max_bw) if max_bw > 0 else 0.0
        # Delay: lower is better, convert to score in (0,1] using 1/(1+delay)
        delay_score = 1.0 / (1.0 + delay)
        return w_reliability * reliability + w_resource * bw_score + w_delay * delay_score

    # helper: render graph using matplotlib only (transparent background, neon path, node "motion" effect)
    def render_pyvis(graph, path=None, height=650):
        # Prefer Plotly interactive network if available
        if go is not None:
            # Layout positions (3D then project)
            try:
                pos3 = nx.spring_layout(graph, dim=3, seed=42)
            except Exception:
                pos2 = nx.spring_layout(graph, seed=42)
                pos3 = {n: (pos2[n][0], pos2[n][1], 0.0) for n in pos2}

            nodes = list(graph.nodes())
            coords = np.array([pos3[n] for n in nodes])
            # subtle z noise
            coords[:, 2] += np.array([np.sin(n * 12.9898) * 0.04 for n in nodes])

            # simple rotation controls
            yaw = st.session_state.get('viz_yaw', 20)
            pitch = st.session_state.get('viz_pitch', 10)
            yaw_rad = np.deg2rad(yaw)
            pitch_rad = np.deg2rad(pitch)
            Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
            Rx = np.array([[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
            R = Rz.dot(Rx)
            rotated = coords.dot(R.T)
            z_vals = rotated[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            z_range = max(z_max - z_min, 1e-6)
            z_norm = (z_vals - z_min) / z_range
            scales = 1.0 / (1.0 + 0.8 * z_norm)
            proj_x = rotated[:, 0] * scales
            proj_y = rotated[:, 1] * scales

            index_map = {n: i for i, n in enumerate(nodes)}

            # build edge traces
            edge_x = []
            edge_y = []
            for u, v in graph.edges():
                i = index_map[u]
                j = index_map[v]
                edge_x += [proj_x[i], proj_x[j], None]
                edge_y += [proj_y[i], proj_y[j], None]

            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.6, color='#9aa0a6'), hoverinfo='none')

            # path edges
            neon_green = '#39FF14'
            path_x = []
            path_y = []
            if path and len(path) > 1:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    iu, iv = index_map.get(u), index_map.get(v)
                    if iu is not None and iv is not None:
                        path_x += [proj_x[iu], proj_x[iv], None]
                        path_y += [proj_y[iu], proj_y[iv], None]

            path_trace = go.Scatter(x=path_x, y=path_y, mode='lines', line=dict(width=3.2, color=neon_green), hoverinfo='none')

            # node traces: glow, trail, main, center
            hover_text = []
            for n in nodes:
                meta = graph.nodes[n]
                hover_text.append(f"Node {n}<br>ProcessingDelay={meta.get('ProcessingDelay',0):.2f} ms<br>Reliability={meta.get('NodeReliability',1.0):.3f}")

            glow_trace = go.Scatter(x=proj_x, y=proj_y, mode='markers', marker=dict(size=40, color='rgba(255,255,255,0.06)'), hoverinfo='none', showlegend=False)

            trail_trace = go.Scatter(x=proj_x + 0.01, y=proj_y + 0.01, mode='markers', marker=dict(size=26, color='rgba(127,179,255,0.12)'), hoverinfo='none', showlegend=False)

            main_colors = ['#00ff77' if (path and n in path) else '#00aaff' for n in nodes]
            main_trace = go.Scatter(x=proj_x, y=proj_y, mode='markers', marker=dict(size=16, color=main_colors, line=dict(width=0.6, color='#ffffff')), text=hover_text, hoverinfo='text')

            center_trace = go.Scatter(x=proj_x, y=proj_y, mode='markers', marker=dict(size=6, color='#ffffff'), hoverinfo='none', showlegend=False)

            layout = go.Layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                height=height
            )

            fig = go.Figure(data=[edge_trace, path_trace, glow_trace, trail_trace, main_trace, center_trace], layout=layout)
            st.plotly_chart(fig, use_container_width=True)
            return

        # fallback: matplotlib renderer (as before)
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='none')
        ax.set_facecolor('none')
        pos = nx.spring_layout(graph, seed=42)
        all_edges = list(graph.edges())
        nx.draw_networkx_edges(graph, pos, edgelist=all_edges, width=0.5, edge_color='#9aa0a6', alpha=0.6, ax=ax)
        neon_green = '#39FF14'
        if path and len(path) > 1:
            path_edge_list = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=path_edge_list, width=2.2, edge_color=neon_green, alpha=0.95, ax=ax)
        node_sizes = 80
        nodes = list(graph.nodes())
        offsets = {}
        for n in nodes:
            rng = np.abs(np.sin(n * 12.9898) * 43758.5453) % 1.0
            angle = rng * 2 * np.pi
            dx = 0.012 * np.cos(angle)
            dy = 0.012 * np.sin(angle)
            offsets[n] = (dx, dy)
        glow_sizes = [node_sizes * 3.5 for _ in nodes]
        glow_pos = {n: (pos[n][0], pos[n][1]) for n in nodes}
        nx.draw_networkx_nodes(graph, glow_pos, nodelist=nodes, node_size=glow_sizes, node_color='#ffffff', alpha=0.05, ax=ax)
        for factor, alpha in [(0.6, 0.12), (0.3, 0.08)]:
            trail_pos = {n: (pos[n][0] + offsets[n][0] * factor, pos[n][1] + offsets[n][1] * factor) for n in nodes}
            nx.draw_networkx_nodes(graph, trail_pos, nodelist=nodes, node_size=int(node_sizes * (1.0 + factor)), node_color='#7fb3ff', alpha=alpha, ax=ax)
        main_colors = ['#00aaff' if not (path and n in path) else '#00ff77' for n in nodes]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=node_sizes, node_color=main_colors, alpha=0.95, ax=ax)
        center_sizes = [int(node_sizes * 0.25) for _ in nodes]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=center_sizes, node_color='#ffffff', alpha=0.9, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=6, font_color='#222222', ax=ax)
        ax.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig)

    # map algorithm name to function
    algo_map = {
        "Genetic Algorithm (GA)": ga,
        "Particle Swarm Optimization (PSO)": pso,
        "Ant Colony Optimization (ACO)": aco,
        "Artificial Bee Colony (ABC)": abc,
        "Simulated Annealing (SA)": sa,
        "Variable Neighborhood Search (VNS)": vns,
    }

    # --- 3-Column Layout: Left | Center | Right ---
    left_col, center_col, right_col = st.columns([1, 2, 1.2])

    # ========== LEFT COLUMN: INPUT PARAMETERS ==========
    with left_col:
        st.markdown('<h3 style="color:var(--accent);">⚙️ Girdi Parametreleri</h3>', unsafe_allow_html=True)
        
    # ========== CENTER COLUMN: GRAPH VISUALIZATION ==========
    with center_col:
        # By default, show the graph
        render_pyvis(G, st.session_state.last_result_path if st.session_state.last_result_path else None)

    # ========== RIGHT COLUMN: RESULTS (Hesap Sonucu) ==========
    with right_col:
        st.markdown('<h3 style="color:var(--accent);">📊 Hesap Sonucu</h3>', unsafe_allow_html=True)
        
        if st.session_state.last_result_metrics is not None:
            metrics = st.session_state.last_result_metrics
            reliability = metrics['reliability']
            total_delay = metrics['total_delay']
            min_bw = metrics['min_bw']
            raw_fitness = metrics['raw_fitness']
            weighted = metrics['weighted']
            result_path = st.session_state.last_result_path
            algorithm_name = st.session_state.last_algorithm
            
            # Display algorithm name
            st.markdown(f'<div class="card neon-outline"><h4 style="color:var(--success);margin:0;">Algoritma</h4><p style="font-family:Roboto Mono;margin:4px 0;">{algorithm_name}</p></div>', unsafe_allow_html=True)
            
            # Display metrics in expandable sections
            with st.expander("📈 Reliability"):
                st.markdown(f'<div style="font-family:Roboto Mono;font-size:16px;color:var(--accent);">{reliability:.6f}</div>', unsafe_allow_html=True)
            with st.expander("⏱️ Delay (ms)"):
                st.markdown(f'<div style="font-family:Roboto Mono;font-size:16px;color:var(--accent);">{total_delay:.2f}</div>', unsafe_allow_html=True)
            with st.expander("📡 Min Bandwidth (Mbps)"):
                st.markdown(f'<div style="font-family:Roboto Mono;font-size:16px;color:var(--accent);">{min_bw:.2f}</div>', unsafe_allow_html=True)
            with st.expander("🎯 Geleneksel Fitness"):
                st.markdown(f'<div style="font-family:Roboto Mono;font-size:16px;color:var(--accent);">{raw_fitness:.6f}</div>', unsafe_allow_html=True)
            with st.expander("⭐ Weighted Score"):
                st.markdown(f'<div style="font-family:Roboto Mono;font-size:16px;color:var(--success);">{weighted:.6f}</div>', unsafe_allow_html=True)
            with st.expander("🛣️ Yol"):
                path_str = " → ".join(map(str, result_path)) if result_path else "N/A"
                st.markdown(f'<div style="font-family:Roboto Mono;font-size:12px;color:var(--text);word-wrap:break-word;">{path_str}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--text);font-style:italic;">Hesaplama sonuçları burada görüntülenecektir.</p>', unsafe_allow_html=True)

    # ========== MOVE INPUT CONTROLS TO LEFT COLUMN (after layout definition) ==========
    left_col.markdown("**Başlangıç & Hedef**")
    source_node = left_col.number_input(
        "Başlangıç Düğümü (S)", 
        min_value=0, 
        max_value=len(G.nodes)-1, 
        value=0,
        step=1,
        key='source_override'
    )
    dest_node = left_col.number_input(
        "Hedef Düğümü (D)", 
        min_value=0, 
        max_value=len(G.nodes)-1, 
        value=len(G.nodes)-1,
        step=1,
        key='dest_override'
    )

    left_col.markdown("**Ağırlıklar (Weight)**")
    w_delay = left_col.slider("W_delay", 0.0, 1.0, st.session_state.w_delay, 0.01, key='w_delay_left')
    w_reliability = left_col.slider("W_reliability", 0.0, 1.0, st.session_state.w_reliability, 0.01, key='w_reliability_left')
    w_resource = left_col.slider("W_resource", 0.0, 1.0, st.session_state.w_resource, 0.01, key='w_resource_left')

    # Normalize weights
    total_w = w_delay + w_reliability + w_resource
    if total_w <= 0:
        st.session_state.w_delay = st.session_state.w_reliability = st.session_state.w_resource = 1.0/3.0
    elif abs(total_w - 1.0) > 1e-6:
        st.session_state.w_delay = round(w_delay / total_w, 3)
        st.session_state.w_reliability = round(w_reliability / total_w, 3)
        st.session_state.w_resource = round(w_resource / total_w, 3)

    left_col.write(f"**Normalized:** D={st.session_state.w_delay}, R={st.session_state.w_reliability}, Bw={st.session_state.w_resource}")

    # Seed input for reproducible stochastic algorithm runs
    left_col.markdown("**Random Seed**")
    seed_value = left_col.number_input(
        "Seed (deterministic runs)",
        min_value=0,
        max_value=2147483647,
        value=42,
        step=1,
        key='seed_value'
    )

    left_col.markdown("**Algoritma Seçimi**")
    algorithm_choice = left_col.selectbox(
        "Algoritma Seçin",
        (
            "Genetic Algorithm (GA)",
            "Particle Swarm Optimization (PSO)",
            "Ant Colony Optimization (ACO)",
            "Artificial Bee Colony (ABC)",
            "Simulated Annealing (SA)",
            "Variable Neighborhood Search (VNS)",
        ),
        key='algo_choice_left'
    )

    calculate_btn = left_col.button("Hesapla", key='calc_btn_left')

    # ========== CALCULATE BUTTON LOGIC ==========
    if calculate_btn:
        if source_node == dest_node:
            st.error("Başlangıç ve hedef düğümler aynı olamaz!")
        else:
            with st.spinner(f"Çalıştırılıyor: {algorithm_choice}..."):
                # Seed RNGs so repeated clicks yield the same result for the same seed
                try:
                    set_global_seed(st.session_state.get('seed_value', seed_value))
                except Exception:
                    pass

                func = algo_map.get(algorithm_choice)
                result_path, raw_fitness = None, 0.0
                try:
                    result = func(G, source_node, dest_node)
                    if isinstance(result, tuple) and len(result) >= 2:
                        result_path, raw_fitness = result[0], result[1]
                    else:
                        result_path = result
                except Exception as e:
                    st.error(f"Algoritma çağrılırken hata: {e}")
                    result_path = None

            if result_path:
                reliability, rel_cost, res_cost, min_bw, total_delay = compute_metrics(G, result_path)
                max_edge_bw = max((d.get('Bandwidth',0.0) for _,_,d in G.edges(data=True)), default=0.0)
                weighted = weighted_score_from_metrics(reliability, total_delay, min_bw, st.session_state.w_delay, st.session_state.w_reliability, st.session_state.w_resource, max_edge_bw)
                
                # Store in session state so it persists
                st.session_state.last_result_path = result_path
                st.session_state.last_algorithm = algorithm_choice
                st.session_state.last_result_metrics = {
                    'reliability': reliability,
                    'total_delay': total_delay,
                    'min_bw': min_bw,
                    'raw_fitness': raw_fitness,
                    'weighted': weighted
                }
                st.rerun()
            else:
                st.error("Yol bulunamadı veya algoritma yakınsamadı.")

    # Compare two algorithms over 5 runs
    if compare_btn:
        with st.spinner("Comparing algorithms over 5 runs..."):
            runs = 5
            records = []
            for algo_name in (algo_A, algo_B):
                func = algo_map.get(algo_name)
                for run in range(1, runs+1):
                    # Use deterministic but different seed per run for reproducible comparisons
                    base_seed = st.session_state.get('seed_value', seed_value)
                    try:
                        set_global_seed(int(base_seed) + (run - 1))
                    except Exception:
                        pass

                    try:
                        result = func(G, source_node, dest_node)
                        if isinstance(result, tuple) and len(result) >= 2:
                            path = result[0]
                        else:
                            path = result
                    except Exception:
                        path = []

                    reliability, rel_cost, res_cost, min_bw, total_delay = compute_metrics(G, path)
                    max_edge_bw = max((d.get('Bandwidth',0.0) for _,_,d in G.edges(data=True)), default=0.0)
                    weighted = weighted_score_from_metrics(reliability, total_delay, min_bw, st.session_state.w_delay, st.session_state.w_reliability, st.session_state.w_resource, max_edge_bw)

                    records.append({
                        'algorithm': algo_name,
                        'run': run,
                        'path_len': len(path) if path else 0,
                        'reliability': reliability,
                        'delay_ms': total_delay,
                        'min_bandwidth': min_bw,
                        'reliability_cost': rel_cost,
                        'resource_cost': res_cost,
                        'weighted_score': weighted
                    })

            df = pd.DataFrame.from_records(records)
            st.subheader("Algorithm Comparison (5 runs each)")
            st.dataframe(df)

            # summary best/worst by algorithm
            summary = df.groupby('algorithm').agg(best_score=('weighted_score','max'), worst_score=('weighted_score','min'), mean_score=('weighted_score','mean')).reset_index()
            st.subheader("Summary (best / worst / mean weighted score)")
            st.table(summary)

            # Plotly interactive performance chart (weighted_score per run)
            if go is not None:
                fig = go.Figure()
                for algo in df['algorithm'].unique():
                    sub = df[df['algorithm'] == algo]
                    fig.add_trace(go.Scatter(x=sub['run'], y=sub['weighted_score'], mode='lines+markers', name=algo, hovertemplate='Run %{x}<br>Score: %{y:.4f}'))
                fig.update_layout(title='Weighted Score per Run', xaxis_title='Run', yaxis_title='Weighted Score', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df.pivot(index='run', columns='algorithm', values='weighted_score'))

            # Radar chart showing normalized weights (Delay, Reliability, Resource)
            weights = [st.session_state.w_delay, st.session_state.w_reliability, st.session_state.w_resource]
            labels = ['Delay', 'Reliability', 'Resource']
            if st_echarts is not None:
                option = {
                    'title': {'text': 'Weights Radar'},
                    'tooltip': {},
                    'radar': {
                        'indicator': [{'name': labels[i], 'max': 1.0} for i in range(len(labels))]
                    },
                    'series': [{'name': 'Weights', 'type': 'radar', 'data': [{'value': weights, 'name': 'Normalized Weights'}]}]
                }
                st_echarts(option, height=350)
            else:
                # simple fallback: plotly polar
                if go is not None:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatterpolar(r=weights + [weights[0]], theta=labels + [labels[0]], fill='toself', name='Weights'))
                    fig2.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=False, title='Normalized Weights (Delay vs Reliability vs Resource)')
                    st.plotly_chart(fig2)
                else:
                    st.write("Weights:")
                    st.write(dict(zip(labels, weights)))

    # Kenar çubuğu alt bilgisi: graf istatistikleri
    st.sidebar.markdown("---")
    st.sidebar.info(f"Graf İstatistikleri:\n- Düğümler: {G.number_of_nodes()}\n- Kenarlar: {G.number_of_edges()}")

if __name__ == "__main__":
    run_streamlit_app()

