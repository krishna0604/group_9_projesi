# -*- coding: utf-8 -*-
"""
Değişken Komşuluk Araması (Variable Neighborhood Search - VNS) ile En İyi Yol Bulma
Dosya: drb_routing_optimization-main/src/algorithms/variable_neighborhood_search.py
"""

import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import itertools # Düğümleri permütasyonlamak için gereklidir

# Tekrarlanabilirlik için tohumu ayarla (Set seed for reproducibility)
random.seed(42)

# --- AĞ VERİSİNİ CSV'DEN YÜKLEME FONKSİYONLARI ---
# (Bu fonksiyon doğrudan ant_colony_optimization.py dosyasından alınmıştır)

def load_network_data():
    """
    Düğüm (node) ve Bağlantı (link) verilerini CSV dosyalarından yükler ve 
    bir NetworkX Grafiği oluşturur.
    Göreceli yollar, proje kök klasöründen çalıştırıldığında sağlam olacak şekilde ayarlanmıştır.
    """

    # 1. Dosya Yollarını Belirle (ÖNEMLİ: Komut dosyasının proje kökünden çalıştırıldığı varsayılır)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

    node_file_path = os.path.join(base_dir, 'data', 'node_properties.csv')
    link_file_path = os.path.join(base_dir, 'data', 'link_properties.csv')

    # Dosyaların var olup olmadığını kontrol et
    if not os.path.exists(node_file_path) or not os.path.exists(link_file_path):
        # Komut dosyasının kök dizinden çalıştırıldığı varsayımıyla tekrar dene (yedek)
        node_file_path_fallback = os.path.join(os.getcwd(), 'data', 'node_properties.csv')
        link_file_path_fallback = os.path.join(os.getcwd(), 'data', 'link_properties.csv')

        if os.path.exists(node_file_path_fallback) and os.path.exists(link_file_path_fallback):
             node_file_path = node_file_path_fallback
             link_file_path = link_file_path_fallback
        else:
             print(f"\n[HATA] Dosya, denenen yolların hiçbirinde bulunamadı.")
             print(f"CSV dosyalarının proje kökündeki 'data' klasöründe olduğundan emin olun.")
             return None, None, None

    print(f"Veriler şu konumlardan yüklenmeye çalışılıyor:\nDüğüm (Node): {node_file_path}\nBağlantı (Link): {link_file_path}")

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

        print(f"Grafik {G.number_of_nodes()} düğüm ve {G.number_of_edges()} bağlantı ile başarıyla oluşturuldu.")

        # Kaynak (Source) ve Hedef (Destination) düğümlerini varsay
        source_node = node_df['NodeID'].min()
        destination_node = node_df['NodeID'].max()

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