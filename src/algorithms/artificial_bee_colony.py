# -*- coding: utf-8 -*-
"""
Yapay Arı Kolonisi (ABC) ile En Uygun Yol Bulma
Dosya: drb_routing_optimization-main/src/algorithms/artificial_bee_colony.py
"""

import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import copy

# Üretkenlik (reprodüktiflik) için tohum (seed) ayarla
random.seed(42)

# ==============================================================================
# --- CSV'DEN AĞ VERİSİ YÜKLEME FONKSİYONU (YENİDEN GEREKLİ) ---
# ==============================================================================

def load_network_data():
    """
    Düğüm (node) ve Bağlantı (link) verilerini CSV dosyasından yükler ve NetworkX Grafiği oluşturur.
    Not: Dosya yolları, test senaryosu için basitleştirilmiş/ayarlanmıştır.
    CSV dosyalarının proje kök dizinine göre 'data' klasöründe olduğundan emin olun.
    """
    
    # 1. Dosya Yollarını Belirle (ÖNEMLİ: Komut dosyasının proje kökünden çalıştığını varsayıyoruz: drb_routing_optimization-main)
    # Erişim sağlanan yollar: 'data/node_properties.csv' ve 'data/link_properties.csv'
    
    # Veriyi mevcut dizin/data/ dizininden yüklemeye çalışılıyor
    node_file_path = os.path.join(os.getcwd(), 'data', 'node_properties.csv')
    link_file_path = os.path.join(os.getcwd(), 'data', 'link_properties.csv')

    if not os.path.exists(node_file_path) or not os.path.exists(link_file_path):
         print(f"\n[HATA] Dosya, denenen hiçbir yolda bulunamadı.")
         print(f"Lütfen CSV dosyalarının proje kökündeki 'data' klasöründe olduğundan emin olun.")
         return None, None, None
    
    print(f"Veri yüklenmeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {link_file_path}")
    
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
    idx1, idx2 = sorted(random.sample(range(1, len(path) - 1), 2))
    
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

    return best_abc_path, best_abc_fitness


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