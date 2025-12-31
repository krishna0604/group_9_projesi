import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import math # SA'daki üstel fonksiyon için gerekli

# Tekrarlanabilirlik (reprodüktivite) için tohumu ayarla
random.seed(42)

# --- AĞ VERİLERİNİ CSV'DEN YÜKLEME FONKSİYONLARI ---
# (ant_colony_optimization.py'dan kopyalandı - Daha taşınabilir olacak şekilde uyarlandı)

def load_network_data():
    """
    Düğüm (node) ve bağlantı (link) verilerini CSV dosyalarından yükler ve bir NetworkX Grafiği oluşturur.
    Varsayım: CSV dosyaları proje kök dizinine göre 'data' klasöründedir.
    """
    
    # Geçerli çalışma dizinine (CWD) göre 'data/' klasöründe dosya bulmaya çalış
    node_file_path = os.path.join(os.getcwd(), 'data', 'node_properties.csv')
    link_file_path = os.path.join(os.getcwd(), 'data', 'link_properties.csv')

    # CWD'ye göre 'data' klasörü bulunamazsa geri dönüş (örneğin: betik doğrudan 'src' klasöründen çalıştırılıyorsa)
    if not os.path.exists(node_file_path) or not os.path.exists(link_file_path):
        # Basit göreceli yol (kök dizinden çalıştırılırsa işe yarayabilir)
        node_file_path = 'data/node_properties.csv'
        link_file_path = 'data/link_properties.csv'
        
        if not os.path.exists(node_file_path) or not os.path.exists(link_file_path):
            print(f"\n[HATA] Dosyalar denenen yolların hiçbirinde bulunamadı.")
            print(f"CSV dosyalarının proje kök dizinindeki 'data' klasöründe olduğundan emin olun.")
            return None, None, None


    print(f"Verileri şuradan yüklemeye çalışılıyor:\nDüğüm: {node_file_path}\nBağlantı: {link_file_path}")
    
    try:
        # 2. Verileri yükle
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
        
        # Kaynak (Source) ve Hedef (Destination) düğümleri varsay (Min/Max NodeID, orijinal betikteki gibi)
        source_node = node_df['NodeID'].min()
        destination_node = node_df['NodeID'].max()
        
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

def fitness_function(path, graph, destination):
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
    fitness = fitness_function(path, graph, destination)
    
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

    current_fitness = fitness_function(current_path, graph, destination)
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
        new_fitness = fitness_function(new_path, graph, destination)

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