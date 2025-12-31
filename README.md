# drb_routing_optimization
Gecikme, güvenilirlik ve bant genişliği metriklerine göre çok amaçlı rotalama optimizasyonu yapan bir proje.

# Projenin Kurulumu ve Çalıştırılması
Bu proje Pipenv kullanılarak oluşturulmuş bir Python sanal ortamında çalışmaktadır. Projeyi kendi bilgisayarınıza kurmak için aşağıdaki adımları takip edin.
## 1. Gerekli Araçların Kurulu Olup Olmadığını Kontrol Etme
Aşağıdaki komutlarla işletim sisteminize göre Git, Python, pip ve Pipenv’in kurulu olup olmadığını kontrol edin.
- ### [<img width="18" src="https://img.icons8.com/?size=100&id=M9BRw0RJZXKi&format=png&color=000000" alt="windows" border="0"> Windows](docs/setup_windows.md)
- ### [<img width="18" src="https://img.icons8.com/?size=100&id=122959&format=png&color=000000" alt="windows" border="0"> MacOS](docs/setup_macos.md)
- ### [<img width="18" src="https://img.icons8.com/?size=100&id=m6O2bFdG70gw&format=png&color=000000" alt="windows" border="0"> Linux](docs/setup_linux.md)

## 2. Repoyu Klonlayın ve Proje Klasörüne Girin
```bash
git clone <repo-link>
cd <proje-klasörü>
```
## 3. Sanal Ortamı Oluşturun ve Bağımlılıkları Yükleyin
Proje klasörü içinde aşağıdaki komutu çalıştırın:
```bash
pipenv install
```
Bu komut ```Pipfile``` ve ```Pipfile.lock``` dosyalarına göre tüm kütüphaneleri yükler.

## 4. Sanal Ortamı Aktifleştirin
```bash
pipenv shell
```

## 5. Uygulamayı 
### Windows
```bash
python src/main.py
```
### MacOS ve Linux
```bash
python3 src/main.py
```