## Ana klasördeki klasör ve dosyaların ne amaçla kullanıldığı
```REAMME.md``` Projenin tanıtımını ve nasıl kurulup çalıştırılacağını anlatan dosyadır.
Yani projeyi ilk kez gören birinin okumaması gereken  ***"ana rehber"*** dir.
Projenin kurulumu ve çalıştırma bilgisi

```requirements.txt``` Projede kullanılan Python kütüphanelerinin listesini içerir.
Başka biri projeyi kurarken bu dosyadan tüm paketleri yükleyebilir.
Yani “*Bu projeyi çalıştırmak için şu kütüphaneleri yükle*” diyen dosyadır.

```.gitignore``` Git tarafından takip edilmesini istemediğin dosya ve klasörleri belirten dosyadır.

```src``` Projenin tüm kaynak kodlarının bulunduğu ana klasördür.

```gui``` Projenin grafik arayüz (kullanıcı arayüzü) ile ilgili dosyalarını içerir.

```docs``` Projenin belgelerinin (dökümantasyonunun) bulunduğu klasördür.

```Pipfile``` Bu dosya, projenin hangi Python kütüphanelerine ihtiyacı olduğunu gösterir.
Yani "Bu projede hangi paketler kullanılacak?" sorusunun cevabıdır.

 ```Pipfile.lock``` Bu dosya, projede kullanılan tüm kütüphanelerin kesin sürümlerini kaydeder.
Yani proje her bilgisayarda aynı şekilde çalışsın diye paket sürümlerini kilitleyen dosyadır.

```LICENSE``` Bu dosya, projenin hangi şartlarla kullanılabileceğini açıklayan lisans belgesidir.
Yani “Bu projeyi isteyen herkes kullanabilir, değiştirebilir ve paylaşabilir” diyen izin metnidir.