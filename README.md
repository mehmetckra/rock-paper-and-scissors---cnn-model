# Taş-Kağıt-Makas Sınıflandırma Projesi

Bu proje, derin öğrenme kullanarak taş-kağıt-makas el hareketlerinin sınıflandırılmasını gerçekleştiren bir uygulamadır. Proje, eğitim ve test süreçlerini içeren bir Python uygulaması ve kullanıcı arayüzü olarak Gradio web arayüzünü içermektedir.
![image](https://github.com/user-attachments/assets/ae28ff17-d99f-47f8-90a4-36b879a432e5)


## Proje İçeriği

- Taş-Kağıt-Makas görsellerini sınıflandıran CNN modeli
- Model eğitimi ve değerlendirmesi
- Web tabanlı kullanıcı arayüzü

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız vardır:

```bash
pip install torch torchvision matplotlib numpy gradio pillow
```

## Veri Seti

Projede kullanılan veri seti `archive` klasörü içinde bulunmalıdır ve şu yapıya sahip olmalıdır:

```
archive/
├── paper/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── rock/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── scissors/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Kurulum

1. Bu repo'yu bilgisayarınıza klonlayın veya ZIP olarak indirin
2. Gerekli kütüphaneleri yükleyin
3. Veri setini `archive` klasörü içinde olduğundan emin olun

## Modeli Eğitme

Modeli eğitmek için `main.py` dosyasını çalıştırın:

```bash
python main.py
```

Bu işlem:
- Veri setini eğitim ve doğrulama olarak ayıracak
- CNN modelini eğitecek
- Eğitim ve doğrulama metrikleri için grafikler oluşturacak
- En iyi modeli `best_model.pth` olarak kaydedecek

## Web Arayüzünü Çalıştırma

Eğitilmiş modeli kullanarak web arayüzünü başlatmak için:

```bash
python gradio_app.py
```

Bu komut, Gradio web arayüzünü başlatacak.
Arayüzde:
1. "Bir Taş, Kağıt veya Makas görseli yükleyin" alanına görselinizi sürükleyin veya tıklayarak yükleyin
2. Model görselinizi analiz edecek ve tahminini "Tahmin" alanında gösterecektir

## Proje Dosyaları

- `main.py`: Model tanımı, eğitim kodları ve değerlendirme grafikleri
- `dataset_utils.py`: Veri seti işleme ve dönüştürme fonksiyonları
- `gradio_app.py`: Web arayüzü uygulama kodu
- `best_model.pth`: Eğitilmiş model dosyası (eğitim sonrası oluşur)

## Notlar

- Eğitim hiperparametreleri `main.py` dosyası içinde ayarlanabilir
- Model eğitimi bilgisayarınızın özelliklerine bağlı olarak zaman alabilir
- CUDA destekli bir GPU'ya sahipseniz eğitim otomatik olarak GPU'yu kullanacaktır
- Model performansını artırmak için hiperparametreler ve model mimarisi değiştirilebilir

## Sorun Giderme

- "best_model.pth not found" hatası: Önce `main.py` çalıştırılarak model eğitilmelidir
- Veri seti yükleme hataları: Klasör yapınızın doğru olduğundan emin olun
- Bellek hataları: Batch size değerini düşürerek tekrar deneyin
