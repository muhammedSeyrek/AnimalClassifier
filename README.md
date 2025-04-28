# 🐾 Yapay Zeka Destekli Hayvan Sınıflandırıcı

Bu proje, hayvan fotoğraflarını yapay zeka kullanarak otomatik olarak sınıflandıran bir uygulamadır. Kullanıcı dostu bir arayüz ile hayvan görüntülerinin hangi türe ait olduğunu tespit edebilir.

![Uygulama Ekran Görüntüsü](./docs/app_screenshot.png)

## 🚀 Özellikler

- Kullanıcının bilgisayarından hayvan görüntüsü yükleyebilmesi
- Yüklenen görüntünün otomatik ön işlemesi ve boyutlandırılması
- Derin öğrenme modeli kullanılarak görüntünün sınıflandırılması
- Tahmin sonuçlarının olasılık değerleriyle birlikte gösterilmesi
- Kullanıcı dostu Streamlit arayüzü
- 10 farklı hayvan sınıfını tanıyabilme yeteneği

## 📊 Desteklenen Hayvan Sınıfları

Model, "Animals-10" veri seti üzerinde eğitilmiştir ve aşağıdaki 10 hayvan türünü tanıyabilir:

1. Köpek (dog)
2. Kedi (cat)
3. At (horse)
4. Kelebek (butterfly)
5. Tavuk (chicken)
6. Koyun (sheep)
7. İnek (cow)
8. Örümcek (spider)
9. Sincap (squirrel)
10. Fil (elephant)

## 🛠️ Teknolojiler

- **PyTorch**: Derin öğrenme modeli için
- **ResNet-50**: Transfer öğrenme için önceden eğitilmiş model
- **Streamlit**: Kullanıcı arayüzü için
- **Python**: Geliştirme dili
- **Scikit-learn**: Model değerlendirme metrikleri için

## 📋 Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız vardır:

```
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
pandas==2.0.2
matplotlib==3.7.1
pillow==9.5.0
streamlit==1.24.0
scikit-learn==1.2.2
tqdm==4.65.0
seaborn==0.12.2
```

## 🚀 Kurulum ve Çalıştırma

1. Bu repo'yu klonlayın:
   ```bash
   git clone https://github.com/kullanici-adi/hayvan-siniflandirici.git
   cd hayvan-siniflandirici
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Veri setini indirin ve hazırlayın:
   ```bash
   # Animals-10 veri setini Kaggle'dan indirin
   # Veri setini ./data/animals-10 klasörüne çıkartın
   ```

4. Modeli eğitin (önceden eğitilmiş model kullanmak isterseniz bu adımı atlayabilirsiniz):
   ```bash
   python src/train.py
   ```

5. Uygulamayı çalıştırın:
   ```bash
   streamlit run src/app.py
   ```

## 📈 Model Performansı

Model, Animals-10 veri setinin test bölümü üzerinde değerlendirilmiştir ve aşağıdaki metriklere ulaşmıştır:

- **Doğruluk (Accuracy)**: ~90%
- **Precision**: ~88%
- **Recall**: ~87%
- **F1-Score**: ~87%

Detaylı performans metrikleri için `models/classification_report.txt` dosyasına bakabilirsiniz.

## 📁 Proje Yapısı

```
hayvan-siniflandirici/
├── data/                   # Veri setinin bulunduğu klasör
├── models/                 # Eğitilmiş modellerin kaydedildiği klasör
├── sample_images/          # Örnek test görüntüleri
├── src/                    # Kaynak kodlar
│   ├── preprocess.py       # Veri ön işleme fonksiyonları
│   ├── model.py            # Model tanımlamaları
│   ├── train.py            # Eğitim kodu
│   ├── evaluate.py         # Değerlendirme kodu
│   └── app.py              # Streamlit arayüzü
├── docs/                   # Dokümantasyon ve ekran görüntüleri
├── requirements.txt        # Gerekli kütüphaneler
└── README.md               # Proje dokümantasyonu
```

## 🧠 Model Mimarisi

Bu projede, ResNet-50 derin öğrenme mimarisi transfer öğrenme yaklaşımıyla kullanılmıştır. Modelin eğitimi iki aşamada gerçekleştirilmiştir:

1. **Özellik Çıkarma**: İlk aşamada, önceden ImageNet üzerinde eğitilmiş ResNet-50 modeli dondurularak yalnızca son sınıflandırıcı katmanları eğitilmiştir.
2. **İnce Ayar (Fine-tuning)**: İkinci aşamada, tüm model parametreleri daha düşük bir öğrenme oranı ile eğitilmiştir.

## 🔍 Veri Ön İşleme

Veri ön işleme aşamasında aşağıdaki adımlar uygulanmıştır:
- Görüntülerin 224x224 piksel boyutuna yeniden boyutlandırılması
- Veri çoğaltma (data augmentation) teknikleri:
  - Rastgele yatay çevirme
  - Rastgele döndürme (±20 derece)
  - Parlaklık, kontrast ve doygunluk değişimleri
- Normalizasyon (ImageNet ortalama ve standart sapma değerleri ile)

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 🧑‍💻 Katkıda Bulunanlar

- [Muhammed Seyrek] - Geliştirici

## 📞 İletişim

Sorularınız veya önerileriniz için lütfen [muhammedseyrek00@gmail.com] adresine e-posta gönderin.