# 🐾 Yapay Zeka Destekli Hayvan Sınıflandırıcı

Bu proje, derin öğrenme kullanarak hayvan fotoğraflarını otomatik olarak sınıflandıran bir web uygulamasıdır. Animals-10 veri seti üzerinde eğitilmiş bir model kullanarak 10 farklı hayvan türünü yüksek doğrulukla tanıyabilir.

## 🚀 Canlı Demo
<div align="center">
  <a href="https://animalclassifier-cloudandai.streamlit.app">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"/>
  </a>
</div>

**[🌐 Uygulamayı Dene - animalclassifier-cloudandai.streamlit.app](https://animalclassifier-cloudandai.streamlit.app)**

## 🚀 Özellikler
- Kullanıcı dostu arayüz
- Gerçek zamanlı görüntü sınıflandırma
- 10 farklı hayvan türünü tanıma (kedi, köpek, at, kelebek, tavuk, inek, fil, koyun, örümcek, sincap)
- Sınıflandırma sonuçlarında olasılık değerleri
- Örnek görüntülerle hızlı test imkanı

## 🛠️ Teknolojiler
- **Backend**: Python, PyTorch, torchvision
- **Frontend**: Streamlit
- **Model**: ResNet18 (transfer öğrenme)
- **Veri İşleme**: PIL, NumPy
- **Görselleştirme**: Matplotlib
- **Deployment**: Streamlit Cloud

## 📋 Gereksinimler
```
torch==2.2.0
torchvision==0.17.0
numpy==1.26.0
pandas>=2.0.0
matplotlib>=3.7.0
pillow>=9.5.0
streamlit>=1.24.0
scikit-learn>=1.2.0
tqdm>=4.65.0
seaborn>=0.12.0
```

## 🚀 Kurulum ve Çalıştırma

### Yerel Çalıştırma
1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/muhammedSeyrek/MusicAnalyzer.git
   cd MusicAnalyzer
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Uygulamayı çalıştırın:
   ```bash
   streamlit run src/app.py
   ```

4. Tarayıcınızda şu adresi açın: `http://localhost:8501`

### Streamlit Cloud'da Çalıştırma
Proje zaten Streamlit Cloud'da çalışıyor: [animalclassifier-cloudandai.streamlit.app](https://animalclassifier-cloudandai.streamlit.app)

## 📂 Klasör Yapısı
```
animal_classifier/
├── src/
│   └── app.py          # Ana uygulama kodu
├── models/
│   ├── model_final.pth # Eğitilmiş model
│   ├── class_info.json # Sınıf bilgileri
│   ├── classification_report.txt
│   └── training_history.png
├── train/
│   ├── animal_classifier.ipynb # Model eğitim notebook'u
│   ├── confusionMatrix.png
│   └── validationGraph.png
├── sample_images/      # Örnek görüntüler
│   ├── cat.jpg
│   ├── dog.jpg
│   └── ...
├── requirements.txt    # Bağımlılıklar
└── README.md           # Bu dosya
```

## 💻 Kullanım
1. Web arayüzünde "Bir hayvan fotoğrafı yükleyin" bölümünden kendi fotoğrafınızı yükleyin veya örnek görüntülerden birini seçin.
2. Uygulama otomatik olarak görüntüyü işleyecek ve sınıflandırma sonuçlarını gösterecektir.
3. Sonuçlar kısmında, modelin tahmin ettiği hayvan türü, olasılık değeri ve diğer olası sınıfların olasılık grafiği görüntülenir.

## 📊 Model Hakkında
Bu uygulama, ImageNet üzerinde önceden eğitilmiş ResNet18 mimarisini kullanarak transfer öğrenme ile oluşturulmuştur. Animals-10 veri seti üzerinde fine-tuning yapılarak 10 farklı hayvan türünü tanıyacak şekilde özelleştirilmiştir.

## 📈 Model Performansı
Modelin Animals-10 test seti üzerindeki performans metrikleri:
- **Doğruluk (Accuracy)**: %95+ (en iyi sonuç)
- **F1 Skoru**: 0.95+
- **Eğitim Süresi**: 6 saat

### Karışıklık Matrisi
Model performansının detaylı analizi için [karışıklık matrisi](train/confusionMatrix.png) görüntüsünü inceleyebilirsiniz.

### Eğitim Grafiği
[Eğitim süreci grafiği](train/validationGraph.png) modelin öğrenme sürecini göstermektedir.

## 🎯 Gelecek Özellikler
- [ ] Daha fazla hayvan türü desteği
- [ ] Model performans metrikleri gösterimi
- [ ] Batch görüntü işleme
- [ ] API entegrasyonu
- [ ] Mobil uygulama versiyonu
- [ ] Görüntü ön işleme seçenekleri
- [ ] Multi-language support

## 🔧 Sorun Giderme

### Yaygın Sorunlar ve Çözümleri
1. **NumPy uyumluluk hatası**: 
   ```bash
   pip install numpy==1.26.0
   ```

2. **Model yükleme hatası**: 
   - Model dosyasının `models/model_final.pth` konumunda olduğundan emin olun
   - Dosya boyutunun 100MB+ olduğunu kontrol edin

3. **Streamlit Cloud hatası**: 
   - requirements.txt dosyasının güncel olduğunu kontrol edin
   - GitHub repository ayarlarındaki dosya boyutu limitini kontrol edin

4. **GPU/CPU uyumluluk**: Uygulama hem CPU hem de GPU ortamlarda çalışacak şekilde tasarlanmıştır.

## 📝 Notlar
- Uygulama Python 3.10 veya 3.11 ile en iyi performansı göstermektedir.
- Görüntü işleme için NumPy kullanmayan bir yaklaşım benimsenmiştir, bu sayede uyumluluk sorunları aşılmıştır.
- Streamlit Cloud deployment sırasında sadece gerekli dosyaların yüklenmesini sağlayın.

## 🤝 Katkıda Bulunma
Projeye katkıda bulunmak isterseniz:
1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans
Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

## 👥 Katkıda Bulunanlar
- [Muhammed Seyrek](https://github.com/muhammedSeyrek) - Geliştirici

## 📞 İletişim
Proje hakkında sorularınız için:
- GitHub Issue açın
- [Streamlit Cloud Demo](https://animalclassifier-cloudandai.streamlit.app) üzerinden geri bildirim gönderin
