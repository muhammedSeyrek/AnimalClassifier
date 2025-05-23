import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import json
from torchvision import transforms

# Sayfa yapısını ayarla
st.set_page_config(
    page_title="Hayvan Sınıflandırıcı",
    page_icon="🐾",
    layout="wide",
)

# Başlık ve açıklama
st.title("🐾 Yapay Zeka Destekli Hayvan Sınıflandırıcı")
st.markdown("""
Bu uygulama, yüklediğiniz hayvan fotoğraflarını otomatik olarak sınıflandırır.
Animals-10 veri seti üzerinde eğitilmiş bir derin öğrenme modeli kullanarak 10 farklı hayvan türünü tanıyabilir.
""")

import torchvision.models as models

class AnimalClassifier(torch.nn.Module):
    """Transfer öğrenme kullanarak hayvan sınıflandırıcı modeli."""
    
    def __init__(self, num_classes, model_name='resnet18', pretrained=False):
        super(AnimalClassifier, self).__init__()
        
        self.model_name = model_name
        
        # Model seçimi
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=False)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = torch.nn.Identity()  # Son katmanı kaldır
        else:
            raise ValueError(f"Desteklenmeyen model: {model_name}")
        
        # Yeni sınıflandırıcı katmanı
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)
    
    @classmethod
    def load_model(cls, path, num_classes):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(num_classes=num_classes, model_name=checkpoint['model_name'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

@st.cache_resource
def load_model():
    """
    Eğitilen modeli ve sınıf bilgilerini yükler ve önbelleğe alır.
    """
    try:
        # Cihazı seç
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        st.write(f"Kullanılan cihaz: {device}")
        
        # Sınıf bilgilerini yükle
        with open('./models/class_info.json', 'r') as f:
            class_info = json.load(f)
        
        class_names_italian = class_info['class_names']
        class_names_english = class_info['english_class_names']
        class_to_idx = class_info['class_to_idx']
        
        # Model dosya yolu
        model_path = './models/model_final.pth'
        
        # Modeli yükle
        model = AnimalClassifier.load_model(model_path, num_classes=len(class_names_italian))
        model = model.to(device)
        model.eval()
        
        return model, device, class_names_italian, class_names_english, class_to_idx
    
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {str(e)}")
        st.warning("Lütfen model dosyalarının doğru konumda olduğundan emin olun.")
        return None, None, None, None, None

def preprocess_image(image, img_size=224):
    """
    Görüntüyü model için PIL kullanarak hazırlar (NumPy kullanmadan).
    """
    # Görüntüyü yeniden boyutlandır
    image = image.resize((img_size, img_size))
    
    # PIL görüntüsünü doğrudan tensöre çevir (NumPy kullanmadan)
    from torch import FloatTensor
    img_tensor = FloatTensor(3, img_size, img_size)
    
    # Görüntü piksellerini manuel olarak tensöre kopyala
    for y in range(img_size):
        for x in range(img_size):
            pixel = image.getpixel((x, y))
            if len(pixel) == 3:  # RGB
                r, g, b = pixel
            else:  # RGBA veya grayscale durumunda
                if len(pixel) == 4:  # RGBA
                    r, g, b, _ = pixel
                else:  # Grayscale
                    r = g = b = pixel
            
            # Normalize et (ImageNet normalizasyon değerleri)
            img_tensor[0][y][x] = (r / 255.0 - 0.485) / 0.229
            img_tensor[1][y][x] = (g / 255.0 - 0.456) / 0.224
            img_tensor[2][y][x] = (b / 255.0 - 0.406) / 0.225
    
    # Batch boyutu ekle
    return img_tensor.unsqueeze(0)

def predict_image(image, model, device, class_names):
    """
    Verilen görüntü için tahmin yapar.
    
    Parametreler:
        image: PIL Image nesnesi
        model: Yüklenmiş model
        device: İşlem cihazı (CPU/GPU)
        class_names: Sınıf isimleri listesi
    
    Returns:
        top_class: En yüksek olasılığa sahip sınıf
        probs: Tüm sınıflar için olasılık değerleri
    """
    # Görüntüyü işle
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # En yüksek olasılığa sahip sınıfı bul
    top_p, top_class = torch.topk(probs, k=1)
    top_prob = top_p.item()
    top_class = top_class.item()
    
    return class_names[top_class], top_prob, probs.cpu().numpy()

def plot_probabilities(probs, class_names):
    """
    Sınıf olasılıklarını görselleştirir.
    
    Parametreler:
        probs: Olasılık değerleri
        class_names: Sınıf isimleri
    
    Returns:
        fig: Matplotlib şekil nesnesi
    """
    # Olasılıklara göre sırala
    indices = np.argsort(probs)[::-1]
    top_probs = probs[indices][:5]  # En yüksek 5 olasılık
    top_classes = [class_names[i] for i in indices[:5]]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(top_classes))
    
    # Yatay çubuk grafiği oluştur
    bars = ax.barh(y_pos, top_probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_classes)
    ax.invert_yaxis()  # Etiketleri yukarıdan aşağıya sıralar
    ax.set_xlabel('Olasılık')
    ax.set_title('En Yüksek 5 Tahmin')
    
    # Çubukların değerlerini ekle
    for i, v in enumerate(top_probs):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    return fig

# Ana fonksiyon
def main():
    # Model ve sınıf bilgilerini yükle
    model, device, class_names_italian, class_names_english, class_to_idx = load_model()
    
    # Kullanıcı arayüzü için iki sütun oluştur
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Görüntü Yükleme")
        uploaded_file = st.file_uploader("Bir hayvan fotoğrafı yükleyin:", type=["jpg", "jpeg", "png"])
        
        # Örnek görüntüler
        st.markdown("### Örnek görüntülerden seçin:")
        sample_images_dir = "./sample_images"
        if os.path.exists(sample_images_dir):
            sample_images = os.listdir(sample_images_dir)
            sample_images = [img for img in sample_images if img.endswith(('.jpg', '.jpeg', '.png'))]
            
            if sample_images:
                selected_sample = st.selectbox(
                    "Örnek görüntüler:",
                    ["Seçiniz..."] + sample_images
                )
                
                if selected_sample != "Seçiniz...":
                    sample_path = os.path.join(sample_images_dir, selected_sample)
                    uploaded_file = sample_path
        else:
            st.info("Örnek görüntüler klasörü bulunamadı.")
    
    # Tahmin işlemi
    if uploaded_file is not None and model is not None:
        # Görüntüyü yükle
        if isinstance(uploaded_file, str):  # Örnek görüntü seçildi
            image = Image.open(uploaded_file)
        else:  # Kullanıcı görüntü yükledi
            image = Image.open(uploaded_file)
        
        # Görüntüyü göster
        with col1:
            st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
        
        # Tahmin yap
        with st.spinner('Sınıflandırma yapılıyor...'):
            # İlerleme çubuğu ekle
            progress_bar = st.progress(0)
            
            # Görüntüyü ön işleme
            progress_bar.progress(25)
            time.sleep(0.5)  # Kullanıcı deneyimi için küçük bir gecikme
            
            # Modeli uygula ve tahmin yap
            progress_bar.progress(50)
            predicted_class, top_prob, probs = predict_image(image, model, device, class_names_english)
            progress_bar.progress(75)
            time.sleep(0.5)  # Kullanıcı deneyimi için küçük bir gecikme
            
            # Sonuçları görselleştir
            progress_bar.progress(100)
            
        # Sonuçları göster
        with col2:
            st.subheader("Tahmin Sonuçları")
            
            # Sınıf adı ve olasılık
            st.markdown(f"### 🏆 Tahmin: **{predicted_class}**")
            st.markdown(f"**Olasılık**: {top_prob:.4f} ({top_prob*100:.2f}%)")
            
            # Olasılık grafiği
            st.subheader("Sınıf Olasılıkları")
            prob_fig = plot_probabilities(probs, class_names_english)
            st.pyplot(prob_fig)
            
            # Güvenirlik seviyesi
            if top_prob > 0.7:
                st.success("✅ Yüksek güvenirlik seviyesi")
            elif top_prob > 0.4:
                st.warning("⚠️ Orta güvenirlik seviyesi")
            else:
                st.error("❌ Düşük güvenirlik seviyesi")
    
    elif uploaded_file is not None:
        st.error("Model yüklenemediği için tahmin yapılamıyor. Lütfen model dosyalarının doğru konumda olduğundan emin olun.")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()

    
