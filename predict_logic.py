# predict_logic.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from color_detector import get_dominant_color_isolated 
from image_enhancer import enhance_image_quality 

# --- Fungsi Helper ---
def load_and_prep_image(image_array, img_width, img_height):
    """Mempersiapkan gambar numpy untuk prediksi model."""
    img_display = image_array.copy()
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_width, img_height))
    img_processed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_processed, axis=0)
    return img_batch, img_display

def get_prediction(model, img_batch, class_names):
    """Menjalankan prediksi pada model."""
    predictions = model.predict(img_batch, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence

# --- FUNGSI UTAMA YANG AKAN DIPANGGIL OLEH app.py ---
def run_all_predictions(image_path, gender, models, rules, classes):
    """
    Menjalankan alur lengkap AI: Enhance, Prediksi Style, 
    Prediksi Kategori, Deteksi Warna, dan Validasi.
    """
    
    # 1. Load & Enhance Gambar (PCD)
    img_original = cv2.imread(image_path)
    if img_original is None:
        raise ValueError(f"Gagal memuat gambar: {image_path}")
    
    img_enhanced = enhance_image_quality(img_original)
    
    # 2. Siapkan gambar untuk model ML (CNN)
    image_batch, _ = load_and_prep_image(img_enhanced, 224, 224)
    
    # 3. Proses C (Style) - (CNN)
    pred_style, conf_style = get_prediction(
        models['style'], image_batch, classes['style']
    )
    
    # 4. Proses A (Kategori) - (CNN)
    if gender == 'male':
        model_kategori = models['men']
        class_kategori = classes['men']
    else:
        model_kategori = models['women']
        class_kategori = classes['women']
        
    pred_kategori, conf_kategori = get_prediction(
        model_kategori, image_batch, class_kategori
    )
        
    # 5. Proses B (Warna) - (Clustering & PCD)
    pred_warna = get_dominant_color_isolated(img_original)
    
    # 6. Logika Validasi (Aturan Bisnis)
    final_pred_style = pred_style
    final_pred_kategori = pred_kategori
    
    try:
        valid_categories = rules[gender][pred_style]
        if pred_kategori not in valid_categories:
            if conf_style < conf_kategori: # Percaya Kategori, Style salah
                final_pred_style = f"<{pred_style}?>"
            else: # Percaya Style, Kategori salah
                final_pred_kategori = f"<{pred_kategori}?>"
    except KeyError:
        pass # Tidak ada aturan

    # Kembalikan prediksi mentah dan yang sudah divalidasi
    return {
        "style": final_pred_style,
        "kategori": final_pred_kategori,
        "warna": pred_warna
    }