# image_enhancer.py
import cv2

def enhance_image_quality(image):
    """
    Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    untuk memperbaiki gambar yang terlalu gelap atau terlalu terang.
    
    Args:
        image (numpy.ndarray): Gambar BGR dari cv2.imread()
    
    Returns:
        numpy.ndarray: Gambar BGR yang sudah ditingkatkan kualitasnya
    """
    try:
        # 1. Konversi ke LAB color space
        # Kita hanya ingin mengubah channel L (Lightness/Intensitas)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 2. Pisahkan channel L, A, B
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        
        # 3. Buat objek CLAHE dan terapkan ke L-channel
        # clipLimit=2.0 dan tileGridSize=(8, 8) adalah nilai standar yang bagus
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l_channel = clahe.apply(l_channel)
        
        # 4. Gabungkan kembali channel LAB yang sudah di-enhance
        merged_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
        
        # 5. Konversi kembali ke BGR
        enhanced_image_bgr = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2BGR)
        
        return enhanced_image_bgr
        
    except Exception as e:
        print(f"Error saat enhance gambar: {e}")
        return image # Kembalikan gambar asli jika gagal