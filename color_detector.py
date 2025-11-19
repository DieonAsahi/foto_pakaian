# color_detector.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# Fungsi get_color_name() tetap sama
def get_color_name(rgb_tuple):
    colors = {
        "Merah": [255, 0, 0], "Hijau": [0, 128, 0], "Biru": [0, 0, 255],
        "Kuning": [255, 255, 0], "Oranye": [255, 165, 0], "Ungu": [128, 0, 128],
        "Pink": [255, 192, 203], "Coklat": [165, 42, 42], "Hitam": [0, 0, 0],
        "Abu-abu": [128, 128, 128], "Putih": [255, 255, 255]
    }
    min_distance = float('inf')
    closest_name = "Tidak Diketahui"
    for name, color_rgb in colors.items():
        distance = np.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(rgb_tuple, color_rgb)]))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

# FUNGSI BARU YANG LEBIH CERDAS
def get_dominant_color_isolated(image, k=5):
    """
    Mendeteksi warna dominan dengan mengisolasi objek dari background putih.
    
    Args:
        image (numpy.ndarray): Gambar BGR dari cv2.imread()
        k (int): Jumlah cluster K-Means
    
    Returns:
        str: Nama warna dominan
    """
    try:
        # 1. Konversi ke Grayscale untuk membuat mask
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Buat mask menggunakan Thresholding
        # Kita anggap background > 240 (hampir putih)
        # THRESH_BINARY_INV: Piksel < 240 jadi putih (255), piksel > 240 jadi hitam (0)
        _val, mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Konversi gambar asli ke RGB (penting untuk K-Means)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. Ubah shape gambar (Height, Width, 3) -> (H*W, 3)
        pixels_rgb = img_rgb.reshape((-1, 3))
        
        # 5. Ubah shape mask (Height, Width) -> (H*W,)
        mask_reshaped = mask.reshape((-1,))
        
        # 6. Ambil HANYA piksel yang ada di dalam mask (bukan background)
        # 'mask_reshaped != 0' akan memilih semua piksel yang BUKAN hitam (0)
        clothing_pixels = pixels_rgb[mask_reshaped != 0]
        
        if len(clothing_pixels) == 0:
            return "Tidak ada objek terdeteksi"

        # 7. Jalankan K-Means HANYA pada piksel pakaian
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(clothing_pixels)
        
        # 8. Dapatkan cluster yang paling umum
        counts = Counter(kmeans.labels_)
        dominant_cluster_index = counts.most_common(1)[0][0]
        
        # 9. Dapatkan nilai RGB dari cluster tsb
        dominant_rgb = kmeans.cluster_centers_[dominant_cluster_index].astype(int)
        
        # 10. Konversi RGB ke Nama Warna
        dominant_name = get_color_name(tuple(dominant_rgb))
        
        return dominant_name
        
    except Exception as e:
        print(f"Error saat deteksi warna: {e}")
        return "Error Deteksi"