# app.py (Versi FINAL: Login + Kamera + AI + Logika Tombol)

from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
from flask_mysqldb import MySQL
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
import base64
import uuid

# --- Variabel Global untuk Model ML ---
model_style, model_men, model_women = None, None, None
AI_MODE_AKTIF = False
predict_logic = None 

# --- Blok TRY...EXCEPT untuk memuat AI ---
try:
    # 1. Coba impor library ML
    import tensorflow as tf
    import predict_logic # Impor file baru kita
    
    # 2. Tentukan path model
    STYLE_MODEL_PATH = 'model_style.h5'
    MEN_MODEL_PATH = 'model_kategori_pria.h5'
    WOMEN_MODEL_PATH = 'model_kategori_wanita.h5'
    
    # 3. Muat model ke variabel global
    print("Memuat model AI (TensorFlow)...")
    model_style = tf.keras.models.load_model(STYLE_MODEL_PATH)
    model_men = tf.keras.models.load_model(MEN_MODEL_PATH)
    model_women = tf.keras.models.load_model(WOMEN_MODEL_PATH)
    
    # Jika semua berhasil
    AI_MODE_AKTIF = True
    print("✅ Model AI berhasil dimuat. Mode AI Aktif.")
    
except ImportError as e:
    print(f"⚠ PERINGATAN: Gagal impor library ML ({e}).")
    print("   Aplikasi akan berjalan dalam 'Mode Manual Saja'.")
except Exception as e:
    print(f"⚠ PERINGATAN: Gagal memuat file model .h5 ({e}).")
    print("   Pastikan file .h5 ada di folder yang sama.")
    print("   Aplikasi akan berjalan dalam 'Mode Manual Saja'.")

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# --- Konfigurasi MySQL (Kode Anda) ---
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_swipeer'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# --- Konfigurasi Session ---
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.config['SECRET_KEY'] = 'kunci_rahasia_anda_yang_sangat_aman_b9'

# Inisialisasi MySQL
mysql = MySQL(app)

# --- Direktori Upload ---
UPLOAD_DIR = os.path.join(app.root_path, 'static/uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 1. LOGIKA ATURAN VALIDASI ---
VALID_RULES = {
    'male': { 'Formal': ['suit', 'shirt', 'pants', 'tshirt'], 'Casual': ['outer', 'pants', 'shirt', 'tshirt'], 'Sport': ['jacket', 'pants', 'tshirt'] },
    'female': { 'Formal': ['blazer', 'blouse', 'dress', 'pants', 'shirt', 'skirt', 'tshirt', 'outer'], 'Casual': ['blouse', 'outer', 'pants', 'shirt', 'skirt', 'tshirt'], 'Sport': ['jacket', 'pants', 'tshirt', 'skirt'] }
}
MODEL_CLASSES = {
    'style': ['Casual', 'Formal', 'Sport'], # Ganti urutan jika perlu
    'men': ['jacket', 'outer', 'pants', 'shirt', 'suit', 'tshirt'], # Ganti urutan jika perlu
    'women': ['jacket', 'blazer', 'blouse', 'dress', 'outer', 'pants', 'shirt', 'skirt', 'tshirt'] # Ganti urutan jika perlu
}

# --- 2. FUNGSI HELPER & RUTE AUTENTIKASI (Kode Anda) ---
def is_valid_email(email):
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(regex, email) is not None

@app.route('/', methods=['GET', 'POST'])
def index():
    if session.get("user_id"): return redirect(url_for('dashboard'))
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        # --- Logika Registrasi (Kode Anda) ---
        if form_type == 'register':
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            gender_form = request.form.get('gender')
            terms = request.form.get('terms')
            if not all([name, email, password, confirm_password, gender_form, terms]):
                flash('Semua field wajib diisi', 'error')
                return redirect(url_for('index'))
            # ... (sisa validasi Anda: email, password match, dll) ...
            gender_db = 'male' if gender_form == 'pria' else 'female'
            hashed_password = generate_password_hash(password)
            try:
                cur = mysql.connection.cursor()
                cur.execute("SELECT * FROM users WHERE username = %s OR email = %s", (name, email))
                if cur.fetchone():
                    flash('Username atau Email sudah terdaftar!', 'error')
                    cur.close()
                    return redirect(url_for('index'))
                cur.execute("INSERT INTO users (username, email, password_hash, gender) VALUES (%s, %s, %s, %s)", (name, email, hashed_password, gender_db))
                mysql.connection.commit()
                cur.close()
                flash(f'Registrasi berhasil untuk {name}!', 'success')
                return redirect(url_for('index'))
            except Exception as e:
                flash(f'Terjadi kesalahan: {e}', 'error')
                return redirect(url_for('index'))

        # --- Logika Login (Kode Anda) ---
        elif form_type == 'login':
            email = request.form.get('email')
            password = request.form.get('password')
            if not email or not password:
                flash('Email dan Password wajib diisi', 'error')
                return redirect(url_for('index'))
            if not is_valid_email(email):
                flash('Format email tidak valid', 'error')
                return redirect(url_for('index'))
            try:
                cur = mysql.connection.cursor()
                cur.execute("SELECT * FROM users WHERE email = %s", [email])
                user = cur.fetchone()
                cur.close()
                if user and check_password_hash(user['password_hash'], password):
                    session["user_id"] = user['user_id']
                    session["username"] = user['username']
                    session["gender"] = user['gender']
                    return redirect(url_for('dashboard'))
                else:
                    flash('Email atau password salah', 'error')
                    return redirect(url_for('index'))
            except Exception as e:
                flash(f'Terjadi kesalahan: {e}', 'error')
                return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# --- 3. RUTE APLIKASI UTAMA (DASHBOARD) ---
@app.route('/dashboard')
def dashboard():
    if not session.get("user_id"):
        return redirect(url_for('index'))
    # Kirim status AI_MODE_AKTIF ke frontend
    return render_template('dashboard.html', 
                           user_gender=session.get("gender"), 
                           user_id=session.get("user_id"),
                           ai_mode=AI_MODE_AKTIF)

# --- 4. RUTE API ---

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """API BARU: Menerima foto, menjalankan AI, mengembalikan prediksi."""
    if not session.get("user_id"): return jsonify(error="Not logged in"), 401
    if not AI_MODE_AKTIF: return jsonify(error="AI mode is not active"), 500
        
    data = request.get_json()
    image_data_url = data.get('image_data')
    
    # Simpan gambar sementara untuk dianalisis
    try:
        header, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        temp_filename = f"temp_{session.get('user_id')}.jpg"
        temp_image_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        with open(temp_image_path, "wb") as f:
            f.write(image_data)
            
        # Panggil "otak" AI kita
        models_dict = {'style': model_style, 'men': model_men, 'women': model_women}
        gender = session.get("gender")
        
        predictions = predict_logic.run_all_predictions(
            temp_image_path, gender, models_dict, VALID_RULES, MODEL_CLASSES
        )
        
        # Hapus file sementara
        os.remove(temp_image_path)
        
        return jsonify(predictions)
        
    except Exception as e:
        print(f"Error saat prediksi AI: {e}")
        return jsonify(style="Error", kategori="AI", warna="Gagal"), 500

@app.route('/get_categories', methods=['POST'])
def get_categories():
    # (Kode SAMA seperti sebelumnya)
    if not session.get("user_id"): return jsonify(error="Not logged in"), 401
    data = request.get_json()
    style = data.get('style')
    gender = session.get("gender")
    try:
        categories = VALID_RULES[gender][style]
        return jsonify(categories=categories)
    except KeyError:
        return jsonify(categories=[]), 404

@app.route('/save_item', methods=['POST'])
def save_item():
    # (Kode SAMA seperti sebelumnya, menggunakan flask_mysqldb)
    if not session.get("user_id"): return jsonify(error="Not logged in"), 401
    data = request.get_json()
    style = data.get('style')
    category_name = data.get('category')
    color_name = data.get('color')
    image_data_url = data.get('image_data')
    
    # 1. Simpan Gambar Base64 ke File
    try:
        header, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_DIR, filename)
        with open(image_path, "wb") as f: f.write(image_data)
        image_db_url = f"static/uploads/{filename}"
    except Exception as e:
        print(f"Error menyimpan gambar: {e}")
        return jsonify(status='error', message="Gagal memproses gambar"), 500

    # 2. Simpan Info ke Database
    try:
        cur = mysql.connection.cursor()
        
        # Cari/Buat Kategori
        cur.execute("SELECT * FROM categories WHERE category_name = %s", [category_name])
        category = cur.fetchone()
        category_id = category['category_id'] if category else None
        if not category:
            cur.execute("INSERT INTO categories (category_name) VALUES (%s)", [category_name])
            mysql.connection.commit()
            category_id = cur.lastrowid

        # Cari/Buat Warna
        cur.execute("SELECT * FROM colors WHERE color_name = %s", [color_name])
        color = cur.fetchone()
        color_id = color['color_id'] if color else None
        if not color:
            cur.execute("INSERT INTO colors (color_name) VALUES (%s)", [color_name])
            mysql.connection.commit()
            color_id = cur.lastrowid
            
        # Masukkan ke user_wardrobe
        cur.execute(
            """INSERT INTO user_wardrobe (user_id, item_name, image_url, category_id, color_id, style) 
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (session.get("user_id"), f"{style} {category_name} {color_name}", 
             image_db_url, category_id, color_id, style)
        )
        mysql.connection.commit()
        cur.close()
        return jsonify(status='success', message=f'Item berhasil disimpan!')
    except Exception as e:
        mysql.connection.rollback()
        cur.close()
        print(f"Error saat menyimpan ke DB: {e}")
        return jsonify(status='error', message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)