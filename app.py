# app.py (Versi FINAL: Login + Kamera + AI + Logika Tombol)

from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
from flask_mysqldb import MySQL
from flask_session import Session
from MySQLdb.cursors import DictCursor
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
import base64
import uuid
from flask_cors import CORS
from predict_skin import detect_skin_tone_ai


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
CORS(app)

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

@app.route('/api/login', methods=['POST'])
def api_login():
    print("=== API LOGIN MASUK ===")   # <-- tambah
    data = request.get_json()
    print("DATA:", data)  

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify(success=False, message="Email & password wajib"), 400

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email=%s", [email])
    user = cur.fetchone()
    cur.close()

    if user and check_password_hash(user['password_hash'], password):
        return jsonify({
            "success": True,
            "user_id": user['user_id'],
            "username": user['username'],
            "gender": user['gender'],
        })

    return jsonify(success=False, message="Login gagal"), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()

    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        return jsonify(success=False, message="Data tidak lengkap"), 400

    hashed = generate_password_hash(password)

    try:
        cur = mysql.connection.cursor()

        # cek email sudah ada
        cur.execute("SELECT user_id FROM users WHERE email=%s", [email])
        if cur.fetchone():
            cur.close()
            return jsonify(success=False, message="Email sudah terdaftar"), 409

        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s,%s,%s)",
            (name, email, hashed)
        )
        mysql.connection.commit()
        cur.close()

        return jsonify(success=True, message="Registrasi berhasil")
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


@app.route('/home')
def home():
    if not session.get("user_id"):
        return redirect(url_for('index'))
    
    return render_template('index.html', 
                           username=session.get("username"),
                           gender=session.get("gender"))

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
    
@app.route('/wardrobe')
def wardrobe_page():
    if not session.get("user_id"):
        return redirect(url_for('index'))
    return render_template('wardrobe.html')

@app.route('/api/wardrobe/<int:user_id>', methods=['GET'])
def api_get_wardrobe(user_id):
    try:
        cur = mysql.connection.cursor()
        query = """
            SELECT uw.item_id, uw.item_name, uw.image_url, uw.style,
                   c.category_name, clr.color_name
            FROM user_wardrobe uw
            LEFT JOIN categories c ON uw.category_id = c.category_id
            LEFT JOIN colors clr ON uw.color_id = clr.color_id
            WHERE uw.user_id = %s
            ORDER BY uw.item_id DESC
        """
        cur.execute(query, [user_id])
        items = cur.fetchall()
        cur.close()

        return jsonify(success=True, items=items)

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


@app.route('/get_wardrobe_items', methods=['GET'])
def get_wardrobe_items():
    if not session.get("user_id"):
        return jsonify(error="Not logged in"), 401

    try:
        cur = mysql.connection.cursor()
        query = """
            SELECT uw.item_id, uw.item_name, uw.image_url, uw.style,
                   c.category_name, clr.color_name
            FROM user_wardrobe uw
            LEFT JOIN categories c ON uw.category_id = c.category_id
            LEFT JOIN colors clr ON uw.color_id = clr.color_id
            WHERE uw.user_id = %s
            ORDER BY uw.item_id DESC
        """
        cur.execute(query, [session.get("user_id")])
        items = cur.fetchall()
        cur.close()

        return jsonify(items)

    except Exception as e:
        print(f"Error get wardrobe: {e}")
        return jsonify(error=str(e)), 500

def classify_body_shape(bust, waist, hip):
    ratio_bust_hip = bust / hip
    ratio_waist_bust = waist / bust
    ratio_waist_hip = waist / hip

    if abs(bust - hip) / hip <= 0.05 and ratio_waist_bust <= 0.75 and ratio_waist_hip <= 0.75:
        return 'Hourglass'
    if hip >= bust * 1.05:
        return 'Pear'
    if bust >= hip * 1.05:
        return 'Inverted Triangle'
    if ratio_waist_bust >= 0.75 and ratio_waist_hip >= 0.75 and abs(bust - hip) / hip <= 0.05:
        return 'Rectangle'
    if waist >= bust * 0.85 and bust > hip:
        return 'Apple'

    return 'Unknown'

@app.route('/api/bodyshape/calculate', methods=['POST'])
def calculate_bodyshape():
    data = request.json
    bust = float(data['bust'])
    waist = float(data['waist'])
    hip = float(data['hip'])

    body_shape = classify_body_shape(bust, waist, hip)

    descriptions = {
        "Hourglass": "Bahu dan pinggul seimbang dengan pinggang ramping.",
        "Pear": "Pinggul lebih besar dari bahu. Cocok fokus ke atasan.",
        "Inverted Triangle": "Bahu lebih lebar dari pinggul.",
        "Rectangle": "Bahu, pinggang, dan pinggul hampir sejajar.",
        "Apple": "Bagian tengah tubuh lebih dominan."
    }

    return jsonify({
        "success": True,
        "body_shape": body_shape,
        "description": descriptions.get(body_shape, "")
    })

BODY_SHAPE_MAP = {
    "Hourglass": 1,
    "Pear": 2,
    "Inverted Triangle": 3,
    "Rectangle": 4,
    "Apple": 5
}

@app.route('/api/bodyshape/save', methods=['POST'])
def save_bodyshape():
    data = request.json
    user_id = data['user_id']
    shape_name = data['body_shape']

    body_shape_id = BODY_SHAPE_MAP.get(shape_name)

    if not body_shape_id:
        return jsonify({"success": False, "message": "Body shape tidak valid"}), 400

    cursor = mysql.connection.cursor()
    cursor.execute("""
        UPDATE users 
        SET body_shape_id = %s
        WHERE user_id = %s
    """, (body_shape_id, user_id))

    mysql.connection.commit()
    cursor.close()

    return jsonify({"success": True})

@app.route('/api/bodyshape/<int:user_id>', methods=['GET'])
def get_bodyshape(user_id):
    cursor = mysql.connection.cursor(DictCursor)

    cursor.execute("""
        SELECT bs.shape_name
        FROM users u
        LEFT JOIN body_shapes bs 
            ON u.body_shape_id = bs.body_shape_id
        WHERE u.user_id = %s
    """, (user_id,))

    row = cursor.fetchone()
    cursor.close()

    return jsonify({
        "success": True,
        "body_shape": row['shape_name'] if row and row['shape_name'] else None
    })

@app.route('/scan_face', methods=['POST'])
def scan_face():
    if not session.get("user_id"):
        return jsonify(error="Not logged in"), 401

    data = request.get_json()
    img_url = data.get('image')

    try:
        # Decode base64
        header, encoded = img_url.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Simpan sementara
        filename = f"user_{session['user_id']}_skin.jpg"
        path = os.path.join(UPLOAD_DIR, filename)
        cv2.imwrite(path, img)

        # Prediksi AI
        skin_tone = detect_skin_tone_ai(path)

        return jsonify(
            status="success",
            skin_tone=skin_tone
        )

    except Exception as e:
        return jsonify(status="error", message=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)