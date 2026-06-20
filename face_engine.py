"""
face_engine.py
Core ML engine: InsightFace ArcFace
Database: PostgreSQL langsung (tabel user, karyawan, absensi)
"""

import os, pickle
from pathlib import Path
from datetime import datetime, date

import numpy as np
import cv2
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

load_dotenv()

# ── InsightFace ───────────────────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# ── Konstanta ─────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH   = Path("registered_faces/embeddings.pkl")
FACES_DIR         = Path("registered_faces")
SIMILARITY_THRESH = float(os.getenv("SIMILARITY_THRESH", "0.40"))
DET_SIZE          = (640, 640)

# PostgreSQL — satu-satunya sumber data (user, karyawan, absensi)
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


#  POSTGRESQL HELPERS

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def test_connection() -> dict:
    """
    Cek koneksi ke PostgreSQL dan status model InsightFace.
    Dipakai oleh endpoint /health.
    """
    result = {
        "database":   {"status": "error", "detail": ""},
        "face_model": {"status": "error", "detail": ""},
        "embeddings": {"jumlah_karyawan": 0},
    }

    # Cek PostgreSQL
    try:
        conn = get_db_connection()
        conn.close()
        result["database"] = {"status": "ok", "detail": f"Terhubung ke {DB_NAME}@{DB_HOST}:{DB_PORT}"}
    except Exception as e:
        result["database"] = {"status": "error", "detail": str(e)}

    # Face model
    result["face_model"] = {
        "status": "ok" if INSIGHTFACE_AVAILABLE else "error",
        "detail": "InsightFace buffalo_l loaded" if INSIGHTFACE_AVAILABLE
                  else "insightface tidak terinstall",
    }

    # Embeddings lokal
    if EMBEDDINGS_PATH.exists():
        with open(EMBEDDINGS_PATH, "rb") as f:
            emb = pickle.load(f)
        result["embeddings"]["jumlah_karyawan"] = len(emb)

    return result


def init_db():
    """
    Pastikan tabel PostgreSQL "user", "karyawan", dan "absensi" sudah dibuat.
    """
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS "user" (
                id SERIAL PRIMARY KEY,
                nip VARCHAR(50),
                username VARCHAR(100),
                email VARCHAR(150) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'karyawan'
            );
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS karyawan (
                id SERIAL PRIMARY KEY,
                nama VARCHAR(150) NOT NULL,
                nip VARCHAR(50),
                divisi VARCHAR(100),
                terdaftar TIMESTAMP NOT NULL DEFAULT NOW()
            );
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS absensi (
                id SERIAL PRIMARY KEY
            );
        ''')
        # Migrasi defensif: tabel 'absensi' mungkin sudah ada dari versi sistem
        # sebelumnya dengan kolom berbeda. ADD COLUMN IF NOT EXISTS aman dijalankan
        # berulang kali, baik untuk tabel baru maupun tabel lama yang perlu dilengkapi.
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS nip VARCHAR(100)')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS nama VARCHAR(150)')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS tipe VARCHAR(20)')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS waktu TIMESTAMP')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS tanggal DATE DEFAULT CURRENT_DATE')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS confidence REAL')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS karyawan_id INTEGER REFERENCES karyawan(id) ON DELETE CASCADE')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS check_in TIMESTAMP')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS check_out TIMESTAMP')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS confidence_in REAL')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS confidence_out REAL')
        # ── Kolom location (baru) ────────────────────────────────────────────
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS location VARCHAR(255)')
        cur.execute('ALTER TABLE absensi ADD COLUMN IF NOT EXISTS shift_id INTEGER REFERENCES shift_master(id) ON DELETE SET NULL')
        cur.execute('ALTER TABLE karyawan ADD COLUMN IF NOT EXISTS foto_urls JSONB DEFAULT \'[]\'')
        conn.commit()
        cur.close()
        conn.close()
        print("[INFO] Tabel 'user', 'karyawan', 'absensi' siap di PostgreSQL")
    except Exception as e:
        print(f"[ERROR] Gagal inisialisasi tabel di PostgreSQL: {e}")


def seed_admin():
    """Buat user login admin default di PostgreSQL kalau belum ada."""
    from werkzeug.security import generate_password_hash
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('SELECT id FROM "user" WHERE email = %s', ("admin@local",))
        if cur.fetchone() is None:
            cur.execute(
                'INSERT INTO "user" (username, email, password, role, nip) '
                'VALUES (%s, %s, %s, %s, %s)',
                ("admin", "admin@local", generate_password_hash("admin123"), "admin", None)
            )
            conn.commit()
            print("[INFO] Akun admin default dibuat: admin@local / admin123")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Gagal seed admin: {e}")

def get_users():
    """List semua user dari tabel 'user' PostgreSQL."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('SELECT id, nip, username, email, role FROM "user" ORDER BY id')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        return rows
    except Exception as e:
        print(f"[ERROR] get_users: {e}")
        return []


def get_user_by_email(email: str):
    """Ambil user dari tabel 'user' PostgreSQL berdasarkan email."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            'SELECT id, nip, username, email, password, role FROM "user" WHERE email = %s',
            (email,)
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[ERROR] get_user_by_email: {e}")
        return None


def get_user_by_id(user_id: int):
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            'SELECT id, nip, username, email, password, role FROM "user" WHERE id = %s',
            (user_id,)
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[ERROR] get_user_by_id: {e}")
        return None


def tambah_user(email, username, password, role="karyawan", nip=None) -> dict:
    """Buat user login baru langsung di PostgreSQL."""
    from werkzeug.security import generate_password_hash
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('SELECT id FROM "user" WHERE email = %s', (email,))
        if cur.fetchone():
            cur.close(); conn.close()
            return {"success": False, "msg": f"Email '{email}' sudah terdaftar"}
        cur.execute(
            'INSERT INTO "user" (nip, username, email, password, role) '
            'VALUES (%s, %s, %s, %s, %s)',
            (nip, username or email, email, generate_password_hash(password), role)
        )
        conn.commit()
        cur.close(); conn.close()
        return {"success": True, "msg": f"User '{email}' berhasil dibuat"}
    except Exception as e:
        return {"success": False, "msg": str(e)}


def hapus_user(user_id: int) -> bool:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('DELETE FROM "user" WHERE id = %s', (user_id,))
        conn.commit()
        cur.close(); conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] hapus_user: {e}")
        return False

def get_karyawan_list():
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            'SELECT id, nip, nama, divisi, terdaftar, foto_urls FROM karyawan ORDER BY id'
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        return [{
            "id":        str(r["id"]),
            "nip":       r["nip"] or "-",
            "nama":      r["nama"],
            "divisi":    r["divisi"] or "-",
            "terdaftar": r["terdaftar"].strftime("%Y-%m-%d") if r["terdaftar"] else "",
            "foto_urls": r["foto_urls"] or [],
        } for r in rows]
    except Exception as e:
        print(f"[ERROR] get_karyawan_list: {e}")
        return []

def register_karyawan_db(nip_input: str, nama: str, divisi: str, foto_urls: list = None) -> int:
    """
    Buat atau update baris karyawan di PostgreSQL.
    Return: id karyawan (dipakai sebagai key embedding wajah).
    """
    import json
    foto_json = json.dumps(foto_urls or [])
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id FROM karyawan WHERE nama = %s", (nama,))
    row = cur.fetchone()
    if row:
        karyawan_id = row[0]
        cur.execute(
            "UPDATE karyawan SET divisi = %s, nip = %s, foto_urls = %s WHERE id = %s",
            (divisi, nip_input or None, foto_json, karyawan_id)
        )
    else:
        cur.execute(
            "INSERT INTO karyawan (nama, nip, divisi, foto_urls) VALUES (%s, %s, %s, %s) RETURNING id",
            (nama, nip_input or None, divisi, foto_json)
        )
        karyawan_id = cur.fetchone()[0]
    conn.commit()
    cur.close(); conn.close()
    return karyawan_id

def delete_karyawan_db(karyawan_id: int) -> bool:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("DELETE FROM karyawan WHERE id = %s", (karyawan_id,))
        conn.commit()
        cur.close(); conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] delete_karyawan_db: {e}")
        return False

def catat_absensi(nip: str, nama: str, tipe: str, confidence: float,
                  location: str = None, shift_id: int = None) -> dict:
    try:
        karyawan_id = int(nip)   # Penyimpanan NIP
    except (ValueError, TypeError):
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("SELECT id FROM karyawan WHERE nama = %s", (nama,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return {"success": False, "msg": f"Karyawan {nama} tidak ditemukan"}
        karyawan_id = row[0]

    now       = datetime.now()
    tanggal   = now.date()
    waktu_str = now.strftime("%H:%M:%S")

    if location and len(location) > 255:
        location = location[:252] + "..."

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT nip, nama FROM karyawan WHERE id = %s", (karyawan_id,))
        k_row = cur.fetchone()
        if not k_row or not k_row[0]:
            return {"success": False, "msg": "NIP karyawan tidak ditemukan, isi NIP terlebih dahulu"}
        nip_val  = k_row[0]
        nama_val = k_row[1]

        if tipe == "masuk":
            cur.execute(
                "SELECT id FROM absensi WHERE karyawan_id = %s AND tanggal = %s "
                "AND check_out IS NULL",
                (karyawan_id, tanggal)
            )
            if cur.fetchone():
                return {"success": False, "msg": f"{nama} sudah absen masuk hari ini"}

            cur.execute(
                "INSERT INTO absensi "
                "(nip, nama, tipe, waktu, confidence, karyawan_id, tanggal, "
                " check_in, confidence_in, location, shift_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (nip_val, nama_val, "masuk", now, confidence,
                 karyawan_id, tanggal, now, confidence, location, shift_id or None)
            )
            conn.commit()
            return {"success": True, "msg": f"Absen masuk berhasil: {nama} pukul {waktu_str}"}

        else:  # pulang
            cur.execute(
                "SELECT id FROM absensi WHERE karyawan_id = %s AND tanggal = %s "
                "AND check_out IS NULL ORDER BY id DESC LIMIT 1",
                (karyawan_id, tanggal)
            )
            row = cur.fetchone()
            if not row:
                return {"success": False, "msg": f"{nama} belum absen masuk hari ini"}

            cur.execute(
                "UPDATE absensi "
                "SET check_out = %s, confidence_out = %s, waktu = %s, "
                "    confidence = %s, location = %s "
                "WHERE id = %s",
                (now, confidence, now, confidence, location, row[0])
            )
            conn.commit()
            return {"success": True, "msg": f"Absen pulang berhasil: {nama} pukul {waktu_str}"}

    except Exception as e:
        conn.rollback()
        return {"success": False, "msg": str(e)}
    finally:
        cur.close()
        conn.close()

def get_absensi_hari_ini():
    tanggal = date.today()

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                a.karyawan_id,
                a.nip,
                k.nama,
                a.check_in,
                a.check_out,
                a.confidence_in,
                a.confidence_out,
                a.location,
                sm.nama_shift,
                sm.jam_masuk,
                sm.jam_pulang
            FROM absensi a
            JOIN karyawan k ON k.id = a.karyawan_id
            LEFT JOIN shift_master sm ON sm.id = a.shift_id
            WHERE a.tanggal = %s
            ORDER BY a.check_in
        """, (tanggal,))

        rows = cur.fetchall()

        cur.close()
        conn.close()

        result = []

        for r in rows:
            nip_val = r["nip"] or str(r["karyawan_id"])

            nama_shift = r["nama_shift"] or "-"
            jam_masuk  = str(r["jam_masuk"])[:5] if r["jam_masuk"] else "-"
            jam_pulang = str(r["jam_pulang"])[:5] if r["jam_pulang"] else "-"

            result.append({
                "nama": r["nama"],
                "nip": nip_val,
                "karyawan_id": r["karyawan_id"],
                "tipe": "masuk",
                "waktu": r["check_in"].strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": round(r["confidence_in"] or 0, 4),
                "location": r["location"] or "",
                "nama_shift": nama_shift,
                "jam_shift_masuk": jam_masuk,
                "jam_shift_pulang": jam_pulang,
            })

            if r["check_out"]:
                result.append({
                    "nama": r["nama"],
                    "nip": nip_val,
                    "karyawan_id": r["karyawan_id"],
                    "tipe": "pulang",
                    "waktu": r["check_out"].strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": round(r["confidence_out"] or 0, 4),
                    "location": r["location"] or "",
                    "nama_shift": nama_shift,
                    "jam_shift_masuk": jam_masuk,
                    "jam_shift_pulang": jam_pulang,
                })

        return sorted(result, key=lambda x: x["waktu"])

    except Exception as e:
        print(f"[ERROR] get_absensi_hari_ini: {e}")
        return []

# def get_absensi_hari_ini():
#     tanggal = date.today()
#     try:
#         conn = get_db_connection()
#         cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
#         cur.execute('''
#             SELECT a.karyawan_id, a.nip, k.nama, a.check_in, a.check_out,
#                    a.confidence_in, a.confidence_out, a.location
#             FROM absensi a
#             JOIN karyawan k ON k.id = a.karyawan_id
#             WHERE a.tanggal = %s
#             ORDER BY a.check_in
#         ''', (tanggal,))
#         rows = cur.fetchall()
#         cur.close(); conn.close()

#         result = []
#         for r in rows:
#             nip_val = r["nip"] or str(r["karyawan_id"])
#             result.append({
#                 "nama":       r["nama"], "nip": nip_val,
#                 "tipe":       "masuk",
#                 "waktu":      r["check_in"].strftime("%Y-%m-%d %H:%M:%S"),
#                 "confidence": round(r["confidence_in"] or 0, 4),
#                 "location":   r["location"] or "",
#             })
#             if r["check_out"]:
#                 result.append({
#                     "nama":       r["nama"], "nip": nip_val,
#                     "tipe":       "pulang",
#                     "waktu":      r["check_out"].strftime("%Y-%m-%d %H:%M:%S"),
#                     "confidence": round(r["confidence_out"] or 0, 4),
#                     "location":   r["location"] or "",
#                 })
#         return sorted(result, key=lambda x: x["waktu"])
#     except Exception as e:
#         print(f"[ERROR] get_absensi_hari_ini: {e}")
#         return []

def get_absensi_range(tgl_awal: str, tgl_akhir: str):
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('''
            SELECT
                a.karyawan_id, a.nip, k.nama, k.divisi,
                a.tanggal, a.check_in, a.check_out,
                a.confidence_in, a.confidence_out, a.location,
                sm.nama_shift, sm.jam_masuk, sm.jam_pulang, sm.toleransi_menit
            FROM absensi a
            JOIN karyawan k ON k.id = a.karyawan_id
            LEFT JOIN shift_master sm ON sm.id = a.shift_id
            WHERE a.tanggal BETWEEN %s AND %s
            ORDER BY a.tanggal DESC, a.check_in DESC
        ''', (tgl_awal, tgl_akhir))
        rows = cur.fetchall()
        cur.close(); conn.close()

        def hitung_status(check_in, jam_masuk, toleransi):
            if not check_in or not jam_masuk:
                return '-'
            from datetime import datetime, timedelta
            batas = (datetime.combine(check_in.date(), jam_masuk)
                     + timedelta(minutes=toleransi or 0))
            if check_in <= batas:
                return 'Tepat Waktu'
            selisih = int((check_in - batas).total_seconds() / 60)
            return f'Terlambat {selisih} mnt'

        result = []
        for r in rows:
            tgl_str    = r['tanggal'].strftime('%Y-%m-%d')
            nip_val    = r['nip'] or str(r['karyawan_id'])
            nama_shift = r['nama_shift'] or '-'
            jam_masuk  = r['jam_masuk']
            jam_pulang = r['jam_pulang']
            toleransi  = r['toleransi_menit']
            jam_masuk_str  = str(jam_masuk)[:5]  if jam_masuk  else '-'
            jam_pulang_str = str(jam_pulang)[:5] if jam_pulang else '-'

            status = hitung_status(r['check_in'], jam_masuk, toleransi)

            # Hitung durasi (detik) untuk kolom Time
            check_in_ts = int(r['check_in'].timestamp()) if r['check_in'] else None
            if r['check_in'] and r['check_out']:
                durasi_detik = int((r['check_out'] - r['check_in']).total_seconds())
            else:
                durasi_detik = None

            def _fmt_durasi(detik):
                if detik is None:
                    return '-'
                h, rem = divmod(detik, 3600)
                m, s   = divmod(rem, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            result.append({
                'tanggal':    tgl_str,
                'nip':        nip_val,
                'nama':       r['nama'],
                'divisi':     r['divisi'] or '-',
                'nama_shift': nama_shift,
                'jam_shift_masuk':  jam_masuk_str,
                'jam_shift_pulang': jam_pulang_str,
                'check_in':   r['check_in'].strftime('%H:%M')  if r['check_in']  else '-',
                'check_out':  r['check_out'].strftime('%H:%M') if r['check_out'] else '-',
                'check_in_ts':  check_in_ts,
                'durasi_detik': durasi_detik,
                'durasi_fmt':   _fmt_durasi(durasi_detik),
                'status':     status,
                'location':   r['location'] or '-',
                # compat lama
                'tipe':  'masuk',
                'waktu': r['check_in'].strftime('%H:%M') if r['check_in'] else '-',
            })
        return result
    except Exception as e:
        print(f'[ERROR] get_absensi_range: {e}')
        return []

class FaceEngine:
    def __init__(self):
        self.app        = None
        self.embeddings = {}
        self._load_model()
        self._load_embeddings()

    def _load_model(self):
        if not INSIGHTFACE_AVAILABLE:
            return
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)

    def _load_embeddings(self):
        if EMBEDDINGS_PATH.exists():
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.embeddings = pickle.load(f)

    def _save_embeddings(self):
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(self.embeddings, f)

    def _to_bgr(self, img):
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _cosine_sim(self, v1, v2):
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)
        return float(np.dot(v1, v2))

    def get_best_embedding(self, img_bgr):
        if self.app is None:
            return None, None, 0.0
        faces = sorted(self.app.get(img_bgr), key=lambda f: f.det_score, reverse=True)
        if not faces:
            return None, None, 0.0
        best = faces[0]
        return best.embedding, best.bbox.astype(int), float(best.det_score)

    def recognize(self, img_bgr) -> dict:
        if self.app is None:
            return _unknown("Model belum dimuat")
        if not self.embeddings:
            return _unknown("Belum ada karyawan terdaftar")
        emb, bbox, det_score = self.get_best_embedding(img_bgr)
        if emb is None:
            return _unknown("Wajah tidak terdeteksi")
        best_nip, best_sim, best_nama = None, -1.0, None
        for nip, data in self.embeddings.items():
            avg = float(np.mean([self._cosine_sim(emb, v) for v in data["vecs"]]))
            if avg > best_sim:
                best_sim, best_nip, best_nama = avg, nip, data["nama"]
        if best_sim >= SIMILARITY_THRESH:
            return {"recognized": True, "nip": best_nip, "nama": best_nama,
                    "confidence": round(best_sim, 4), "bbox": bbox.tolist(),
                    "det_score": round(det_score, 4)}
        return _unknown(f"Similarity {best_sim:.3f} < threshold {SIMILARITY_THRESH}")

    def draw_result(self, img_bgr, result) -> np.ndarray:
        out = img_bgr.copy()
        if not result.get("bbox"):
            return out
        x1, y1, x2, y2 = result["bbox"]
        nama  = result.get("nama", "Unknown")
        conf  = result.get("confidence", 0)
        color = (0, 200, 80) if result["recognized"] else (0, 0, 220)
        label = f"{nama} {conf:.2f}" if result["recognized"] else "Unknown"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, y1 - 28), (x2, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def register(self, nip: str, nama: str, divisi: str, images: list, base_url: str = "") -> dict:
        if self.app is None:
            return {"success": False, "msg": "Model belum dimuat"}
        vecs = []
        for img in images:
            bgr = self._to_bgr(img)
            emb, _, score = self.get_best_embedding(bgr)
            if emb is not None and score > 0.5:
                vecs.append(emb)
        if not vecs:
            return {"success": False, "msg": "Tidak ada wajah terdeteksi"}

        try:
            # Daftar dulu untuk dapat karyawan_id, foto_urls diupdate setelah simpan file
            karyawan_id = register_karyawan_db(nip, nama, divisi, [])
            nip_key = str(karyawan_id)
        except Exception as e:
            return {"success": False, "msg": f"Gagal daftar ke database: {e}"}

        # Simpan file foto
        face_dir = FACES_DIR / nip_key
        face_dir.mkdir(parents=True, exist_ok=True)
        foto_urls = []
        for i, img in enumerate(images[:5]):
            filename = f"ref_{i}.jpg"
            cv2.imwrite(str(face_dir / filename), self._to_bgr(img))
            foto_urls.append(f"{base_url}/registered_faces/{karyawan_id}/{filename}")

        # Update foto_urls di DB
        register_karyawan_db(nip, nama, divisi, foto_urls)

        self.embeddings[nip_key] = {"nama": nama, "vecs": vecs}
        self._save_embeddings()

        return {"success": True, "msg": f"Berhasil mendaftarkan {nama} ({len(vecs)} foto)"}

    def delete_karyawan(self, nip: str) -> bool:
        import shutil
        if nip in self.embeddings:
            del self.embeddings[nip]
            self._save_embeddings()
        # Hapus folder foto
        face_dir = FACES_DIR / nip
        if face_dir.exists():
            shutil.rmtree(face_dir)
        try:
            return delete_karyawan_db(int(nip))
        except Exception:
            return False


def _unknown(msg=""):
    return {"recognized": False, "nip": None, "nama": None,
            "confidence": 0.0, "bbox": None, "det_score": 0.0, "msg": msg}


#  SHIFT MASTER

def get_shift_list():
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('SELECT * FROM shift_master ORDER BY jam_masuk')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        for r in rows:
            r['jam_masuk']  = str(r['jam_masuk'])[:5]
            r['jam_pulang'] = str(r['jam_pulang'])[:5]
        return rows
    except Exception as e:
        print(f"[ERROR] get_shift_list: {e}")
        return []


def tambah_shift(nama_shift, jam_masuk, jam_pulang, toleransi_menit=15,
                 melewati_tengah_malam=False, keterangan='') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'INSERT INTO shift_master (nama_shift, jam_masuk, jam_pulang, '
            'toleransi_menit, melewati_tengah_malam, keterangan) '
            'VALUES (%s, %s, %s, %s, %s, %s)',
            (nama_shift, jam_masuk, jam_pulang,
             int(toleransi_menit), bool(melewati_tengah_malam), keterangan)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f"Shift '{nama_shift}' berhasil ditambahkan"}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def hapus_shift(shift_id: int) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('DELETE FROM shift_master WHERE id = %s', (shift_id,))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Shift berhasil dihapus'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def edit_shift(shift_id: int, nama_shift, jam_masuk, jam_pulang,
              toleransi_menit=15, melewati_tengah_malam=False, keterangan='') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'UPDATE shift_master SET nama_shift=%s, jam_masuk=%s, jam_pulang=%s, '
            'toleransi_menit=%s, melewati_tengah_malam=%s, keterangan=%s WHERE id=%s',
            (nama_shift, jam_masuk, jam_pulang,
             int(toleransi_menit), bool(melewati_tengah_malam), keterangan, shift_id)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f"Shift '{nama_shift}' berhasil diupdate"}
    except Exception as e:
        return {'success': False, 'msg': str(e)}

def get_shift_karyawan():
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('''
            SELECT k.id AS karyawan_id, k.nama, k.nip, k.divisi,
                   sm.id AS shift_id, sm.nama_shift,
                   sm.jam_masuk, sm.jam_pulang,
                   sk.berlaku_dari, sk.berlaku_sampai
            FROM karyawan k
            LEFT JOIN shift_karyawan sk
                ON sk.karyawan_id = k.id
                AND sk.berlaku_dari <= CURRENT_DATE
                AND (sk.berlaku_sampai IS NULL OR sk.berlaku_sampai >= CURRENT_DATE)
            LEFT JOIN shift_master sm ON sm.id = sk.shift_id
            ORDER BY k.nama
        ''')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        for r in rows:
            if r['jam_masuk']:  r['jam_masuk']  = str(r['jam_masuk'])[:5]
            if r['jam_pulang']: r['jam_pulang'] = str(r['jam_pulang'])[:5]
        return rows
    except Exception as e:
        print(f"[ERROR] get_shift_karyawan: {e}")
        return []


def assign_shift(karyawan_id: int, shift_id: int, berlaku_dari: str,
                 berlaku_sampai: str = None) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('''
            INSERT INTO shift_karyawan (karyawan_id, shift_id, berlaku_dari, berlaku_sampai)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (karyawan_id, berlaku_dari)
            DO UPDATE SET shift_id = EXCLUDED.shift_id,
                          berlaku_sampai = EXCLUDED.berlaku_sampai
        ''', (karyawan_id, shift_id, berlaku_dari, berlaku_sampai or None))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Shift karyawan berhasil diassign'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def hapus_shift_karyawan(sk_id: int) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('DELETE FROM shift_karyawan WHERE id = %s', (sk_id,))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Assignment shift dihapus'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}

def get_absensi_karyawan(karyawan_id: int, limit: int = 30) -> list:
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('''
            SELECT a.tanggal, a.check_in, a.check_out, a.location,
                   sm.nama_shift, sm.jam_masuk, sm.jam_pulang
            FROM absensi a
            LEFT JOIN shift_master sm ON sm.id = a.shift_id
            WHERE a.karyawan_id = %s
            ORDER BY a.tanggal DESC, a.check_in DESC
            LIMIT %s
        ''', (karyawan_id, limit))
        rows = cur.fetchall()
        cur.close(); conn.close()
        result = []
        for r in rows:
            ci = r['check_in']
            co = r['check_out']
            # hitung durasi
            if ci and co:
                diff = int((co - ci).total_seconds())
                h, rem = divmod(diff, 3600)
                m, s   = divmod(rem, 60)
                durasi = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                durasi = None
            result.append({
                'tanggal':    r['tanggal'].strftime('%Y-%m-%d'),
                'check_in':   ci.strftime('%H:%M') if ci else '-',
                'check_in_ts': int(ci.timestamp()) if ci else None,
                'check_out':  co.strftime('%H:%M') if co else '-',
                'durasi':     durasi,
                'nama_shift': r['nama_shift'] or '-',
                'jam_shift':  (str(r['jam_masuk'])[:5] + '–' + str(r['jam_pulang'])[:5]) if r['jam_masuk'] else '-',
                'location':   r['location'] or '-',
            })
        return result
    except Exception as e:
        print(f'[ERROR] get_absensi_karyawan: {e}')
        return []


def get_shift_aktif_karyawan(karyawan_id: int) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('''
            SELECT sm.* FROM shift_karyawan sk
            JOIN shift_master sm ON sm.id = sk.shift_id
            WHERE sk.karyawan_id = %s
              AND sk.berlaku_dari <= CURRENT_DATE
              AND (sk.berlaku_sampai IS NULL OR sk.berlaku_sampai >= CURRENT_DATE)
            ORDER BY sk.berlaku_dari DESC LIMIT 1
        ''', (karyawan_id,))
        row = cur.fetchone()
        cur.close(); conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[ERROR] get_shift_aktif_karyawan: {e}")
        return None