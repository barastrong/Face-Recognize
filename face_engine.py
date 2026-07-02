import os, pickle, json
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


def _append_audit_log(event_type: str, payload: dict):
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "system.log"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **payload,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] append audit log: {e}")


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
    """Cek koneksi ke PostgreSQL. Untuk setup tabel jalankan: python table.py"""
    try:
        conn = get_db_connection()
        conn.close()
        print("[INFO] Koneksi PostgreSQL OK")
    except Exception as e:
        print(f"[ERROR] Koneksi PostgreSQL gagal: {e}")


def seed_admin():
    """Buat user login admin default di PostgreSQL kalau belum ada."""
    from werkzeug.security import generate_password_hash
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('SELECT id FROM "user" WHERE email = %s', ("admin@local",))
        if cur.fetchone() is None:
            cur.execute(
                'INSERT INTO "user" (username, email, password, role, nip, is_login, address) '
                'VALUES (%s, %s, %s, %s, %s, %s, %s)',
                ("admin", "admin@local", generate_password_hash("admin123"), "admin", None, False, None)
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
        cur.execute('SELECT id, nip, username, email, role, is_login, address FROM "user" ORDER BY id')
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
            'SELECT id, nip, username, email, password, role, is_login, address FROM "user" WHERE email = %s',
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
            'SELECT id, nip, username, email, password, role, is_login, address FROM "user" WHERE id = %s',
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
            _append_audit_log("absensi", {
                "status": "error",
                "reason": "karyawan tidak ditemukan",
                "input_nip": nip,
                "input_nama": nama,
                "tipe": tipe,
                "location": location,
                "shift_id": shift_id,
            })
            return {"success": False, "msg": f"Karyawan {nama} tidak ditemukan"}
        karyawan_id = row[0]

    now       = datetime.now()
    tanggal   = now.date()
    waktu_str = now.strftime("%H:%M:%S")

    if location and len(location) > 255:
        location = location[:252] + "..."

    log_payload = {
        "input_nip": nip,
        "input_nama": nama,
        "tipe": tipe,
        "location": location,
        "shift_id": shift_id,
        "karyawan_id": karyawan_id,
    }

    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT nip, nama FROM karyawan WHERE id = %s", (karyawan_id,))
        k_row = cur.fetchone()
        if not k_row or not k_row[0]:
            _append_audit_log("absensi", {**log_payload,
                "status": "error",
                "reason": "NIP karyawan tidak ditemukan",
            })
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
                _append_audit_log("absensi", {**log_payload,
                    "status": "error",
                    "reason": "sudah absen masuk hari ini",
                    "nip": nip_val,
                    "nama": nama_val,
                })
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
            _append_audit_log("absensi", {**log_payload,
                "status": "success",
                "action": "masuk",
                "nip": nip_val,
                "nama": nama_val,
                "message": f"Absen masuk berhasil: {nama} pukul {waktu_str}",
            })
            return {"success": True, "msg": f"Absen masuk berhasil: {nama} pukul {waktu_str}"}

        else:  # pulang
            cur.execute(
                "SELECT id FROM absensi WHERE karyawan_id = %s AND tanggal = %s "
                "AND check_out IS NULL ORDER BY id DESC LIMIT 1",
                (karyawan_id, tanggal)
            )
            row = cur.fetchone()
            if not row:
                _append_audit_log("absensi", {**log_payload,
                    "status": "error",
                    "reason": "belum absen masuk hari ini",
                    "nip": nip_val,
                    "nama": nama_val,
                })
                return {"success": False, "msg": f"{nama} belum absen masuk hari ini"}

            cur.execute(
                "UPDATE absensi "
                "SET check_out = %s, confidence_out = %s, waktu = %s, "
                "    confidence = %s, location = %s "
                "WHERE id = %s",
                (now, confidence, now, confidence, location, row[0])
            )
            conn.commit()
            _append_audit_log("absensi", {**log_payload,
                "status": "success",
                "action": "pulang",
                "nip": nip_val,
                "nama": nama_val,
                "message": f"Absen pulang berhasil: {nama} pukul {waktu_str}",
            })
            return {"success": True, "msg": f"Absen pulang berhasil: {nama} pukul {waktu_str}"}

    except Exception as e:
        conn.rollback()
        _append_audit_log("absensi", {**log_payload,
            "status": "error",
            "reason": str(e),
        })
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
                'lupa_absen': bool(r['check_in'] and not r['check_out']),
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
                   k.nip, sm.nama_shift, sm.jam_masuk, sm.jam_pulang
            FROM absensi a
            LEFT JOIN karyawan k ON k.id = a.karyawan_id
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
            nip_val = r['nip'] or str(karyawan_id)
            # hitung durasi
            if ci and co:
                diff = int((co - ci).total_seconds())
                h, rem = divmod(diff, 3600)
                m, s   = divmod(rem, 60)
                durasi = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                durasi = None
            # status: Lupa Absen jika check_in ada tapi check_out tidak
            if ci and not co:
                status_absen = 'Lupa Absen'
            elif not ci:
                status_absen = 'Tidak Hadir'
            else:
                status_absen = 'Selesai'
            result.append({
                'tanggal':    r['tanggal'].strftime('%Y-%m-%d'),
                'check_in':   ci.strftime('%H:%M') if ci else '-',
                'check_in_ts': int(ci.timestamp()) if ci else None,
                'check_out':  co.strftime('%H:%M') if co else '-',
                'durasi':     durasi,
                'nama_shift': r['nama_shift'] or '-',
                'jam_shift':  (str(r['jam_masuk'])[:5] + '–' + str(r['jam_pulang'])[:5]) if r['jam_masuk'] else '-',
                'location':   r['location'] or '-',
                'nip':        nip_val,
                'status_absen': status_absen,
            })
        return result
    except Exception as e:
        print(f'[ERROR] get_absensi_karyawan: {e}')
        return []


# ── OVERTIME REQUEST ─────────────────────────────────────────────────────────

def buat_overtime_request(karyawan_id: int, nip: str, nama: str,
                           tanggal: str, jam_mulai: str, jam_selesai: str,
                           alasan: str) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'INSERT INTO overtime_request '
            '(karyawan_id, nip, nama, tanggal, jam_mulai, jam_selesai, alasan) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (karyawan_id, nip, nama, tanggal, jam_mulai, jam_selesai, alasan)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Permintaan lembur berhasil diajukan'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def get_overtime_requests(karyawan_id: int = None) -> list:
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if karyawan_id:
            cur.execute('SELECT * FROM overtime_request WHERE karyawan_id = %s ORDER BY dibuat DESC', (karyawan_id,))
        else:
            cur.execute('SELECT * FROM overtime_request ORDER BY dibuat DESC')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        for r in rows:
            r['tanggal']          = r['tanggal'].strftime('%Y-%m-%d') if r['tanggal'] else '-'
            r['jam_mulai']        = str(r['jam_mulai'])[:5] if r['jam_mulai'] else '-'
            r['jam_selesai']      = str(r['jam_selesai'])[:5] if r['jam_selesai'] else '-'
            r['dibuat']           = r['dibuat'].strftime('%Y-%m-%d %H:%M') if r['dibuat'] else '-'
        return rows
    except Exception as e:
        print(f'[ERROR] get_overtime_requests: {e}')
        return []


def update_overtime_status(req_id: int, status: str, catatan: str = '') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'UPDATE overtime_request SET status = %s, catatan_admin = %s WHERE id = %s',
            (status, catatan, req_id)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f'Status diubah ke {status}'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


# ── HOME EARLY REQUEST ────────────────────────────────────────────────────────

def buat_home_early_request(karyawan_id: int, nip: str, nama: str,
                             tanggal: str, jam_pulang_normal: str,
                             jam_pulang_awal: str, alasan: str) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'INSERT INTO home_early_request '
            '(karyawan_id, nip, nama, tanggal, jam_pulang_normal, jam_pulang_awal, alasan) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (karyawan_id, nip, nama, tanggal, jam_pulang_normal or None, jam_pulang_awal, alasan)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Permintaan pulang awal berhasil diajukan'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def get_home_early_requests(karyawan_id: int = None) -> list:
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if karyawan_id:
            cur.execute('SELECT * FROM home_early_request WHERE karyawan_id = %s ORDER BY dibuat DESC', (karyawan_id,))
        else:
            cur.execute('SELECT * FROM home_early_request ORDER BY dibuat DESC')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        for r in rows:
            r['tanggal']           = r['tanggal'].strftime('%Y-%m-%d') if r['tanggal'] else '-'
            r['jam_pulang_normal'] = str(r['jam_pulang_normal'])[:5] if r['jam_pulang_normal'] else '-'
            r['jam_pulang_awal']   = str(r['jam_pulang_awal'])[:5] if r['jam_pulang_awal'] else '-'
            r['dibuat']            = r['dibuat'].strftime('%Y-%m-%d %H:%M') if r['dibuat'] else '-'
        return rows
    except Exception as e:
        print(f'[ERROR] get_home_early_requests: {e}')
        return []


def update_home_early_status(req_id: int, status: str, catatan: str = '') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'UPDATE home_early_request SET status = %s, catatan_admin = %s WHERE id = %s',
            (status, catatan, req_id)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f'Status diubah ke {status}'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


# ── EDIT KARYAWAN ────────────────────────────────────────────────────────────

def edit_karyawan(karyawan_id: int, nama: str, nip: str, divisi: str) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'UPDATE karyawan SET nama=%s, nip=%s, divisi=%s WHERE id=%s',
            (nama, nip or None, divisi or None, karyawan_id)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f"Karyawan '{nama}' berhasil diupdate"}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


# ── EDIT USER ─────────────────────────────────────────────────────────────────

def edit_user(user_id: int, email: str, username: str, role: str, nip: str = None,
              password: str = None) -> dict:
    from werkzeug.security import generate_password_hash
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        if password:
            cur.execute(
                'UPDATE "user" SET email=%s, username=%s, role=%s, nip=%s, password=%s WHERE id=%s',
                (email, username, role, nip or None, generate_password_hash(password), user_id)
            )
        else:
            cur.execute(
                'UPDATE "user" SET email=%s, username=%s, role=%s, nip=%s WHERE id=%s',
                (email, username, role, nip or None, user_id)
            )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f"User '{email}' berhasil diupdate"}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


# ── GAJI POKOK ────────────────────────────────────────────────────────────────

def get_gaji_list():
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('''
            SELECT g.id, g.karyawan_id, k.nama, k.nip, k.divisi, g.gaji_harian, g.diupdate
            FROM gaji_pokok g
            JOIN karyawan k ON k.id = g.karyawan_id
            ORDER BY k.nama
        ''')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        return rows
    except Exception as e:
        print(f'[ERROR] get_gaji_list: {e}')
        return []


def upsert_gaji(karyawan_id: int, gaji_harian: float) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('''
            INSERT INTO gaji_pokok (karyawan_id, gaji_harian)
            VALUES (%s, %s)
            ON CONFLICT (karyawan_id)
            DO UPDATE SET gaji_harian=EXCLUDED.gaji_harian, diupdate=NOW()
        ''', (karyawan_id, gaji_harian))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Gaji pokok berhasil disimpan'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def hapus_gaji(gaji_id: int) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('DELETE FROM gaji_pokok WHERE id=%s', (gaji_id,))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Gaji pokok berhasil dihapus'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


# ── POTONGAN CONFIG ───────────────────────────────────────────────────────────

def get_potongan_list():
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('SELECT * FROM potongan_config ORDER BY jenis')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        return rows
    except Exception as e:
        print(f'[ERROR] get_potongan_list: {e}')
        return []


def upsert_potongan(jenis: str, nominal: float, keterangan: str = '') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('''
            INSERT INTO potongan_config (jenis, nominal, keterangan)
            VALUES (%s, %s, %s)
            ON CONFLICT (jenis)
            DO UPDATE SET nominal=EXCLUDED.nominal, keterangan=EXCLUDED.keterangan, diupdate=NOW()
        ''', (jenis, nominal, keterangan))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f"Potongan '{jenis}' berhasil disimpan"}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def hapus_potongan(potongan_id: int) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('DELETE FROM potongan_config WHERE id=%s', (potongan_id,))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Potongan berhasil dihapus'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


# ── LEMBUR RATE ───────────────────────────────────────────────────────────────

def get_lembur_rate_list():
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('SELECT * FROM lembur_rate ORDER BY jabatan')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        return rows
    except Exception as e:
        print(f'[ERROR] get_lembur_rate_list: {e}')
        return []


def upsert_lembur_rate(jabatan: str, rate_per_jam: float, keterangan: str = '') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('''
            INSERT INTO lembur_rate (jabatan, rate_per_jam, keterangan)
            VALUES (%s, %s, %s)
            ON CONFLICT (jabatan)
            DO UPDATE SET rate_per_jam=EXCLUDED.rate_per_jam, keterangan=EXCLUDED.keterangan, diupdate=NOW()
        ''', (jabatan, rate_per_jam, keterangan))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': f"Rate lembur '{jabatan}' berhasil disimpan"}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def hapus_lembur_rate(rate_id: int) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('DELETE FROM lembur_rate WHERE id=%s', (rate_id,))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Rate lembur berhasil dihapus'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


# ── LAPORAN GAJI ──────────────────────────────────────────────────────────────

def get_laporan_gaji(tgl_awal: str, tgl_akhir: str) -> list:
    """
    Hitung laporan gaji per karyawan dalam rentang tanggal.
    Kolom: nama, divisi, gaji_harian, hari_masuk, izin_sakit, lupa_absen,
           telat_pulang_cepat, lembur_jam, gaji_kotor,
           potongan_izin, potongan_lupa_absen, potongan_terlambat, bonus_lembur, total_terima
    """
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Ambil semua absensi dalam range
        cur.execute('''
            SELECT a.karyawan_id, k.nama, k.nip, k.divisi,
                   a.tanggal, a.check_in, a.check_out,
                   sm.jam_masuk, sm.jam_pulang, sm.toleransi_menit
            FROM absensi a
            JOIN karyawan k ON k.id = a.karyawan_id
            LEFT JOIN shift_master sm ON sm.id = a.shift_id
            WHERE a.tanggal BETWEEN %s AND %s
            ORDER BY a.karyawan_id, a.tanggal
        ''', (tgl_awal, tgl_akhir))
        absensi_rows = cur.fetchall()

        # Ambil izin sakit dalam range
        cur.execute('''
            SELECT karyawan_id, COUNT(*) as jumlah
            FROM izin_sakit
            WHERE tanggal BETWEEN %s AND %s
            GROUP BY karyawan_id
        ''', (tgl_awal, tgl_akhir))
        izin_map = {r['karyawan_id']: int(r['jumlah']) for r in cur.fetchall()}

        # Ambil overtime approved
        cur.execute('''
            SELECT karyawan_id, tanggal, jam_mulai, jam_selesai
            FROM overtime_request
            WHERE status='approved'
              AND tanggal BETWEEN %s AND %s
        ''', (tgl_awal, tgl_akhir))
        overtime_rows = cur.fetchall()

        # Ambil gaji pokok
        cur.execute('SELECT karyawan_id, gaji_harian FROM gaji_pokok')
        gaji_map = {r['karyawan_id']: float(r['gaji_harian']) for r in cur.fetchall()}

        # Ambil potongan config
        cur.execute('SELECT jenis, nominal FROM potongan_config')
        pot_map = {r['jenis']: float(r['nominal']) for r in cur.fetchall()}
        pot_terlambat   = pot_map.get('terlambat', 0)
        pot_lupa_absen  = pot_map.get('lupa_absen', 0)

        # Ambil lembur rate
        cur.execute('SELECT jabatan, rate_per_jam FROM lembur_rate')
        lembur_map = {r['jabatan'].lower(): float(r['rate_per_jam']) for r in cur.fetchall()}

        cur.close(); conn.close()

        # Kelompokkan overtime per karyawan
        ot_per_kar = {}
        for ot in overtime_rows:
            kid = ot['karyawan_id']
            if kid not in ot_per_kar:
                ot_per_kar[kid] = 0.0
            try:
                from datetime import datetime as dt
                t1 = dt.strptime(str(ot['jam_mulai'])[:5], '%H:%M')
                t2 = dt.strptime(str(ot['jam_selesai'])[:5], '%H:%M')
                jam = max(0, (t2 - t1).total_seconds() / 3600)
                ot_per_kar[kid] += jam
            except Exception:
                pass

        # Kelompokkan absensi per karyawan
        kar_map = {}
        for r in absensi_rows:
            kid = r['karyawan_id']
            if kid not in kar_map:
                kar_map[kid] = {
                    'nama': r['nama'], 'nip': r['nip'], 'divisi': r['divisi'] or '-',
                    'masuk': 0, 'lupa_absen': 0, 'terlambat': 0,
                }
            ci = r['check_in']
            co = r['check_out']
            if ci:
                kar_map[kid]['masuk'] += 1
                if not co:
                    kar_map[kid]['lupa_absen'] += 1
                # cek terlambat
                if r['jam_masuk'] and r['toleransi_menit'] is not None:
                    from datetime import datetime as dt, timedelta
                    batas = (dt.combine(ci.date(), r['jam_masuk'])
                             + timedelta(minutes=int(r['toleransi_menit'])))
                    if ci > batas:
                        kar_map[kid]['terlambat'] += 1
            else:
                kar_map[kid]['lupa_absen'] += 1

        result = []
        for i, (kid, d) in enumerate(sorted(kar_map.items(), key=lambda x: x[1]['nama'])):
            gaji_harian  = gaji_map.get(kid, 0)
            lembur_jam   = round(ot_per_kar.get(kid, 0), 2)
            divisi_lower = d['divisi'].lower()
            rate_lembur  = lembur_map.get(divisi_lower, list(lembur_map.values())[0] if lembur_map else 0)

            izin_sakit_count = izin_map.get(kid, 0)
            gaji_kotor        = gaji_harian * d['masuk']
            p_lupa_absen      = pot_lupa_absen * d['lupa_absen']
            p_terlambat       = pot_terlambat * d['terlambat']
            bonus_lembur      = rate_lembur * lembur_jam
            total_terima      = gaji_kotor - p_lupa_absen - p_terlambat + bonus_lembur

            result.append({
                'no':               i + 1,
                'karyawan_id':      kid,
                'nama':             d['nama'],
                'nip':              d['nip'] or '-',
                'divisi':           d['divisi'],
                'gaji_harian':      gaji_harian,
                'masuk':            d['masuk'],
                'izin_sakit':       izin_sakit_count,
                'lupa_absen':       d['lupa_absen'],
                'terlambat':        d['terlambat'],
                'lembur_jam':       lembur_jam,
                'gaji_kotor':       gaji_kotor,
                'potongan_lupa':    p_lupa_absen,
                'potongan_terlambat': p_terlambat,
                'bonus_lembur':     bonus_lembur,
                'total_terima':     total_terima,
            })
        return result
    except Exception as e:
        print(f'[ERROR] get_laporan_gaji: {e}')
        return []


# ── IZIN SAKIT ───────────────────────────────────────────────────────────────

def get_izin_sakit_list(karyawan_id: int = None, tgl_awal: str = None, tgl_akhir: str = None) -> list:
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if karyawan_id:
            cur.execute('''
                SELECT iz.id, iz.karyawan_id, k.nama, k.nip, k.divisi,
                       iz.tanggal, iz.jenis, iz.keterangan, iz.dibuat
                FROM izin_sakit iz JOIN karyawan k ON k.id = iz.karyawan_id
                WHERE iz.karyawan_id = %s ORDER BY iz.tanggal DESC
            ''', (karyawan_id,))
        elif tgl_awal and tgl_akhir:
            cur.execute('''
                SELECT iz.id, iz.karyawan_id, k.nama, k.nip, k.divisi,
                       iz.tanggal, iz.jenis, iz.keterangan, iz.dibuat
                FROM izin_sakit iz JOIN karyawan k ON k.id = iz.karyawan_id
                WHERE iz.tanggal BETWEEN %s AND %s ORDER BY iz.tanggal DESC
            ''', (tgl_awal, tgl_akhir))
        else:
            cur.execute('''
                SELECT iz.id, iz.karyawan_id, k.nama, k.nip, k.divisi,
                       iz.tanggal, iz.jenis, iz.keterangan, iz.dibuat
                FROM izin_sakit iz JOIN karyawan k ON k.id = iz.karyawan_id
                ORDER BY iz.tanggal DESC
            ''')
        rows = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        for r in rows:
            r['tanggal'] = r['tanggal'].strftime('%Y-%m-%d') if r['tanggal'] else '-'
            r['dibuat']  = r['dibuat'].strftime('%Y-%m-%d %H:%M') if r['dibuat'] else '-'
        return rows
    except Exception as e:
        print(f'[ERROR] get_izin_sakit_list: {e}')
        return []


def tambah_izin_sakit(karyawan_id: int, tanggal: str, jenis: str, keterangan: str = '') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'INSERT INTO izin_sakit (karyawan_id, tanggal, jenis, keterangan) VALUES (%s, %s, %s, %s)',
            (karyawan_id, tanggal, jenis, keterangan or None)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Data izin/sakit berhasil ditambahkan'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def edit_izin_sakit(izin_id: int, tanggal: str, jenis: str, keterangan: str = '') -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            'UPDATE izin_sakit SET tanggal=%s, jenis=%s, keterangan=%s WHERE id=%s',
            (tanggal, jenis, keterangan or None, izin_id)
        )
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Data izin/sakit berhasil diupdate'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


def hapus_izin_sakit(izin_id: int) -> dict:
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('DELETE FROM izin_sakit WHERE id=%s', (izin_id,))
        conn.commit(); cur.close(); conn.close()
        return {'success': True, 'msg': 'Data izin/sakit berhasil dihapus'}
    except Exception as e:
        return {'success': False, 'msg': str(e)}


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