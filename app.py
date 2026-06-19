"""
app.py  –  Flask Absensi Face Recognition
Database: PostgreSQL langsung (user, karyawan, absensi)
Jalankan: python app.py
"""

import os, base64, io
from datetime import date, timedelta, datetime
from functools import wraps

import numpy as np
import cv2
from PIL import Image
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, send_file)
from werkzeug.security import check_password_hash
from dotenv import load_dotenv
import pandas as pd

from face_engine import (
    FaceEngine, init_db, test_connection, seed_admin,
    catat_absensi, get_absensi_hari_ini, get_absensi_range,
    get_karyawan_list, get_user_by_email, get_user_by_id,
    tambah_user, hapus_user, get_users,
    get_shift_list, tambah_shift, hapus_shift, edit_shift,
    get_shift_karyawan, assign_shift, hapus_shift_karyawan,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-ganti-ini")

engine = FaceEngine()


# ═════════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Akses ditolak. Halaman ini khusus admin.", "danger")
            return redirect(url_for("absensi"))
        return f(*args, **kwargs)
    return decorated


def decode_b64_image(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def bgr_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ═════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    result      = test_connection()
    db_ok       = result["database"]["status"] == "ok"
    model_ok    = result["face_model"]["status"] == "ok"
    status_code = 200 if (db_ok and model_ok) else 503
    return jsonify({
        "status":    "ok" if status_code == 200 else "degraded",
        "timestamp": datetime.now().isoformat(),
        "checks":    result,
    }), status_code


# ═════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("admin_dashboard") if session.get("role") == "admin" else url_for("absensi"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_user_by_email(email)
        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["email"]    = user["email"]
            session["username"] = user.get("username")
            session["role"]     = user["role"]
            session["nip"]      = user.get("nip")
            flash(f"Selamat datang, {user.get('username') or user['email']}!", "success")
            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("absensi"))
        flash("Email atau password salah.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ═════════════════════════════════════════════════════════════════════════════
#  ABSENSI
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/absensi")
@login_required
def absensi():
    return render_template("absensi.html", absensi=get_absensi_hari_ini())


@app.route("/api/absen", methods=["POST"])
@login_required
def api_absen():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Field image wajib ada"}), 400

    tipe = data.get("tipe", "masuk")
    if tipe not in ("masuk", "pulang"):
        return jsonify({"error": "Tipe harus masuk atau pulang"}), 400

    # Ambil lokasi dari request (hasil reverse-geocode di browser)
    location = data.get("location") or None

    try:
        img_bgr = decode_b64_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"Gagal decode gambar: {e}"}), 400

    result       = engine.recognize(img_bgr)
    out_bgr      = engine.draw_result(img_bgr, result)
    face_b64     = bgr_to_b64(out_bgr)
    absen_result = {"success": False, "msg": "Wajah tidak dikenali"}

    if result["recognized"]:
        absen_result = catat_absensi(
            result["nip"], result["nama"], tipe,
            result["confidence"], location
        )

    return jsonify({
        "recognized":   result["recognized"],
        "nama":         result.get("nama"),
        "nip":          result.get("nip"),
        "confidence":   result.get("confidence", 0),
        "absen_result": absen_result,
        "face_image":   face_b64,
        "location":     location,
    })


@app.route("/api/absensi-hari-ini")
@login_required
def api_absensi_hari_ini():
    return jsonify(get_absensi_hari_ini())


# ═════════════════════════════════════════════════════════════════════════════
#  ADMIN
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/admin")
@admin_required
def admin_dashboard():
    absensi_hari_ini = get_absensi_hari_ini()
    karyawan         = get_karyawan_list()
    n_masuk  = sum(1 for a in absensi_hari_ini if a["tipe"] == "masuk")
    n_pulang = sum(1 for a in absensi_hari_ini if a["tipe"] == "pulang")
    return render_template("admin/admin_dashboard.html",
        absensi=absensi_hari_ini, karyawan=karyawan,
        n_masuk=n_masuk, n_pulang=n_pulang, total_karyawan=len(karyawan))


@app.route("/admin/karyawan")
@admin_required
def admin_karyawan():
    return render_template("admin/admin_karyawan.html", karyawan=get_karyawan_list())


@app.route("/admin/karyawan/daftar", methods=["GET", "POST"])
@admin_required
def admin_daftar_karyawan():
    if request.method == "POST":
        nama   = request.form.get("nama", "").strip()
        nip    = request.form.get("nip", "").strip()
        divisi = request.form.get("divisi", "").strip()
        photos = request.files.getlist("photos")
        if not nama or not photos:
            flash("Nama dan foto wajib diisi.", "danger")
            return redirect(request.url)
        try:
            images = [np.array(Image.open(f).convert("RGB")) for f in photos[:5]]
        except Exception as e:
            flash(f"Gagal membaca foto: {e}", "danger")
            return redirect(request.url)
        result = engine.register(nip, nama, divisi, images)
        flash(result["msg"], "success" if result["success"] else "danger")
        return redirect(url_for("admin_karyawan"))
    return render_template("admin/admin_daftar.html")


@app.route("/admin/karyawan/hapus/<nip>", methods=["POST"])
@admin_required
def admin_hapus_karyawan(nip):
    ok = engine.delete_karyawan(nip)
    flash("Karyawan berhasil dihapus." if ok else "Gagal menghapus.", "success" if ok else "danger")
    return redirect(url_for("admin_karyawan"))


@app.route("/admin/laporan")
@admin_required
def admin_laporan():
    tgl_awal  = request.args.get("dari",   (date.today() - timedelta(days=7)).isoformat())
    tgl_akhir = request.args.get("sampai", date.today().isoformat())
    return render_template("admin/admin_laporan.html",
        data=get_absensi_range(tgl_awal, tgl_akhir),
        tgl_awal=tgl_awal, tgl_akhir=tgl_akhir)


@app.route("/admin/laporan/export")
@admin_required
def admin_export():
    tgl_awal  = request.args.get("dari",   (date.today() - timedelta(days=7)).isoformat())
    tgl_akhir = request.args.get("sampai", date.today().isoformat())

    def _hitung_durasi(ci, co):
        """Hitung durasi kerja dari string HH:MM. Return '7j 30m' atau '-'."""
        if not ci or not co or ci == "-" or co == "-":
            return "-"
        try:
            from datetime import datetime as dt
            fmt = "%H:%M"
            t_in  = dt.strptime(ci, fmt)
            t_out = dt.strptime(co, fmt)
            # Jika pulang < masuk = melewati tengah malam
            if t_out < t_in:
                t_out += timedelta(days=1)
            total_menit = int((t_out - t_in).total_seconds() / 60)
            jam, menit  = divmod(total_menit, 60)
            return f"{jam}j {menit}m"
        except Exception:
            return "-"
    rows = get_absensi_range(tgl_awal, tgl_akhir)

    # ── Sheet 1: Detail absensi ──────────────────────────────────
    df = pd.DataFrame([
        {
            "No":               i + 1,
            "Tanggal":          r["tanggal"],
            "NIP":              r["nip"],
            "Nama":             r["nama"],
            "Divisi":           r["divisi"],
            "Shift":            r["nama_shift"],
            "Jam Shift Masuk":  r["jam_shift_masuk"],
            "Jam Shift Pulang": r["jam_shift_pulang"],
            "Check In":         r["check_in"],
            "Check Out":        r["check_out"],
            "Total Kerja":      _hitung_durasi(r["check_in"], r["check_out"]),
            "Status":           r["status"],
            "Lokasi":           r["location"],
        }
        for i, r in enumerate(rows)
    ])

    # ── Sheet 2: Rekap per karyawan ──────────────────────────────
    rekap = {}
    for r in rows:
        key = (r["nip"], r["nama"], r["divisi"])
        if key not in rekap:
            rekap[key] = {"total_hadir": 0, "total_checkin": 0,
                          "total_checkout": 0, "terlambat": 0,
                          "tepat_waktu": 0, "total_menit": 0}
        rekap[key]["total_hadir"]    += 1
        rekap[key]["total_checkin"]  += 1 if r["check_in"]  != "-" else 0
        rekap[key]["total_checkout"] += 1 if r["check_out"] != "-" else 0
        if r["status"] == "Tepat Waktu":
            rekap[key]["tepat_waktu"] += 1
        elif r["status"] != "-":
            rekap[key]["terlambat"] += 1
        # Akumulasi total menit kerja
        if r["check_in"] != "-" and r["check_out"] != "-":
            try:
                from datetime import datetime as dt
                t_in  = dt.strptime(r["check_in"],  "%H:%M")
                t_out = dt.strptime(r["check_out"], "%H:%M")
                if t_out < t_in:
                    t_out += timedelta(days=1)
                rekap[key]["total_menit"] += int((t_out - t_in).total_seconds() / 60)
            except Exception:
                pass

    def _menit_ke_jam(total_menit):
        if total_menit == 0:
            return "-"
        jam, menit = divmod(total_menit, 60)
        return f"{jam}j {menit}m"

    df_rekap = pd.DataFrame([
        {
            "No":              i + 1,
            "NIP":             k[0],
            "Nama":            k[1],
            "Divisi":          k[2],
            "Total Hadir":     v["total_hadir"],
            "Total Check In":  v["total_checkin"],
            "Total Check Out": v["total_checkout"],
            "Tepat Waktu":     v["tepat_waktu"],
            "Terlambat":       v["terlambat"],
        }
        for i, (k, v) in enumerate(sorted(rekap.items(), key=lambda x: x[0][1]))
    ])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Detail Absensi")
        df_rekap.to_excel(writer, index=False, sheet_name="Rekap Karyawan")
        # Auto-width kedua sheet
        for sheet_name in ["Detail Absensi", "Rekap Karyawan"]:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"absensi_{tgl_awal}_{tgl_akhir}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/admin/users")
@admin_required
def admin_users():
    return render_template("admin/admin_users.html", users=get_users())


@app.route("/admin/users/tambah", methods=["POST"])
@admin_required
def admin_tambah_user():
    email    = request.form.get("email", "").strip().lower()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role     = request.form.get("role", "karyawan")
    nip      = request.form.get("nip", "").strip() or None
    if not email or not password:
        flash("Email dan password wajib diisi.", "danger")
        return redirect(url_for("admin_users"))
    result = tambah_user(email, username, password, role, nip)
    flash(result["msg"], "success" if result["success"] else "danger")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/hapus/<int:user_id>", methods=["POST"])
@admin_required
def admin_hapus_user(user_id):
    if user_id == session.get("user_id"):
        flash("Tidak bisa hapus akun sendiri.", "danger")
        return redirect(url_for("admin_users"))
    hapus_user(user_id)
    flash("User berhasil dihapus.", "success")
    return redirect(url_for("admin_users"))


# ═════════════════════════════════════════════════════════════════════════════
#  SHIFT
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/admin/shift")
@admin_required
def admin_shift():
    return render_template("admin/admin_shift.html",
        shifts=get_shift_list(),
        shift_karyawan=get_shift_karyawan(),
        karyawan=get_karyawan_list(),
        today=date.today().isoformat())


@app.route("/admin/shift/tambah", methods=["POST"])
@admin_required
def admin_tambah_shift():
    r = tambah_shift(
        nama_shift           = request.form.get("nama_shift", "").strip(),
        jam_masuk            = request.form.get("jam_masuk"),
        jam_pulang           = request.form.get("jam_pulang"),
        toleransi_menit      = request.form.get("toleransi_menit", 15),
        melewati_tengah_malam= request.form.get("melewati_tengah_malam") == "1",
        keterangan           = request.form.get("keterangan", "").strip(),
    )
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_shift"))


@app.route("/admin/shift/edit/<int:shift_id>", methods=["POST"])
@admin_required
def admin_edit_shift(shift_id):
    r = edit_shift(
        shift_id             = shift_id,
        nama_shift           = request.form.get("nama_shift", "").strip(),
        jam_masuk            = request.form.get("jam_masuk"),
        jam_pulang           = request.form.get("jam_pulang"),
        toleransi_menit      = request.form.get("toleransi_menit", 15),
        melewati_tengah_malam= request.form.get("melewati_tengah_malam") == "1",
        keterangan           = request.form.get("keterangan", "").strip(),
    )
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_shift"))


@app.route("/admin/shift/hapus/<int:shift_id>", methods=["POST"])
@admin_required
def admin_hapus_shift(shift_id):
    r = hapus_shift(shift_id)
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_shift"))


@app.route("/admin/shift/assign", methods=["POST"])
@admin_required
def admin_assign_shift():
    r = assign_shift(
        karyawan_id    = int(request.form.get("karyawan_id")),
        shift_id       = int(request.form.get("shift_id")),
        berlaku_dari   = request.form.get("berlaku_dari"),
        berlaku_sampai = request.form.get("berlaku_sampai") or None,
    )
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_shift"))


@app.route("/admin/shift/assign/hapus/<int:sk_id>", methods=["POST"])
@admin_required
def admin_hapus_assign_shift(sk_id):
    r = hapus_shift_karyawan(sk_id)
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_shift"))


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    seed_admin()
    print("=" * 55)
    print("  Sistem Absensi Face Recognition")
    print("  http://localhost:5000")
    print("  Health: http://localhost:5000/health")
    print("  Login: admin@local / admin123")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000,
            debug=os.getenv("FLASK_ENV") == "development")