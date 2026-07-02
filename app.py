import os, base64, io, json, mimetypes
from datetime import date, timedelta, datetime
from functools import wraps
import socket
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, send_file, abort)
from werkzeug.security import check_password_hash
from dotenv import load_dotenv
import pandas as pd

from face_engine import (
    FaceEngine, init_db, test_connection, seed_admin,
    catat_absensi, get_absensi_hari_ini, get_absensi_range,
    get_karyawan_list, get_user_by_email, get_user_by_id,
    tambah_user, hapus_user, get_users,
    get_shift_list, tambah_shift, hapus_shift, edit_shift,
    get_absensi_karyawan,
    buat_overtime_request, get_overtime_requests, update_overtime_status,
    buat_home_early_request, get_home_early_requests, update_home_early_status,
    get_shift_aktif_karyawan, get_db_connection,
    _append_audit_log,
    edit_karyawan, edit_user,
    get_gaji_list, upsert_gaji, hapus_gaji,
    get_potongan_list, upsert_potongan, hapus_potongan,
    get_lembur_rate_list, upsert_lembur_rate, hapus_lembur_rate,
    get_laporan_gaji,
    get_izin_sakit_list, tambah_izin_sakit, edit_izin_sakit, hapus_izin_sakit,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-ganti-ini")

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    safe_root = Path('templates') / 'assets'
    asset_path = safe_root / filename
    if not asset_path.exists() or not asset_path.is_file() or safe_root not in asset_path.parents:
        abort(404)
    mime_type, _ = mimetypes.guess_type(str(asset_path))
    return send_file(str(asset_path), mimetype=mime_type or 'application/octet-stream')

engine = FaceEngine()

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
            try:
                hostname = socket.gethostname()
                ipv4_address = socket.gethostbyname(hostname)
            except:
                ipv4_address = "Tidak diketahui"
            session["user_id"]  = user["id"]
            session["email"]    = user["email"]
            session["username"] = user.get("username")
            session["role"]     = user["role"]
            session["nip"]      = user.get("nip")
            session["is_login"] = True
            session['address'] = ipv4_address
            flash(f"Selamat datang, {user.get('username') or user['email']}!", "success")
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE "user"
                SET
                    is_login = %s,
                    address = %s
                WHERE id = %s
            """, (True, ipv4_address, user["id"]))
        
            conn.commit()
            cur.close()
            conn.close()
            _append_audit_log("user_login", {
                "user_id": user["id"],
                "email": user["email"],
                "role": user["role"],
                "ip_address": ipv4_address,
                "success": True
            })
            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("absensi"))
        flash("Email atau password salah.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():

    user_id = session.get("user_id")

    if user_id:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(""" 
                UPDATE "user" 
                SET 
                    is_login = %s,
                    address = %s
                WHERE id = %s
        """, (False, "None", user_id))

        conn.commit()
        cur.close()
        conn.close()
    _append_audit_log("user_logout", {
        "user_id": user_id,
        "ip_address": session.get('address'),
        "success": True
    })
    session.clear()
    return redirect(url_for("login"))

@app.route("/profile")
def profile():
    return {
        "username": session.get("username"),
        "email": session.get("email"),
        "role": session.get("role"),
        "is_login": session.get("is_login"),
        "ipv4_address": session.get("address")
    }

@app.route("/absensi")
@login_required
def absensi():
    return render_template("absensi.html", absensi=get_absensi_hari_ini(),
                           shifts=get_shift_list())

@app.route("/api/absen", methods=["POST"])
@login_required
def api_absen():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Field image wajib ada"}), 400

    tipe = data.get("tipe", "masuk")
    if tipe not in ("masuk", "pulang", "__preview__"):
        return jsonify({"error": "Tipe harus masuk atau pulang"}), 400

    location = data.get("location") or None
    shift_id  = data.get("shift_id") or None
    if shift_id:
        try: shift_id = int(shift_id)
        except: shift_id = None

    try:
        img_bgr = decode_b64_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"Gagal decode gambar: {e}"}), 400

    result       = engine.recognize(img_bgr)
    out_bgr      = engine.draw_result(img_bgr, result)
    face_b64     = bgr_to_b64(out_bgr)
    absen_result = {"success": False, "msg": "Wajah tidak dikenali"}

    if result["recognized"] and tipe != "__preview__":
        absen_result = catat_absensi(
            result["nip"], result["nama"], tipe,
            result["confidence"], location, shift_id
        )
    elif result["recognized"]:
        absen_result = {"success": True, "msg": "Wajah dikenali"}

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


@app.route("/riwayat")
@login_required
def riwayat_absensi():
    nip = session.get("nip")
    karyawan_id = None
    if nip:
        # nip di session bisa berupa karyawan_id (integer string) atau NIP string
        # cari karyawan_id dari DB berdasarkan NIP
        from face_engine import get_db_connection
        import psycopg2.extras
        try:
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute("SELECT id FROM karyawan WHERE nip = %s LIMIT 1", (nip,))
            row = cur.fetchone()
            if not row:
                # fallback: nip mungkin langsung karyawan_id integer
                try: karyawan_id = int(nip)
                except: pass
            else:
                karyawan_id = row[0]
            cur.close(); conn.close()
        except Exception:
            pass
    riwayat = get_absensi_karyawan(karyawan_id) if karyawan_id else []
    return render_template("riwayat.html", riwayat=riwayat, karyawan_id=karyawan_id)


@app.route("/registered_faces/<int:karyawan_id>/<filename>")
@login_required
def serve_face_photo(karyawan_id, filename):
    """Serve foto referensi wajah karyawan: /registered_faces/{id}/{file}"""
    from pathlib import Path
    safe = Path(filename).name  # cegah path traversal
    path = Path("registered_faces") / str(karyawan_id) / safe
    if not path.exists() or not path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/karyawan/<int:karyawan_id>/photos")
@login_required
def api_karyawan_photos(karyawan_id):
    """Return list URL foto untuk karyawan tertentu."""
    from pathlib import Path
    folder = Path("registered_faces") / str(karyawan_id)
    if not folder.exists():
        return jsonify([])
    urls = [
        url_for("serve_face_photo", karyawan_id=karyawan_id, filename=f.name)
        for f in sorted(folder.iterdir())
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    return jsonify(urls)


@app.route("/admin")
@admin_required
def admin_dashboard():
    absensi_hari_ini = get_absensi_hari_ini()
    karyawan         = get_karyawan_list()
    n_masuk  = sum(1 for a in absensi_hari_ini if a["tipe"] == "masuk")
    n_pulang = sum(1 for a in absensi_hari_ini if a["tipe"] == "pulang")

    # Data 7 hari terakhir untuk grafik
    import json as _json
    labels_7 = []
    data_masuk_7 = []
    data_izin_7  = []
    for i in range(6, -1, -1):
        d = date.today() - timedelta(days=i)
        rows = get_absensi_range(d.isoformat(), d.isoformat())
        labels_7.append(d.strftime('%d/%m'))
        data_masuk_7.append(len(rows))
        izin = get_izin_sakit_list(tgl_awal=d.isoformat(), tgl_akhir=d.isoformat())
        data_izin_7.append(len(izin))

    # Rekap per divisi hari ini
    divisi_count = {}
    seen_nips = set()
    for a in absensi_hari_ini:
        if a['tipe'] == 'masuk' and a['nip'] not in seen_nips:
            seen_nips.add(a['nip'])
            div = next((k['divisi'] for k in karyawan if k['nip'] == a['nip']), '-')
            divisi_count[div] = divisi_count.get(div, 0) + 1

    # Pending requests
    ot_pending = sum(1 for r in get_overtime_requests() if r['status'] == 'pending')
    he_pending = sum(1 for r in get_home_early_requests() if r['status'] == 'pending')

    return render_template("admin/admin_dashboard.html",
        absensi=absensi_hari_ini, karyawan=karyawan,
        n_masuk=n_masuk, n_pulang=n_pulang, total_karyawan=len(karyawan),
        ot_pending=ot_pending, he_pending=he_pending,
        chart_labels=_json.dumps(labels_7),
        chart_masuk=_json.dumps(data_masuk_7),
        chart_izin=_json.dumps(data_izin_7),
        chart_divisi_labels=_json.dumps(list(divisi_count.keys())),
        chart_divisi_data=_json.dumps(list(divisi_count.values())),
    )

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
        result = engine.register(nip, nama, divisi, images,
                                   base_url=request.host_url.rstrip("/"))
        _append_audit_log("admin_add_karyawan", {
            "nip": nip, "nama": nama, "divisi": divisi,
            "success": result["success"], "message": result["msg"]
        })
        flash(result["msg"], "success" if result["success"] else "danger")
        return redirect(url_for("admin_karyawan"))
    return render_template("admin/admin_daftar.html")


@app.route("/admin/karyawan/edit/<int:karyawan_id>", methods=["POST"])
@admin_required
def admin_edit_karyawan(karyawan_id):
    nama   = request.form.get("nama", "").strip()
    nip    = request.form.get("nip", "").strip()
    divisi = request.form.get("divisi", "").strip()
    if not nama:
        flash("Nama wajib diisi.", "danger")
        return redirect(url_for("admin_karyawan"))
    r = edit_karyawan(karyawan_id, nama, nip, divisi)
    _append_audit_log("admin_edit_karyawan", {
        "karyawan_id": karyawan_id, "nama": nama, "nip": nip, "divisi": divisi,
        "success": r["success"], "message": r["msg"]
    })
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_karyawan"))


@app.route("/admin/karyawan/hapus/<int:karyawan_id>", methods=["POST"])
@admin_required
def admin_hapus_karyawan(karyawan_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, nip, nama, divisi FROM karyawan WHERE id = %s", (karyawan_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        flash("Karyawan tidak ditemukan.", "danger")
        return redirect(url_for("admin_karyawan"))

    _, nip, nama, divisi = row
    ok = engine.delete_karyawan(str(karyawan_id))
    _append_audit_log("admin_delete_karyawan", {
        "karyawan_id": karyawan_id,
        "nip": nip,
        "nama": nama,
        "divisi": divisi,
        "deleted_by": session.get("user_id"),
        "success": ok,
    })
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

@app.route("/admin/logs")
@admin_required
def admin_logs():
    from pathlib import Path
    log_path = Path("logs") / "system.log"
    entries = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        entries.append(data)
                    except Exception:
                        continue
        except Exception:
            entries = []
    entries = list(reversed(entries[-100:]))
    return render_template("admin/admin_logs.html", entries=entries)

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
            if t_out < t_in:
                t_out += timedelta(days=1)
            total_menit = int((t_out - t_in).total_seconds() / 60)
            jam, menit  = divmod(total_menit, 60)
            return f"{jam}j {menit}m"
        except Exception:
            return "-"

    rows = get_absensi_range(tgl_awal, tgl_akhir)

    df = pd.DataFrame([
        {
            "No":               i + 1,
            "Tanggal":          r["tanggal"],
            "NIP":              r["nip"],
            "Nama":             r["nama"],
            "Divisi":           r["divisi"],
            "Shift":            r["nama_shift"],
            "Jam Shift": f'{r.get("jam_shift_masuk", "-")} - {r.get("jam_shift_pulang", "-")}',
            "Check In":         r["check_in"],
            "Check Out":        r["check_out"],
            "Time":             r.get("durasi_fmt", "-"),
            "Total Kerja":      _hitung_durasi(r["check_in"], r["check_out"]),
            "Status":           r["status"],
            "Lokasi":           r["location"],
        }
        for i, r in enumerate(rows)
    ])

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
    _append_audit_log("admin_export_report", {
        "export_from": tgl_awal,
        "export_to": tgl_akhir,
        "requested_by": session.get("user_id"),
        "success": True
    })
    return send_file(buf, as_attachment=True,
                     download_name=f"absensi_{tgl_awal}_{tgl_akhir}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.route("/admin/logs/download")
@admin_required
def admin_download_logs():
    from pathlib import Path
    log_path = Path("logs") / "system.log"
    if not log_path.exists():
        flash("File log belum tersedia.", "danger")
        return redirect(url_for("admin_laporan"))
    _append_audit_log("admin_download_logs", {
        "requested_by": session.get("user_id"),
        "success": True
    })
    return send_file(str(log_path), as_attachment=True,
                     download_name="system.log",
                     mimetype="text/plain")

@app.route("/admin/users")
@admin_required
def admin_users():
    return render_template("admin/admin_users.html", users=get_users(), karyawan=get_karyawan_list())

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
    _append_audit_log("admin_add_user", {
        "email": email,
        "username": username,
        "role": role,
        "nip": nip,
        "success": result["success"],
        "message": result["msg"]
    })
    flash(result["msg"], "success" if result["success"] else "danger")
    return redirect(url_for("admin_users"))

@app.route("/admin/users/edit/<int:user_id>", methods=["POST"])
@admin_required
def admin_edit_user(user_id):
    email    = request.form.get("email", "").strip().lower()
    username = request.form.get("username", "").strip()
    role     = request.form.get("role", "karyawan")
    nip      = request.form.get("nip", "").strip() or None
    password = request.form.get("password", "").strip() or None
    if not email:
        flash("Email wajib diisi.", "danger")
        return redirect(url_for("admin_users"))
    r = edit_user(user_id, email, username, role, nip, password)
    _append_audit_log("admin_edit_user", {
        "user_id": user_id, "email": email, "role": role,
        "success": r["success"], "message": r["msg"]
    })
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/hapus/<int:user_id>", methods=["POST"])
@admin_required
def admin_hapus_user(user_id):
    if user_id == session.get("user_id"):
        flash("Tidak bisa hapus akun sendiri.", "danger")
        return redirect(url_for("admin_users"))
    ok = hapus_user(user_id)
    _append_audit_log("admin_delete_user", {
        "user_id": user_id,
        "deleted_by": session.get("user_id"),
        "success": ok,
    })
    flash("User berhasil dihapus." if ok else "Gagal menghapus user.", "success" if ok else "danger")
    return redirect(url_for("admin_users"))

@app.route("/admin/shift")
@admin_required
def admin_shift():
    return render_template("admin/admin_shift.html", shifts=get_shift_list())

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
    _append_audit_log("admin_add_shift", {
        "nama_shift": request.form.get("nama_shift", "").strip(),
        "jam_masuk": request.form.get("jam_masuk"),
        "jam_pulang": request.form.get("jam_pulang"),
        "success": r["success"],
        "message": r["msg"]
    })
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
    _append_audit_log("admin_edit_shift", {
        "shift_id": shift_id,
        "nama_shift": request.form.get("nama_shift", "").strip(),
        "success": r["success"],
        "message": r["msg"]
    })
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_shift"))

@app.route("/admin/shift/hapus/<int:shift_id>", methods=["POST"])
@admin_required
def admin_hapus_shift(shift_id):
    r = hapus_shift(shift_id)
    _append_audit_log("admin_delete_shift", {
        "shift_id": shift_id,
        "success": r["success"],
        "message": r["msg"]
    })
    flash(r["msg"], "success" if r["success"] else "danger")
    return redirect(url_for("admin_shift"))

# ── helper: resolve karyawan_id dari session ──────────────────────────────
def _get_karyawan_id_from_session():
    nip = session.get('nip')
    if not nip:
        return None, None, None
    from face_engine import get_db_connection
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute('SELECT id, nip, nama FROM karyawan WHERE nip = %s LIMIT 1', (nip,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if row:
            return row[0], row[1], row[2]
        try:
            kid = int(nip)
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute('SELECT id, nip, nama FROM karyawan WHERE id = %s LIMIT 1', (kid,))
            row = cur.fetchone()
            cur.close(); conn.close()
            return (row[0], row[1], row[2]) if row else (None, None, None)
        except Exception:
            return None, None, None
    except Exception:
        return None, None, None


# ── Overtime Request (User) ───────────────────────────────────────────────
@app.route('/overtime', methods=['GET', 'POST'])
@login_required
def overtime():
    karyawan_id, nip, nama = _get_karyawan_id_from_session()
    shift = get_shift_aktif_karyawan(karyawan_id) if karyawan_id else None
    if request.method == 'POST':
        if not karyawan_id:
            flash('NIP belum diisi di akun. Hubungi admin.', 'danger')
            return redirect(request.url)
        tanggal     = request.form.get('tanggal', '')
        jam_mulai   = request.form.get('jam_mulai', '')
        jam_selesai = request.form.get('jam_selesai', '')
        alasan      = request.form.get('alasan', '').strip()
        if not (tanggal and jam_mulai and jam_selesai and alasan):
            flash('Semua field wajib diisi.', 'danger')
            return redirect(request.url)
        r = buat_overtime_request(karyawan_id, nip, nama, tanggal, jam_mulai, jam_selesai, alasan)
        flash(r['msg'], 'success' if r['success'] else 'danger')
        return redirect(url_for('overtime'))
    riwayat = get_overtime_requests(karyawan_id) if karyawan_id else []
    return render_template('overtime.html', riwayat=riwayat, shift=shift,
                           today=date.today().isoformat())


# ── Home Early Request (User) ─────────────────────────────────────────────
@app.route('/home-early', methods=['GET', 'POST'])
@login_required
def home_early():
    karyawan_id, nip, nama = _get_karyawan_id_from_session()
    shift = get_shift_aktif_karyawan(karyawan_id) if karyawan_id else None
    if request.method == 'POST':
        if not karyawan_id:
            flash('NIP belum diisi di akun. Hubungi admin.', 'danger')
            return redirect(request.url)
        tanggal            = request.form.get('tanggal', '')
        jam_pulang_normal  = request.form.get('jam_pulang_normal', '').strip() or None
        jam_pulang_awal    = request.form.get('jam_pulang_awal', '')
        alasan             = request.form.get('alasan', '').strip()
        if not (tanggal and jam_pulang_awal and alasan):
            flash('Semua field wajib diisi.', 'danger')
            return redirect(request.url)
        r = buat_home_early_request(karyawan_id, nip, nama, tanggal,
                                     jam_pulang_normal, jam_pulang_awal, alasan)
        flash(r['msg'], 'success' if r['success'] else 'danger')
        return redirect(url_for('home_early'))
    riwayat = get_home_early_requests(karyawan_id) if karyawan_id else []
    return render_template('home_early.html', riwayat=riwayat, shift=shift,
                           today=date.today().isoformat())


# ── Admin Overtime Dashboard ──────────────────────────────────────────────
@app.route('/admin/overtime')
@admin_required
def admin_overtime():
    status = request.args.get('status', '')
    data   = get_overtime_requests()
    if status:
        data = [d for d in data if d['status'] == status]
    return render_template('admin/admin_overtime.html', data=data, status=status)


@app.route('/admin/overtime/<int:req_id>/update', methods=['POST'])
@admin_required
def admin_update_overtime(req_id):
    status = request.form.get('status', 'pending')
    catatan = request.form.get('catatan', '')
    r = update_overtime_status(
        req_id,
        status,
        catatan
    )
    _append_audit_log("admin_update_overtime", {
        "request_id": req_id,
        "status": status,
        "catatan": catatan,
        "updated_by": session.get("user_id"),
        "success": r['success']
    })
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_overtime'))


# ── Admin Home Early Dashboard ────────────────────────────────────────────
@app.route('/admin/home-early')
@admin_required
def admin_home_early():
    status = request.args.get('status', '')
    data   = get_home_early_requests()
    if status:
        data = [d for d in data if d['status'] == status]
    return render_template('admin/admin_home_early.html', data=data, status=status)


@app.route('/admin/home-early/<int:req_id>/update', methods=['POST'])
@admin_required
def admin_update_home_early(req_id):
    status = request.form.get('status', 'pending')
    catatan = request.form.get('catatan', '')
    r = update_home_early_status(
        req_id,
        status,
        catatan
    )
    _append_audit_log("admin_update_home_early", {
        "request_id": req_id,
        "status": status,
        "catatan": catatan,
        "updated_by": session.get("user_id"),
        "success": r['success']
    })
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_home_early'))


# ── Admin Gaji Pokok ─────────────────────────────────────────────────────────
@app.route('/admin/gaji')
@admin_required
def admin_gaji():
    return render_template('admin/admin_gaji.html',
        gaji_list=get_gaji_list(), karyawan=get_karyawan_list())


@app.route('/admin/gaji/simpan', methods=['POST'])
@admin_required
def admin_simpan_gaji():
    karyawan_id = request.form.get('karyawan_id', type=int)
    gaji_harian = request.form.get('gaji_harian', type=float)
    if not karyawan_id or gaji_harian is None:
        flash('Data tidak lengkap.', 'danger')
        return redirect(url_for('admin_gaji'))
    r = upsert_gaji(karyawan_id, gaji_harian)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_gaji'))


@app.route('/admin/gaji/hapus/<int:gaji_id>', methods=['POST'])
@admin_required
def admin_hapus_gaji(gaji_id):
    r = hapus_gaji(gaji_id)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_gaji'))


# ── Admin Potongan Config ─────────────────────────────────────────────────────
@app.route('/admin/potongan')
@admin_required
def admin_potongan():
    return render_template('admin/admin_potongan.html', potongan=get_potongan_list())


@app.route('/admin/potongan/simpan', methods=['POST'])
@admin_required
def admin_simpan_potongan():
    jenis      = request.form.get('jenis', '').strip()
    nominal    = request.form.get('nominal', type=float)
    keterangan = request.form.get('keterangan', '').strip()
    if not jenis or nominal is None:
        flash('Jenis dan nominal wajib diisi.', 'danger')
        return redirect(url_for('admin_potongan'))
    r = upsert_potongan(jenis, nominal, keterangan)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_potongan'))


@app.route('/admin/potongan/hapus/<int:potongan_id>', methods=['POST'])
@admin_required
def admin_hapus_potongan(potongan_id):
    r = hapus_potongan(potongan_id)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_potongan'))


# ── Admin Lembur Rate ─────────────────────────────────────────────────────────
@app.route('/admin/lembur-rate')
@admin_required
def admin_lembur_rate():
    return render_template('admin/admin_lembur_rate.html', rates=get_lembur_rate_list())


@app.route('/admin/lembur-rate/simpan', methods=['POST'])
@admin_required
def admin_simpan_lembur_rate():
    jabatan     = request.form.get('jabatan', '').strip()
    rate_per_jam = request.form.get('rate_per_jam', type=float)
    keterangan  = request.form.get('keterangan', '').strip()
    if not jabatan or rate_per_jam is None:
        flash('Jabatan dan rate wajib diisi.', 'danger')
        return redirect(url_for('admin_lembur_rate'))
    r = upsert_lembur_rate(jabatan, rate_per_jam, keterangan)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_lembur_rate'))


@app.route('/admin/lembur-rate/hapus/<int:rate_id>', methods=['POST'])
@admin_required
def admin_hapus_lembur_rate(rate_id):
    r = hapus_lembur_rate(rate_id)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_lembur_rate'))


# ── Admin Laporan Gaji ────────────────────────────────────────────────────────
@app.route('/admin/laporan-gaji')
@admin_required
def admin_laporan_gaji():
    tgl_awal  = request.args.get('dari',   (date.today().replace(day=1)).isoformat())
    tgl_akhir = request.args.get('sampai', date.today().isoformat())
    data = get_laporan_gaji(tgl_awal, tgl_akhir)
    return render_template('admin/admin_laporan_gaji.html',
        data=data, tgl_awal=tgl_awal, tgl_akhir=tgl_akhir)


@app.route('/admin/laporan-gaji/export')
@admin_required
def admin_export_gaji():
    tgl_awal  = request.args.get('dari',   (date.today().replace(day=1)).isoformat())
    tgl_akhir = request.args.get('sampai', date.today().isoformat())
    data = get_laporan_gaji(tgl_awal, tgl_akhir)
    df = pd.DataFrame([{
        'No':                   d['no'],
        'Nama Karyawan':        d['nama'],
        'Divisi/Jabatan':       d['divisi'],
        'Gaji Pokok/Hari':      d['gaji_harian'],
        'Masuk':                d['masuk'],
        'Izin/Sakit':           d['izin_sakit'],
        'Lupa Absen':           d['lupa_absen'],
        'Telat/Pulang Cepat':   d['terlambat'],
        'Lembur (Jam)':         d['lembur_jam'],
        'Gaji Kotor':           d['gaji_kotor'],
        'Potongan Lupa Absen':  d['potongan_lupa'],
        'Potongan Terlambat':   d['potongan_terlambat'],
        'Lembur':               d['bonus_lembur'],
        'Total Terima':         d['total_terima'],
    } for d in data])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Laporan Gaji')
        ws = writer.sheets['Laporan Gaji']
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(
                max(len(str(c.value or '')) for c in col) + 4, 40)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'gaji_{tgl_awal}_{tgl_akhir}.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# ── Admin Izin Sakit ───────────────────────────────────────────────────────────────
@app.route('/admin/izin-sakit')
@admin_required
def admin_izin_sakit():
    tgl_awal  = request.args.get('dari',   (date.today() - timedelta(days=30)).isoformat())
    tgl_akhir = request.args.get('sampai', date.today().isoformat())
    data = get_izin_sakit_list(tgl_awal=tgl_awal, tgl_akhir=tgl_akhir)
    return render_template('admin/admin_izin_sakit.html',
        data=data, karyawan=get_karyawan_list(),
        tgl_awal=tgl_awal, tgl_akhir=tgl_akhir)


@app.route('/admin/izin-sakit/tambah', methods=['POST'])
@admin_required
def admin_tambah_izin_sakit():
    karyawan_id = request.form.get('karyawan_id', type=int)
    tanggal     = request.form.get('tanggal', '').strip()
    jenis       = request.form.get('jenis', 'izin').strip()
    keterangan  = request.form.get('keterangan', '').strip()
    if not karyawan_id or not tanggal:
        flash('Karyawan dan tanggal wajib diisi.', 'danger')
        return redirect(url_for('admin_izin_sakit'))
    r = tambah_izin_sakit(karyawan_id, tanggal, jenis, keterangan)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_izin_sakit'))


@app.route('/admin/izin-sakit/edit/<int:izin_id>', methods=['POST'])
@admin_required
def admin_edit_izin_sakit(izin_id):
    tanggal    = request.form.get('tanggal', '').strip()
    jenis      = request.form.get('jenis', 'izin').strip()
    keterangan = request.form.get('keterangan', '').strip()
    if not tanggal:
        flash('Tanggal wajib diisi.', 'danger')
        return redirect(url_for('admin_izin_sakit'))
    r = edit_izin_sakit(izin_id, tanggal, jenis, keterangan)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_izin_sakit'))


@app.route('/admin/izin-sakit/hapus/<int:izin_id>', methods=['POST'])
@admin_required
def admin_hapus_izin_sakit(izin_id):
    r = hapus_izin_sakit(izin_id)
    flash(r['msg'], 'success' if r['success'] else 'danger')
    return redirect(url_for('admin_izin_sakit'))


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