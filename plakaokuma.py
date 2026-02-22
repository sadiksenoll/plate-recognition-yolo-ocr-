"""
Plaka Okuma Sistemi
YOLO ve EasyOCR tabanlı akıllı plaka tanıma sistemi.

Özellikler:
- YOLO modeli ile otomatik plaka tespiti
- EasyOCR ile plaka metinlerini okuma
- Gerçek zamanlı webcam desteği
- Tkinter tabanlı kullanıcı arayüzü
- GPU hızlandırma desteği

Author: [Your Name]
License: MIT
GitHub: https://github.com/[your-username]/plaka-okuma-sistemi
"""

import json
import os
import base64
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
try:
    import cv2  # pip install opencv-python
except Exception:
    cv2 = None
try:
    from ultralytics import YOLO  # pip install ultralytics
    HAS_YOLO = True
except BaseException as e:
    YOLO = None
    HAS_YOLO = False
try:
    import torch
    HAS_TORCH = True
    TORCH_HAS_CUDA = torch.cuda.is_available()
except BaseException:
    torch = None
    HAS_TORCH = False
    TORCH_HAS_CUDA = False
try:
    import easyocr  # pip install easyocr
    HAS_EASYOCR = True
except BaseException as e:
    easyocr = None
    HAS_EASYOCR = False
try:
    from PIL import Image, ImageTk  # pip install pillow
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageTk = None
    PIL_AVAILABLE = False


DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")


class DataStore:
    def __init__(self, path: str):
        self.path = path
        self.data = {"settings": {"camera1": "", "camera2": "", "relay_ip": "", "relay_port": 1590, "open_ms": 1000, "auto_open": True, "yolo_model_path": "", "relay_command_open": "10", "roi_cam1": [], "roi_cam2": []}, "plates": [], "passes": []}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def get_settings(self):
        return self.data.get("settings", {})

    def update_settings(self, updates: dict):
        self.data["settings"].update(updates)
        return self.save()

    def list_plates(self):
        return list(self.data.get("plates", []))

    def add_plate(self, plate: str, owner: str = "", note: str = "", extra: dict | None = None):
        items = self.data.setdefault("plates", [])
        rec = {"plate": plate.strip().upper(), "owner": owner.strip(), "note": note.strip()}
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in rec:
                    rec[k] = v
        items.append(rec)
        return self.save()

    def is_registered(self, plate: str) -> bool:
        p = plate.strip().upper()
        for item in self.data.get("plates", []):
            if item.get("plate", "").upper() == p:
                return True
        return False

    def get_owner_by_plate(self, plate: str) -> str:
        p = plate.strip().upper()
        for item in self.data.get("plates", []):
            if (item.get("plate", "") or "").upper() == p:
                return item.get("owner", "")
        return ""

    def add_pass(self, plate: str, ts: str, source: str = "cam1"):
        items = self.data.setdefault("passes", [])
        items.append({"time": ts, "plate": plate.strip().upper(), "registered": self.is_registered(plate), "source": source})
        return self.save()

    def list_passes(self):
        return list(self.data.get("passes", []))


class PlakaOkumaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plaka Okuma Sistemi")
        self.geometry("1100x700")
        self.minsize(980, 620)

        self.store = DataStore(DATA_FILE)
        if not self.store.get_settings().get("camera1"):
            self.store.update_settings({
                "camera1": "rtsp://admin:Bema2019@192.168.1.46:554/Streaming/Channels/101"
            })
        # Röle varsayılanlarını doldur
        s0 = self.store.get_settings()
        if not s0.get("relay_ip") or not s0.get("relay_port"):
            self.store.update_settings({"relay_ip": "192.168.1.248", "relay_port": 1590})

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        # Gelişmiş stil ayarları
        accent = "#4f46e5"
        bg = "#0f172a"
        surface = "#1e293b"
        surface_light = "#334155"
        fg = "#f1f5f9"
        muted = "#94a3b8"
        success = "#10b981"
        warning = "#f59e0b"
        error = "#ef4444"
        info = "#3b82f6"
        
        self.configure(bg=bg)
        self.style.configure("TNotebook", background=bg, borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            padding=(16, 10),
            background="#0b1222",
            foreground=muted,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[
                ("selected", surface),
                ("active", "#1a2238"),
                ("!selected", "#0b1222"),
            ],
            foreground=[
                ("selected", fg),
                ("active", fg),
                ("!selected", muted),
            ],
        )
        self.style.configure("TFrame", background=bg)
        self.style.configure("Card.TLabelframe", background=surface, foreground=fg, relief="flat", borderwidth=1)
        self.style.configure("Card.TLabelframe.Label", background=surface, foreground=muted, font=("Segoe UI", 11, "bold"))
        self.style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
        self.style.configure("Status.TLabel", background=surface, foreground=fg, font=("Segoe UI", 9))
        self.style.configure("Success.TLabel", background=bg, foreground=success, font=("Segoe UI", 10, "bold"))
        self.style.configure("Error.TLabel", background=bg, foreground=error, font=("Segoe UI", 10, "bold"))
        self.style.configure("Info.TLabel", background=bg, foreground=info, font=("Segoe UI", 10, "bold"))
        # Renkli başlık stilleri ekle
        self.style.configure("Title.TLabel", background=bg, foreground="#60a5fa", font=("Segoe UI", 16, "bold"))
        self.style.configure("Subtitle.TLabel", background=bg, foreground="#94a3b8", font=("Segoe UI", 11))
        self.style.configure("Field.TLabel", background=bg, foreground=fg, font=("Segoe UI", 10, "bold"))
        self.style.configure("Emoji.TLabel", background=bg, foreground="#fbbf24", font=("Segoe UI", 10, "bold"))  # Sarı emojiler
        self.style.configure("Card.TLabelframe.Label", background=surface, foreground="#60a5fa", font=("Segoe UI", 11, "bold"))  # Mavi başlıklar
        self.style.configure("Accent.TButton", padding=(16, 12), foreground="#ffffff", background=accent, borderwidth=0, font=("Segoe UI", 11, "bold"))
        self.style.map(
            "Accent.TButton",
            background=[
                ("active", "#6366f1"),
                ("pressed", "#4338ca"),
                ("!disabled", accent),
            ],
            relief=[("pressed", "flat"), ("!pressed", "flat")],
        )
        self.style.configure("TButton", padding=(14, 10), foreground=fg, background=surface_light, borderwidth=0, font=("Segoe UI", 10))
        self.style.map(
            "TButton",
            background=[
                ("active", "#475569"),
                ("pressed", surface),
                ("!disabled", surface_light),
            ],
            relief=[("pressed", "flat"), ("!pressed", "flat")],
        )
        
        # Özel buton stilleri
        self.style.configure("Success.TButton", padding=(14, 10), foreground="#ffffff", background=success, borderwidth=0, font=("Segoe UI", 11, "bold"))
        self.style.configure("Error.TButton", padding=(14, 10), foreground="#ffffff", background=error, borderwidth=0, font=("Segoe UI", 11, "bold"))
        self.style.configure("Warning.TButton", padding=(14, 10), foreground="#ffffff", background=warning, borderwidth=0, font=("Segoe UI", 11, "bold"))
        
        # Entry stilleri
        self.style.configure("TEntry", fieldbackground=surface_light, foreground=fg, borderwidth=1, font=("Segoe UI", 10))
        self.style.map("TEntry", focuscolor=[("focus", accent)])

        # Son geçiş metin değişkenleri (UI'de kullanılacak) – sayfalar oluşturulmadan önce tanımla
        self.var_cam1_last = tk.StringVar(value="—")
        self.var_cam2_last = tk.StringVar(value="—")
        
        # Renkli font ayarları
        self.font_large = ("Segoe UI", 11, "bold")
        self.font_normal = ("Segoe UI", 10)

        self._build_root()
        self._build_pages()
        self._log("Uygulama hazır.")
        # Terminale ortam bilgisini yaz
        self._print_env_info()
        # Kamera 1 çalışma durum değişkenleri
        self._cam1_running = False
        self._cam1_thread = None
        self._cam1_cap = None
        self._cam1_photo = None
        self._cam1_latest = None
        self._cam1_display_job = None
        # Kamera 2 çalışma durum değişkenleri
        self._cam2_running = False
        self._cam2_thread = None
        self._cam2_cap = None
        self._cam2_photo = None
        self._cam2_latest = None
        self._cam2_display_job = None
        # ROI dikdörtgenleri (x1, y1, x2, y2) – normalleştirilmiş koordinatlar (0-1 arası)
        self._roi_rect_cam1 = self._load_roi_rect("roi_cam1")
        self._roi_rect_cam2 = self._load_roi_rect("roi_cam2")
        # ROI görünürlük bayrakları (Kaydet'e basınca çizgiyi gizlemek için)
        self._roi_visible_cam1 = True
        self._roi_visible_cam2 = True
        # Canvas üzerinde sürükleme ile geçici ROI seçimi
        self._roi_drag_active_cam1 = False
        self._roi_drag_active_cam2 = False
        self._roi_drag_start_cam1 = None
        self._roi_drag_start_cam2 = None
        self._roi_drag_cur_cam1 = None
        self._roi_drag_cur_cam2 = None
        # ANPR arka plan
        self._anpr_running = False
        self._anpr_thread = None
        self._yolo_model = None
        self._ocr_reader = None
        # Varsayılan GPU tercihi: sadece CUDA uygunsa etkin
        self._gpu_enabled = True if 'TORCH_HAS_CUDA' in globals() and TORCH_HAS_CUDA else False
        self._conf_thres = 0.35  # Daha yüksek confidence için daha kesin tespit
        self._iou_thres = 0.5    # NMS IoU threshold
        self._max_detections = 3 # Maksimum plaka sayısı
        self._min_plate_area = 5000  # Minimum plaka alanı (piksel)
        self._max_plate_area = 150000 # Maksimum plaka alanı
        
        # Yeni ayar değişkenleri
        self._min_aspect_ratio = 1.5  # Minimum en-boy oranı
        self._max_aspect_ratio = 6.0  # Maksimum en-boy oranı
        self._ocr_confidence = 0.6    # OCR güven eşiği
        self._min_char_count = 6      # Minimum karakter sayısı
        self._max_char_count = 8      # Maksimum karakter sayısı
        self._vote_window = 3.0       # Oylama penceresi (saniye)
        self._min_votes = 3           # Minimum oylama sayısı
        self._roi_height = 80         # ROI yüksekliği
        self._clahe_clip_limit = 2.0  # CLAHE clip limit
        
        # Varsayılan ayar değerlerini tanımla
        self._default_settings = {
            'gpu_enabled': True if 'TORCH_HAS_CUDA' in globals() and TORCH_HAS_CUDA else False,
            'conf_thres': 0.25,        # Daha düşük threshold - daha fazla tespit
            'iou_thres': 0.45,         # Daha iyi NMS
            'max_detections': 5,       # Daha fazla tespit imkanı
            'min_plate_area': 3000,    # Daha küçük plakaları da yakala
            'max_plate_area': 500000,  # Büyük plakaları da yakala
            'min_aspect_ratio': 1.8,   # Daha geniş aralık
            'max_aspect_ratio': 8.0,   # Daha geniş aralık
            'ocr_confidence': 0.35,    # Daha düşük OCR threshold
            'min_char_count': 5,       # Daha az karakter kabul et
            'max_char_count': 9,       # Daha fazla karakter kabul et
            'vote_window': 2.0,        # Daha hızlı oylama
            'min_votes': 3,            # Daha az oyla kabul et
            'roi_height': 300,         # Daha yüksek ROI - daha iyi görüntü
            'clahe_clip_limit': 3.0,   # Daha iyi kontrast
            'plate_cooldown_s': 2.0    # Daha hızlı tekrar okuma
        }
        
        # Ayarları yükle
        self._load_settings()
        
        # Röle ayarlarını debug et
        s = self.store.get_settings()
        relay_ip = s.get("relay_ip", "")
        relay_port = s.get("relay_port", 1590)
        relay_cmd = s.get("relay_command_open", "10")
        self._log(f"🔧 [RELAY-INIT] Röle ayarları: IP='{relay_ip}', Port={relay_port}, Komut='{relay_cmd}'")
        
        self._anpr_last_emit = {}
        # Oylama: plaka -> zaman damgaları
        self._votes = {}
        # Onaylanmış son plaka ve soğuma süresi
        self._last_confirmed_plate = None
        self._last_confirmed_ts = 0.0
        self._plate_cooldown_s = 8.0  # Daha uzun cooldown ile yanlışları azalt

    def _load_settings(self):
        """Kaydedilmiş ayarları dosyadan yükle"""
        settings_file = "anpr_settings.json"
        try:
            import os
            if os.path.exists(settings_file):
                import json
                with open(settings_file, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                
                # YOLO ayarları
                self._gpu_enabled = saved_settings.get('gpu_enabled', self._default_settings['gpu_enabled'])
                self._conf_thres = saved_settings.get('conf_thres', self._default_settings['conf_thres'])
                self._iou_thres = saved_settings.get('iou_thres', self._default_settings['iou_thres'])
                self._max_detections = saved_settings.get('max_detections', self._default_settings['max_detections'])
                self._imgsz = saved_settings.get('imgsz', 640)
                self._half_precision = saved_settings.get('half_precision', True)
                
                # Plaka filtreleme ayarları
                self._min_plate_area = saved_settings.get('min_plate_area', self._default_settings['min_plate_area'])
                self._max_plate_area = saved_settings.get('max_plate_area', self._default_settings['max_plate_area'])
                self._min_aspect_ratio = saved_settings.get('min_aspect_ratio', self._default_settings['min_aspect_ratio'])
                self._max_aspect_ratio = saved_settings.get('max_aspect_ratio', self._default_settings['max_aspect_ratio'])
                self._min_y_percent = saved_settings.get('min_y_percent', 0.15)
                self._margin = saved_settings.get('margin', 20)
                self._nms_iou_threshold = saved_settings.get('nms_iou_threshold', 0.3)
                
                # OCR ayarları
                self._ocr_confidence = saved_settings.get('ocr_confidence', self._default_settings['ocr_confidence'])
                self._min_char_count = saved_settings.get('min_char_count', self._default_settings['min_char_count'])
                self._max_char_count = saved_settings.get('max_char_count', self._default_settings['max_char_count'])
                self._allowlist = saved_settings.get('allowlist', "ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
                self._ocr_detail = saved_settings.get('ocr_detail', 1)
                
                # Zamanlama ayarları
                self._vote_window = saved_settings.get('vote_window', self._default_settings['vote_window'])
                self._min_votes = saved_settings.get('min_votes', self._default_settings['min_votes'])
                self._plate_cooldown_s = saved_settings.get('plate_cooldown_s', self._default_settings['plate_cooldown_s'])
                self._emit_interval = saved_settings.get('emit_interval', 8.0)
                
                # Görüntü işleme ayarları
                self._roi_height = saved_settings.get('roi_height', self._default_settings['roi_height'])
                self._min_roi_width = saved_settings.get('min_roi_width', 90)
                self._min_roi_height = saved_settings.get('min_roi_height', 30)
                self._clahe_clip_limit = saved_settings.get('clahe_clip_limit', self._default_settings['clahe_clip_limit'])
                self._clahe_grid_size = saved_settings.get('clahe_grid_size', 8)
                self._interpolation = saved_settings.get('interpolation', "INTER_CUBIC")
                
                # Performans ayarları
                self._gpu_wait_time = saved_settings.get('gpu_wait_time', 15.0)
                self._cpu_wait_time = saved_settings.get('cpu_wait_time', 25.0)
                
                # Hata ayıklama ayarları
                self._verbose = saved_settings.get('verbose', False)
                self._show_boxes = saved_settings.get('show_boxes', False)
                
                self._log("💾 Kaydedilmiş ayarlar yüklendi.")
            else:
                self._log("📝 Kaydedilmiş ayar bulunamadı, varsayılan ayarlar kullanılıyor.")
        except Exception as e:
            self._log(f"⚠️ Ayarlar yüklenirken hata: {e}")
            self._log("📝 Varsayılan ayarlar kullanılıyor.")
    
    def _save_settings(self):
        """Mevcut ayarları dosyaya kaydet"""
        settings_file = "anpr_settings.json"
        try:
            import json
            current_settings = {
                # YOLO ayarları
                'gpu_enabled': self._gpu_enabled,
                'conf_thres': self._conf_thres,
                'iou_thres': self._iou_thres,
                'max_detections': self._max_detections,
                'imgsz': getattr(self, '_imgsz', 640),
                'half_precision': getattr(self, '_half_precision', True),
                
                # Plaka filtreleme ayarları
                'min_plate_area': self._min_plate_area,
                'max_plate_area': self._max_plate_area,
                'min_aspect_ratio': self._min_aspect_ratio,
                'max_aspect_ratio': self._max_aspect_ratio,
                'min_y_percent': getattr(self, '_min_y_percent', 0.15),
                'margin': getattr(self, '_margin', 20),
                'nms_iou_threshold': getattr(self, '_nms_iou_threshold', 0.3),
                
                # OCR ayarları
                'ocr_confidence': self._ocr_confidence,
                'min_char_count': self._min_char_count,
                'max_char_count': self._max_char_count,
                'allowlist': getattr(self, '_allowlist', "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"),
                'ocr_detail': getattr(self, '_ocr_detail', 1),
                
                # Zamanlama ayarları
                'vote_window': self._vote_window,
                'min_votes': self._min_votes,
                'plate_cooldown_s': self._plate_cooldown_s,
                'emit_interval': getattr(self, '_emit_interval', 8.0),
                
                # Görüntü işleme ayarları
                'roi_height': self._roi_height,
                'min_roi_width': getattr(self, '_min_roi_width', 90),
                'min_roi_height': getattr(self, '_min_roi_height', 30),
                'clahe_clip_limit': self._clahe_clip_limit,
                'clahe_grid_size': getattr(self, '_clahe_grid_size', 8),
                'interpolation': getattr(self, '_interpolation', "INTER_CUBIC"),
                
                # Performans ayarları
                'gpu_wait_time': getattr(self, '_gpu_wait_time', 15.0),
                'cpu_wait_time': getattr(self, '_cpu_wait_time', 25.0),
                
                # Hata ayıklama ayarları
                'verbose': getattr(self, '_verbose', False),
                'show_boxes': getattr(self, '_show_boxes', False)
            }
            
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(current_settings, f, ensure_ascii=False, indent=2)
            
            self._log("💾 Ayarlar kaydedildi.")
        except Exception as e:
            self._log(f"⚠️ Ayarlar kaydedilirken hata: {e}")

    def _load_roi_rect(self, key: str):
        """Ayarlar sözlüğünden normalleştirilmiş dikdörtgen ROI'yi (x1,y1,x2,y2) oku."""
        try:
            s = self.store.get_settings() or {}
        except Exception:
            return None
        v = s.get(key) or []
        if (isinstance(v, (list, tuple)) and len(v) == 4):
            try:
                x1 = float(v[0]); y1 = float(v[1]); x2 = float(v[2]); y2 = float(v[3])
            except Exception:
                return None
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
        return None

    def _save_roi_rect(self, key: str, rect):
        """Dikdörtgen ROI'yi ayarlara kaydet."""
        if not rect or len(rect) != 4:
            return
        try:
            x1, y1, x2, y2 = [float(v) for v in rect]
        except Exception:
            return
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return
        self.store.update_settings({key: [x1, y1, x2, y2]})

    def _canvas_to_frame_norm(self, event, cam: str):
        """Canvas üzerindeki bir noktayı, ilgili kameranın frame koordinatlarına (0-1 normalize) çevir."""
        if cam == 'cam1':
            frame = self._cam1_latest
            canvas = self.canvas_cam1
        else:
            frame = self._cam2_latest
            canvas = self.canvas_cam2
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        cw = max(canvas.winfo_width(), 1)
        ch = max(canvas.winfo_height(), 1)
        scale = min(cw / max(fw, 1), ch / max(fh, 1))
        disp_w = fw * scale
        disp_h = fh * scale
        x0 = (cw - disp_w) / 2.0
        y0 = (ch - disp_h) / 2.0
        ex = float(event.x)
        ey = float(event.y)
        if ex < x0 or ex > x0 + disp_w or ey < y0 or ey > y0 + disp_h:
            return None
        fx = (ex - x0) / scale
        fy = (ey - y0) / scale
        nx = fx / float(fw)
        ny = fy / float(fh)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return nx, ny

    def _on_canvas_down_rect(self, event, cam: str):
        """Mouse basılınca ROI dikdörtgeni sürüklemeyi başlat."""
        pt = self._canvas_to_frame_norm(event, cam)
        if pt is None:
            return
        if cam == 'cam1':
            self._roi_drag_active_cam1 = True
            self._roi_drag_start_cam1 = pt
            self._roi_drag_cur_cam1 = pt
        else:
            self._roi_drag_active_cam2 = True
            self._roi_drag_start_cam2 = pt
            self._roi_drag_cur_cam2 = pt

    def _on_canvas_drag_rect(self, event, cam: str):
        """Mouse hareket ederken geçici ROI dikdörtgenini güncelle."""
        pt = self._canvas_to_frame_norm(event, cam)
        if pt is None:
            return
        if cam == 'cam1':
            if not self._roi_drag_active_cam1:
                return
            self._roi_drag_cur_cam1 = pt
        else:
            if not self._roi_drag_active_cam2:
                return
            self._roi_drag_cur_cam2 = pt

    def _on_canvas_up_rect(self, event, cam: str):
        """Mouse bırakılınca ROI dikdörtgenini sabitle ve ayarlara kaydet."""
        pt = self._canvas_to_frame_norm(event, cam)
        if pt is None:
            # Dışarı bırakıldıysa sürüklemeyi iptal et
            if cam == 'cam1':
                self._roi_drag_active_cam1 = False
                self._roi_drag_start_cam1 = None
                self._roi_drag_cur_cam1 = None
            else:
                self._roi_drag_active_cam2 = False
                self._roi_drag_start_cam2 = None
                self._roi_drag_cur_cam2 = None
            return
        if cam == 'cam1':
            if not self._roi_drag_active_cam1 or self._roi_drag_start_cam1 is None:
                return
            x1, y1 = self._roi_drag_start_cam1
            x2, y2 = pt
            self._roi_drag_active_cam1 = False
            self._roi_drag_start_cam1 = None
            self._roi_drag_cur_cam1 = None
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            if (x_max - x_min) < 0.02 or (y_max - y_min) < 0.02:
                return
            self._roi_rect_cam1 = [x_min, y_min, x_max, y_max]
            self._save_roi_rect("roi_cam1", self._roi_rect_cam1)
            # Yeni ROI çizildiğinde çizgileri yeniden göster
            self._roi_visible_cam1 = True
            self._log("Kamera 1 ROI alanı güncellendi.")
        else:
            if not self._roi_drag_active_cam2 or self._roi_drag_start_cam2 is None:
                return
            x1, y1 = self._roi_drag_start_cam2
            x2, y2 = pt
            self._roi_drag_active_cam2 = False
            self._roi_drag_start_cam2 = None
            self._roi_drag_cur_cam2 = None
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            if (x_max - x_min) < 0.02 or (y_max - y_min) < 0.02:
                return
            self._roi_rect_cam2 = [x_min, y_min, x_max, y_max]
            self._save_roi_rect("roi_cam2", self._roi_rect_cam2)
            # Yeni ROI çizildiğinde çizgileri yeniden göster
            self._roi_visible_cam2 = True
            self._log("Kamera 2 ROI alanı güncellendi.")

    def _get_roi_rect_for_cam(self, cam: str):
        if cam == 'cam1':
            return self._roi_rect_cam1
        else:
            return self._roi_rect_cam2

    def on_roi_save(self):
        """ROI çizgilerini gizle; alanlar zaten seçim sırasında kaydedildiği için ANPR kullanmaya devam eder."""
        self._roi_visible_cam1 = False
        self._roi_visible_cam2 = False
        self._log("ROI alanları kaydedildi, mavi çizgiler gizlendi.")

    def _print_env_info(self):
        import sys
        print("==== Ortam Bilgisi ====")
        print(f"Python: {sys.version.split()[0]}")
        print(f"OpenCV: {'VAR' if cv2 is not None else 'YOK'}  (pip install opencv-python)")
        print(f"YOLO (ultralytics): {'VAR' if HAS_YOLO else 'YOK'}  (pip install ultralytics)")
        print(f"EasyOCR: {'VAR' if HAS_EASYOCR else 'YOK'}  (pip install easyocr)")
        print(f"Pillow: {'VAR' if PIL_AVAILABLE else 'YOK'}  (pip install pillow)")
        print(f"PyTorch: {'VAR' if HAS_TORCH else 'YOK'}  (pip install torch)")
        if HAS_TORCH:
            print(f"CUDA kullanılabilir: {TORCH_HAS_CUDA}")
        else:
            print("CUDA bilgisi: PyTorch olmadığı için kontrol edilemedi.")
        if (not HAS_YOLO) or (not HAS_EASYOCR) or (cv2 is None):
            print("!! ANPR için gerekli paketler eksik olabilir. En az şunlar olmalı:")
            print("   pip install ultralytics easyocr opencv-python pillow")
        if HAS_TORCH and not TORCH_HAS_CUDA:
            print("!! PyTorch var ama CUDA aktif değil. GPU hızlandırma için uygun sürücü ve CUDA gerekir.")
        print("=======================\n")

    def _build_root(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.Frame(self, padding=(12, 6))
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Hazır")
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w", style="Status.TLabel").grid(row=0, column=0, sticky="ew")

    def _build_pages(self):
        self.page_home = ttk.Frame(self.notebook)
        self.page_cam_settings = ttk.Frame(self.notebook)
        self.page_gate_settings = ttk.Frame(self.notebook)
        self.page_plate_list = ttk.Frame(self.notebook)
        self.page_passes = ttk.Frame(self.notebook)
        self.page_form = ttk.Frame(self.notebook)

        self.notebook.add(self.page_home, text="Ana Sayfa")
        self.notebook.add(self.page_cam_settings, text="Kamera Ayarları")
        self.notebook.add(self.page_gate_settings, text="Kapı Ayarları")
        self.notebook.add(self.page_plate_list, text="Plaka Listeleri")
        self.notebook.add(self.page_form, text="Kayıt Formu")
        self.notebook.add(self.page_passes, text="Geçişler")

        self._build_home()
        self._build_cam_settings()
        self._build_gate_settings()
        self._build_plate_list()
        self._build_form()
        self._build_passes()

    def _build_home(self):
        self.page_home.columnconfigure(0, weight=1)
        self.page_home.columnconfigure(1, weight=1)
        self.page_home.rowconfigure(0, weight=1)
        self.page_home.rowconfigure(1, weight=0)

        left_card = ttk.LabelFrame(self.page_home, text="Kamera 1", style="Card.TLabelframe")
        right_card = ttk.LabelFrame(self.page_home, text="Kamera 2", style="Card.TLabelframe")
        left_card.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=(12, 6))
        right_card.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=(12, 6))

        left_inner = ttk.Frame(left_card, padding=(8, 8))
        left_inner.pack(fill="both", expand=True)
        self.canvas_cam1 = tk.Canvas(left_inner, bg="#0b1020", highlightthickness=2, highlightbackground="#4f46e5", highlightcolor="#4f46e5")
        self.canvas_cam1.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        # ROI dikdörtgeni seçimi için mouse olayları
        self.canvas_cam1.bind("<Button-1>", lambda e: self._on_canvas_down_rect(e, 'cam1'))
        self.canvas_cam1.bind("<B1-Motion>", lambda e: self._on_canvas_drag_rect(e, 'cam1'))
        self.canvas_cam1.bind("<ButtonRelease-1>", lambda e: self._on_canvas_up_rect(e, 'cam1'))
        left_info = ttk.LabelFrame(left_card, text="Son Geçiş", style="Card.TLabelframe")
        self.lbl_cam1_last = ttk.Label(left_info, textvariable=self.var_cam1_last, font=self.font_large, foreground="#10b981")
        self.lbl_cam1_last.pack(anchor="w", padx=12, pady=8)
        left_info.pack(fill="x", padx=8, pady=(0, 8))

        right_inner = ttk.Frame(right_card, padding=(8, 8))
        right_inner.pack(fill="both", expand=True)
        self.canvas_cam2 = tk.Canvas(right_inner, bg="#0b1020", highlightthickness=2, highlightbackground="#4f46e5", highlightcolor="#4f46e5")
        self.canvas_cam2.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        # ROI dikdörtgeni seçimi için mouse olayları
        self.canvas_cam2.bind("<Button-1>", lambda e: self._on_canvas_down_rect(e, 'cam2'))
        self.canvas_cam2.bind("<B1-Motion>", lambda e: self._on_canvas_drag_rect(e, 'cam2'))
        self.canvas_cam2.bind("<ButtonRelease-1>", lambda e: self._on_canvas_up_rect(e, 'cam2'))
        right_info = ttk.LabelFrame(right_card, text="Son Geçiş", style="Card.TLabelframe")
        self.lbl_cam2_last = ttk.Label(right_info, textvariable=self.var_cam2_last, font=self.font_large, foreground="#10b981")
        self.lbl_cam2_last.pack(anchor="w", padx=12, pady=8)
        right_info.pack(fill="x", padx=8, pady=(0, 8))

        controls = ttk.Frame(self.page_home)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        for i in range(10):
            controls.columnconfigure(i, weight=1)

        ttk.Button(controls, text="Kamera 1 Başlat", style="Accent.TButton", command=self.on_cam1_start).grid(row=0, column=0, padx=4)
        ttk.Button(controls, text="Kamera 1 Durdur", style="TButton", command=self.on_cam1_stop).grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="Kamera 2 Başlat", style="Accent.TButton", command=self.on_cam2_start).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Kamera 2 Durdur", style="TButton", command=self.on_cam2_stop).grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="Kapıyı Aç", style="Accent.TButton", command=self.on_gate_open).grid(row=0, column=4, padx=4)
        ttk.Button(controls, text="ANPR Başlat", style="Accent.TButton", command=self.on_anpr_start).grid(row=0, column=5, padx=4)
        ttk.Button(controls, text="ANPR Durdur", style="TButton", command=self.on_anpr_stop).grid(row=0, column=6, padx=4)
        ttk.Button(controls, text="Ekranı Temizle", style="TButton", command=self.on_clear_preview).grid(row=0, column=7, padx=4)
        ttk.Button(controls, text="ROI Kaydet", style="TButton", command=self.on_roi_save).grid(row=0, column=8, padx=4)
        ttk.Button(controls, text="⚙️ Ayarlar", style="Accent.TButton", command=self.open_settings_dialog).grid(row=0, column=9, padx=4)
        

    def _build_cam_settings(self):
        frm = self.page_cam_settings
        frm.columnconfigure(1, weight=1)

        # Başlık alanı
        header_frame = ttk.Frame(frm)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(12, 8))
        header_frame.columnconfigure(0, weight=1)
        
        # Renkli başlık
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(anchor="w")
        
        emoji_label = ttk.Label(title_frame, text="📹", style="Emoji.TLabel")
        emoji_label.pack(side="left")
        
        title_label = ttk.Label(title_frame, text=" Kamera Ayarları", style="Title.TLabel")
        title_label.pack(side="left")
        
        # Alt başlık
        subtitle_label = ttk.Label(header_frame, text="✨ Kamera URL'leri ve YOLO model yapılandırması", style="Subtitle.TLabel")
        subtitle_label.pack(anchor="w", pady=(5, 0))

        # Kamera 1
        cam1_emoji = ttk.Label(frm, text="📹", style="Emoji.TLabel")
        cam1_emoji.grid(row=1, column=0, sticky="w", padx=12, pady=(12, 6))
        cam1_label = ttk.Label(frm, text=" Kamera 1 URL", style="Field.TLabel")
        cam1_label.grid(row=1, column=0, sticky="w", padx=(35, 12), pady=(12, 6))
        self.var_cam1 = tk.StringVar(value=self.store.get_settings().get("camera1", ""))
        cam1_entry = ttk.Entry(frm, textvariable=self.var_cam1, font=("Segoe UI", 11))
        cam1_entry.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(12, 6))

        # Kamera 2
        cam2_emoji = ttk.Label(frm, text="📹", style="Emoji.TLabel")
        cam2_emoji.grid(row=2, column=0, sticky="w", padx=12, pady=6)
        cam2_label = ttk.Label(frm, text=" Kamera 2 URL", style="Field.TLabel")
        cam2_label.grid(row=2, column=0, sticky="w", padx=(35, 12), pady=6)
        self.var_cam2 = tk.StringVar(value=self.store.get_settings().get("camera2", ""))
        cam2_entry = ttk.Entry(frm, textvariable=self.var_cam2, font=("Segoe UI", 11))
        cam2_entry.grid(row=2, column=1, sticky="ew", padx=(0, 12), pady=6)

        # YOLO model yolu
        yolo_emoji = ttk.Label(frm, text="🤖", style="Emoji.TLabel")
        yolo_emoji.grid(row=3, column=0, sticky="w", padx=12, pady=6)
        yolo_label = ttk.Label(frm, text=" YOLO Model (.pt) Yolu", style="Field.TLabel")
        yolo_label.grid(row=3, column=0, sticky="w", padx=(35, 12), pady=6)
        self.var_yolo = tk.StringVar(value=self.store.get_settings().get("yolo_model_path", ""))
        ent_yolo = ttk.Entry(frm, textvariable=self.var_yolo, font=("Segoe UI", 11))
        ent_yolo.grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=6)
        
        def _browse_yolo():
            from tkinter import filedialog
            p = filedialog.askopenfilename(title="YOLO .pt dosyasını seç", filetypes=[["YOLO Weights", "*.pt"], ["Tüm Dosyalar", "*.*"]])
            if p:
                self.var_yolo.set(p)
        browse_btn = ttk.Button(frm, text="📁 Gözat", style="TButton", command=_browse_yolo)
        browse_btn.grid(row=3, column=2, sticky="w", padx=(0, 12), pady=6)

        # Butonlar
        actions_frame = ttk.Frame(frm)
        actions_frame.grid(row=4, column=0, columnspan=3, sticky="e", padx=12, pady=(12, 12))
        
        ttk.Button(actions_frame, text="💾 Kaydet", style="Success.TButton", command=self.on_save_cam_settings).pack(side="left", padx=(0, 8))
        ttk.Button(actions_frame, text="🔄 Test Et", style="Info.TButton", command=self.on_cam1_start).pack(side="left")

    def _build_gate_settings(self):
        frm = self.page_gate_settings
        frm.columnconfigure(1, weight=1)

        # Başlık alanı
        header_frame = ttk.Frame(frm)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(12, 8))
        header_frame.columnconfigure(0, weight=1)
        
        # Renkli başlık
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(anchor="w")
        
        emoji_label = ttk.Label(title_frame, text="🚪", style="Emoji.TLabel")
        emoji_label.pack(side="left")
        
        title_label = ttk.Label(title_frame, text=" Kapı Ayarları", style="Title.TLabel")
        title_label.pack(side="left")
        
        # Alt başlık
        subtitle_label = ttk.Label(header_frame, text="✨ Röle ve kapı kontrol yapılandırması", style="Subtitle.TLabel")
        subtitle_label.pack(anchor="w", pady=(5, 0))

        s = self.store.get_settings()
        
        # Röle IP
        ip_emoji = ttk.Label(frm, text="🌐", style="Emoji.TLabel")
        ip_emoji.grid(row=1, column=0, sticky="w", padx=12, pady=(12, 6))
        ip_label = ttk.Label(frm, text=" Röle IP", style="Field.TLabel")
        ip_label.grid(row=1, column=0, sticky="w", padx=(35, 12), pady=(12, 6))
        self.var_ip = tk.StringVar(value=s.get("relay_ip", ""))
        ip_entry = ttk.Entry(frm, textvariable=self.var_ip, font=("Segoe UI", 11))
        ip_entry.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(12, 6))

        # Röle Port
        port_emoji = ttk.Label(frm, text="🔌", style="Emoji.TLabel")
        port_emoji.grid(row=2, column=0, sticky="w", padx=12, pady=6)
        port_label = ttk.Label(frm, text=" Röle Port", style="Field.TLabel")
        port_label.grid(row=2, column=0, sticky="w", padx=(35, 12), pady=6)
        self.var_port = tk.IntVar(value=int(s.get("relay_port", 1590) or 1590))
        port_entry = ttk.Entry(frm, textvariable=self.var_port, font=("Segoe UI", 11))
        port_entry.grid(row=2, column=1, sticky="ew", padx=(0, 12), pady=6)

        # Açık Kalma Süresi
        time_emoji = ttk.Label(frm, text="⏱️", style="Emoji.TLabel")
        time_emoji.grid(row=3, column=0, sticky="w", padx=12, pady=6)
        time_label = ttk.Label(frm, text=" Açık Kalma (ms)", style="Field.TLabel")
        time_label.grid(row=3, column=0, sticky="w", padx=(35, 12), pady=6)
        self.var_open_ms = tk.IntVar(value=int(s.get("open_ms", 1000) or 1000))
        time_entry = ttk.Entry(frm, textvariable=self.var_open_ms, font=("Segoe UI", 11))
        time_entry.grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=6)

        # Aç Komutu
        cmd_emoji = ttk.Label(frm, text="📡", style="Emoji.TLabel")
        cmd_emoji.grid(row=4, column=0, sticky="w", padx=12, pady=6)
        cmd_label = ttk.Label(frm, text=" Aç Komutu", style="Field.TLabel")
        cmd_label.grid(row=4, column=0, sticky="w", padx=(35, 12), pady=6)
        self.var_relay_cmd = tk.StringVar(value=s.get("relay_command_open", "10"))
        cmd_entry = ttk.Entry(frm, textvariable=self.var_relay_cmd, font=("Segoe UI", 11))
        cmd_entry.grid(row=4, column=1, sticky="ew", padx=(0, 12), pady=6)

        # Butonlar
        actions_frame = ttk.Frame(frm)
        actions_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=12, pady=(12, 12))
        actions_frame.columnconfigure(1, weight=1)
        
        ttk.Button(actions_frame, text="🚪 Test Aç", style="Warning.TButton", command=self.on_gate_open).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Button(actions_frame, text="💾 Kaydet", style="Success.TButton", command=self.on_save_gate_settings).grid(row=0, column=2, sticky="e")

    def _build_plate_list(self):
        frm = self.page_plate_list
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        # Başlık alanı
        header_frame = ttk.Frame(frm)
        header_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        header_frame.columnconfigure(0, weight=1)
        
        # Renkli başlık
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(anchor="w")
        
        emoji_label = ttk.Label(title_frame, text="📋", style="Emoji.TLabel")
        emoji_label.pack(side="left")
        
        title_label = ttk.Label(title_frame, text=" Kayıtlı Plakalar", style="Title.TLabel")
        title_label.pack(side="left")
        
        # Alt başlık
        subtitle_label = ttk.Label(header_frame, text="✨ Tüm kayıtlı plakalar ve izin bilgileri", style="Subtitle.TLabel")
        subtitle_label.pack(anchor="w", pady=(5, 0))
        
        # Butonlar
        actions_frame = ttk.Frame(frm)
        actions_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(12, 12))
        actions_frame.columnconfigure(1, weight=1)
        
        # Sol butonlar
        left_frame = ttk.Frame(actions_frame)
        left_frame.grid(row=0, column=0, sticky="w")
        
        ttk.Button(left_frame, text="✏️ Düzenle", style="Info.TButton", command=self.on_edit_plate).pack(side="left", padx=(0, 8))
        ttk.Button(left_frame, text="🗑️ Sil", style="Error.TButton", command=self.on_delete_plate).pack(side="left")
        
        # Sağ buton
        ttk.Button(actions_frame, text="🔄 Yenile", style="Accent.TButton", command=self.refresh_plate_list).grid(row=0, column=2, sticky="e")
        
        # Detaylı sütunlar
        cols = ("plate", "owner", "phone", "block", "flat", "brand", "model", "color", "rfid", "valid_from", "valid_to", "auto_gate", "entry_perm", "exit_perm", "note")
        
        # Gelişmiş Treeview stili
        try:
            self.style.configure(
                "Plate.Treeview",
                background="#1a2238",
                fieldbackground="#1a2238",
                foreground="#f3f4f6",
                rowheight=32,
                borderwidth=0,
                font=("Segoe UI", 9)
            )
            self.style.configure(
                "Plate.Treeview.Heading",
                background="#0f172a",
                foreground="#60a5fa",
                borderwidth=0,
                font=("Segoe UI", 10, "bold")
            )
            self.style.map(
                "Plate.Treeview",
                background=[("selected", "#2563eb")],
                foreground=[("selected", "#ffffff")]
            )
        except Exception:
            pass

        self.tree = ttk.Treeview(frm, columns=cols, show="headings", style="Plate.Treeview")
        
        # Emoji başlıklar ve genişlikler
        self.tree.heading("plate", text="🚗 Plaka")
        self.tree.heading("owner", text="👤 Sahip")
        self.tree.heading("phone", text="📞 Telefon")
        self.tree.heading("block", text="🏢 Blok")
        self.tree.heading("flat", text="🔑 Daire")
        self.tree.heading("brand", text="🚘 Marka")
        self.tree.heading("model", text="� Model")
        self.tree.heading("color", text="🎨 Renk")
        self.tree.heading("rfid", text="🪪 RFID")
        self.tree.heading("valid_from", text="📅 Başlangıç")
        self.tree.heading("valid_to", text="📅 Bitiş")
        self.tree.heading("auto_gate", text="🚪 Otomatik")
        self.tree.heading("entry_perm", text="⬅️ Giriş")
        self.tree.heading("exit_perm", text="➡️ Çıkış")
        self.tree.heading("note", text="🗒️ Not")
        
        # Sütun genişlikleri
        self.tree.column("plate", width=100, anchor="center")
        self.tree.column("owner", width=120)
        self.tree.column("phone", width=100, anchor="center")
        self.tree.column("block", width=60, anchor="center")
        self.tree.column("flat", width=60, anchor="center")
        self.tree.column("brand", width=80)
        self.tree.column("model", width=80)
        self.tree.column("color", width=70, anchor="center")
        self.tree.column("rfid", width=80, anchor="center")
        self.tree.column("valid_from", width=90, anchor="center")
        self.tree.column("valid_to", width=90, anchor="center")
        self.tree.column("auto_gate", width=70, anchor="center")
        self.tree.column("entry_perm", width=60, anchor="center")
        self.tree.column("exit_perm", width=60, anchor="center")
        self.tree.column("note", width=150)
        
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=0)

        # Vertical scrollbar
        scrolly = ttk.Scrollbar(frm, orient="vertical", command=self.tree.yview)
        scrolly.grid(row=1, column=1, sticky="ns", padx=(0, 12), pady=0)
        
        self.tree.configure(yscrollcommand=scrolly.set)

        # Renkli tag'ler
        try:
            self.tree.tag_configure("alt", background="#0b1222")  # Alternatif satır
            self.tree.tag_configure("has_data", foreground="#10b981", font=("Segoe UI", 9, "bold"))  # Veri varsa yeşil
            self.tree.tag_configure("no_data", foreground="#94a3b8")  # Veri yoksa gri
            self.tree.tag_configure("permission_yes", foreground="#10b981", font=("Segoe UI", 9, "bold"))  # İzin varsa yeşil
            self.tree.tag_configure("permission_no", foreground="#ef4444", font=("Segoe UI", 9, "bold"))  # İzin yoksa kırmızı
            self.tree.tag_configure("expired", foreground="#f59e0b", font=("Segoe UI", 9, "bold"))  # Süresi dolmuşsa turuncu
        except Exception:
            pass

        # Eski yenile butonunu kaldır (başlıkta zaten var)
        # ttk.Button(frm, text="Yenile", command=self.refresh_plate_list).grid(row=2, column=0, sticky="e", padx=12, pady=(6, 12))
        self.refresh_plate_list()

    def _build_form(self):
        frm = self.page_form
        for c in range(4):
            frm.columnconfigure(c, weight=1)

        # Ana başlık alanı
        header_frame = ttk.Frame(frm)
        header_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=12, pady=(12, 8))
        header_frame.columnconfigure(0, weight=1)
        
        # Renkli başlık
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(anchor="w")
        
        emoji_label = ttk.Label(title_frame, text="📋", style="Emoji.TLabel")
        emoji_label.pack(side="left")
        
        title_label = ttk.Label(title_frame, text=" Yeni Plaka Kaydı", style="Title.TLabel")
        title_label.pack(side="left")
        
        # Alt başlık
        subtitle_label = ttk.Label(header_frame, text="✨ Site sakini ve araç bilgilerini eksiksiz doldurun", style="Subtitle.TLabel")
        subtitle_label.pack(anchor="w", pady=(5, 0))

        info = ttk.LabelFrame(frm, text="Kayıt Formu", style="Card.TLabelframe")
        info.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=12, pady=(0, 12))
        for c in range(4):
            info.columnconfigure(c, weight=1)

        # Form state değişkenleri
        self.var_plate = tk.StringVar()
        self.var_owner = tk.StringVar()
        self.var_phone = tk.StringVar()
        self.var_block = tk.StringVar()
        self.var_flat = tk.StringVar()
        self.var_brand = tk.StringVar()
        self.var_model = tk.StringVar()
        self.var_color = tk.StringVar()
        self.var_rfid = tk.StringVar()
        self.var_valid_from = tk.StringVar()
        self.var_valid_to = tk.StringVar()
        self.var_note = tk.StringVar()
        self.var_allow_auto = tk.BooleanVar(value=True)
        self.var_allow_in = tk.BooleanVar(value=True)
        self.var_allow_out = tk.BooleanVar(value=True)

        # Satır 1: Plaka, İsim
        plate_emoji = ttk.Label(info, text="🚗", style="Emoji.TLabel")
        plate_emoji.grid(row=1, column=0, sticky="w", padx=12, pady=(12, 6))
        plate_label = ttk.Label(info, text=" Plaka", style="Field.TLabel")
        plate_label.grid(row=1, column=0, sticky="w", padx=(35, 12), pady=(12, 6))
        plate_entry = ttk.Entry(info, textvariable=self.var_plate, font=("Segoe UI", 11, "bold"))
        plate_entry.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(12, 6))
        
        owner_emoji = ttk.Label(info, text="👤", style="Emoji.TLabel")
        owner_emoji.grid(row=1, column=2, sticky="w", padx=12, pady=(12, 6))
        owner_label = ttk.Label(info, text=" İsim Soyisim", style="Field.TLabel")
        owner_label.grid(row=1, column=2, sticky="w", padx=(35, 12), pady=(12, 6))
        owner_entry = ttk.Entry(info, textvariable=self.var_owner, font=("Segoe UI", 11))
        owner_entry.grid(row=1, column=3, sticky="ew", padx=(0, 12), pady=(12, 6))

        # Satır 2: Telefon, Blok
        phone_emoji = ttk.Label(info, text="📞", style="Emoji.TLabel")
        phone_emoji.grid(row=2, column=0, sticky="w", padx=12, pady=6)
        phone_label = ttk.Label(info, text=" Telefon", style="Field.TLabel")
        phone_label.grid(row=2, column=0, sticky="w", padx=(35, 12), pady=6)
        phone_entry = ttk.Entry(info, textvariable=self.var_phone, font=("Segoe UI", 11))
        phone_entry.grid(row=2, column=1, sticky="ew", padx=(0, 12), pady=6)
        
        block_emoji = ttk.Label(info, text="🏢", style="Emoji.TLabel")
        block_emoji.grid(row=2, column=2, sticky="w", padx=12, pady=6)
        block_label = ttk.Label(info, text=" Blok", style="Field.TLabel")
        block_label.grid(row=2, column=2, sticky="w", padx=(35, 12), pady=6)
        block_entry = ttk.Entry(info, textvariable=self.var_block, font=("Segoe UI", 11))
        block_entry.grid(row=2, column=3, sticky="ew", padx=(0, 12), pady=6)

        # Satır 3: Daire, Araç Marka
        flat_emoji = ttk.Label(info, text="🔑", style="Emoji.TLabel")
        flat_emoji.grid(row=3, column=0, sticky="w", padx=12, pady=6)
        flat_label = ttk.Label(info, text=" Daire", style="Field.TLabel")
        flat_label.grid(row=3, column=0, sticky="w", padx=(35, 12), pady=6)
        flat_entry = ttk.Entry(info, textvariable=self.var_flat, font=("Segoe UI", 11))
        flat_entry.grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=6)
        
        brand_emoji = ttk.Label(info, text="🚘", style="Emoji.TLabel")
        brand_emoji.grid(row=3, column=2, sticky="w", padx=12, pady=6)
        brand_label = ttk.Label(info, text=" Araç Marka", style="Field.TLabel")
        brand_label.grid(row=3, column=2, sticky="w", padx=(35, 12), pady=6)
        brand_entry = ttk.Entry(info, textvariable=self.var_brand, font=("Segoe UI", 11))
        brand_entry.grid(row=3, column=3, sticky="ew", padx=(0, 12), pady=6)

        # Satır 4: Model, Renk
        model_emoji = ttk.Label(info, text="🔧", style="Emoji.TLabel")
        model_emoji.grid(row=4, column=0, sticky="w", padx=12, pady=6)
        model_label = ttk.Label(info, text=" Model", style="Field.TLabel")
        model_label.grid(row=4, column=0, sticky="w", padx=(35, 12), pady=6)
        model_entry = ttk.Entry(info, textvariable=self.var_model, font=("Segoe UI", 11))
        model_entry.grid(row=4, column=1, sticky="ew", padx=(0, 12), pady=6)
        
        color_emoji = ttk.Label(info, text="🎨", style="Emoji.TLabel")
        color_emoji.grid(row=4, column=2, sticky="w", padx=12, pady=6)
        color_label = ttk.Label(info, text=" Renk", style="Field.TLabel")
        color_label.grid(row=4, column=2, sticky="w", padx=(35, 12), pady=6)
        color_entry = ttk.Entry(info, textvariable=self.var_color, font=("Segoe UI", 11))
        color_entry.grid(row=4, column=3, sticky="ew", padx=(0, 12), pady=6)

        # Satır 5: Kart / RFID, Geçerlilik Başlangıç
        rfid_emoji = ttk.Label(info, text="🪪", style="Emoji.TLabel")
        rfid_emoji.grid(row=5, column=0, sticky="w", padx=12, pady=6)
        rfid_label = ttk.Label(info, text=" Kart / RFID", style="Field.TLabel")
        rfid_label.grid(row=5, column=0, sticky="w", padx=(35, 12), pady=6)
        rfid_entry = ttk.Entry(info, textvariable=self.var_rfid, font=("Segoe UI", 11))
        rfid_entry.grid(row=5, column=1, sticky="ew", padx=(0, 12), pady=6)
        
        valid_from_emoji = ttk.Label(info, text="📅", style="Emoji.TLabel")
        valid_from_emoji.grid(row=5, column=2, sticky="w", padx=12, pady=6)
        valid_from_label = ttk.Label(info, text=" Geçerlilik Başlangıç", style="Field.TLabel")
        valid_from_label.grid(row=5, column=2, sticky="w", padx=(35, 12), pady=6)
        valid_from_entry = ttk.Entry(info, textvariable=self.var_valid_from, font=("Segoe UI", 11))
        valid_from_entry.grid(row=5, column=3, sticky="ew", padx=(0, 12), pady=6)

        # Satır 6: Geçerlilik Bitiş, Not
        valid_to_emoji = ttk.Label(info, text="📅", style="Emoji.TLabel")
        valid_to_emoji.grid(row=6, column=0, sticky="w", padx=12, pady=6)
        valid_to_label = ttk.Label(info, text=" Geçerlilik Bitiş", style="Field.TLabel")
        valid_to_label.grid(row=6, column=0, sticky="w", padx=(35, 12), pady=6)
        valid_to_entry = ttk.Entry(info, textvariable=self.var_valid_to, font=("Segoe UI", 11))
        valid_to_entry.grid(row=6, column=1, sticky="ew", padx=(0, 12), pady=6)
        
        note_emoji = ttk.Label(info, text="🗒️", style="Emoji.TLabel")
        note_emoji.grid(row=6, column=2, sticky="w", padx=12, pady=6)
        note_label = ttk.Label(info, text=" Not", style="Field.TLabel")
        note_label.grid(row=6, column=2, sticky="w", padx=(35, 12), pady=6)
        note_entry = ttk.Entry(info, textvariable=self.var_note, font=("Segoe UI", 11))
        note_entry.grid(row=6, column=3, sticky="ew", padx=(0, 12), pady=6)

        # İzinler bloğu
        opts = ttk.LabelFrame(frm, text="🔐 İzinler", style="Card.TLabelframe")
        opts.grid(row=2, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 12))
        for c in range(3):
            opts.columnconfigure(c, weight=1)
        
        # Renkli checkbox'lar
        auto_emoji = ttk.Label(opts, text="🚪", style="Emoji.TLabel")
        auto_emoji.grid(row=0, column=0, sticky="w", padx=12, pady=8)
        auto_cb = ttk.Checkbutton(opts, text=" Otomatik kapı aç", variable=self.var_allow_auto)
        auto_cb.grid(row=0, column=0, sticky="w", padx=(35, 12), pady=8)
        
        in_emoji = ttk.Label(opts, text="⬅️", style="Emoji.TLabel")
        in_emoji.grid(row=0, column=1, sticky="w", padx=12, pady=8)
        in_cb = ttk.Checkbutton(opts, text=" Girişe izin", variable=self.var_allow_in)
        in_cb.grid(row=0, column=1, sticky="w", padx=(35, 12), pady=8)
        
        out_emoji = ttk.Label(opts, text="➡️", style="Emoji.TLabel")
        out_emoji.grid(row=0, column=2, sticky="w", padx=12, pady=8)
        out_cb = ttk.Checkbutton(opts, text=" Çıkışa izin", variable=self.var_allow_out)
        out_cb.grid(row=0, column=2, sticky="w", padx=(35, 12), pady=8)

        # Alt butonlar
        actions = ttk.Frame(frm)
        actions.grid(row=3, column=0, columnspan=4, sticky="e", padx=12, pady=(0, 12))
        
        ttk.Button(actions, text="🗑️ Temizle", style="TButton", command=self._clear_form).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="💾 Kaydet", style="Success.TButton", command=self.on_save_plate).grid(row=0, column=1)

    def _build_passes(self):
        frm = self.page_passes
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)
        
        # Başlık alanı
        header_frame = ttk.Frame(frm)
        header_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header_frame.columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="🚗 Okunan Geçişler", font=("Segoe UI", 14, "bold"), foreground="#e5e7eb").grid(row=0, column=0, sticky="w")
        ttk.Button(header_frame, text="🔄 Yenile", style="Accent.TButton", command=self.refresh_passes).grid(row=0, column=2, sticky="e", padx=(10, 0))
        
        # Gelişmiş Treeview stili
        try:
            self.style.configure("Pass.Treeview", 
                background="#1a2238", 
                fieldbackground="#1a2238", 
                foreground="#f3f4f6", 
                rowheight=32, 
                borderwidth=0,
                font=("Segoe UI", 10))
            self.style.configure("Pass.Treeview.Heading", 
                background="#0f172a", 
                foreground="#60a5fa", 
                borderwidth=0,
                font=("Segoe UI", 11, "bold"))
            self.style.map("Pass.Treeview", 
                background=[("selected", "#2563eb")],
                foreground=[("selected", "#ffffff")])
        except Exception:
            pass
            
        cols = ("time", "plate", "registered", "source")
        self.tree_pass = ttk.Treeview(frm, columns=cols, show="headings", style="Pass.Treeview")
        
        # Emoji başlıklar ve genişlikler
        self.tree_pass.heading("time", text="🕒 Tarih/Saat")
        self.tree_pass.heading("plate", text="🚗 Plaka")
        self.tree_pass.heading("registered", text="📋 Durum")
        self.tree_pass.heading("source", text="↔️ Yön")
        
        self.tree_pass.column("time", width=180)
        self.tree_pass.column("plate", width=140, anchor="center")
        self.tree_pass.column("registered", width=120, anchor="center")
        self.tree_pass.column("source", width=100, anchor="center")
        
        self.tree_pass.grid(row=1, column=0, sticky="nsew", padx=12)
        
        # Scrollbar
        sc = ttk.Scrollbar(frm, orient="vertical", command=self.tree_pass.yview)
        sc.grid(row=1, column=1, sticky="ns")
        self.tree_pass.configure(yscrollcommand=sc.set)
        
        # Renkli tag'ler
        try:
            self.tree_pass.tag_configure("registered", foreground="#10b981", font=("Segoe UI", 10, "bold"))  # Yeşil
            self.tree_pass.tag_configure("unregistered", foreground="#ef4444", font=("Segoe UI", 10, "bold"))  # Kırmızı
            self.tree_pass.tag_configure("entry", foreground="#3b82f6", font=("Segoe UI", 10, "bold"))  # Mavi
            self.tree_pass.tag_configure("exit", foreground="#f59e0b", font=("Segoe UI", 10, "bold"))  # Turuncu
            self.tree_pass.tag_configure("alt", background="#0b1222")  # Alternatif satır
        except Exception:
            pass
            
        self.refresh_passes()

    def _log(self, msg: str):
        """Hem durum çubuğuna, hem de terminale mesaj yaz."""
        try:
            self.status_var.set(msg)
            self.update_idletasks()
        except Exception:
            pass
        try:
            print(f"[LOG] {msg}")
        except Exception:
            pass

    def refresh_plate_list(self):
        """Detaylı plaka listesini yenile - tüm bilgiler ve izin durumu"""
        from datetime import datetime
        
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for idx, row in enumerate(self.store.list_plates()):
            plate = row.get("plate", "")
            owner = row.get("owner", "")
            phone = row.get("phone", "")
            block = row.get("block", "")
            flat = row.get("flat", "")
            brand = row.get("brand", "")
            model = row.get("model", "")
            color = row.get("color", "")
            rfid = row.get("rfid", "")
            valid_from = row.get("valid_from", "")
            valid_to = row.get("valid_to", "")
            note = row.get("note", "")
            
            # İzin bilgileri
            allow_auto = row.get("allow_auto", True)
            allow_in = row.get("allow_in", True)
            allow_out = row.get("allow_out", True)
            
            # Veri durumunu kontrol et
            tags = []
            if idx % 2 == 1:
                tags.append("alt")
                
            # Geçerlilik kontrolü
            is_expired = False
            if valid_to:
                try:
                    exp_date = datetime.strptime(valid_to, "%Y-%m-%d")
                    if datetime.now() > exp_date:
                        is_expired = True
                        tags.append("expired")
                except Exception:
                    pass
            
            # Değerleri hazırla
            plate_val = plate if plate else "—"
            owner_val = owner if owner else "—"
            phone_val = phone if phone else "—"
            block_val = block if block else "—"
            flat_val = flat if flat else "—"
            brand_val = brand if brand else "—"
            model_val = model if model else "—"
            color_val = color if color else "—"
            rfid_val = rfid if rfid else "—"
            valid_from_val = valid_from if valid_from else "—"
            valid_to_val = valid_to if valid_from else "—"
            note_val = note if note else "—"
            
            # İzin durumları
            auto_gate_val = "✅ Evet" if allow_auto else "❌ Hayır"
            entry_perm_val = "✅ Evet" if allow_in else "❌ Hayır"
            exit_perm_val = "✅ Evet" if allow_out else "❌ Hayır"
            
            # Tag'leri ekle
            if plate:
                tags.append("has_data")
            if not allow_auto or not allow_in or not allow_out:
                tags.append("permission_no")
            else:
                tags.append("permission_yes")
                
            values = (
                plate_val, owner_val, phone_val, block_val, flat_val,
                brand_val, model_val, color_val, rfid_val,
                valid_from_val, valid_to_val,
                auto_gate_val, entry_perm_val, exit_perm_val, note_val
            )
            
            self.tree.insert("", "end", values=values, tags=tuple(tags) if tags else None)

    def refresh_passes(self):
        """Geçiş listesini yenile - en son geçiş en üstte"""
        for item in self.tree_pass.get_children():
            self.tree_pass.delete(item)
            
        passes = self.store.list_passes()
        # Ters sırala - en son geçiş en üstte
        passes.reverse()
        
        for idx, row in enumerate(passes):
            time_disp = f"🕒 {row.get('time', '')}"
            plate_disp = f"🚗 {row.get('plate', '')}"
            reg_ok = bool(row.get("registered"))
            
            if reg_ok:
                reg_disp = "✅ Kayıtlı"
                reg_tag = "registered"
            else:
                reg_disp = "🚫 Kayıtsız"
                reg_tag = "unregistered"
                
            src_raw = (row.get("source", "") or "").lower()
            if src_raw in ("cam1", "giris", "entry", "in", "inbound"):
                src_disp = "⬅️ Giriş"
                src_tag = "entry"
            elif src_raw in ("cam2", "cikis", "exit", "out", "outbound"):
                src_disp = "➡️ Çıkış"
                src_tag = "exit"
            else:
                src_disp = "❓"
                src_tag = ""
                
            # Tag'leri birleştir
            tags = [reg_tag, src_tag]
            if idx % 2 == 1:
                tags.append("alt")
                
            self.tree_pass.insert("", "end", values=(time_disp, plate_disp, reg_disp, src_disp), tags=tuple(tags))

    def _update_last_info(self, source: str, plate: str):
        src = (source or "").lower()
        direction = "Giriş" if src in ("cam1", "giris", "entry", "in", "inbound") else "Çıkış"
        reg = self.store.is_registered(plate)
        owner = self.store.get_owner_by_plate(plate) if reg else ""
        
        if reg and owner:
            text = f"🟢 {direction}: {plate} — 👤 {owner}"
            color = "#10b981"  # Yeşil
        elif reg:
            text = f"🟢 {direction}: {plate} — 👤 Kayıtlı"
            color = "#10b981"  # Yeşil
        else:
            text = f"🔴 {direction}: {plate} — 🚫 Kayıtsız"
            color = "#ef4444"  # Kırmızı
            
        if src in ("cam1", "giris", "entry", "in", "inbound"):
            self.var_cam1_last.set(text)
            self.lbl_cam1_last.configure(foreground=color)
        else:
            self.var_cam2_last.set(text)
            self.lbl_cam2_last.configure(foreground=color)

    def on_save_cam_settings(self):
        ok = self.store.update_settings({
            "camera1": self.var_cam1.get().strip(),
            "camera2": self.var_cam2.get().strip(),
            "yolo_model_path": self.var_yolo.get().strip(),
        })
        self._log("Kamera ayarları kaydedildi." if ok else "Kamera ayarları kaydedilemedi.")
        if ok:
            messagebox.showinfo("Bilgi", "Kamera ayarları kaydedildi.")
        else:
            messagebox.showerror("Hata", "Kamera ayarları kaydedilemedi.")

    def on_edit_plate(self):
        """Seçili plakayı düzenle"""
        try:
            selection = self.tree.selection()
            if not selection:
                messagebox.showwarning("Uyarı", "Lütfen düzenlenecek bir plaka seçin.")
                return
                
            item = selection[0]
            values = self.tree.item(item, "values")
            if not values:
                return
                
            plate = values[0]
            if plate == "—":
                messagebox.showwarning("Uyarı", "Geçersiz plaka seçildi.")
                return
                
            # Mevcut veriyi getir
            plates = self.store.list_plates()
            plate_data = None
            for p in plates:
                if p.get("plate", "") == plate:
                    plate_data = p
                    break
                    
            if not plate_data:
                messagebox.showerror("Hata", "Plaka bulunamadı.")
                return
                
            # Form alanlarını doldur
            self.var_plate.set(plate_data.get("plate", ""))
            self.var_owner.set(plate_data.get("owner", ""))
            self.var_phone.set(plate_data.get("phone", ""))
            self.var_block.set(plate_data.get("block", ""))
            self.var_flat.set(plate_data.get("flat", ""))
            self.var_brand.set(plate_data.get("brand", ""))
            self.var_model.set(plate_data.get("model", ""))
            self.var_color.set(plate_data.get("color", ""))
            self.var_rfid.set(plate_data.get("rfid", ""))
            self.var_valid_from.set(plate_data.get("valid_from", ""))
            self.var_valid_to.set(plate_data.get("valid_to", ""))
            self.var_note.set(plate_data.get("note", ""))
            self.var_allow_auto.set(plate_data.get("allow_auto", True))
            self.var_allow_in.set(plate_data.get("allow_in", True))
            self.var_allow_out.set(plate_data.get("allow_out", True))
            
            # Form sayfasına geç
            self.notebook.select(1)  # Form sayfası
            
            messagebox.showinfo("Bilgi", f"'{plate}' plakası form'a yüklendi. Düzenlemeyi tamamlayıp kaydedin.")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Plaka düzenlenirken hata: {e}")
    
    def on_delete_plate(self):
        """Seçili plakayı sil"""
        try:
            selection = self.tree.selection()
            if not selection:
                messagebox.showwarning("Uyarı", "Lütfen silinecek bir plaka seçin.")
                return
                
            item = selection[0]
            values = self.tree.item(item, "values")
            if not values:
                return
                
            plate = values[0]
            if plate == "—":
                messagebox.showwarning("Uyarı", "Geçersiz plaka seçildi.")
                return
                
            # Onay al
            result = messagebox.askyesno("Onay", f"'{plate}' plakasını silmek istediğinizden emin misiniz?\n\nBu işlem geri alınamaz!")
            if not result:
                return
                
            # Plakayı sil
            plates = self.store.data.get("plates", [])
            success = False
            for i, p in enumerate(plates):
                if p.get("plate", "") == plate:
                    # Store'dan sil
                    del plates[i]
                    success = self.store.save()
                    break
            
            if success:
                self._log(f"'{plate}' plakası silindi.")
                messagebox.showinfo("Başarılı", f"'{plate}' plakası başarıyla silindi.")
                self.refresh_plate_list()  # Listeyi yenile
            else:
                messagebox.showerror("Hata", "Plaka silinirken bir hata oluştu.")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Plaka silinirken hata: {e}")
    
    def on_save_gate_settings(self):
        ok = self.store.update_settings({
            "relay_ip": self.var_ip.get().strip(),
            "relay_port": int(self.var_port.get() or 1590),
            "open_ms": int(self.var_open_ms.get() or 1000),
            "relay_command_open": self.var_relay_cmd.get().strip() or "10",
        })
        self._log("Kapı ayarları kaydedildi." if ok else "Kapı ayarları kaydedilemedi.")
        if ok:
            messagebox.showinfo("Bilgi", "Kapı ayarları kaydedildi.")
        else:
            messagebox.showerror("Hata", "Kapı ayarları kaydedilemedi.")

    def on_save_plate(self):
        plate = self.var_plate.get().strip().upper()
        if not plate:
            messagebox.showwarning("Uyarı", "Plaka boş olamaz.")
            return
        extra = {
            "phone": self.var_phone.get().strip(),
            "block": self.var_block.get().strip(),
            "flat": self.var_flat.get().strip(),
            "brand": self.var_brand.get().strip(),
            "model": self.var_model.get().strip(),
            "color": self.var_color.get().strip(),
            "rfid": self.var_rfid.get().strip(),
            "valid_from": self.var_valid_from.get().strip(),
            "valid_to": self.var_valid_to.get().strip(),
            "allow_auto": bool(self.var_allow_auto.get()),
            "allow_in": bool(self.var_allow_in.get()),
            "allow_out": bool(self.var_allow_out.get()),
        }
        ok = self.store.add_plate(plate, self.var_owner.get(), self.var_note.get(), extra=extra)
        if ok:
            self.var_plate.set("")
            self.var_owner.set("")
            self.var_phone.set("")
            self.var_block.set("")
            self.var_flat.set("")
            self.var_brand.set("")
            self.var_model.set("")
            self.var_color.set("")
            self.var_rfid.set("")
            self.var_valid_from.set("")
            self.var_valid_to.set("")
            self.var_note.set("")
            self.var_allow_auto.set(True)
            self.var_allow_in.set(True)
            self.var_allow_out.set(True)
            self._log("Plaka kaydedildi.")
            messagebox.showinfo("Bilgi", "Plaka kaydedildi.")
        else:
            messagebox.showerror("Hata", "Plaka kaydedilemedi.")

    def _clear_form(self):
        self.var_plate.set("")
        self.var_owner.set("")
        self.var_phone.set("")
        self.var_block.set("")
        self.var_flat.set("")
        self.var_brand.set("")
        self.var_model.set("")
        self.var_color.set("")
        self.var_rfid.set("")
        self.var_valid_from.set("")
        self.var_valid_to.set("")
        self.var_note.set("")
        self.var_allow_auto.set(True)
        self.var_allow_in.set(True)
        self.var_allow_out.set(True)

    def on_cam1_start(self):
        if self._cam1_running:
            return  # Log yazma, sadece sessizce çık
        url = self.store.get_settings().get("camera1", "")
        if not url:
            messagebox.showwarning("Uyarı", "Kamera 1 URL ayarlarda boş.")
            return
        if cv2 is None:
            messagebox.showerror("Hata", "OpenCV yüklü değil. Lütfen 'pip install opencv-python' kurun.")
            return
        self._log("Kamera 1 başlatılıyor...")
        # FFMPEG düşük gecikme ayarları
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0"
        # Backend seçimi ve buffer küçültme
        try:
            self._cam1_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        except Exception:
            self._cam1_cap = cv2.VideoCapture(url)
        try:
            self._cam1_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self._cam1_cap or not self._cam1_cap.isOpened():
            self._cam1_cap = None
            messagebox.showerror("Hata", "Kamera 1 açılamadı. URL veya ağ bağlantısını kontrol edin.")
            return
        self._cam1_running = True
        # Ayrı okuma thread'i (son frame stratejisi)
        self._cam1_thread = threading.Thread(target=self._cam1_reader_loop, daemon=True)
        self._cam1_thread.start()
        # Çizim döngüsü
        self._schedule_cam1_render()

    def on_cam1_stop(self):
        if not self._cam1_running:
            self._log("Kamera 1 zaten durdu.")
            return
        self._cam1_running = False
        time.sleep(0.05)
        try:
            if self._cam1_cap is not None:
                self._cam1_cap.release()
        finally:
            self._cam1_cap = None
        self._log("Kamera 1 durduruldu.")
        if self._cam1_display_job is not None:
            try:
                self.after_cancel(self._cam1_display_job)
            except Exception:
                pass
            self._cam1_display_job = None
        self._cam1_latest = None

    def _cam1_reader_loop(self):
        # Kamera'dan sürekli oku, sadece son frame'i tut
        while self._cam1_running and self._cam1_cap is not None:
            ok, frame = self._cam1_cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            self._cam1_latest = frame
            # Kuyruğu şişirmemek için bekleme yok; son frame politikası

    def _schedule_cam1_render(self):
        if not self._cam1_running:
            return
        try:
            frame = self._cam1_latest
            if frame is not None and cv2 is not None:
                # Ölçekle ve çiz
                cw = max(self.canvas_cam1.winfo_width(), 10)
                ch = max(self.canvas_cam1.winfo_height(), 10)
                fh, fw = frame.shape[:2]
                scale = min(cw / max(fw, 1), ch / max(fh, 1))
                nw, nh = max(int(fw * scale), 1), max(int(fh * scale), 1)
                if nw != fw or nh != fh:
                    frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame
                # ROI dikdörtgenini mavi çiz (görünürlük açıksa)
                if self._roi_visible_cam1:
                    rect = self._get_roi_rect_for_cam('cam1')
                    if rect is not None:
                        try:
                            rx1 = int(rect[0] * nw)
                            ry1 = int(rect[1] * nh)
                            rx2 = int(rect[2] * nw)
                            ry2 = int(rect[3] * nh)
                            if rx2 > rx1 and ry2 > ry1:
                                if getattr(self, '_show_boxes', False):
                                    cv2.rectangle(frame_resized, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                        except Exception:
                            pass
                if PIL_AVAILABLE:
                    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    photo = ImageTk.PhotoImage(image=img)
                else:
                    ok2, buf = cv2.imencode('.png', frame_resized)
                    if ok2:
                        b64 = base64.b64encode(buf)
                        photo = tk.PhotoImage(data=b64)
                    else:
                        photo = None
                if photo is not None:
                    self._update_cam1_image(photo)
        except Exception as e:
            # Hata durumunda loglama ama çökme
            pass
        # 30-40 fps hedefle (yaklaşık 25-33 ms)
        if self._cam1_running:  # Tekrar kontrol et
            self._cam1_display_job = self.after(25, self._schedule_cam1_render)

    def _update_cam1_image(self, photo: tk.PhotoImage):
        self._cam1_photo = photo
        self.canvas_cam1.delete("all")
        cw = self.canvas_cam1.winfo_width()
        ch = self.canvas_cam1.winfo_height()
        self.canvas_cam1.create_image(cw // 2, ch // 2, image=self._cam1_photo)

    def on_cam2_start(self):
        if self._cam2_running:
            self._log("Kamera 2 zaten çalışıyor.")
            return
        url = self.store.get_settings().get("camera2", "")
        if not url:
            messagebox.showwarning("Uyarı", "Kamera 2 URL ayarlarda boş.")
            return
        if cv2 is None:
            messagebox.showerror("Hata", "OpenCV yüklü değil. Lütfen 'pip install opencv-python' kurun.")
            return
        self._log("Kamera 2 başlatılıyor...")
        # FFMPEG düşük gecikme ayarları (aynı global seçenekleri kullan)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;0"
        try:
            self._cam2_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        except Exception:
            self._cam2_cap = cv2.VideoCapture(url)
        try:
            self._cam2_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self._cam2_cap or not self._cam2_cap.isOpened():
            self._cam2_cap = None
            messagebox.showerror("Hata", "Kamera 2 açılamadı. URL veya ağ bağlantısını kontrol edin.")
            return
        self._cam2_running = True
        self._cam2_thread = threading.Thread(target=self._cam2_reader_loop, daemon=True)
        self._cam2_thread.start()
        self._schedule_cam2_render()

    def on_cam2_stop(self):
        if not self._cam2_running:
            self._log("Kamera 2 zaten durdu.")
            return
        self._cam2_running = False
        time.sleep(0.05)
        try:
            if self._cam2_cap is not None:
                self._cam2_cap.release()
        finally:
            self._cam2_cap = None
        self._log("Kamera 2 durduruldu.")
        if self._cam2_display_job is not None:
            try:
                self.after_cancel(self._cam2_display_job)
            except Exception:
                pass
            self._cam2_display_job = None
        self._cam2_latest = None

    def _cam2_reader_loop(self):
        # Kamera 2'den sürekli oku, sadece son frame'i tut
        while self._cam2_running and self._cam2_cap is not None:
            ok, frame = self._cam2_cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            self._cam2_latest = frame

    def _schedule_cam2_render(self):
        if not self._cam2_running:
            return
        frame = self._cam2_latest
        if frame is not None and cv2 is not None:
            try:
                cw = max(self.canvas_cam2.winfo_width(), 10)
                ch = max(self.canvas_cam2.winfo_height(), 10)
                fh, fw = frame.shape[:2]
                scale = min(cw / max(fw, 1), ch / max(fh, 1))
                nw, nh = max(int(fw * scale), 1), max(int(fh * scale), 1)
                if nw != fw or nh != fh:
                    frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame
                # ROI dikdörtgenini mavi çiz (görünürlük açıksa)
                if self._roi_visible_cam2:
                    rect = self._get_roi_rect_for_cam('cam2')
                    if rect is not None:
                        try:
                            rx1 = int(rect[0] * nw)
                            ry1 = int(rect[1] * nh)
                            rx2 = int(rect[2] * nw)
                            ry2 = int(rect[3] * nh)
                            if rx2 > rx1 and ry2 > ry1:
                                if getattr(self, '_show_boxes', False):
                                    cv2.rectangle(frame_resized, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                        except Exception:
                            pass
                if PIL_AVAILABLE:
                    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    photo = ImageTk.PhotoImage(image=img)
                else:
                    ok2, buf = cv2.imencode('.png', frame_resized)
                    if ok2:
                        b64 = base64.b64encode(buf)
                        photo = tk.PhotoImage(data=b64)
                    else:
                        photo = None
                if photo is not None:
                    self._update_cam2_image(photo)
            except Exception:
                pass
        self._cam2_display_job = self.after(25, self._schedule_cam2_render)

    def _update_cam2_image(self, photo: tk.PhotoImage):
        self._cam2_photo = photo
        self.canvas_cam2.delete("all")
        cw = self.canvas_cam2.winfo_width()
        ch = self.canvas_cam2.winfo_height()
        self.canvas_cam2.create_image(cw // 2, ch // 2, image=self._cam2_photo)

    def on_anpr_start(self):
        if self._anpr_running:
            return  # Sessizce çık, log yazma
        if not HAS_YOLO or not HAS_EASYOCR:
            messagebox.showerror("Hata", "ANPR için 'ultralytics' ve 'easyocr' paketleri gerekli.\nKurulum: pip install ultralytics easyocr")
            return
        if cv2 is None:
            messagebox.showerror("Hata", "OpenCV gerekli. 'pip install opencv-python'")
            return
        device = 'cuda' if (self._gpu_enabled and 'TORCH_HAS_CUDA' in globals() and TORCH_HAS_CUDA) else 'cpu'
        self._log(f"ANPR başlatılıyor... device={device}")
        self._anpr_running = True
        self._anpr_thread = threading.Thread(target=self._anpr_loop, daemon=True)
        self._anpr_thread.start()

    def on_anpr_stop(self):
        if not self._anpr_running:
            return  # Sessizce çık
        self._anpr_running = False
        self._log("ANPR durduruldu.")

    def _ensure_models(self):
        # YOLO modeli
        if self._yolo_model is None and HAS_YOLO:
            device = 'cuda' if (self._gpu_enabled and 'TORCH_HAS_CUDA' in globals() and TORCH_HAS_CUDA) else 'cpu'
            last_error = None
            # Önce ayarlardaki yerel .pt denenir
            local_path = (self.store.get_settings() or {}).get('yolo_model_path', '')
            candidates = []
            # Uygulama klasöründeki varsayılan dosya
            try:
                default_local = os.path.join(os.path.dirname(__file__), 'license_plate_detector.pt')
                if os.path.exists(default_local):
                    candidates.append(default_local)
            except Exception:
                pass
            if local_path:
                candidates.append(local_path)
            # Ardından bilinen ID'ler
            candidates.extend(['keremberke/yolov8m-license-plate', 'keremberke/yolov8n-license-plate'])
            for model_id in candidates:
                try:
                    self._log(f"YOLO modeli yükleniyor: {model_id} ({device})")
                    self._yolo_model = YOLO(model_id)
                    self._yolo_model.to(device)
                    self._log(f"YOLO modeli yüklendi: {model_id}")
                    break
                except Exception as e:
                    last_error = e
                    self._yolo_model = None
                    self._log(f"YOLO yüklenemedi: {model_id} -> {e}")
            if self._yolo_model is None and last_error is not None:
                messagebox.showerror("Model Hatası", f"YOLO modeli yüklenemedi. İnternet bağlantısını ve paketleri kontrol edin.\nHata: {last_error}")
        # EasyOCR
        if self._ocr_reader is None and HAS_EASYOCR:
            try:
                gpu_ok = bool(self._gpu_enabled and 'TORCH_HAS_CUDA' in globals() and TORCH_HAS_CUDA)
                self._log(f"EasyOCR Reader yükleniyor (gpu={gpu_ok})...")
                # FutureWarning bastır
                warnings.filterwarnings("ignore", category=FutureWarning, module=r"easyocr.*")
                self._ocr_reader = easyocr.Reader(['tr', 'en'], gpu=gpu_ok)
            except Exception as e:
                self._log(f"EasyOCR GPU ile yüklenemedi: {e}. CPU deneniyor...")
                try:
                    warnings.filterwarnings("ignore", category=FutureWarning, module=r"easyocr.*")
                    self._ocr_reader = easyocr.Reader(['tr', 'en'], gpu=False)
                    self._log("EasyOCR Reader CPU ile yüklendi.")
                except Exception as e2:
                    self._ocr_reader = None
                    messagebox.showerror("OCR Hatası", f"EasyOCR yüklenemedi. Hata: {e2}")

    def _normalize_plate(self, text: str) -> str:
        t = (text or '').upper()
        # Türkçe karakterleri dönüştür
        tr_map = str.maketrans({'Ç':'C','Ğ':'G','İ':'I','Ö':'O','Ş':'S','Ü':'U'})
        t = t.translate(tr_map)
        # Boşluk ve ayraçları kaldır
        for ch in [' ', '-', '_', '.', ':', '/']:
            t = t.replace(ch, '')
        # Sadece A-Z,0-9 tut
        t = ''.join(ch for ch in t if ('A' <= ch <= 'Z') or ('0' <= ch <= '9'))
        # Segment bazlı düzeltme: 2 digit + 1-3 letter + 2-4 digit
        import re
        m = re.match(r'^([0-9]{2})([A-Z]{1,3})([0-9]{2,4})$', t)
        if not m:
            # Kısmi eşleşmeler için kaba düzeltme
            # Harf->rakam (sadece rakam segmentinde varsayım yoksa genel):
            repl_num = str.maketrans({'O':'0','Q':'0','D':'0','S':'5','B':'8','Z':'2','I':'1','L':'1','G':'6','T':'7'})
            t = t.translate(repl_num)
            return t
        g1, g2, g3 = m.group(1), m.group(2), m.group(3)
        # Rakam segmentleri için harf->rakam düzeltmesi
        num_map = str.maketrans({'O':'0','Q':'0','D':'0','S':'5','B':'8','Z':'2','I':'1','L':'1','G':'6','T':'7'})
        # Harf segmenti için rakam->harf düzeltmesi
        let_map = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','8':'B'})
        g1 = g1.translate(num_map)
        g3 = g3.translate(num_map)
        g2 = g2.translate(let_map)
        return f"{g1}{g2}{g3}"

    def _coerce_plate_to_tr(self, text: str) -> str:
        """Ambiguity giderme: Konumsal kurala göre (2 rakam, 1-3 harf, 2-4 rakam) dönüştür.
        Geçerliyse dönüştürülmüş değeri, değilse girişi döndür."""
        import re
        t = (text or '').upper()
        # Hızlı çıkış
        if self._is_valid_tr_plate(t):
            return t
        # Temizle
        t = ''.join(ch for ch in t if ('A' <= ch <= 'Z') or ('0' <= ch <= '9'))
        if not (5 <= len(t) <= 9):
            return text or ''
        # Harita tabloları
        num_map = str.maketrans({'O':'0','Q':'0','D':'0','S':'5','B':'8','Z':'2','I':'1','L':'1','G':'6','T':'7'})
        let_map = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G','7':'T'})
        # İl kodu (2 char) -> rakam
        head = t[:2].translate(num_map)
        rest = t[2:]
        # Orta (1-3) harf; son rakam
        # 1-3 arasında farklı bölmeler deneyelim (en çok 3 deneme)
        for mid_len in (3, 2, 1):
            if len(rest) < (mid_len + 2):
                continue
            mid = rest[:mid_len].translate(let_map)
            tail = rest[mid_len:].translate(num_map)
            cand = f"{head}{mid}{tail}"
            if self._is_valid_tr_plate(cand):
                return cand
        return text or ''

    def _is_valid_tr_plate(self, plate: str) -> bool:
        import re
        # TR Plaka: 01-81 il kodu, 1-3 harf, 2-4 rakam
        return re.match(r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,4}$', plate) is not None

    def _select_plate_from_ocr(self, ocr_res):
        """Gelişmiş OCR sonuç seçimi - yanlış pozitifleri azalt"""
        import re
        candidates = []  # (plate, score, confidence)
        
        for item in ocr_res:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            
            # EasyOCR detay=1 ise (bbox, text, confidence)
            # detay=0 ise (text, confidence)
            if len(item) == 3 and isinstance(item[0], (list, tuple)):
                # detay=1 formatı
                _, text, score = item
            else:
                # detay=0 formatı veya diğer
                text, score = item[0], item[1] if len(item) >= 2 else (item[0], 0.0)
                
            if not text:
                continue
                
            try:
                score = float(score)
            except Exception:
                continue
                
            # Ayarlanabilir güven eşiği
            if score < self._ocr_confidence:
                continue
                
            plate0 = self._normalize_plate(text)
            if not plate0:
                continue
                
            # Ayarlanabilir uzunluk kontrolü
            if len(plate0) < self._min_char_count or len(plate0) > self._max_char_count:
                continue
                
            # TR plaka formatı kontrolü
            if not self._is_valid_tr_plate(plate0):
                continue
                
            # Skor hesapla - güven + format bonusu
            final_score = score
            if self._is_valid_tr_plate(plate0):
                final_score += 2.0  # Format bonusu
                
            candidates.append((plate0, final_score, score))
                
        # Adayları skora göre sırala
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # En yüksek skorlu adayı döndür
        if candidates:
            return candidates[0][0]
            
        return None

    def _enhance_plate_detection(self, frame, boxes):
        """Gelişmiş plaka tespit filtreleme - yanlış pozitifleri azalt"""
        enhanced_boxes = []
        frame_h, frame_w = frame.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Alan kontrolü
            area = (x2 - x1) * (y2 - y1)
            if area < self._min_plate_area or area > self._max_plate_area:
                continue
                
            # En-boy oranı kontrolü (plakalar genelde yataydır)
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            
            # Türk plakaları için ayarlanabilir en-boy oranları
            if not (self._min_aspect_ratio <= aspect_ratio <= self._max_aspect_ratio):
                continue
                
            # Konum kontrolü - plakalar genelde frame'in alt yarısında olur
            center_y = (y1 + y2) / 2
            if center_y < frame_h * 0.15:  # Frame'in en üst 15%'inde plaka olmaz
                continue
                
            # Çerçeve kenarından çok uzakta olmamalı
            margin = 20
            if x1 < margin or y1 < margin or x2 > frame_w - margin or y2 > frame_h - margin:
                # Eğer ROI içindeyse kenara yakın olabilir
                continue
                
            enhanced_boxes.append([x1, y1, x2, y2])
            
        # Overlapping box'ları temizle (gelişmiş NMS)
        if len(enhanced_boxes) > 1:
            enhanced_boxes = self._non_max_suppression(enhanced_boxes, 0.3)
            
        # En fazla 3 en iyi kutuyu al
        if len(enhanced_boxes) > self._max_detections:
            # Alanlarına göre sırala ve en büyüklerini al
            enhanced_boxes.sort(key=lambda b: (b[2]-b[1])*(b[3]-b[0]), reverse=True)
            enhanced_boxes = enhanced_boxes[:self._max_detections]
            
        return enhanced_boxes
    
    def _non_max_suppression(self, boxes, iou_threshold):
        """Basit Non-Maximum Suppression implementasyonu"""
        if not boxes:
            return []
            
        # Alanlarına göre sırala
        boxes = sorted(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
        keep = []
        
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            remaining = []
            for box in boxes:
                iou = self._calculate_iou(current, box)
                if iou < iou_threshold:
                    remaining.append(box)
            boxes = remaining
            
        return keep
    
    def _calculate_iou(self, box1, box2):
        """İki kutu arasında IoU hesapla"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1)

    def _vote_plate(self, plate: str, window_s: float = None, min_hits: int = None, confidence: float = 0.5) -> bool:
        """Gelişmiş oylama sistemi - güven skoruna göre dinamik oylama"""
        import time as _t
        if window_s is None:
            window_s = self._vote_window
        if min_hits is None:
            min_hits = self._min_votes
            
        # Yüksek güven skorlu plakalar için daha az oylama gereksinimi
        if confidence >= 0.9:
            min_hits = max(1, min_hits // 2)  # Yüksek güven için yarı oylama
            window_s = max(1.0, window_s * 0.7)  # Daha kısa zaman penceresi
        elif confidence >= 0.7:
            min_hits = max(2, min_hits - 1)  # Orta güven için 1 az oylama
            window_s = max(1.5, window_s * 0.85)  # Biraz daha kısa zaman penceresi
            
        now = _t.time()
        dq = self._votes.get(plate, [])
        dq.append(now)
        # pencene dışını temizle
        dq = [t for t in dq if now - t <= window_s]
        self._votes[plate] = dq
        
        vote_result = len(dq) >= min_hits
        if getattr(self, '_verbose', False):
            self._log(f"🗳️ [VOTE-DEBUG] Plaka: '{plate}', Güven: {confidence:.3f}, Oylar: {len(dq)}/{min_hits}, Pencere: {window_s:.1f}s, Sonuç: {vote_result}")
        
        return vote_result

    def _should_emit(self, plate: str, interval_s: float = 8.0, confidence: float = 0.5) -> bool:
        """Plaka yayını kontrolü - güven skoruna göre dinamik aralık"""
        import time as _t
        now = _t.time()
        
        # Yüksek güven skorlu plakalar için daha kısa yayın aralığı
        if confidence >= 0.9:
            interval_s = max(2.0, interval_s * 0.3)  # %70 daha kısa
        elif confidence >= 0.7:
            interval_s = max(3.0, interval_s * 0.5)  # %50 daha kısa
        elif confidence >= 0.5:
            interval_s = max(4.0, interval_s * 0.7)  # %30 daha kısa
            
        last = self._anpr_last_emit.get(plate)
        if last is None or now - last >= interval_s:
            self._anpr_last_emit[plate] = now
            if getattr(self, '_verbose', False):
                self._log(f"📢 [EMIT-DEBUG] Plaka: '{plate}', Güven: {confidence:.3f}, Aralık: {interval_s:.1f}s, İzin: VERİLDİ")
            return True
        
        if getattr(self, '_verbose', False):
            self._log(f"📢 [EMIT-DEBUG] Plaka: '{plate}', Güven: {confidence:.3f}, Aralık: {interval_s:.1f}s, İzin: REDDEDİ (son: {now-last:.1f}s)")
        return False

    def _anpr_loop(self):
        self._ensure_models()
        if self._yolo_model is None or self._ocr_reader is None:
            self._log("ANPR modelleri yüklenemedi. Ayrıntılar yukarıdaki durum mesajlarında.")
            self._anpr_running = False
            return
        device = 'cuda' if self._gpu_enabled else 'cpu'
        while self._anpr_running:
            frame1 = self._cam1_latest
            frame2 = getattr(self, "_cam2_latest", None)
            # İşlenecek frame listesi: (frame, kaynak-etiketi)
            tasks = []
            if frame1 is not None:
                tasks.append((frame1, 'cam1'))
            if frame2 is not None:
                tasks.append((frame2, 'cam2'))
            if not tasks:
                self._log("ANPR bekliyor: Kamera akışı yok. Lütfen en az bir kamerayı başlatın.")
                time.sleep(0.2)  # Daha kısa bekleme
                continue
            try:
                # Her aktif kamera için tespit + OCR
                for frame, src in tasks:
                    # ROI dikdörtgenini piksel cinsine çevir
                    roi_rect = self._get_roi_rect_for_cam(src)
                    roi_px = None
                    if roi_rect is not None:
                        fh, fw = frame.shape[:2]
                        x1n, y1n, x2n, y2n = roi_rect
                        rx1 = max(int(x1n * fw), 0)
                        ry1 = max(int(y1n * fh), 0)
                        rx2 = min(int(x2n * fw), fw - 1)
                        ry2 = min(int(y2n * fh), fh - 1)
                        if rx2 > rx1 and ry2 > ry1:
                            roi_px = (rx1, ry1, rx2, ry2)
                    # Tespit - gelişmiş parametrelerle
                    res = self._yolo_model.predict(
                        source=frame, 
                        conf=getattr(self, '_conf_thres', 0.25), 
                        iou=getattr(self, '_iou_thres', 0.45),
                        max_det=getattr(self, '_max_detections', 5),
                        imgsz=getattr(self, '_imgsz', 1024), 
                        device=device, 
                        verbose=False,
                        half=getattr(self, '_half_precision', True) if device == 'cuda' else False
                    )
                    detections = []
                    boxes = []
                    for r in res:
                        if r.boxes is None:
                            continue
                        for b in r.boxes.xyxy.tolist():
                            boxes.append(b)
                            detections.append(b)
                    
                    # Gelişmiş kutu filtreleme
                    boxes = self._enhance_plate_detection(frame, boxes)
                    
                    # OCR - Gelişmiş ön işleme ile
                    for (x1, y1, x2, y2) in boxes:
                        # Eğer ROI tanımlıysa, kutu merkezinin ROI içinde olup olmadığını kontrol et
                        if roi_px is not None:
                            rx1, ry1, rx2, ry2 = roi_px
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                                continue
                        x1 = max(int(x1), 0); y1 = max(int(y1), 0)
                        x2 = min(int(x2), frame.shape[1]-1); y2 = min(int(y2), frame.shape[0]-1)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        roi = frame[y1:y2, x1:x2]
                        # ROI boyutu çok küçükse (çok az piksel) OCR yapma
                        h_roi, w_roi = roi.shape[:2]
                        if h_roi < 30 or w_roi < 90:  # Daha katı boyut kontrolü
                            continue
                        # Sabit bir hedef yüksekliğe normalize et (en-boy oranını koru)
                        target_h = self._roi_height
                        scale = target_h / float(max(h_roi, 1))
                        new_w = max(int(w_roi * scale), 1)
                        roi_resized = cv2.resize(roi, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
                        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=self._clahe_clip_limit, tileGridSize=(8,8))
                        gray = clahe.apply(gray)
                        # Gelişmiş ön işleme pipeline
                        processed_images = self._enhanced_preprocessing(gray)
                        
                        if getattr(self, '_verbose', False):
                            self._log(f"🔍 [DEBUG] ROI işlendi: {len(processed_images)} ön işleme görüntüsü")
                        
                        ocr_res = []
                        allowlist = getattr(self, '_allowlist', "ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
                        
                        # Çoklu ölçek ve ön işleme ile OCR
                        for i, proc_img in enumerate(processed_images):
                            try:
                                img_ocr_res = self._ocr_reader.readtext(proc_img, allowlist=allowlist, detail=1)
                                ocr_res.extend(img_ocr_res)
                                if getattr(self, '_verbose', False):
                                    self._log(f"📝 [OCR] Görüntü {i+1}: {len(img_ocr_res)} sonuç bulundu")
                                    for j, item in enumerate(img_ocr_res):
                                        try:
                                            text = item[1] if isinstance(item[1], str) else str(item[0])
                                            conf = float(item[2]) if len(item) >= 3 else 0.0
                                            self._log(f"📝 [OCR-{i+1}-{j+1}] Text: '{text}', Conf: {conf:.3f}")
                                        except Exception:
                                            pass
                            except Exception as e:
                                if getattr(self, '_verbose', False):
                                    self._log(f"❌ [OCR] Görüntü {i+1} hatası: {e}")
                        
                        if getattr(self, '_verbose', False):
                            self._log(f"🔍 [DEBUG] Toplam OCR sonuçları: {len(ocr_res)}")
                        
                        # Gelişmiş plaka seçimi
                        plate = self._select_plate_from_ocr(ocr_res)
                        
                        if getattr(self, '_verbose', False):
                            self._log(f"🎯 [SELECT] Seçilen plaka: '{plate}'")
                        
                        # Sadece çok yüksek güven skorlu plakaları işle
                        if plate and self._is_valid_tr_plate(plate):
                            if getattr(self, '_verbose', False):
                                self._log(f"✅ [VALID] Plaka geçerli: '{plate}'")
                            
                            # Ek doğrulama: OCR güven skoru kontrolü
                            max_confidence = 0.0
                            best_item = None
                            for item in ocr_res:
                                try:
                                    if len(item) >= 3:
                                        text = item[1] if isinstance(item[1], str) else str(item[0])
                                        conf = float(item[2]) if isinstance(item[1], str) else float(item[1])
                                        if conf > max_confidence:
                                            max_confidence = conf
                                            best_item = item
                                except Exception:
                                    continue
                            
                            if getattr(self, '_verbose', False):
                                self._log(f"📊 [CONF] En yüksek güven skoru: {max_confidence:.3f}")
                            
                            # Sadece yüksek güven skorlu plakaları oylamaya gönder
                            ocr_threshold = getattr(self, '_ocr_confidence', 0.35)
                            if max_confidence >= ocr_threshold:
                                if getattr(self, '_verbose', False):
                                    self._log(f"✅ [THRESHOLD] OCR güven skoru yeterli: {max_confidence:.3f} >= {ocr_threshold}")
                                
                                # Oylama süreci
                                if self._vote_plate(plate, confidence=max_confidence):
                                    if getattr(self, '_verbose', False):
                                        self._log(f"🗳️ [VOTE] Plaka oylamadan geçti: '{plate}'")
                                    
                                    # Yakın zamanda aynı plakayı işlemiş miyiz? (global soğuma ve ardışık kopya bastırma)
                                    import time as _t
                                    now = _t.time()
                                    cooldown_s = getattr(self, '_plate_cooldown_s', 2.0)
                                    
                                    if self._last_confirmed_plate == plate and (now - self._last_confirmed_ts) < cooldown_s:
                                        if getattr(self, '_verbose', False):
                                            self._log(f"❄️ [COOLDOWN] Plaka soğumada: '{plate}' ({cooldown_s:.1f}s)")
                                        continue
                                    
                                    if not self._should_emit(plate, interval_s=cooldown_s, confidence=max_confidence):
                                        if getattr(self, '_verbose', False):
                                            self._log(f"⏱️ [EMIT] Plaka yayın aralığında: '{plate}'")
                                        continue
                                    
                                    if getattr(self, '_verbose', False):
                                        self._log(f"📢 [EMIT] Plaka yayınlanıyor: '{plate}'")
                                    
                                    from datetime import datetime
                                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    # Son kayıtla aynıysa ekleme (ardışık kopya) – datastore üzerinden kontrol
                                    last_list = self.store.list_passes()
                                    if last_list and last_list[-1].get('plate') == plate:
                                        if getattr(self, '_verbose', False):
                                            self._log(f"🚫 [DUPLICATE] Ardışık kopya engellendi: '{plate}'")
                                        continue
                                    
                                    self.store.add_pass(plate, ts, source=src)
                                    self._last_confirmed_plate = plate
                                    self.after(0, lambda p=plate, s=src: self._update_last_info(s, p))
                                    self._last_confirmed_ts = now
                                    
                                    if getattr(self, '_verbose', False):
                                        self._log(f"✅ [SUCCESS] Plaka başarıyla kaydedildi: '{plate}'")
                                    
                                    # Otomatik kapı açma kuralı: sadece giriş izni OLAN plakalar için kapıyı aç
                                    try:
                                        auto_open_enabled = self.store.get_settings().get('auto_open', True)
                                        if getattr(self, '_verbose', False):
                                            self._log(f"🔧 [AUTO-OPEN] Auto-open ayarı: {auto_open_enabled}")
                                        
                                        if auto_open_enabled:
                                            # Plaka izinlerini kontrol et
                                            plate_data = None
                                            plates = self.store.data.get("plates", [])
                                            if getattr(self, '_verbose', False):
                                                self._log(f"🔍 [AUTO-OPEN] Kayıtlı plakalar: {len(plates)} adet")
                                            
                                            for p in plates:
                                                if p.get("plate", "") == plate:
                                                    plate_data = p
                                                    if getattr(self, '_verbose', False):
                                                        self._log(f"✅ [AUTO-OPEN] Plaka bulundu: '{plate}' - İzin: {p.get('allow_in', True)}")
                                                    break
                                            
                                            if plate_data:
                                                # Kayıtlı plaka - izinleri kontrol et
                                                allow_in = plate_data.get("allow_in", True)
                                                if getattr(self, '_verbose', False):
                                                    self._log(f"🔑 [AUTO-OPEN] Kayıtlı plaka izni: '{plate}' -> {allow_in}")
                                                
                                                if allow_in:
                                                    # Giriş izni var - kapıyı aç
                                                    if getattr(self, '_verbose', False):
                                                        self._log(f"🚪 [AUTO-OPEN] Kapı açılıyor (izinli plaka): '{plate}'")
                                                    self.after(0, self.on_gate_open)
                                                else:
                                                    # Giriş izni yok - uyarı ver
                                                    if getattr(self, '_verbose', False):
                                                        self._log(f"🚫 [AUTO-OPEN] Kapı açılmıyor (izinsiz plaka): '{plate}'")
                                                    self.after(0, lambda p=plate: self._show_permission_warning(p, "Giriş izni yok"))
                                            else:
                                                # Kayıtsız plaka - kapıyı aç (misafir araç)
                                                if getattr(self, '_verbose', False):
                                                    self._log(f"👤 [AUTO-OPEN] Kapı açılıyor (kayıtsız/misafir plaka): '{plate}'")
                                                self.after(0, self.on_gate_open)
                                        else:
                                            if getattr(self, '_verbose', False):
                                                self._log(f"❌ [AUTO-OPEN] Auto-open kapalı: '{plate}'")
                                    except Exception as e:
                                        self._log(f"❌ [AUTO-OPEN] Otomatik kapı açma hatası: {e}")
                                    # UI'yi güncelle
                                    self.after(0, self.refresh_passes)
                                else:
                                    if getattr(self, '_verbose', False):
                                        self._log(f"❌ [VOTE] Plaka oylamadan geçemedi: '{plate}'")
                            else:
                                if getattr(self, '_verbose', False):
                                    self._log(f"❌ [THRESHOLD] OCR güven skoru yetersiz: {max_confidence:.3f} < {ocr_threshold}")
                        else:
                            if getattr(self, '_verbose', False):
                                self._log(f"❌ [VALID] Plaka geçersiz: '{plate}'")
                        
                        if plate and not self._is_valid_tr_plate(plate):
                            if getattr(self, '_verbose', False):
                                self._log(f"❌ [EMPTY] Plaka boş veya None")
                    else:
                        if getattr(self, '_verbose', False):
                            self._log(f"❌ [DETECT] Plaka bulunamadı - {len(detections)} tespit var")
            except Exception as e:
                self._log(f"ANPR işleme hatası: {e}")
                pass
            # FPS optimizasyonu - GPU'da daha az bekleme
            wait_time = 0.015 if device == 'cuda' else 0.025
            time.sleep(wait_time)

    def _show_permission_warning(self, plate: str, message: str):
        """Kamera altında izin uyarısı göster"""
        try:
            # Kamera 1 altında uyarı göster
            warning_text = f"🚫 {message}: {plate}"
            self.var_cam1_last.set(warning_text)
            self.lbl_cam1_last.configure(foreground="#ef4444")  # Kırmızı renk
            
            # 3 saniye sonra normale döndür
            def reset_warning():
                try:
                    self.var_cam1_last.set("Son geçiş bekleniyor...")
                    self.lbl_cam1_last.configure(foreground="#94a3b8")
                except Exception:
                    pass
            
            self.after(3000, reset_warning)
            
            # Log'a da ekle
            self._log(f"Uyarı: {message} - {plate}")
            
        except Exception as e:
            pass
    
    def _enhanced_preprocessing(self, gray_img):
        """Gelişmiş ön işleme pipeline'ı - çoklu yöntemler"""
        processed = []
        
        # 1. Orijinal CLAHE görüntü
        processed.append(gray_img)
        
        # 2. Gaussian Blur + Adaptive Threshold
        try:
            blur = cv2.GaussianBlur(gray_img, (3,3), 0)
            adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed.append(adaptive)
        except Exception:
            pass
            
        # 3. Median Blur + Otsu Threshold
        try:
            median = cv2.medianBlur(gray_img, 3)
            _, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed.append(otsu)
        except Exception:
            pass
            
        # 4. Morfolojik işlemler
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            morph = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
            processed.append(morph)
        except Exception:
            pass
            
        # 5. Normalizasyon + CLAHE (ayarlanabilir clip limit)
        try:
            norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
            clahe2 = cv2.createCLAHE(clipLimit=self._clahe_clip_limit, tileGridSize=(8,8))
            enhanced = clahe2.apply(norm)
            processed.append(enhanced)
        except Exception:
            pass
            
        return processed

    def on_gate_open(self):
        """Güvenilir kapı açma sistemi - retry ve hata yönetimi ile"""
        s = self.store.get_settings()
        ip = s.get("relay_ip", "")
        port = int(s.get("relay_port", 1590) or 1590)
        cmd = (s.get("relay_command_open", "10") or "10").strip()
        
        self._log(f"🔧 [RELAY-DEBUG] Ayarlar: IP='{ip}', Port={port}, Komut='{cmd}'")
        
        # Test için farklı komutları dene
        test_commands = [
            cmd,                    # Orijinal komut
            "10",                   # Standart '10'
            "1",                    # Basit '1'
            "ON",                   # 'ON'
            "OPEN",                 # 'OPEN'
            "\x31\x30",             # HEX '10'
            "\x01\x10\x0D"          # Kontrol formatı
        ]
        
        if not ip:
            self._log("❌ [RELAY] Röle IP ayarı boş")
            messagebox.showwarning("Uyarı", "Röle IP ayarlarda boş.")
            return
        
        # Spam'i önlemek için son gönderim zamanını kontrol et
        import time as _t
        now = _t.time()
        last_gate_time = getattr(self, '_last_gate_time', 0)
        if now - last_gate_time < 1.0:  # 1 saniyeden sık gönderme
            self._log(f"⏱️ [RELAY] Spam koruması: {now - last_gate_time:.1f}s geçti")
            return
        self._last_gate_time = now
        
        self._log(f"🚪 [RELAY] Kapı açılıyor: {ip}:{port} -> {cmd}")
        
        def _worker():
            import socket, time as _t
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                if success:
                    break
                    
                self._log(f"🔄 [RELAY] Deneme {attempt+1}/{max_retries}")
                    
                try:
                    self._log(f"🔌 [RELAY] Bağlanıyor: {ip}:{port}")
                    with socket.create_connection((ip, port), timeout=5.0) as sock:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        self._log(f"✅ [RELAY] Bağlantı başarılı")
                        
                        # Çoklu gönderim stratejisi - test komutları
                        for test_cmd in test_commands:
                            payloads = [
                                test_cmd.encode('ascii'),           # Sadece komut
                                (test_cmd + "\r").encode('ascii'),   # Komut+\r
                                (test_cmd + "\n").encode('ascii'),   # Komut+\n
                                (test_cmd + "\r\n").encode('ascii')  # Komut+\r\n
                            ]
                            
                            for i, payload in enumerate(payloads):
                                try:
                                    self._log(f"📤 [RELAY] Gönderiliyor: {payload} (komut: '{test_cmd}', format {i+1})")
                                    sock.send(payload)
                                    self._log(f"✅ [RELAY] Komut gönderildi: {payload}")
                                    
                                    # Röle yanıtını bekle
                                    sock.settimeout(1.0)
                                    try:
                                        response = sock.recv(1024)
                                        self._log(f"📥 [RELAY] Röle yanıtı: {response}")
                                    except socket.timeout:
                                        self._log(f"⏰ [RELAY] Röle yanıtı yok (timeout)")
                                    except Exception as e:
                                        self._log(f"❌ [RELAY] Yanıt okuma hatası: {e}")
                                    
                                    # Komut işlenmesi için bekle
                                    _t.sleep(0.2)
                                    
                                    success = True
                                    break
                                except Exception as e:
                                    self._log(f"❌ [RELAY] Gönderim hatası: {e}")
                                    continue
                            
                            if success:
                                break
                    
                    if success:
                        self._log(f"🎉 [RELAY] Kapı açma başarılı!")
                    else:
                        self._log(f"❌ [RELAY] Tüm formatlar başarısız")
                
                except socket.timeout:
                    self._log(f"⏰ [RELAY] Bağlantı zaman aşımı (deneme {attempt+1})")
                except socket.connection_refused:
                    self._log(f"🚫 [RELAY] Bağlantı reddedildi (deneme {attempt+1})")
                except socket.gaierror as e:
                    self._log(f"🔍 [RELAY] DNS çözümleme hatası: {e}")
                except Exception as e:
                    self._log(f"❌ [RELAY] Genel hata (deneme {attempt+1}): {e}")
                
                if not success and attempt < max_retries - 1:
                    self._log(f"⏳ [RELAY] {0.5}s bekleniyor...")
                    _t.sleep(0.5)
                
            if not success:
                self._log("❌ [RELAY] Kapı açma başarısız - tüm denemeler reddedildi")
                
        threading.Thread(target=_worker, daemon=True).start()

    def on_clear_preview(self):
        self.canvas_cam1.delete("all")
        self.canvas_cam2.delete("all")
        self._log("Önizleme alanları temizlendi.")

    def open_settings_dialog(self):
        """Tek sayfalı kapsamlı plaka okuma ayarları diyalog penceresi"""
        dialog = tk.Toplevel(self)
        dialog.title("⚙️ Plaka Okuma Ayarları")
        dialog.geometry("950x750")
        dialog.configure(bg="#0f172a")
        dialog.resizable(True, True)
        
        # Dialog'u ana pencerenin ortasına konumlandır
        dialog.transient(self)
        dialog.grab_set()
        
        # Ana frame
        main_frame = ttk.Frame(dialog, padding=(0, 0))
        main_frame.pack(fill="both", expand=True)
        
        # Başlık frame - gradient efekt
        header_frame = tk.Frame(main_frame, bg="#1e293b", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        # Başlık içeriği
        header_content = ttk.Frame(header_frame, padding=(20, 15))
        header_content.pack(fill="both", expand=True)
        
        # Sol taraf - başlık
        title_section = ttk.Frame(header_content)
        title_section.pack(side="left", fill="y")
        
        # Büyük icon ve başlık
        icon_label = tk.Label(title_section, text="⚙️", font=("Segoe UI", 24), 
                            fg="#60a5fa", bg="#1e293b")
        icon_label.pack(side="left", padx=(0, 12))
        
        title_text = tk.Frame(title_section, bg="#1e293b")
        title_text.pack(side="left", fill="y")
        
        title_label = tk.Label(title_text, text="Plaka Okuma Ayarları", 
                             font=("Segoe UI", 16, "bold"), 
                             fg="#f1f5f9", bg="#1e293b")
        title_label.pack(anchor="w")
        
        subtitle_label = tk.Label(title_text, 
                                text="🎯 Tüm ANPR parametreleri - optimize edilmiş performans için", 
                                font=("Segoe UI", 9), 
                                fg="#94a3b8", bg="#1e293b")
        subtitle_label.pack(anchor="w", pady=(2, 0))
        
        # Ana içerik alanı - 3 kolonlu grid
        content_frame = tk.Frame(main_frame, bg="#0f172a")
        content_frame.pack(fill="both", expand=True, padx=20, pady=15)
        
        # Kolonlar oluştur
        left_column = tk.Frame(content_frame, bg="#0f172a")
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        middle_column = tk.Frame(content_frame, bg="#0f172a")
        middle_column.pack(side="left", fill="both", expand=True, padx=10)
        
        right_column = tk.Frame(content_frame, bg="#0f172a")
        right_column.pack(side="left", fill="both", expand=True, padx=(10, 0))
        
        # Ayar değişkenleri
        settings_vars = {}
        
        # Mevcut ayar değerlerini UI kontrollerine yükle
        # YOLO ayarları
        settings_vars['gpu_enabled'] = tk.BooleanVar(value=self._gpu_enabled)
        settings_vars['conf_thres'] = tk.DoubleVar(value=self._conf_thres)
        settings_vars['iou_thres'] = tk.DoubleVar(value=self._iou_thres)
        settings_vars['max_detections'] = tk.IntVar(value=self._max_detections)
        settings_vars['imgsz'] = tk.IntVar(value=getattr(self, '_imgsz', 640))
        settings_vars['half_precision'] = tk.BooleanVar(value=getattr(self, '_half_precision', True))
        
        # Plaka filtreleme ayarları
        settings_vars['min_plate_area'] = tk.IntVar(value=self._min_plate_area)
        settings_vars['max_plate_area'] = tk.IntVar(value=self._max_plate_area)
        settings_vars['min_aspect_ratio'] = tk.DoubleVar(value=self._min_aspect_ratio)
        settings_vars['max_aspect_ratio'] = tk.DoubleVar(value=self._max_aspect_ratio)
        settings_vars['min_y_percent'] = tk.DoubleVar(value=getattr(self, '_min_y_percent', 0.15))
        settings_vars['margin'] = tk.IntVar(value=getattr(self, '_margin', 20))
        settings_vars['nms_iou_threshold'] = tk.DoubleVar(value=getattr(self, '_nms_iou_threshold', 0.3))
        
        # OCR ayarları
        settings_vars['ocr_confidence'] = tk.DoubleVar(value=self._ocr_confidence)
        settings_vars['min_char_count'] = tk.IntVar(value=self._min_char_count)
        settings_vars['max_char_count'] = tk.IntVar(value=self._max_char_count)
        settings_vars['allowlist'] = tk.StringVar(value=getattr(self, '_allowlist', "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"))
        settings_vars['ocr_detail'] = tk.IntVar(value=getattr(self, '_ocr_detail', 1))
        
        # Zamanlama ayarları
        settings_vars['vote_window'] = tk.DoubleVar(value=self._vote_window)
        settings_vars['min_votes'] = tk.IntVar(value=self._min_votes)
        settings_vars['plate_cooldown'] = tk.DoubleVar(value=self._plate_cooldown_s)
        settings_vars['emit_interval'] = tk.DoubleVar(value=getattr(self, '_emit_interval', 8.0))
        
        # Görüntü işleme ayarları
        settings_vars['roi_height'] = tk.IntVar(value=self._roi_height)
        settings_vars['min_roi_width'] = tk.IntVar(value=getattr(self, '_min_roi_width', 90))
        settings_vars['min_roi_height'] = tk.IntVar(value=getattr(self, '_min_roi_height', 30))
        settings_vars['clahe_clip_limit'] = tk.DoubleVar(value=self._clahe_clip_limit)
        settings_vars['clahe_grid_size'] = tk.IntVar(value=getattr(self, '_clahe_grid_size', 8))
        settings_vars['interpolation'] = tk.StringVar(value=getattr(self, '_interpolation', "INTER_CUBIC"))
        
        # Performans ayarları
        settings_vars['gpu_wait_time'] = tk.DoubleVar(value=getattr(self, '_gpu_wait_time', 15.0))
        settings_vars['cpu_wait_time'] = tk.DoubleVar(value=getattr(self, '_cpu_wait_time', 25.0))
        
        # Hata ayıklama ayarları
        settings_vars['verbose'] = tk.BooleanVar(value=getattr(self, '_verbose', False))
        settings_vars['show_boxes'] = tk.BooleanVar(value=getattr(self, '_show_boxes', False))
        
        # SOL KOLON - YOLO ve Tespit Ayarları
        # 1. YOLO Tespit Ayarları
        yolo_frame = ttk.LabelFrame(left_column, text="🤖 YOLO Tespit Ayarları")
        yolo_frame.pack(fill="x", pady=(0, 10))
        
        # GPU Kullanımı
        gpu_frame = tk.Frame(yolo_frame, bg="#1e293b")
        gpu_frame.pack(fill="x", pady=2)
        gpu_icon = tk.Label(gpu_frame, text="⚡", font=("Segoe UI", 12), 
                           fg="#fbbf24", bg="#1e293b", width=2)
        gpu_icon.pack(side="left")
        gpu_label = tk.Label(gpu_frame, text="GPU Hızlandırma:", font=("Segoe UI", 8, "bold"), 
                            fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        gpu_label.pack(side="left")
        gpu_check = ttk.Checkbutton(gpu_frame, text="GPU", 
                                    variable=settings_vars['gpu_enabled'])
        gpu_check.pack(side="left")
        
        # Confidence Threshold
        conf_frame = tk.Frame(yolo_frame, bg="#1e293b")
        conf_frame.pack(fill="x", pady=2)
        conf_icon = tk.Label(conf_frame, text="🎯", font=("Segoe UI", 12), 
                            fg="#10b981", bg="#1e293b", width=2)
        conf_icon.pack(side="left")
        conf_label = tk.Label(conf_frame, text="Güven Eşiği:", font=("Segoe UI", 8, "bold"), 
                             fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        conf_label.pack(side="left")
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=settings_vars['conf_thres'], 
                              orient="horizontal", length=80)
        conf_scale.pack(side="left", padx=(3, 3))
        conf_label_val = tk.Label(conf_frame, text=f"{settings_vars['conf_thres'].get():.2f}", 
                                 font=("Segoe UI", 7, "bold"), fg="#60a5fa", bg="#1e293b", width=3)
        conf_label_val.pack(side="left")
        conf_scale.configure(command=lambda v: conf_label_val.configure(text=f"{float(v):.2f}"))
        
        # IoU Threshold
        iou_frame = tk.Frame(yolo_frame, bg="#1e293b")
        iou_frame.pack(fill="x", pady=2)
        iou_icon = tk.Label(iou_frame, text="🔄", font=("Segoe UI", 12), 
                           fg="#8b5cf6", bg="#1e293b", width=2)
        iou_icon.pack(side="left")
        iou_label = tk.Label(iou_frame, text="IoU Eşiği (NMS):", font=("Segoe UI", 8, "bold"), 
                            fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        iou_label.pack(side="left")
        iou_scale = ttk.Scale(iou_frame, from_=0.1, to=0.9, variable=settings_vars['iou_thres'], 
                             orient="horizontal", length=80)
        iou_scale.pack(side="left", padx=(3, 3))
        iou_label_val = tk.Label(iou_frame, text=f"{settings_vars['iou_thres'].get():.2f}", 
                                font=("Segoe UI", 7, "bold"), fg="#8b5cf6", bg="#1e293b", width=3)
        iou_label_val.pack(side="left")
        iou_scale.configure(command=lambda v: iou_label_val.configure(text=f"{float(v):.2f}"))
        
        # Maksimum Tespit Sayısı
        max_det_frame = tk.Frame(yolo_frame, bg="#1e293b")
        max_det_frame.pack(fill="x", pady=2)
        max_det_icon = tk.Label(max_det_frame, text="📊", font=("Segoe UI", 12), 
                               fg="#ef4444", bg="#1e293b", width=2)
        max_det_icon.pack(side="left")
        max_det_label = tk.Label(max_det_frame, text="Maks. Tespit:", font=("Segoe UI", 8, "bold"), 
                                fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        max_det_label.pack(side="left")
        max_det_spin = ttk.Spinbox(max_det_frame, from_=1, to=10, 
                                   textvariable=settings_vars['max_detections'], width=5)
        max_det_spin.pack(side="left")
        
        # Görüntü Boyutu
        imgsz_frame = tk.Frame(yolo_frame, bg="#1e293b")
        imgsz_frame.pack(fill="x", pady=2)
        imgsz_icon = tk.Label(imgsz_frame, text="📷", font=("Segoe UI", 12), 
                             fg="#06b6d4", bg="#1e293b", width=2)
        imgsz_icon.pack(side="left")
        imgsz_label = tk.Label(imgsz_frame, text="Görüntü Boyutu:", font=("Segoe UI", 8, "bold"), 
                              fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        imgsz_label.pack(side="left")
        imgsz_combo = ttk.Combobox(imgsz_frame, textvariable=settings_vars['imgsz'], 
                                   values=[320, 416, 512, 640, 832, 1024, 1280], 
                                   width=5, state="readonly")
        imgsz_combo.pack(side="left")
        
        # Half Precision
        half_frame = tk.Frame(yolo_frame, bg="#1e293b")
        half_frame.pack(fill="x", pady=2)
        half_icon = tk.Label(half_frame, text="⚡", font=("Segoe UI", 12), 
                            fg="#f59e0b", bg="#1e293b", width=2)
        half_icon.pack(side="left")
        half_label = tk.Label(half_frame, text="Half Precision:", font=("Segoe UI", 8, "bold"), 
                             fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        half_label.pack(side="left")
        half_check = ttk.Checkbutton(half_frame, text="FP16", 
                                    variable=settings_vars['half_precision'])
        half_check.pack(side="left")
        
        # 2. Plaka Filtreleme Ayarları
        filter_frame = ttk.LabelFrame(left_column, text="🔍 Plaka Filtreleme")
        filter_frame.pack(fill="x", pady=(0, 10))
        
        # Minimum Plaka Alanı
        min_area_frame = tk.Frame(filter_frame, bg="#1e293b")
        min_area_frame.pack(fill="x", pady=2)
        min_area_icon = tk.Label(min_area_frame, text="📏", font=("Segoe UI", 12), 
                                fg="#06b6d4", bg="#1e293b", width=2)
        min_area_icon.pack(side="left")
        min_area_label = tk.Label(min_area_frame, text="Min. Alan (px):", font=("Segoe UI", 8, "bold"), 
                                 fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        min_area_label.pack(side="left")
        settings_vars['min_plate_area'] = tk.IntVar(value=self._min_plate_area)
        min_area_spin = ttk.Spinbox(min_area_frame, from_=1000, to=50000, increment=500,
                                   textvariable=settings_vars['min_plate_area'], width=5)
        min_area_spin.pack(side="left")
        
        # Maksimum Plaka Alanı
        max_area_frame = tk.Frame(filter_frame, bg="#1e293b")
        max_area_frame.pack(fill="x", pady=2)
        max_area_icon = tk.Label(max_area_frame, text="📐", font=("Segoe UI", 12), 
                                fg="#06b6d4", bg="#1e293b", width=2)
        max_area_icon.pack(side="left")
        max_area_label = tk.Label(max_area_frame, text="Maks. Alan (px):", font=("Segoe UI", 8, "bold"), 
                                 fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        max_area_label.pack(side="left")
        settings_vars['max_plate_area'] = tk.IntVar(value=self._max_plate_area)
        max_area_spin = ttk.Spinbox(max_area_frame, from_=10000, to=1000000, increment=5000,
                                   textvariable=settings_vars['max_plate_area'], width=5)
        max_area_spin.pack(side="left")
        
        # En-Boy Oranları
        aspect_frame = tk.Frame(filter_frame, bg="#1e293b")
        aspect_frame.pack(fill="x", pady=2)
        aspect_icon = tk.Label(aspect_frame, text="📏", font=("Segoe UI", 12), 
                              fg="#f59e0b", bg="#1e293b", width=2)
        aspect_icon.pack(side="left")
        aspect_label = tk.Label(aspect_frame, text="En-Boy Oranı:", font=("Segoe UI", 8, "bold"), 
                               fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        aspect_label.pack(side="left")
        settings_vars['min_aspect_ratio'] = tk.DoubleVar(value=self._min_aspect_ratio)
        settings_vars['max_aspect_ratio'] = tk.DoubleVar(value=self._max_aspect_ratio)
        
        min_aspect_spin = ttk.Spinbox(aspect_frame, from_=0.5, to=5.0, increment=0.1,
                                      textvariable=settings_vars['min_aspect_ratio'], width=3)
        min_aspect_spin.pack(side="left", padx=(3, 1))
        
        tk.Label(aspect_frame, text="-", font=("Segoe UI", 8, "bold"), 
                fg="#94a3b8", bg="#1e293b").pack(side="left")
        
        max_aspect_spin = ttk.Spinbox(aspect_frame, from_=2.0, to=15.0, increment=0.1,
                                      textvariable=settings_vars['max_aspect_ratio'], width=3)
        max_aspect_spin.pack(side="left", padx=(1, 0))
        
        # Minimum Konum Yüzdesi
        min_y_frame = tk.Frame(filter_frame, bg="#1e293b")
        min_y_frame.pack(fill="x", pady=2)
        min_y_icon = tk.Label(min_y_frame, text="📍", font=("Segoe UI", 12), 
                             fg="#ef4444", bg="#1e293b", width=2)
        min_y_icon.pack(side="left")
        min_y_label = tk.Label(min_y_frame, text="Min. Konum (%):", font=("Segoe UI", 8, "bold"), 
                              fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        min_y_label.pack(side="left")
        min_y_scale = ttk.Scale(min_y_frame, from_=0.0, to=0.5, 
                               variable=settings_vars['min_y_percent'], 
                               orient="horizontal", length=60)
        min_y_scale.pack(side="left", padx=(3, 3))
        min_y_label_val = tk.Label(min_y_frame, text=f"{settings_vars['min_y_percent'].get():.2f}", 
                                  font=("Segoe UI", 7, "bold"), fg="#ef4444", bg="#1e293b", width=3)
        min_y_label_val.pack(side="left")
        min_y_scale.configure(command=lambda v: min_y_label_val.configure(text=f"{float(v):.2f}"))
        
        # Kenar Boşluğu
        margin_frame = tk.Frame(filter_frame, bg="#1e293b")
        margin_frame.pack(fill="x", pady=2)
        margin_icon = tk.Label(margin_frame, text="📏", font=("Segoe UI", 12), 
                              fg="#8b5cf6", bg="#1e293b", width=2)
        margin_icon.pack(side="left")
        margin_label = tk.Label(margin_frame, text="Kenar Boşluk:", font=("Segoe UI", 8, "bold"), 
                               fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        margin_label.pack(side="left")
        settings_vars['margin'] = tk.IntVar(value=20)
        margin_spin = ttk.Spinbox(margin_frame, from_=0, to=100, increment=5,
                                 textvariable=settings_vars['margin'], width=5)
        margin_spin.pack(side="left")
        
        # NMS IoU Threshold
        nms_iou_frame = tk.Frame(filter_frame, bg="#1e293b")
        nms_iou_frame.pack(fill="x", pady=2)
        nms_iou_icon = tk.Label(nms_iou_frame, text="🔄", font=("Segoe UI", 12), 
                               fg="#10b981", bg="#1e293b", width=2)
        nms_iou_icon.pack(side="left")
        nms_iou_label = tk.Label(nms_iou_frame, text="NMS IoU:", font=("Segoe UI", 8, "bold"), 
                                fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        nms_iou_label.pack(side="left")
        nms_iou_scale = ttk.Scale(nms_iou_frame, from_=0.1, to=0.8, 
                                 variable=settings_vars['nms_iou_threshold'], 
                                 orient="horizontal", length=60)
        nms_iou_scale.pack(side="left", padx=(3, 3))
        nms_iou_label_val = tk.Label(nms_iou_frame, text=f"{settings_vars['nms_iou_threshold'].get():.2f}", 
                                    font=("Segoe UI", 7, "bold"), fg="#10b981", bg="#1e293b", width=3)
        nms_iou_label_val.pack(side="left")
        nms_iou_scale.configure(command=lambda v: nms_iou_label_val.configure(text=f"{float(v):.2f}"))
        
        # ORTA KOLON - OCR ve Zamanlama
        # 3. OCR Ayarları
        ocr_frame = ttk.LabelFrame(middle_column, text="📝 OCR Ayarları")
        ocr_frame.pack(fill="x", pady=(0, 10))
        
        # OCR Güven Eşiği
        ocr_conf_frame = tk.Frame(ocr_frame, bg="#1e293b")
        ocr_conf_frame.pack(fill="x", pady=2)
        ocr_conf_icon = tk.Label(ocr_conf_frame, text="🔤", font=("Segoe UI", 12), 
                                fg="#10b981", bg="#1e293b", width=2)
        ocr_conf_icon.pack(side="left")
        ocr_conf_label = tk.Label(ocr_conf_frame, text="OCR Güven:", font=("Segoe UI", 8, "bold"), 
                                 fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        ocr_conf_label.pack(side="left")
        ocr_conf_scale = ttk.Scale(ocr_conf_frame, from_=0.1, to=1.0, 
                                  variable=settings_vars['ocr_confidence'], 
                                  orient="horizontal", length=80)
        ocr_conf_scale.pack(side="left", padx=(3, 3))
        ocr_conf_label_val = tk.Label(ocr_conf_frame, text=f"{settings_vars['ocr_confidence'].get():.2f}", 
                                     font=("Segoe UI", 7, "bold"), fg="#10b981", bg="#1e293b", width=3)
        ocr_conf_label_val.pack(side="left")
        ocr_conf_scale.configure(command=lambda v: ocr_conf_label_val.configure(text=f"{float(v):.2f}"))
        
        # Karakter Sayıları
        char_frame = tk.Frame(ocr_frame, bg="#1e293b")
        char_frame.pack(fill="x", pady=2)
        char_icon = tk.Label(char_frame, text="🔠", font=("Segoe UI", 12), 
                            fg="#f59e0b", bg="#1e293b", width=2)
        char_icon.pack(side="left")
        char_label = tk.Label(char_frame, text="Karakter Sayısı:", font=("Segoe UI", 8, "bold"), 
                             fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        char_label.pack(side="left")
        settings_vars['min_char_count'] = tk.IntVar(value=self._min_char_count)
        settings_vars['max_char_count'] = tk.IntVar(value=self._max_char_count)
        
        min_char_spin = ttk.Spinbox(char_frame, from_=4, to=12, 
                                   textvariable=settings_vars['min_char_count'], width=3)
        min_char_spin.pack(side="left", padx=(3, 1))
        
        tk.Label(char_frame, text="-", font=("Segoe UI", 8, "bold"), 
                fg="#94a3b8", bg="#1e293b").pack(side="left")
        
        max_char_spin = ttk.Spinbox(char_frame, from_=6, to=15, 
                                   textvariable=settings_vars['max_char_count'], width=3)
        max_char_spin.pack(side="left", padx=(1, 0))
        
        # İzinli Karakterler
        allowlist_frame = tk.Frame(ocr_frame, bg="#1e293b")
        allowlist_frame.pack(fill="x", pady=2)
        allowlist_icon = tk.Label(allowlist_frame, text="🔡", font=("Segoe UI", 12), 
                                 fg="#8b5cf6", bg="#1e293b", width=2)
        allowlist_icon.pack(side="left")
        allowlist_label = tk.Label(allowlist_frame, text="İzinli Karakterler:", font=("Segoe UI", 8, "bold"), 
                                  fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        allowlist_label.pack(side="left")
        settings_vars['allowlist'] = tk.StringVar(value="ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
        allowlist_entry = ttk.Entry(allowlist_frame, textvariable=settings_vars['allowlist'], 
                                   font=("Segoe UI", 7), width=14)
        allowlist_entry.pack(side="left")
        
        # OCR Detay Seviyesi
        detail_frame = tk.Frame(ocr_frame, bg="#1e293b")
        detail_frame.pack(fill="x", pady=2)
        detail_icon = tk.Label(detail_frame, text="🔍", font=("Segoe UI", 12), 
                              fg="#06b6d4", bg="#1e293b", width=2)
        detail_icon.pack(side="left")
        detail_label = tk.Label(detail_frame, text="Detay Seviyesi:", font=("Segoe UI", 8, "bold"), 
                               fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        detail_label.pack(side="left")
        settings_vars['ocr_detail'] = tk.IntVar(value=1)
        detail_combo = ttk.Combobox(detail_frame, textvariable=settings_vars['ocr_detail'], 
                                   values=[0, 1], width=3, state="readonly")
        detail_combo.pack(side="left")
        tk.Label(detail_frame, text="(0=hızlı,1=detaylı)", font=("Segoe UI", 6), 
                fg="#94a3b8", bg="#1e293b").pack(side="left", padx=(3, 0))
        
        # 4. Zamanlama Ayarları
        timing_frame = ttk.LabelFrame(middle_column, text="⏱️ Zamanlama")
        timing_frame.pack(fill="x", pady=(0, 10))
        
        # Oylama Penceresi
        vote_frame = tk.Frame(timing_frame, bg="#1e293b")
        vote_frame.pack(fill="x", pady=2)
        vote_icon = tk.Label(vote_frame, text="🗳️", font=("Segoe UI", 12), 
                            fg="#8b5cf6", bg="#1e293b", width=2)
        vote_icon.pack(side="left")
        vote_label = tk.Label(vote_frame, text="Oylama Penceresi:", font=("Segoe UI", 8, "bold"), 
                             fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        vote_label.pack(side="left")
        vote_scale = ttk.Scale(vote_frame, from_=0.5, to=15.0, 
                              variable=settings_vars['vote_window'], 
                              orient="horizontal", length=60)
        vote_scale.pack(side="left", padx=(3, 3))
        vote_label_val = tk.Label(vote_frame, text=f"{settings_vars['vote_window'].get():.1f}", 
                                 font=("Segoe UI", 7, "bold"), fg="#8b5cf6", bg="#1e293b", width=3)
        vote_label_val.pack(side="left")
        vote_scale.configure(command=lambda v: vote_label_val.configure(text=f"{float(v):.1f}"))
        
        # Minimum Oylama
        min_votes_frame = tk.Frame(timing_frame, bg="#1e293b")
        min_votes_frame.pack(fill="x", pady=2)
        min_votes_icon = tk.Label(min_votes_frame, text="🗳️", font=("Segoe UI", 12), 
                                 fg="#8b5cf6", bg="#1e293b", width=2)
        min_votes_icon.pack(side="left")
        min_votes_label = tk.Label(min_votes_frame, text="Min. Oylama:", font=("Segoe UI", 8, "bold"), 
                                  fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        min_votes_label.pack(side="left")
        settings_vars['min_votes'] = tk.IntVar(value=self._min_votes)
        min_votes_spin = ttk.Spinbox(min_votes_frame, from_=1, to=20, 
                                    textvariable=settings_vars['min_votes'], width=5)
        min_votes_spin.pack(side="left")
        
        # Plaka Soğuma Süresi
        cooldown_frame = tk.Frame(timing_frame, bg="#1e293b")
        cooldown_frame.pack(fill="x", pady=2)
        cooldown_icon = tk.Label(cooldown_frame, text="❄️", font=("Segoe UI", 12), 
                                fg="#06b6d4", bg="#1e293b", width=2)
        cooldown_icon.pack(side="left")
        cooldown_label = tk.Label(cooldown_frame, text="Soğuma Süresi:", font=("Segoe UI", 8, "bold"), 
                                 fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        cooldown_label.pack(side="left")
        cooldown_scale = ttk.Scale(cooldown_frame, from_=0.5, to=60.0, 
                                  variable=settings_vars['plate_cooldown'], 
                                  orient="horizontal", length=60)
        cooldown_scale.pack(side="left", padx=(3, 3))
        cooldown_label_val = tk.Label(cooldown_frame, text=f"{settings_vars['plate_cooldown'].get():.1f}", 
                                     font=("Segoe UI", 7, "bold"), fg="#06b6d4", bg="#1e293b", width=3)
        cooldown_label_val.pack(side="left")
        cooldown_scale.configure(command=lambda v: cooldown_label_val.configure(text=f"{float(v):.1f}"))
        
        # Yayın Aralığı
        emit_frame = tk.Frame(timing_frame, bg="#1e293b")
        emit_frame.pack(fill="x", pady=2)
        emit_icon = tk.Label(emit_frame, text="📢", font=("Segoe UI", 12), 
                            fg="#ef4444", bg="#1e293b", width=2)
        emit_icon.pack(side="left")
        emit_label = tk.Label(emit_frame, text="Yayın Aralığı:", font=("Segoe UI", 8, "bold"), 
                             fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        emit_label.pack(side="left")
        emit_scale = ttk.Scale(emit_frame, from_=1.0, to=30.0, 
                              variable=settings_vars['emit_interval'], 
                              orient="horizontal", length=60)
        emit_scale.pack(side="left", padx=(3, 3))
        emit_label_val = tk.Label(emit_frame, text=f"{settings_vars['emit_interval'].get():.1f}", 
                                 font=("Segoe UI", 7, "bold"), fg="#ef4444", bg="#1e293b", width=3)
        emit_label_val.pack(side="left")
        emit_scale.configure(command=lambda v: emit_label_val.configure(text=f"{float(v):.1f}"))
        
        # SAĞ KOLON - Görüntü ve Performans
        # 5. Görüntü İşleme
        image_frame = ttk.LabelFrame(right_column, text="🖼️ Görüntü İşleme")
        image_frame.pack(fill="x", pady=(0, 10))
        
        # ROI Yüksekliği
        roi_frame = tk.Frame(image_frame, bg="#1e293b")
        roi_frame.pack(fill="x", pady=2)
        roi_icon = tk.Label(roi_frame, text="📐", font=("Segoe UI", 12), 
                           fg="#ef4444", bg="#1e293b", width=2)
        roi_icon.pack(side="left")
        roi_label = tk.Label(roi_frame, text="ROI Yüksekliği:", font=("Segoe UI", 8, "bold"), 
                            fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        roi_label.pack(side="left")
        settings_vars['roi_height'] = tk.IntVar(value=self._roi_height)
        roi_spin = ttk.Spinbox(roi_frame, from_=20, to=300, increment=10,
                               textvariable=settings_vars['roi_height'], width=5)
        roi_spin.pack(side="left")
        
        # Minimum ROI Boyutları
        min_roi_width_frame = tk.Frame(image_frame, bg="#1e293b")
        min_roi_width_frame.pack(fill="x", pady=2)
        min_roi_width_icon = tk.Label(min_roi_width_frame, text="📏", font=("Segoe UI", 12), 
                                     fg="#06b6d4", bg="#1e293b", width=2)
        min_roi_width_icon.pack(side="left")
        min_roi_width_label = tk.Label(min_roi_width_frame, text="Min. ROI (GxY):", font=("Segoe UI", 8, "bold"), 
                                      fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        min_roi_width_label.pack(side="left")
        settings_vars['min_roi_width'] = tk.IntVar(value=90)
        settings_vars['min_roi_height'] = tk.IntVar(value=30)
        
        min_roi_width_spin = ttk.Spinbox(min_roi_width_frame, from_=30, to=500, increment=10,
                                        textvariable=settings_vars['min_roi_width'], width=3)
        min_roi_width_spin.pack(side="left", padx=(3, 1))
        
        tk.Label(min_roi_width_frame, text="x", font=("Segoe UI", 8, "bold"), 
                fg="#94a3b8", bg="#1e293b").pack(side="left")
        
        min_roi_height_spin = ttk.Spinbox(min_roi_width_frame, from_=10, to=200, increment=5,
                                         textvariable=settings_vars['min_roi_height'], width=3)
        min_roi_height_spin.pack(side="left", padx=(1, 0))
        
        # CLAHE Clip Limit
        clahe_frame = tk.Frame(image_frame, bg="#1e293b")
        clahe_frame.pack(fill="x", pady=2)
        clahe_icon = tk.Label(clahe_frame, text="🎨", font=("Segoe UI", 12), 
                             fg="#f59e0b", bg="#1e293b", width=2)
        clahe_icon.pack(side="left")
        clahe_label = tk.Label(clahe_frame, text="CLAHE Clip:", font=("Segoe UI", 8, "bold"), 
                              fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        clahe_label.pack(side="left")
        clahe_scale = ttk.Scale(clahe_frame, from_=0.5, to=10.0, 
                               variable=settings_vars['clahe_clip_limit'], 
                               orient="horizontal", length=60)
        clahe_scale.pack(side="left", padx=(3, 3))
        clahe_label_val = tk.Label(clahe_frame, text=f"{settings_vars['clahe_clip_limit'].get():.1f}", 
                                  font=("Segoe UI", 7, "bold"), fg="#f59e0b", bg="#1e293b", width=3)
        clahe_label_val.pack(side="left")
        clahe_scale.configure(command=lambda v: clahe_label_val.configure(text=f"{float(v):.1f}"))
        
        # CLAHE Grid Size
        clahe_grid_frame = tk.Frame(image_frame, bg="#1e293b")
        clahe_grid_frame.pack(fill="x", pady=2)
        clahe_grid_icon = tk.Label(clahe_grid_frame, text="🔲", font=("Segoe UI", 12), 
                                  fg="#8b5cf6", bg="#1e293b", width=2)
        clahe_grid_icon.pack(side="left")
        clahe_grid_label = tk.Label(clahe_grid_frame, text="CLAHE Grid:", font=("Segoe UI", 8, "bold"), 
                                   fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        clahe_grid_label.pack(side="left")
        settings_vars['clahe_grid_size'] = tk.IntVar(value=8)
        clahe_grid_spin = ttk.Spinbox(clahe_grid_frame, from_=4, to=32, increment=2,
                                     textvariable=settings_vars['clahe_grid_size'], width=5)
        clahe_grid_spin.pack(side="left")
        
        # Enterpolasyon
        interp_frame = tk.Frame(image_frame, bg="#1e293b")
        interp_frame.pack(fill="x", pady=2)
        interp_icon = tk.Label(interp_frame, text="🔄", font=("Segoe UI", 12), 
                             fg="#10b981", bg="#1e293b", width=2)
        interp_icon.pack(side="left")
        interp_label = tk.Label(interp_frame, text="Enterpolasyon:", font=("Segoe UI", 8, "bold"), 
                              fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        interp_label.pack(side="left")
        settings_vars['interpolation'] = tk.StringVar(value="INTER_CUBIC")
        interp_combo = ttk.Combobox(interp_frame, textvariable=settings_vars['interpolation'], 
                                   values=["INTER_LINEAR", "INTER_CUBIC", "INTER_AREA", "INTER_LANCZOS4"], 
                                   width=8, state="readonly")
        interp_combo.pack(side="left")
        
        # 6. Performans Ayarları
        perf_frame = ttk.LabelFrame(right_column, text="⚡ Performans")
        perf_frame.pack(fill="x", pady=(0, 10))
        
        # GPU Bekleme Süresi
        gpu_wait_frame = tk.Frame(perf_frame, bg="#1e293b")
        gpu_wait_frame.pack(fill="x", pady=2)
        gpu_wait_icon = tk.Label(gpu_wait_frame, text="⏱️", font=("Segoe UI", 12), 
                                fg="#fbbf24", bg="#1e293b", width=2)
        gpu_wait_icon.pack(side="left")
        gpu_wait_label = tk.Label(gpu_wait_frame, text="GPU Bekleme (ms):", font=("Segoe UI", 8, "bold"), 
                                 fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        gpu_wait_label.pack(side="left")
        gpu_wait_scale = ttk.Scale(gpu_wait_frame, from_=5.0, to=100.0, 
                                  variable=settings_vars['gpu_wait_time'], 
                                  orient="horizontal", length=60)
        gpu_wait_scale.pack(side="left", padx=(3, 3))
        gpu_wait_label_val = tk.Label(gpu_wait_frame, text=f"{settings_vars['gpu_wait_time'].get():.1f}", 
                                     font=("Segoe UI", 7, "bold"), fg="#fbbf24", bg="#1e293b", width=3)
        gpu_wait_label_val.pack(side="left")
        gpu_wait_scale.configure(command=lambda v: gpu_wait_label_val.configure(text=f"{float(v):.1f}"))
        
        # CPU Bekleme Süresi
        cpu_wait_frame = tk.Frame(perf_frame, bg="#1e293b")
        cpu_wait_frame.pack(fill="x", pady=2)
        cpu_wait_icon = tk.Label(cpu_wait_frame, text="⏱️", font=("Segoe UI", 12), 
                                fg="#f59e0b", bg="#1e293b", width=2)
        cpu_wait_icon.pack(side="left")
        cpu_wait_label = tk.Label(cpu_wait_frame, text="CPU Bekleme (ms):", font=("Segoe UI", 8, "bold"), 
                                 fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        cpu_wait_label.pack(side="left")
        cpu_wait_scale = ttk.Scale(cpu_wait_frame, from_=10.0, to=200.0, 
                                  variable=settings_vars['cpu_wait_time'], 
                                  orient="horizontal", length=60)
        cpu_wait_scale.pack(side="left", padx=(3, 3))
        cpu_wait_label_val = tk.Label(cpu_wait_frame, text=f"{settings_vars['cpu_wait_time'].get():.1f}", 
                                     font=("Segoe UI", 7, "bold"), fg="#f59e0b", bg="#1e293b", width=3)
        cpu_wait_label_val.pack(side="left")
        cpu_wait_scale.configure(command=lambda v: cpu_wait_label_val.configure(text=f"{float(v):.1f}"))
        
        # 7. Hata Ayıklama
        debug_frame = ttk.LabelFrame(right_column, text="🐛 Hata Ayıklama")
        debug_frame.pack(fill="x", pady=(0, 10))
        
        # Detaylı Log
        verbose_frame = tk.Frame(debug_frame, bg="#1e293b")
        verbose_frame.pack(fill="x", pady=2)
        verbose_icon = tk.Label(verbose_frame, text="📝", font=("Segoe UI", 12), 
                               fg="#10b981", bg="#1e293b", width=2)
        verbose_icon.pack(side="left")
        verbose_label = tk.Label(verbose_frame, text="Detaylı Log:", font=("Segoe UI", 8, "bold"), 
                                fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        verbose_label.pack(side="left")
        settings_vars['verbose'] = tk.BooleanVar(value=False)
        verbose_check = ttk.Checkbutton(verbose_frame, text="Tüm adımları logla", 
                                       variable=settings_vars['verbose'])
        verbose_check.pack(side="left")
        
        # Tespit Kutularını Göster
        show_boxes_frame = tk.Frame(debug_frame, bg="#1e293b")
        show_boxes_frame.pack(fill="x", pady=2)
        show_boxes_icon = tk.Label(show_boxes_frame, text="📦", font=("Segoe UI", 12), 
                                  fg="#ef4444", bg="#1e293b", width=2)
        show_boxes_icon.pack(side="left")
        show_boxes_label = tk.Label(show_boxes_frame, text="Tespit Kutuları:", font=("Segoe UI", 8, "bold"), 
                                   fg="#f1f5f9", bg="#1e293b", width=12, anchor="w")
        show_boxes_label.pack(side="left")
        settings_vars['show_boxes'] = tk.BooleanVar(value=False)
        show_boxes_check = ttk.Checkbutton(show_boxes_frame, text="Canvas'da göster", 
                                          variable=settings_vars['show_boxes'])
        show_boxes_check.pack(side="left")
        
        # Alt butonlar frame
        button_frame = tk.Frame(main_frame, bg="#1e293b", height=60)
        button_frame.pack(fill="x", side="bottom")
        button_frame.pack_propagate(False)
        
        button_content = ttk.Frame(button_frame, padding=(20, 15))
        button_content.pack(fill="both", expand=True)
        
        def apply_settings():
            """Tüm ayarları uygula"""
            try:
                # YOLO ayarları
                self._gpu_enabled = settings_vars['gpu_enabled'].get()
                self._conf_thres = settings_vars['conf_thres'].get()
                self._iou_thres = settings_vars['iou_thres'].get()
                self._max_detections = settings_vars['max_detections'].get()
                self._imgsz = settings_vars['imgsz'].get()
                self._half_precision = settings_vars['half_precision'].get()
                
                # Plaka filtreleme ayarları
                self._min_plate_area = settings_vars['min_plate_area'].get()
                self._max_plate_area = settings_vars['max_plate_area'].get()
                self._min_aspect_ratio = settings_vars['min_aspect_ratio'].get()
                self._max_aspect_ratio = settings_vars['max_aspect_ratio'].get()
                self._min_y_percent = settings_vars['min_y_percent'].get()
                self._margin = settings_vars['margin'].get()
                self._nms_iou_threshold = settings_vars['nms_iou_threshold'].get()
                
                # OCR ayarları
                self._ocr_confidence = settings_vars['ocr_confidence'].get()
                self._min_char_count = settings_vars['min_char_count'].get()
                self._max_char_count = settings_vars['max_char_count'].get()
                self._allowlist = settings_vars['allowlist'].get()
                self._ocr_detail = settings_vars['ocr_detail'].get()
                
                # Zamanlama ayarları
                self._vote_window = settings_vars['vote_window'].get()
                self._min_votes = settings_vars['min_votes'].get()
                self._plate_cooldown_s = settings_vars['plate_cooldown'].get()
                self._emit_interval = settings_vars['emit_interval'].get()
                
                # Görüntü işleme ayarları
                self._roi_height = settings_vars['roi_height'].get()
                self._min_roi_width = settings_vars['min_roi_width'].get()
                self._min_roi_height = settings_vars['min_roi_height'].get()
                self._clahe_clip_limit = settings_vars['clahe_clip_limit'].get()
                self._clahe_grid_size = settings_vars['clahe_grid_size'].get()
                self._interpolation = settings_vars['interpolation'].get()
                
                # Performans ayarları
                self._gpu_wait_time = settings_vars['gpu_wait_time'].get()
                self._cpu_wait_time = settings_vars['cpu_wait_time'].get()
                
                # Hata ayıklama ayarları
                self._verbose = settings_vars['verbose'].get()
                self._show_boxes = settings_vars['show_boxes'].get()
                
                # Ayarları kaydet
                self._save_settings()
                
                # Tüm ayarları log'a yaz
                self._log("🎯 Tüm ayarlar uygulandı:")
                self._log(f"  ⚡ GPU: {self._gpu_enabled}, ImgSz: {self._imgsz}, Half: {self._half_precision}")
                self._log(f"  🎯 Conf: {self._conf_thres:.2f}, IoU: {self._iou_thres:.2f}, MaxDet: {self._max_detections}")
                self._log(f"  📏 Alan: {self._min_plate_area}-{self._max_plate_area}, En-Boy: {self._min_aspect_ratio:.2f}-{self._max_aspect_ratio:.2f}")
                self._log(f"  📍 Konum: {self._min_y_percent:.2f}, Margin: {self._margin}, NMS-IoU: {self._nms_iou_threshold:.2f}")
                self._log(f"  🔤 OCR: {self._ocr_confidence:.2f}, Char: {self._min_char_count}-{self._max_char_count}, Detail: {self._ocr_detail}")
                self._log(f"  🔡 Allowlist: {self._allowlist}")
                self._log(f"  ⏱️ Oylama: {self._vote_window:.1f}s, Min: {self._min_votes}, Soğuma: {self._plate_cooldown_s:.1f}s")
                self._log(f"  📢 Yayın: {self._emit_interval:.1f}s")
                self._log(f"  🖼️ ROI: {self._roi_height}, MinROI: {self._min_roi_width}x{self._min_roi_height}")
                self._log(f"  🎨 CLAHE: {self._clahe_clip_limit:.1f}, Grid: {self._clahe_grid_size}, Interp: {self._interpolation}")
                self._log(f"  ⚡ GPU/CPU Wait: {self._gpu_wait_time:.1f}ms/{self._cpu_wait_time:.1f}ms")
                self._log(f"  🐛 Debug: Verbose={self._verbose}, ShowBoxes={self._show_boxes}")
                
                messagebox.showinfo("✅ Başarılı", "Tüm ayarlar uygulandı ve kaydedildi!\n\nANPR'i yeniden başlatarak değişiklikleri test edebilirsiniz.")
                
            except Exception as e:
                messagebox.showerror("❌ Hata", f"Ayarlar uygulanırken hata oluştu: {e}")
        
        def reset_to_defaults():
            """Varsayılan ayarlara dön"""
            if messagebox.askyesno("⚠️ Onay", "Tüm ayarları varsayılan değerlere sıfırlamak istediğinizden emin misiniz?"):
                # Yeni optimize edilmiş varsayılan değerleri uygula
                settings_vars['gpu_enabled'].set(self._default_settings['gpu_enabled'])
                settings_vars['conf_thres'].set(self._default_settings['conf_thres'])
                settings_vars['iou_thres'].set(self._default_settings['iou_thres'])
                settings_vars['max_detections'].set(self._default_settings['max_detections'])
                settings_vars['imgsz'].set(1024)
                settings_vars['half_precision'].set(True)
                settings_vars['min_plate_area'].set(self._default_settings['min_plate_area'])
                settings_vars['max_plate_area'].set(self._default_settings['max_plate_area'])
                settings_vars['min_aspect_ratio'].set(self._default_settings['min_aspect_ratio'])
                settings_vars['max_aspect_ratio'].set(self._default_settings['max_aspect_ratio'])
                settings_vars['min_y_percent'].set(0.10)  # Daha düşük konum
                settings_vars['margin'].set(15)           # Daha az margin
                settings_vars['nms_iou_threshold'].set(0.45)
                settings_vars['ocr_confidence'].set(self._default_settings['ocr_confidence'])
                settings_vars['min_char_count'].set(self._default_settings['min_char_count'])
                settings_vars['max_char_count'].set(self._default_settings['max_char_count'])
                settings_vars['allowlist'].set("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
                settings_vars['ocr_detail'].set(1)
                settings_vars['vote_window'].set(self._default_settings['vote_window'])
                settings_vars['min_votes'].set(self._default_settings['min_votes'])
                settings_vars['plate_cooldown'].set(self._default_settings['plate_cooldown_s'])
                settings_vars['emit_interval'].set(3.0)    # Daha hızlı yayın
                settings_vars['roi_height'].set(self._default_settings['roi_height'])
                settings_vars['min_roi_width'].set(80)     # Daha küçük minimum genişlik
                settings_vars['min_roi_height'].set(25)    # Daha küçük minimum yükseklik
                settings_vars['clahe_clip_limit'].set(self._default_settings['clahe_clip_limit'])
                settings_vars['clahe_grid_size'].set(8)
                settings_vars['interpolation'].set("INTER_CUBIC")
                settings_vars['gpu_wait_time'].set(3.0)    # Daha hızlı GPU bekleme
                settings_vars['cpu_wait_time'].set(8.0)    # Daha hızlı CPU bekleme
                settings_vars['verbose'].set(True)         # Detaylı log aktif
                settings_vars['show_boxes'].set(True)      # Tespit kutuları aktif
                
                # Label'ları güncelle
                conf_label_val.configure(text=f"{self._default_settings['conf_thres']:.2f}")
                iou_label_val.configure(text=f"{self._default_settings['iou_thres']:.2f}")
                ocr_conf_label_val.configure(text=f"{self._default_settings['ocr_confidence']:.2f}")
                vote_label_val.configure(text=f"{self._default_settings['vote_window']:.1f}")
                cooldown_label_val.configure(text=f"{self._default_settings['plate_cooldown_s']:.1f}")
                clahe_label_val.configure(text=f"{self._default_settings['clahe_clip_limit']:.1f}")
                min_y_label_val.configure(text="0.10")
                nms_iou_label_val.configure(text="0.45")
                emit_label_val.configure(text="3.0")
                gpu_wait_label_val.configure(text="3.0")
                cpu_wait_label_val.configure(text="8.0")
                
                self._log("🔄 Ayarlar varsayılan değerlere sıfırlandı.")
        
        def test_current_settings():
            """Mevcut ayarlarla test yap"""
            self._log("🧪 Ayarlar test ediliyor...")
            if self._anpr_running:
                messagebox.showinfo("🧪 Test", "ANPR zaten çalışıyor. Ayarlar gerçek zamanlı etkili olacak.")
            else:
                if messagebox.askyesno("🧪 Test", "ANPR'i başlatarak ayarları test etmek istiyor musunuz?"):
                    self.on_anpr_start()
                    messagebox.showinfo("🧪 Test", "ANPR başlatıldı. Kamera görüntüsünde plaka okumayı test edin.")
        
        def export_settings():
            """Ayarları dışa aktar"""
            try:
                settings_data = {
                    "gpu_enabled": settings_vars['gpu_enabled'].get(),
                    "conf_thres": settings_vars['conf_thres'].get(),
                    "iou_thres": settings_vars['iou_thres'].get(),
                    "max_detections": settings_vars['max_detections'].get(),
                    "imgsz": settings_vars['imgsz'].get(),
                    "half_precision": settings_vars['half_precision'].get(),
                    "min_plate_area": settings_vars['min_plate_area'].get(),
                    "max_plate_area": settings_vars['max_plate_area'].get(),
                    "min_aspect_ratio": settings_vars['min_aspect_ratio'].get(),
                    "max_aspect_ratio": settings_vars['max_aspect_ratio'].get(),
                    "min_y_percent": settings_vars['min_y_percent'].get(),
                    "margin": settings_vars['margin'].get(),
                    "nms_iou_threshold": settings_vars['nms_iou_threshold'].get(),
                    "ocr_confidence": settings_vars['ocr_confidence'].get(),
                    "min_char_count": settings_vars['min_char_count'].get(),
                    "max_char_count": settings_vars['max_char_count'].get(),
                    "allowlist": settings_vars['allowlist'].get(),
                    "ocr_detail": settings_vars['ocr_detail'].get(),
                    "vote_window": settings_vars['vote_window'].get(),
                    "min_votes": settings_vars['min_votes'].get(),
                    "plate_cooldown": settings_vars['plate_cooldown'].get(),
                    "emit_interval": settings_vars['emit_interval'].get(),
                    "roi_height": settings_vars['roi_height'].get(),
                    "min_roi_width": settings_vars['min_roi_width'].get(),
                    "min_roi_height": settings_vars['min_roi_height'].get(),
                    "clahe_clip_limit": settings_vars['clahe_clip_limit'].get(),
                    "clahe_grid_size": settings_vars['clahe_grid_size'].get(),
                    "interpolation": settings_vars['interpolation'].get(),
                    "gpu_wait_time": settings_vars['gpu_wait_time'].get(),
                    "cpu_wait_time": settings_vars['cpu_wait_time'].get(),
                    "verbose": settings_vars['verbose'].get(),
                    "show_boxes": settings_vars['show_boxes'].get()
                }
                
                import json
                from tkinter import filedialog
                filename = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Ayarları Kaydet"
                )
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(settings_data, f, ensure_ascii=False, indent=2)
                    self._log(f"💾 Ayarlar dışa aktarıldı: {filename}")
                    messagebox.showinfo("✅ Başarılı", f"Ayarlar kaydedildi:\n{filename}")
            except Exception as e:
                messagebox.showerror("❌ Hata", f"Ayarlar kaydedilirken hata: {e}")
        
        # Butonlar
        ttk.Button(button_content, text="🔄 Varsayılana Dön", style="TButton", 
                  command=reset_to_defaults).pack(side="left", padx=(0, 10))
        ttk.Button(button_content, text="💾 Dışa Aktar", style="TButton", 
                  command=export_settings).pack(side="left", padx=(0, 10))
        ttk.Button(button_content, text="🧪 Test Et", style="Info.TButton", 
                  command=test_current_settings).pack(side="left", padx=(0, 10))
        ttk.Button(button_content, text="❌ İptal", style="Error.TButton", 
                  command=dialog.destroy).pack(side="right", padx=(10, 0))
        ttk.Button(button_content, text="✅ Uygula", style="Success.TButton", 
                  command=apply_settings).pack(side="right")
        
        # Klavye navigasyonu
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        dialog.bind('<Return>', lambda e: apply_settings())
        
        # Dialog'u focus et
        dialog.focus_set()
        dialog.wait_window()


def main():
    """
    Gelişmiş Plaka Tanıma Sistemi
    
    Özellikler:
    - Çoklu ön işleme yöntemleri (CLAHE, Adaptive Threshold, Otsu, Morfolojik işlemler)
    - Adaptif güven eşikleri
    - Gelişmiş OCR sonuç seçimi
    - Performans izleme
    - ROI (Region of Interest) desteği
    - Çift kamera desteği
    - Otomatik kapı kontrolü
    
    Kullanım:
    1. Kamera ayarlarından URL'leri yapılandırın
    2. Gelişmiş ANPR ayarlarından güven eşiklerini ayarlayın
    3. ROI seçimi için kamera görüntüsünde sürükle-bırak yapın
    4. ANPR'i başlatın ve performansı izleyin
    
    Performans İpuçları:
    - GPU destekli sistemde daha yüksek FPS
    - Adaptif güven ayarı ile otomatik optimizasyon
    - ROI kullanarak işlem yükünü azaltın
    - Güven eşiklerini ortam koşullarına göre ayarlayın
    """
    app = PlakaOkumaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
