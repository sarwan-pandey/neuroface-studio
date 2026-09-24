# ═══════════════════════════════════════════════════════════════════
#  NeuroFace Studio — Professional Real-Time Face Recognition GUI
#  Usage: python face_gui.py
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import time
import pickle
import shutil
import threading
import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFilter

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from insightface.app import FaceAnalysis

# ═══════════════════════════════════════════════════════════════════
#  PALETTE & DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════

DARK_BG       = "#0b0e14"
PANEL_BG      = "#111620"
CARD_BG       = "#161b27"
INPUT_BG      = "#1c2333"
BORDER        = "#242d3d"
BORDER_LIGHT  = "#2e3a50"

BLUE          = "#3b82f6"
BLUE_HOVER    = "#2563eb"
BLUE_GLOW     = "#3b82f620"
GREEN         = "#10b981"
GREEN_HOVER   = "#059669"
RED           = "#ef4444"
RED_HOVER     = "#dc2626"
AMBER         = "#f59e0b"
PURPLE        = "#8b5cf6"

TEXT_WHITE    = "#f1f5f9"
TEXT_LIGHT    = "#94a3b8"
TEXT_DIM      = "#64748b"
TEXT_MUTED    = "#475569"


# ═══════════════════════════════════════════════════════════════════
#  FACE ENGINE (core logic from your working realtime_inface.py)
# ═══════════════════════════════════════════════════════════════════

def l2_normalize(v, eps=1e-12):
    return v / (np.linalg.norm(v) + eps)

def choose_largest_face(faces):
    if not faces:
        return None
    areas = [max(0, f.bbox[2]-f.bbox[0]) * max(0, f.bbox[3]-f.bbox[1]) for f in faces]
    return faces[int(np.argmax(areas))]

def gallery_signature(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    num, latest = 0, 0.0
    if not root.exists():
        return {"num_files": 0, "latest_mtime": 0.0}
    for pdir in root.iterdir():
        if not pdir.is_dir() or pdir.name.startswith("."):
            continue
        for p in pdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                num += 1
                try:
                    latest = max(latest, os.path.getmtime(p))
                except OSError:
                    pass
    return {"num_files": num, "latest_mtime": latest}


class FaceEngine:
    def __init__(self):
        self.app = None
        self.labels = []
        self.gallery_embs = None
        self.gallery_root = None
        self.cache_path = None
        self.loaded = False

    def load_model(self, name="buffalo_l", det_size=640):
        self.app = FaceAnalysis(name=name)
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        self.loaded = True

    def set_gallery(self, path):
        self.gallery_root = Path(path)
        self.gallery_root.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.gallery_root / "gallery_cache.pkl"

    def persons(self):
        if not self.gallery_root or not self.gallery_root.exists():
            return []
        return sorted([d.name for d in self.gallery_root.iterdir()
                       if d.is_dir() and not d.name.startswith(".")])

    def person_images(self, name):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        pdir = self.gallery_root / name
        if not pdir.exists():
            return []
        return sorted([p for p in pdir.iterdir() if p.suffix.lower() in exts])

    def save_capture(self, person, frame_bgr):
        pdir = self.gallery_root / person
        pdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fp = pdir / f"{person}_{ts}.jpg"
        cv2.imwrite(str(fp), frame_bgr)
        return fp

    def add_files(self, person, paths):
        pdir = self.gallery_root / person
        pdir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dst = pdir / Path(src).name
            if dst.exists():
                ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                dst = pdir / f"{ts}_{Path(src).name}"
            shutil.copy2(src, dst)

    def delete_person(self, name):
        pdir = self.gallery_root / name
        if pdir.exists():
            shutil.rmtree(pdir)

    def delete_image(self, path):
        p = Path(path)
        if p.exists():
            p.unlink()

    def build_gallery(self, force=False, on_progress=None):
        if not self.app or not self.gallery_root:
            return
        sig = gallery_signature(self.gallery_root)

        # Try cache
        if self.cache_path and self.cache_path.exists() and not force:
            try:
                with open(self.cache_path, "rb") as f:
                    c = pickle.load(f)
                if isinstance(c, dict) and "labels" in c and "embeddings" in c:
                    m = c.get("meta")
                    if m and m.get("num_files") == sig["num_files"] and \
                       abs(m.get("latest_mtime", 0) - sig["latest_mtime"]) < 1e-6:
                        self.labels, self.gallery_embs = c["labels"], c["embeddings"]
                        if on_progress:
                            on_progress(f"Cache loaded — {len(self.labels)} identities")
                        return
            except Exception:
                pass

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        embeds_all, labels_all = [], []
        persons = self.persons()

        for i, pname in enumerate(persons):
            pdir = self.gallery_root / pname
            imgs = []
            for ext in exts:
                imgs.extend(sorted(pdir.glob(ext)))
            if not imgs:
                continue

            embeds = []
            for p in imgs:
                img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                faces = self.app.get(img)
                if not faces:
                    continue
                face = choose_largest_face(faces)
                if getattr(face, "normed_embedding", None) is None:
                    continue
                embeds.append(l2_normalize(np.asarray(face.normed_embedding, dtype=np.float32)))

            if embeds:
                embeds_all.append(l2_normalize(np.mean(np.stack(embeds), axis=0)))
                labels_all.append(pname)

            if on_progress:
                on_progress(f"Encoding {i+1}/{len(persons)}: {pname} ({len(embeds)} faces)")

        self.labels = labels_all
        self.gallery_embs = np.stack(embeds_all).astype(np.float32) if embeds_all else None

        cache = {"labels": labels_all, "embeddings": self.gallery_embs, "meta": sig}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(cache, f)

        if on_progress:
            on_progress(f"Gallery ready — {len(labels_all)} identities")

    def recognize(self, frame, threshold=0.40):
        if not self.app or self.gallery_embs is None or not self.labels:
            return []
        faces = self.app.get(frame)
        if not faces:
            return []
        G = self.gallery_embs.T
        out = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            if getattr(f, "normed_embedding", None) is None:
                continue
            e = np.asarray(f.normed_embedding, dtype=np.float32)
            sims = np.dot(G.T, e)
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            name = self.labels[idx] if sim >= threshold else "Unknown"
            out.append({"bbox": (x1, y1, x2, y2), "name": name, "score": sim,
                        "kps": getattr(f, "kps", None)})
        return out


# ═══════════════════════════════════════════════════════════════════
#  CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════

class StatusDot(ctk.CTkFrame):
    """Tiny colored status indicator dot."""
    def __init__(self, master, color=TEXT_DIM, size=8, **kw):
        super().__init__(master, width=size, height=size,
                         corner_radius=size//2, fg_color=color, **kw)
        self.configure(width=size, height=size)
        self.pack_propagate(False)

    def set_color(self, c):
        self.configure(fg_color=c)


class MetricCard(ctk.CTkFrame):
    """Small stat card with label and value."""
    def __init__(self, master, label, value="—", color=BLUE, **kw):
        super().__init__(master, fg_color=CARD_BG, corner_radius=10,
                         border_width=1, border_color=BORDER, **kw)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=label, font=ctk.CTkFont(size=10),
                     text_color=TEXT_DIM).grid(row=0, column=0, sticky="w",
                                               padx=14, pady=(10, 0))
        self.val = ctk.CTkLabel(self, text=value,
                                 font=ctk.CTkFont(size=22, weight="bold"),
                                 text_color=color)
        self.val.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

    def set(self, v):
        self.val.configure(text=v)


class NavButton(ctk.CTkButton):
    """Sidebar navigation button with icon + text."""
    def __init__(self, master, icon, text, active=False, **kw):
        fg = BLUE if active else "transparent"
        hv = BLUE_HOVER if active else BORDER
        txt = TEXT_WHITE if active else TEXT_LIGHT
        super().__init__(master, text=f"  {icon}   {text}", anchor="w",
                         fg_color=fg, hover_color=hv, text_color=txt,
                         font=ctk.CTkFont(size=13), height=42,
                         corner_radius=8, **kw)
        self._active = active

    def set_active(self, on):
        self._active = on
        if on:
            self.configure(fg_color=BLUE, hover_color=BLUE_HOVER,
                           text_color=TEXT_WHITE)
        else:
            self.configure(fg_color="transparent", hover_color=BORDER,
                           text_color=TEXT_LIGHT)


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════

class FaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        self.title("FaceID Studio")
        self.geometry("1380x800")
        self.minsize(1100, 680)
        self.configure(fg_color=DARK_BG)

        self.engine = FaceEngine()
        self.cap = None
        self.running = False
        self.enroll_active = False
        self.snap_frame = None
        self.threshold = 0.40
        self.current_page = "live"

        # Default gallery = ./gallery relative to this script
        self.gallery_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")

        self._build_sidebar()
        self._build_pages()
        self._show_page("live")

        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ──────────────────────────────────────────────────────────
    #  SIDEBAR
    # ──────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=260, fg_color=PANEL_BG,
                                     corner_radius=0, border_width=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Brand
        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=20, pady=(22, 2))
        ctk.CTkLabel(brand, text="◉",
                     font=ctk.CTkFont(size=24), text_color=BLUE).pack(side="left")
        ctk.CTkLabel(brand, text=" FaceID Studio",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=TEXT_WHITE).pack(side="left", padx=(6, 0))

        ctk.CTkLabel(self.sidebar, text="Real-time recognition engine",
                     font=ctk.CTkFont(size=11), text_color=TEXT_DIM).pack(
                         padx=20, anchor="w", pady=(0, 16))

        # Status row
        stat_row = ctk.CTkFrame(self.sidebar, fg_color=CARD_BG, corner_radius=8,
                                 border_width=1, border_color=BORDER)
        stat_row.pack(fill="x", padx=16, pady=(0, 16))
        stat_inner = ctk.CTkFrame(stat_row, fg_color="transparent")
        stat_inner.pack(fill="x", padx=12, pady=10)
        self.model_dot = StatusDot(stat_inner, color=RED, size=9)
        self.model_dot.pack(side="left")
        self.model_status_lbl = ctk.CTkLabel(stat_inner, text="  Model not loaded",
                                              font=ctk.CTkFont(size=11),
                                              text_color=TEXT_LIGHT)
        self.model_status_lbl.pack(side="left")

        # Separator
        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(
            fill="x", padx=20, pady=(0, 12))

        # Nav section label
        ctk.CTkLabel(self.sidebar, text="NAVIGATION",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_MUTED).pack(padx=22, anchor="w", pady=(0, 6))

        self.nav_btns = {}
        for key, icon, label in [("live", "📡", "Live Recognition"),
                                  ("enroll", "➕", "Enroll Faces"),
                                  ("gallery", "📂", "Gallery"),
                                  ("settings", "⚙", "Settings")]:
            b = NavButton(self.sidebar, icon, label,
                          command=lambda k=key: self._show_page(k))
            b.pack(fill="x", padx=16, pady=2)
            self.nav_btns[key] = b

        # Spacer
        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(
            fill="x", padx=20, pady=12)

        # Quick actions
        ctk.CTkLabel(self.sidebar, text="QUICK ACTIONS",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_MUTED).pack(padx=22, anchor="w", pady=(0, 8))

        self.load_btn = ctk.CTkButton(
            self.sidebar, text="⚡  Load Model", height=38,
            fg_color=BLUE, hover_color=BLUE_HOVER, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._load_model)
        self.load_btn.pack(fill="x", padx=16, pady=(0, 5))

        self.build_btn = ctk.CTkButton(
            self.sidebar, text="🔨  Build Gallery", height=38,
            fg_color=INPUT_BG, hover_color=BORDER_LIGHT, corner_radius=8,
            border_width=1, border_color=BORDER,
            font=ctk.CTkFont(size=12),
            command=self._build_gallery)
        self.build_btn.pack(fill="x", padx=16, pady=(0, 5))

        # Bottom info
        bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=16, pady=16)
        self.status_text = ctk.CTkLabel(bottom, text="Ready",
                                         font=ctk.CTkFont(size=11),
                                         text_color=TEXT_DIM,
                                         wraplength=220, justify="left")
        self.status_text.pack(anchor="w")

    # ──────────────────────────────────────────────────────────
    #  PAGES
    # ──────────────────────────────────────────────────────────
    def _build_pages(self):
        self.page_container = ctk.CTkFrame(self, fg_color=DARK_BG, corner_radius=0)
        self.page_container.pack(side="right", fill="both", expand=True)

        self.pages = {}
        self._build_live_page()
        self._build_enroll_page()
        self._build_gallery_page()
        self._build_settings_page()

    def _show_page(self, key):
        self.current_page = key
        for k, b in self.nav_btns.items():
            b.set_active(k == key)
        for k, f in self.pages.items():
            if k == key:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()
        if key == "gallery":
            self._refresh_gallery()

    # ──────────────────────────────────────────────────────────
    #  PAGE: LIVE RECOGNITION
    # ──────────────────────────────────────────────────────────
    def _build_live_page(self):
        page = ctk.CTkFrame(self.page_container, fg_color=DARK_BG, corner_radius=0)
        self.pages["live"] = page

        # Header
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.pack(fill="x", padx=24, pady=(20, 12))
        ctk.CTkLabel(hdr, text="Live Recognition",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_WHITE).pack(side="left")

        self.live_toggle = ctk.CTkButton(
            hdr, text="▶  Start Camera", height=40, width=180,
            fg_color=GREEN, hover_color=GREEN_HOVER, corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_live)
        self.live_toggle.pack(side="right")

        # Metrics row
        metrics = ctk.CTkFrame(page, fg_color="transparent")
        metrics.pack(fill="x", padx=24, pady=(0, 12))
        self.m_fps = MetricCard(metrics, "FPS", "—", BLUE)
        self.m_fps.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.m_faces = MetricCard(metrics, "FACES", "0", PURPLE)
        self.m_faces.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.m_ids = MetricCard(metrics, "IDENTITIES", "0", GREEN)
        self.m_ids.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.m_status = MetricCard(metrics, "STATUS", "Idle", AMBER)
        self.m_status.pack(side="left", fill="x", expand=True)

        # Video feed
        self.video_frame = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=12,
                                          border_width=1, border_color=BORDER)
        self.video_frame.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Camera feed will appear here\n\n"
                 "Load Model  →  Build Gallery  →  Start Camera",
            font=ctk.CTkFont(size=15), text_color=TEXT_DIM,
            fg_color="transparent")
        self.video_label.pack(fill="both", expand=True, padx=2, pady=2)

    # ──────────────────────────────────────────────────────────
    #  PAGE: ENROLL FACES
    # ──────────────────────────────────────────────────────────
    def _build_enroll_page(self):
        page = ctk.CTkFrame(self.page_container, fg_color=DARK_BG, corner_radius=0)
        self.pages["enroll"] = page

        # Header
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.pack(fill="x", padx=24, pady=(20, 12))
        ctk.CTkLabel(hdr, text="Enroll New Person",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_WHITE).pack(side="left")

        # Name + controls card
        ctrl_card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=12,
                                  border_width=1, border_color=BORDER)
        ctrl_card.pack(fill="x", padx=24, pady=(0, 12))
        ctrl_inner = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        ctrl_inner.pack(fill="x", padx=20, pady=16)

        ctk.CTkLabel(ctrl_inner, text="Person Name",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_DIM).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.enroll_name = ctk.CTkEntry(ctrl_inner, height=38, width=220,
                                         fg_color=INPUT_BG, border_color=BORDER,
                                         text_color=TEXT_WHITE,
                                         placeholder_text="e.g. John Smith",
                                         font=ctk.CTkFont(size=13))
        self.enroll_name.grid(row=1, column=0, padx=(0, 12))

        self.cam_btn = ctk.CTkButton(
            ctrl_inner, text="📷  Start Capture", height=38,
            fg_color=BLUE, hover_color=BLUE_HOVER, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._start_enroll)
        self.cam_btn.grid(row=1, column=1, padx=(0, 8))

        ctk.CTkButton(ctrl_inner, text="📁  Upload Photos", height=38,
                      fg_color=INPUT_BG, hover_color=BORDER_LIGHT,
                      border_width=1, border_color=BORDER, corner_radius=8,
                      font=ctk.CTkFont(size=12),
                      command=self._upload_photos).grid(row=1, column=2, padx=(0, 8))

        self.snap_btn = ctk.CTkButton(
            ctrl_inner, text="📸  Snap", height=38, width=90,
            fg_color=GREEN, hover_color=GREEN_HOVER, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._snap_face)
        self.snap_btn.grid(row=1, column=3, padx=(0, 8))

        self.enroll_stop = ctk.CTkButton(
            ctrl_inner, text="⏹ Stop", height=38, width=80,
            fg_color=RED, hover_color=RED_HOVER, corner_radius=8,
            font=ctk.CTkFont(size=12),
            command=self._stop_enroll)
        self.enroll_stop.grid(row=1, column=4)

        # Body: camera preview + thumbnail sidebar
        body = ctk.CTkFrame(page, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        # Camera preview
        cam_card = ctk.CTkFrame(body, fg_color=CARD_BG, corner_radius=12,
                                 border_width=1, border_color=BORDER)
        cam_card.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.enroll_video = ctk.CTkLabel(
            cam_card,
            text="Enter a name → click Start Capture\n\n"
                 "Position your face and click Snap to save frames\n"
                 "Capture 5–10 images with varied angles",
            font=ctk.CTkFont(size=14), text_color=TEXT_DIM)
        self.enroll_video.pack(fill="both", expand=True, padx=2, pady=2)

        # Thumbnails sidebar
        thumb_card = ctk.CTkFrame(body, width=200, fg_color=CARD_BG, corner_radius=12,
                                   border_width=1, border_color=BORDER)
        thumb_card.pack(side="right", fill="y")
        thumb_card.pack_propagate(False)

        ctk.CTkLabel(thumb_card, text="Captured",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=TEXT_WHITE).pack(pady=(14, 2))
        self.cap_count = ctk.CTkLabel(thumb_card, text="0 images",
                                       font=ctk.CTkFont(size=11),
                                       text_color=TEXT_DIM)
        self.cap_count.pack(pady=(0, 8))

        ctk.CTkFrame(thumb_card, height=1, fg_color=BORDER).pack(fill="x", padx=12)

        self.thumb_scroll = ctk.CTkScrollableFrame(thumb_card, fg_color="transparent",
                                                    scrollbar_button_color=BORDER)
        self.thumb_scroll.pack(fill="both", expand=True, padx=6, pady=6)

    # ──────────────────────────────────────────────────────────
    #  PAGE: GALLERY
    # ──────────────────────────────────────────────────────────
    def _build_gallery_page(self):
        page = ctk.CTkFrame(self.page_container, fg_color=DARK_BG, corner_radius=0)
        self.pages["gallery"] = page

        # Header
        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.pack(fill="x", padx=24, pady=(20, 12))
        ctk.CTkLabel(hdr, text="Gallery Manager",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_WHITE).pack(side="left")
        ctk.CTkButton(hdr, text="🔄 Refresh", height=34, width=100,
                      fg_color=BLUE, hover_color=BLUE_HOVER, corner_radius=8,
                      command=self._refresh_gallery).pack(side="right", padx=(8, 0))
        ctk.CTkButton(hdr, text="🗑  Delete Person", height=34,
                      fg_color=RED, hover_color=RED_HOVER, corner_radius=8,
                      command=self._delete_person_action).pack(side="right")

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        # Person list
        list_card = ctk.CTkFrame(body, width=220, fg_color=CARD_BG, corner_radius=12,
                                  border_width=1, border_color=BORDER)
        list_card.pack(side="left", fill="y", padx=(0, 12))
        list_card.pack_propagate(False)

        ctk.CTkLabel(list_card, text="Enrolled Persons",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=TEXT_WHITE).pack(pady=(14, 8))
        ctk.CTkFrame(list_card, height=1, fg_color=BORDER).pack(fill="x", padx=12)
        self.person_scroll = ctk.CTkScrollableFrame(list_card, fg_color="transparent",
                                                     scrollbar_button_color=BORDER)
        self.person_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # Image grid
        self.img_grid_card = ctk.CTkFrame(body, fg_color=CARD_BG, corner_radius=12,
                                           border_width=1, border_color=BORDER)
        self.img_grid_card.pack(side="right", fill="both", expand=True)
        self.img_grid_scroll = ctk.CTkScrollableFrame(self.img_grid_card,
                                                       fg_color="transparent",
                                                       scrollbar_button_color=BORDER)
        self.img_grid_scroll.pack(fill="both", expand=True, padx=8, pady=8)

        self._sel_person = None

    # ──────────────────────────────────────────────────────────
    #  PAGE: SETTINGS
    # ──────────────────────────────────────────────────────────
    def _build_settings_page(self):
        page = ctk.CTkFrame(self.page_container, fg_color=DARK_BG, corner_radius=0)
        self.pages["settings"] = page

        hdr = ctk.CTkFrame(page, fg_color="transparent")
        hdr.pack(fill="x", padx=24, pady=(20, 16))
        ctk.CTkLabel(hdr, text="Settings",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_WHITE).pack(side="left")

        # Settings card
        card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=12,
                             border_width=1, border_color=BORDER)
        card.pack(fill="x", padx=24, pady=(0, 12))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=24, pady=24)

        row = 0

        # Gallery path
        ctk.CTkLabel(inner, text="Gallery Folder",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=TEXT_LIGHT).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        gf = ctk.CTkFrame(inner, fg_color="transparent")
        gf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 16))
        self.gal_entry = ctk.CTkEntry(gf, height=36, fg_color=INPUT_BG,
                                       border_color=BORDER, text_color=TEXT_WHITE,
                                       font=ctk.CTkFont(size=12))
        self.gal_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.gal_entry.insert(0, self.gallery_path)
        ctk.CTkButton(gf, text="Browse", height=36, width=80,
                      fg_color=INPUT_BG, hover_color=BORDER_LIGHT,
                      border_width=1, border_color=BORDER,
                      command=self._browse_gallery).pack(side="right")
        row += 1

        # Model
        ctk.CTkLabel(inner, text="Model",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=TEXT_LIGHT).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        self.model_var = ctk.StringVar(value="buffalo_l")
        ctk.CTkOptionMenu(inner, variable=self.model_var,
                          values=["buffalo_l", "buffalo_s", "antelopev2"],
                          fg_color=INPUT_BG, button_color=BLUE,
                          button_hover_color=BLUE_HOVER,
                          width=200).grid(row=row, column=0, sticky="w", pady=(0, 16))
        row += 1

        # Threshold
        ctk.CTkLabel(inner, text="Recognition Threshold",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=TEXT_LIGHT).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        tf = ctk.CTkFrame(inner, fg_color="transparent")
        tf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 16))
        self.thresh_slider = ctk.CTkSlider(
            tf, from_=0.20, to=0.70, number_of_steps=50,
            progress_color=BLUE, button_color=BLUE,
            button_hover_color=BLUE_HOVER, width=300,
            command=self._on_thresh)
        self.thresh_slider.set(0.40)
        self.thresh_slider.pack(side="left")
        self.thresh_val = ctk.CTkLabel(tf, text="0.40",
                                        font=ctk.CTkFont(size=14, weight="bold"),
                                        text_color=BLUE)
        self.thresh_val.pack(side="left", padx=(12, 0))
        row += 1

        # Camera
        ctk.CTkLabel(inner, text="Camera Index",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=TEXT_LIGHT).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        self.cam_var = ctk.StringVar(value="0")
        ctk.CTkEntry(inner, textvariable=self.cam_var, width=80, height=36,
                     fg_color=INPUT_BG, border_color=BORDER,
                     text_color=TEXT_WHITE).grid(row=row, column=0, sticky="w", pady=(0, 16))

        # Descriptions
        desc = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=12,
                             border_width=1, border_color=BORDER)
        desc.pack(fill="x", padx=24, pady=(0, 20))
        desc_inner = ctk.CTkFrame(desc, fg_color="transparent")
        desc_inner.pack(fill="x", padx=24, pady=16)
        tips = ("Threshold Guide:  0.30 = lenient  •  0.40 = balanced  •  0.55 = strict\n"
                "Model Guide:  buffalo_l = best accuracy (~3 FPS)  •  buffalo_s = faster (~6 FPS)\n"
                "Camera:  0 = default webcam  •  1 = external camera")
        ctk.CTkLabel(desc_inner, text=tips, font=ctk.CTkFont(size=12),
                     text_color=TEXT_DIM, justify="left").pack(anchor="w")

    # ──────────────────────────────────────────────────────────
    #  STATUS HELPER
    # ──────────────────────────────────────────────────────────
    def _status(self, msg):
        self.status_text.configure(text=msg)

    # ──────────────────────────────────────────────────────────
    #  SETTINGS ACTIONS
    # ──────────────────────────────────────────────────────────
    def _browse_gallery(self):
        d = filedialog.askdirectory(title="Select Gallery Folder")
        if d:
            self.gal_entry.delete(0, "end")
            self.gal_entry.insert(0, d)
            self.gallery_path = d

    def _on_thresh(self, val):
        self.threshold = round(val, 2)
        self.thresh_val.configure(text=f"{self.threshold:.2f}")

    # ──────────────────────────────────────────────────────────
    #  LOAD MODEL
    # ──────────────────────────────────────────────────────────
    def _load_model(self):
        self.load_btn.configure(state="disabled", text="Loading…")
        self._status("Loading model — first time may download ~300 MB…")
        self.model_dot.set_color(AMBER)
        self.model_status_lbl.configure(text="  Loading…")

        def _work():
            try:
                self.engine.load_model(name=self.model_var.get())
                self.after(0, lambda: (
                    self.load_btn.configure(state="normal", text="✓  Model Loaded",
                                            fg_color=GREEN, hover_color=GREEN_HOVER),
                    self.model_dot.set_color(GREEN),
                    self.model_status_lbl.configure(text="  Model ready"),
                    self._status("Model loaded successfully.")
                ))
            except Exception as e:
                self.after(0, lambda: (
                    messagebox.showerror("Error", str(e)),
                    self.load_btn.configure(state="normal", text="⚡  Load Model",
                                            fg_color=BLUE, hover_color=BLUE_HOVER),
                    self.model_dot.set_color(RED),
                    self.model_status_lbl.configure(text="  Load failed"),
                    self._status(f"Error: {e}")
                ))

        threading.Thread(target=_work, daemon=True).start()

    # ──────────────────────────────────────────────────────────
    #  BUILD GALLERY
    # ──────────────────────────────────────────────────────────
    def _build_gallery(self):
        gp = self.gal_entry.get().strip()
        if not gp:
            messagebox.showwarning("Warning", "Set the gallery folder in Settings.")
            return
        self.gallery_path = gp
        self.engine.set_gallery(gp)

        if not self.engine.loaded:
            messagebox.showwarning("Warning", "Load the model first.")
            return

        self.build_btn.configure(state="disabled", text="Building…")

        def _work():
            try:
                self.engine.build_gallery(
                    force=True,
                    on_progress=lambda m: self.after(0, self._status, m))
                n = len(self.engine.labels)
                self.after(0, lambda: (
                    self.m_ids.set(str(n)),
                    self.build_btn.configure(state="normal", text="🔨  Build Gallery"),
                    self._status(f"Gallery built — {n} identities ready.")
                ))
            except Exception as e:
                self.after(0, lambda: (
                    messagebox.showerror("Error", str(e)),
                    self.build_btn.configure(state="normal", text="🔨  Build Gallery"),
                    self._status(f"Error: {e}")
                ))

        threading.Thread(target=_work, daemon=True).start()

    # ──────────────────────────────────────────────────────────
    #  LIVE RECOGNITION
    # ──────────────────────────────────────────────────────────
    def _toggle_live(self):
        if self.running:
            self._stop_live()
            return

        if not self.engine.loaded:
            messagebox.showwarning("Warning", "Load the model first.")
            return
        if self.engine.gallery_embs is None:
            messagebox.showwarning("Warning", "Build gallery first.")
            return

        cid = int(self.cam_var.get())
        self.cap = cv2.VideoCapture(cid, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cid)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {cid}")
            return

        self.running = True
        self.live_toggle.configure(text="⏹  Stop Camera", fg_color=RED,
                                    hover_color=RED_HOVER)
        self.m_status.set("Live")
        self._status("Recognition running.")
        threading.Thread(target=self._live_loop, daemon=True).start()

    def _live_loop(self):
        t0 = time.time()
        cnt = 0
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            results = self.engine.recognize(frame, self.threshold)
            cnt += 1
            fps = cnt / max(1e-6, time.time() - t0)

            # Draw
            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                known = r["name"] != "Unknown"
                c = (16, 185, 129) if known else (239, 68, 68)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

                label = f"{r['name']}  {r['score']:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                sc, th = 0.55, 2
                (tw, tht), _ = cv2.getTextSize(label, font, sc, th)
                ly = max(0, y1 - tht - 12)
                cv2.rectangle(frame, (x1, ly), (x1+tw+10, ly+tht+10), c, -1)
                cv2.putText(frame, label, (x1+5, ly+tht+5), font, sc,
                            (255, 255, 255), th, cv2.LINE_AA)

                if r["kps"] is not None:
                    for px, py in r["kps"]:
                        cv2.circle(frame, (int(px), int(py)), 2, (255, 255, 255), -1)

            # Convert to Tk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            try:
                lw = self.video_label.winfo_width()
                lh = self.video_label.winfo_height()
                if lw > 20 and lh > 20:
                    img.thumbnail((lw - 4, lh - 4), Image.LANCZOS)
            except:
                pass
            photo = ctk.CTkImage(light_image=img, dark_image=img,
                                  size=(img.width, img.height))

            def _up(p=photo, f=fps, n=len(results)):
                try:
                    self.video_label.configure(image=p, text="")
                    self.video_label._img = p
                    self.m_fps.set(f"{f:.1f}")
                    self.m_faces.set(str(n))
                except:
                    pass

            self.after(0, _up)
            time.sleep(0.005)

        self.after(0, self._live_stopped)

    def _stop_live(self):
        self.running = False

    def _live_stopped(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.live_toggle.configure(text="▶  Start Camera", fg_color=GREEN,
                                    hover_color=GREEN_HOVER)
        self.m_status.set("Idle")
        self.m_fps.set("—")
        self.video_label.configure(image=None, text="Camera stopped")
        self._status("Camera stopped.")

    # ──────────────────────────────────────────────────────────
    #  ENROLLMENT
    # ──────────────────────────────────────────────────────────
    def _start_enroll(self):
        name = self.enroll_name.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Enter a person name.")
            return
        if not self.engine.loaded:
            messagebox.showwarning("Warning", "Load the model first.")
            return

        gp = self.gal_entry.get().strip()
        if not gp:
            messagebox.showwarning("Warning", "Set gallery folder in Settings.")
            return
        self.gallery_path = gp
        self.engine.set_gallery(gp)
        self.engine.set_gallery(gp)
        (self.engine.gallery_root / name).mkdir(parents=True, exist_ok=True)

        if self.cap and self.cap.isOpened():
            self.running = False
            time.sleep(0.15)
            self.cap.release()

        cid = int(self.cam_var.get())
        self.cap = cv2.VideoCapture(cid, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cid)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {cid}")
            return

        self.enroll_active = True
        self.running = True
        self.snap_frame = None
        self._enroll_person = name
        self._status(f"Enrolling: {name}  — click Snap to capture")
        threading.Thread(target=self._enroll_loop, daemon=True).start()

    def _enroll_loop(self):
        while self.running and self.enroll_active and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.snap_frame = frame.copy()

            display = frame.copy()
            try:
                faces = self.engine.app.get(display)
                for f in faces:
                    x1, y1, x2, y2 = [int(v) for v in f.bbox]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (59, 130, 246), 2)
                    cv2.putText(display, "Face", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 130, 246), 2)
            except:
                pass

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            try:
                lw = self.enroll_video.winfo_width()
                lh = self.enroll_video.winfo_height()
                if lw > 20 and lh > 20:
                    img.thumbnail((lw-4, lh-4), Image.LANCZOS)
            except:
                pass
            photo = ctk.CTkImage(light_image=img, dark_image=img,
                                  size=(img.width, img.height))

            def _up(p=photo):
                try:
                    self.enroll_video.configure(image=p, text="")
                    self.enroll_video._img = p
                except:
                    pass

            self.after(0, _up)
            time.sleep(0.03)

        self.after(0, self._enroll_stopped)

    def _snap_face(self):
        if not self.enroll_active or self.snap_frame is None:
            messagebox.showinfo("Info", "Start capture first.")
            return
        fp = self.engine.save_capture(self._enroll_person, self.snap_frame)
        self._status(f"Saved: {fp.name}")
        self._update_thumbs()

    def _update_thumbs(self):
        imgs = self.engine.person_images(self._enroll_person)
        self.cap_count.configure(text=f"{len(imgs)} images")
        for w in self.thumb_scroll.winfo_children():
            w.destroy()
        for p in imgs[-12:]:
            try:
                im = Image.open(p)
                im.thumbnail((80, 80))
                ph = ctk.CTkImage(light_image=im, dark_image=im, size=(80, 80))
                lbl = ctk.CTkLabel(self.thumb_scroll, image=ph, text="",
                                   fg_color=INPUT_BG, corner_radius=6)
                lbl._ph = ph
                lbl.pack(pady=3)
            except:
                pass

    def _stop_enroll(self):
        self.enroll_active = False
        self.running = False

    def _enroll_stopped(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.enroll_video.configure(image=None,
                                     text="Capture stopped.\n\nRemember to Build Gallery after enrolling.")
        self._status("Enrollment stopped. Build Gallery to update embeddings.")

    def _upload_photos(self):
        name = self.enroll_name.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Enter a person name.")
            return
        gp = self.gal_entry.get().strip()
        if not gp:
            messagebox.showwarning("Warning", "Set gallery folder in Settings.")
            return
        self.gallery_path = gp
        self.engine.set_gallery(gp)

        files = filedialog.askopenfilenames(
            title="Select Face Photos",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not files:
            return
        self.engine.add_files(name, files)
        self._enroll_person = name
        self._status(f"Added {len(files)} images for '{name}'")
        self._update_thumbs()

    # ──────────────────────────────────────────────────────────
    #  GALLERY
    # ──────────────────────────────────────────────────────────
    def _refresh_gallery(self):
        gp = self.gal_entry.get().strip()
        if not gp:
            return
        self.gallery_path = gp
        self.engine.set_gallery(gp)

        for w in self.person_scroll.winfo_children():
            w.destroy()

        persons = self.engine.persons()
        for pname in persons:
            n = len(self.engine.person_images(pname))
            is_sel = pname == self._sel_person
            btn = ctk.CTkButton(
                self.person_scroll,
                text=f"  {pname}  ({n})",
                anchor="w", height=38, corner_radius=6,
                fg_color=BLUE if is_sel else INPUT_BG,
                hover_color=BLUE_HOVER if is_sel else BORDER_LIGHT,
                font=ctk.CTkFont(size=12),
                command=lambda n=pname: self._select_person(n))
            btn.pack(fill="x", pady=2)

    def _select_person(self, name):
        self._sel_person = name
        self._refresh_gallery()

        for w in self.img_grid_scroll.winfo_children():
            w.destroy()

        imgs = self.engine.person_images(name)
        ctk.CTkLabel(self.img_grid_scroll,
                     text=f"{name}  —  {len(imgs)} enrolled images",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=TEXT_WHITE).pack(anchor="w", padx=4, pady=(4, 8))

        row_frame = None
        for i, p in enumerate(imgs):
            if i % 6 == 0:
                row_frame = ctk.CTkFrame(self.img_grid_scroll, fg_color="transparent")
                row_frame.pack(fill="x", pady=4)
            try:
                im = Image.open(p)
                im.thumbnail((120, 120))
                ph = ctk.CTkImage(light_image=im, dark_image=im, size=(120, 120))
                lbl = ctk.CTkLabel(row_frame, image=ph, text="",
                                   fg_color=INPUT_BG, corner_radius=8)
                lbl._ph = ph
                lbl.pack(side="left", padx=4)
            except:
                pass

    def _delete_person_action(self):
        if not self._sel_person:
            messagebox.showinfo("Info", "Select a person from the list.")
            return
        if messagebox.askyesno("Confirm",
                                f"Delete '{self._sel_person}' and all images?"):
            self.engine.delete_person(self._sel_person)
            self._sel_person = None
            self._refresh_gallery()
            for w in self.img_grid_scroll.winfo_children():
                w.destroy()
            self._status("Person deleted. Rebuild gallery to update.")

    # ──────────────────────────────────────────────────────────
    #  CLEANUP
    # ──────────────────────────────────────────────────────────
    def _quit(self):
        self.running = False
        self.enroll_active = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = FaceApp()
    app.mainloop()
