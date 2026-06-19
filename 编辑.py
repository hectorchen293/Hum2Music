#!/usr/bin/env python3
"""
哼唱编曲助手 Pro v2.1 - 完整多轨版本（含录音键、旋律编辑器）
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import os, sys, subprocess, tempfile, time, shutil, re
import urllib.request
import platform
import warnings

# ---------- 自动依赖安装 ----------
def install_package(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def check_and_install():
    required = {
        "numpy": "numpy", "scipy": "scipy", "librosa": "librosa",
        "sounddevice": "sounddevice", "soundfile": "soundfile",
        "pydub": "pydub", "mido": "mido",
        "whisper": "openai-whisper",
        "moviepy": "moviepy",
        "PIL": "pillow",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("正在安装缺失依赖：", missing)
        for pkg in missing:
            install_package(pkg)

check_and_install()

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from pydub import AudioSegment
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import whisper

# 兼容 moviepy 2.x
try:
    from moviepy.editor import ColorClip, AudioFileClip, CompositeVideoClip, TextClip
except ImportError:
    from moviepy import ColorClip, AudioFileClip, CompositeVideoClip, TextClip

from PIL import Image, ImageDraw, ImageTk

# ---------- 全局配置 ----------
SAMPLE_RATE = 44100
TEMPO = 120
SOUNDFONT_FILE = "MS_Basic-v2.0.0.sf3"
SOUNDFONT_URL = "https://github.com/GM-Sound-Fonts/Timbres-of-Heaven/raw/main/Timbres_Of_Heaven_GM.sf2"

# ---------- 工具函数 ----------
def get_fluid_path():
    system = platform.system()
    if system == "Windows":
        possible = [r"C:\Program Files\FluidSynth\bin\fluidsynth.exe", "fluidsynth"]
    else:
        possible = ["fluidsynth"]
    for p in possible:
        if shutil.which(p):
            return p
    return None

def download_soundfont():
    local = os.path.join(os.path.dirname(__file__), SOUNDFONT_FILE)
    if os.path.exists(local):
        return local
    try:
        print(f"正在下载 SoundFont: {SOUNDFONT_URL}")
        urllib.request.urlretrieve(SOUNDFONT_URL, local)
        print("下载完成")
        return local
    except:
        messagebox.showwarning("下载失败", "自动下载 SoundFont 失败，请手动放置 Timbres_Of_Heaven_GM.sf2")
        return None

# ---------- 合成引擎 ----------
class SynthEngine:
    def __init__(self, soundfont):
        self.sf = soundfont
        self.cmd = get_fluid_path()
        if not self.cmd:
            raise RuntimeError("请安装 FluidSynth 并将 fluidsynth 加入 PATH")

    def render(self, midi_path, program=0, tempo=TEMPO, timeout=60):
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        cmd = [
            self.cmd, "-ni",
            "-g", "2.0",
            "-r", str(SAMPLE_RATE),
            "-T", "wav",
            "-F", out,
            self.sf,
            midi_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            return out
        except Exception as e:
            messagebox.showerror("合成失败", f"FluidSynth 错误：{e}")
            return None

# ---------- MIDI 生成 ----------
def create_midi(notes, times, tempo=TEMPO, transpose=0, instrument=0, is_drum=False):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    tempo_us = int(60_000_000 / tempo)
    track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    if is_drum:
        track.append(Message('program_change', channel=9, program=0, time=0))
    else:
        track.append(Message('program_change', channel=0, program=instrument, time=0))

    ticks_per_beat = 480
    sec_per_tick = 60 / (tempo * ticks_per_beat)

    last_tick = 0
    for i, (note, onset) in enumerate(zip(notes, times)):
        if note is None:
            continue
        real_note = note + transpose
        if i < len(times)-1:
            dur_sec = times[i+1] - onset
        else:
            dur_sec = 0.5
        dur_ticks = max(1, int(dur_sec / sec_per_tick))
        start_tick = int(onset / sec_per_tick)
        delta = max(0, start_tick - last_tick)

        channel = 9 if is_drum else 0
        track.append(Message('note_on', channel=channel, note=real_note, velocity=80, time=delta))
        track.append(Message('note_off', channel=channel, note=real_note, velocity=80, time=dur_ticks))
        last_tick = start_tick + dur_ticks

    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False).name
    midi.save(tmp)
    return tmp

# ---------- 鼓轨生成 ----------
DRUM_MAP = {'kick': 36, 'snare': 38, 'hihat': 42, 'crash': 49}
def generate_drum_pattern(duration_sec, start_beat=0, tempo=TEMPO):
    beat_sec = 60 / tempo
    total_beats = int(duration_sec / beat_sec) + 1
    notes, times = [], []
    for beat in range(total_beats):
        abs_time = beat * beat_sec
        if abs_time < start_beat * beat_sec:
            continue
        if beat % 4 == 0:
            notes.append(DRUM_MAP['kick'])
            times.append(abs_time)
        if beat % 4 == 2:
            notes.append(DRUM_MAP['snare'])
            times.append(abs_time)
        notes.append(DRUM_MAP['hihat'])
        times.append(abs_time)
        notes.append(DRUM_MAP['hihat'])
        times.append(abs_time + beat_sec/2)
    return notes, times

# ---------- 歌词识别 ----------
class LyricRecognizer:
    def __init__(self):
        self.model = None
    def load_model(self, model_size="small"):
        try:
            self.model = whisper.load_model(model_size)
            return True
        except:
            return False
    def transcribe(self, audio_path, language="zh"):
        if not self.model:
            if not self.load_model("small"):
                raise RuntimeError("无法加载 Whisper 模型")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.model.transcribe(audio_path, word_timestamps=True, language=language)
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append((w["word"], w["start"], w["end"]))
        return words

# ---------- 主程序 ----------
class Hum2MusicPro:
    def __init__(self, root):
        self.root = root
        self.root.title("哼唱编曲助手 Pro v2.1")
        self.root.geometry("1200x850")

        # 合成器
        self.synth = None
        sf = download_soundfont()
        if sf and get_fluid_path():
            try:
                self.synth = SynthEngine(sf)
            except RuntimeError as e:
                messagebox.showwarning("FluidSynth", str(e))

        # 轨道数据
        self.tracks = []
        self.pitches = []
        self.onset_times = []
        self.lyric_words = []

        # 录音相关
        self.recording = False
        self.audio_data = []
        self.record_target_track = None

        # UI 变量
        self.status_var = tk.StringVar(value="就绪")
        self.wave_images = {}

        self.setup_ui()

    def setup_ui(self):
        # 顶部工具栏
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_rec = ttk.Button(toolbar, text="● 开始录音", command=self.toggle_record)
        self.btn_rec.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="新建重唱轨 (指定起始)", command=self.new_audio_track).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="分析选中旋律", command=self.analyze_selected_track).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="添加乐器轨", command=self.add_instrument_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="添加鼓轨", command=self.add_drum_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="识别歌词", command=self.auto_lyrics).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="▶ 播放", command=self.play_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="⏹ 停止", command=self.stop_audio).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="导出音频", command=self.export_audio).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="导出 MP4", command=self.export_mp4).pack(side=tk.RIGHT, padx=2)

        # 歌词显示区
        lyric_frame = ttk.LabelFrame(self.root, text="歌词 (识别后可编辑)")
        lyric_frame.pack(fill=tk.X, padx=5, pady=2)
        self.lyric_text = tk.Text(lyric_frame, height=3, font=("Microsoft YaHei", 10))
        self.lyric_text.pack(fill=tk.X, padx=5, pady=2)

        # 旋律编辑器
        melody_frame = ttk.LabelFrame(self.root, text="旋律编辑器（可手动增删音符，双击修改）")
        melody_frame.pack(fill=tk.X, padx=5, pady=2)
        mel_tool = ttk.Frame(melody_frame)
        mel_tool.pack(fill=tk.X)
        ttk.Button(mel_tool, text="添加音符", command=self.add_note).pack(side=tk.LEFT, padx=2)
        ttk.Button(mel_tool, text="删除选中", command=self.delete_note).pack(side=tk.LEFT, padx=2)
        columns = ("time", "note", "midi")
        self.tree_pitch = ttk.Treeview(melody_frame, columns=columns, show="headings", height=5)
        self.tree_pitch.heading("time", text="起始时间(s)")
        self.tree_pitch.heading("note", text="音名")
        self.tree_pitch.heading("midi", text="MIDI值")
        self.tree_pitch.column("time", width=100)
        self.tree_pitch.column("note", width=100)
        self.tree_pitch.column("midi", width=80)
        self.tree_pitch.pack(fill=tk.X, padx=5, pady=2)
        self.tree_pitch.bind("<Double-1>", self.edit_pitch_cell)

        # 音轨列表（可滚动）
        track_container = ttk.LabelFrame(self.root, text="音轨列表")
        track_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.track_canvas = tk.Canvas(track_container, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(track_container, orient=tk.VERTICAL, command=self.track_canvas.yview)
        self.track_frame = ttk.Frame(self.track_canvas)

        self.track_frame.bind("<Configure>", lambda e: self.track_canvas.configure(scrollregion=self.track_canvas.bbox("all")))
        self.track_canvas.create_window((0,0), window=self.track_frame, anchor="nw")
        self.track_canvas.configure(yscrollcommand=scrollbar.set)

        self.track_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- 录音控制 ----------
    def toggle_record(self):
        if not self.recording:
            self.recording = True
            self.btn_rec.config(text="■ 停止录音")
            self.status_var.set("录音中...")
            self.audio_data = []
            threading.Thread(target=self._direct_record_thread, daemon=True).start()
        else:
            self.recording = False
            self.btn_rec.config(text="● 开始录音")
            self.status_var.set("录音完成")

    def _direct_record_thread(self):
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)
        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)
            file_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(file_path, audio, SAMPLE_RATE)
            new_track = {
                'type': 'audio',
                'name': f'录音 {len(self.tracks)+1}',
                'file': file_path,
                'vol': 1.0,
                'offset': 0.0,
                'mute': False,
                'solo': False,
            }
            self.tracks.append(new_track)
            self.root.after(0, self.redraw_tracks)
            self.status_var.set(f"录音完成，长度 {len(audio)/SAMPLE_RATE:.1f} 秒")

    def new_audio_track(self):
        offset = simpledialog.askfloat("新建重唱轨", "输入起始时间（秒），默认0：", initialvalue=0.0, minvalue=0.0)
        if offset is None:
            return
        self.record_target_track = len(self.tracks)
        self._start_recording_with_offset(offset)

    def _start_recording_with_offset(self, offset):
        if self.recording:
            return
        self.recording = True
        self.status_var.set(f"录音中... (偏移 {offset:.1f}s)")
        self.audio_data = []
        threading.Thread(target=self._record_thread_offset, args=(offset,), daemon=True).start()

    def _record_thread_offset(self, offset):
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            while self.recording:
                sd.sleep(100)
        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)
            file_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(file_path, audio, SAMPLE_RATE)
            new_track = {
                'type': 'audio',
                'name': f'重唱 {len(self.tracks)+1}',
                'file': file_path,
                'vol': 1.0,
                'offset': offset,
                'mute': False,
                'solo': False,
            }
            self.tracks.append(new_track)
            self.root.after(0, self.redraw_tracks)
            self.status_var.set(f"重唱完成，长度 {len(audio)/SAMPLE_RATE:.1f} 秒")

    # ---------- 轨道 UI ----------
    def redraw_tracks(self):
        for widget in self.track_frame.winfo_children():
            widget.destroy()
        self.wave_images.clear()
        for idx, track in enumerate(self.tracks):
            self._create_track_row(idx, track)

    def _create_track_row(self, idx, track):
        row_frame = ttk.Frame(self.track_frame)
        row_frame.pack(fill=tk.X, padx=5, pady=2)

        wave_canvas = tk.Canvas(row_frame, width=200, height=40, bg='black', highlightthickness=0)
        wave_canvas.pack(side=tk.LEFT, padx=2)
        self._draw_wave_on_canvas(wave_canvas, track.get('file'))

        info_frame = ttk.Frame(row_frame)
        info_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(info_frame, text=f"{track['name']}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"类型: {track.get('type','')}  偏移: {track.get('offset',0):.1f}s", 
                  font=("Arial",8)).pack(anchor=tk.W)

        ctrl_frame = ttk.Frame(row_frame)
        ctrl_frame.pack(side=tk.RIGHT, padx=5)

        vol_var = tk.DoubleVar(value=track.get('vol',1.0))
        ttk.Label(ctrl_frame, text="音量").pack(side=tk.LEFT)
        vol_scale = ttk.Scale(ctrl_frame, from_=0, to=2, variable=vol_var, orient=tk.HORIZONTAL, length=80,
                              command=lambda v, i=idx: self.update_vol(i, float(v)))
        vol_scale.pack(side=tk.LEFT, padx=2)

        mute_var = tk.BooleanVar(value=track.get('mute', False))
        ttk.Checkbutton(ctrl_frame, text="静音", variable=mute_var,
                        command=lambda i=idx: self.toggle_mute(i)).pack(side=tk.LEFT, padx=2)

        solo_var = tk.BooleanVar(value=track.get('solo', False))
        ttk.Checkbutton(ctrl_frame, text="独奏", variable=solo_var,
                        command=lambda i=idx: self.toggle_solo(i)).pack(side=tk.LEFT, padx=2)

        ttk.Button(ctrl_frame, text="删除", command=lambda i=idx: self.delete_track(i)).pack(side=tk.LEFT, padx=2)

        track['vol_var'] = vol_var
        track['mute_var'] = mute_var
        track['solo_var'] = solo_var

    def _draw_wave_on_canvas(self, canvas, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            canvas.create_text(100,20, text="无波形", fill='white')
            return
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            w, h = 200, 40
            if len(y) > w:
                indices = np.linspace(0, len(y)-1, w, dtype=int)
                data = y[indices]
            else:
                data = y
            data = np.int16(data / (np.max(np.abs(data)) + 1e-10) * (h//2 - 2))
            img = Image.new('RGB', (w, h), 'black')
            draw = ImageDraw.Draw(img)
            center = h//2
            for x, val in enumerate(data):
                draw.line([(x, center - val), (x, center + val)], fill='#00ff88', width=1)
            self.wave_images[canvas] = ImageTk.PhotoImage(img)
            canvas.create_image(0,0, anchor=tk.NW, image=self.wave_images[canvas])
        except:
            canvas.create_text(100,20, text="读取失败", fill='white')

    def toggle_mute(self, idx):
        self.tracks[idx]['mute'] = not self.tracks[idx].get('mute', False)

    def toggle_solo(self, idx):
        self.tracks[idx]['solo'] = not self.tracks[idx].get('solo', False)

    def update_vol(self, idx, val):
        self.tracks[idx]['vol'] = val

    def delete_track(self, idx):
        del self.tracks[idx]
        self.redraw_tracks()

    # ---------- 旋律分析 ----------
    def analyze_selected_track(self):
        audio_track = None
        for t in self.tracks:
            if t['type'] == 'audio':
                audio_track = t
                break
        if not audio_track:
            messagebox.showerror("错误", "请先录制至少一个音频轨道")
            return
        self.status_var.set("分析中...")
        y, sr = librosa.load(audio_track['file'], sr=SAMPLE_RATE)
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        times = librosa.times_like(f0, sr=sr)
        pitches, onsets = [], []
        last_p = None
        grp_start = None
        for t, f, v in zip(times, f0, voiced):
            if v and not np.isnan(f):
                midi = int(round(librosa.hz_to_midi(f)))
                if last_p != midi:
                    if last_p is not None and grp_start is not None:
                        onsets.append(grp_start)
                        pitches.append(last_p)
                    last_p = midi
                    grp_start = t
            else:
                if last_p is not None and grp_start is not None:
                    onsets.append(grp_start)
                    pitches.append(last_p)
                last_p = None
                grp_start = None
        if last_p is not None and grp_start is not None:
            onsets.append(grp_start)
            pitches.append(last_p)
        self.pitches = pitches
        self.onset_times = onsets
        self.refresh_pitch_tree()
        self.status_var.set(f"识别到 {len(pitches)} 个音符")

    # ---------- 旋律编辑器 ----------
    def refresh_pitch_tree(self):
        self.tree_pitch.delete(*self.tree_pitch.get_children())
        for t, p in zip(self.onset_times, self.pitches):
            note_name = librosa.midi_to_note(p) if p is not None else "R"
            self.tree_pitch.insert("", "end", values=(f"{t:.2f}", note_name, p))

    def edit_pitch_cell(self, event):
        col = self.tree_pitch.identify_column(event.x)
        item = self.tree_pitch.selection()[0]
        if col == "#2":
            new = simpledialog.askstring("修改音高", "输入音名 (如 C4, D#5):")
            if new:
                try:
                    midi = librosa.note_to_midi(new)
                    idx = self.tree_pitch.index(item)
                    self.pitches[idx] = midi
                    self.tree_pitch.set(item, column="note", value=new)
                    self.tree_pitch.set(item, column="midi", value=midi)
                except:
                    messagebox.showerror("错误", "无效的音名")
        elif col == "#1":
            new = simpledialog.askfloat("修改时间", "起始时间(秒):")
            if new is not None:
                idx = self.tree_pitch.index(item)
                self.onset_times[idx] = new
                self.tree_pitch.set(item, column="time", value=f"{new:.2f}")

    def add_note(self):
        time = simpledialog.askfloat("添加音符", "起始时间(秒):", minvalue=0)
        if time is None:
            return
        note = simpledialog.askstring("添加音符", "音名 (如 C4):")
        if not note:
            return
        try:
            midi = librosa.note_to_midi(note)
        except:
            messagebox.showerror("错误", "无效音名")
            return
        self.onset_times.append(time)
        self.pitches.append(midi)
        sorted_pairs = sorted(zip(self.onset_times, self.pitches))
        self.onset_times, self.pitches = map(list, zip(*sorted_pairs))
        self.refresh_pitch_tree()

    def delete_note(self):
        sel = self.tree_pitch.selection()
        if sel:
            idx = self.tree_pitch.index(sel[0])
            del self.pitches[idx]
            del self.onset_times[idx]
            self.refresh_pitch_tree()

    # ---------- 乐器轨 ----------
    def add_instrument_dialog(self):
        if not self.pitches:
            messagebox.showerror("错误", "请先分析旋律或手动添加音符")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("添加乐器")
        dialog.geometry("250x180")
        ttk.Label(dialog, text="乐器:").pack(pady=5)
        instr = ttk.Combobox(dialog, values=["piano", "guitar", "violin", "flute", "trumpet"])
        instr.pack()
        instr.current(0)
        ttk.Label(dialog, text="移调 (半音):").pack(pady=5)
        trans_var = tk.IntVar(value=0)
        trans_scale = tk.Scale(dialog, from_=-12, to=12, orient=tk.HORIZONTAL, variable=trans_var)
        trans_scale.pack()
        def confirm():
            name = instr.get()
            trans = trans_var.get()
            self._add_instrument(name, trans)
            dialog.destroy()
        ttk.Button(dialog, text="确定", command=confirm).pack(pady=10)

    def _add_instrument(self, name, transpose):
        prog_map = {"piano":0, "guitar":24, "violin":40, "flute":73, "trumpet":56}
        prog = prog_map.get(name, 0)
        midi_path = create_midi(self.pitches, self.onset_times, tempo=TEMPO,
                                transpose=transpose, instrument=prog)
        if not self.synth:
            messagebox.showerror("错误", "合成器未就绪")
            return
        wav = self.synth.render(midi_path, program=prog)
        if wav:
            track = {
                'type': 'instrument',
                'name': f'{name} (移调{transpose:+d})',
                'file': wav,
                'vol': 1.0,
                'offset': 0.0,
                'mute': False,
                'solo': False,
            }
            self.tracks.append(track)
            self.redraw_tracks()
            self.status_var.set(f"已添加 {name}")

    # ---------- 鼓轨 ----------
    def add_drum_dialog(self):
        if not self.pitches:
            messagebox.showerror("错误", "请先分析旋律或添加音符")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("添加鼓轨")
        dialog.geometry("250x120")
        ttk.Label(dialog, text="入鼓小节:").pack(pady=5)
        bar_var = tk.IntVar(value=1)
        bar_scale = tk.Scale(dialog, from_=1, to=32, orient=tk.HORIZONTAL, variable=bar_var)
        bar_scale.pack()
        def confirm():
            bar = bar_var.get() - 1
            self._add_drum(bar)
            dialog.destroy()
        ttk.Button(dialog, text="确定", command=confirm).pack(pady=10)

    def _add_drum(self, start_bar):
        beat_sec = 60 / TEMPO
        duration = max(self.onset_times) + 2 if self.onset_times else 10
        d_notes, d_times = generate_drum_pattern(duration, start_beat=start_bar*4, tempo=TEMPO)
        midi_path = create_midi(d_notes, d_times, tempo=TEMPO, is_drum=True)
        wav = self.synth.render(midi_path)
        if wav:
            track = {
                'type': 'drum',
                'name': f'鼓 (从{start_bar+1}小节)',
                'file': wav,
                'vol': 0.8,
                'offset': start_bar * 4 * beat_sec,
                'mute': False,
                'solo': False,
            }
            self.tracks.append(track)
            self.redraw_tracks()
            self.status_var.set("鼓轨已添加")

    # ---------- 歌词识别 ----------
    def auto_lyrics(self):
        audio_track = None
        for t in self.tracks:
            if t['type'] == 'audio':
                audio_track = t
                break
        if not audio_track:
            messagebox.showerror("错误", "请先录制人声轨")
            return
        self.status_var.set("识别歌词中...")
        try:
            rec = LyricRecognizer()
            words = rec.transcribe(audio_track['file'], language="zh")
            self.lyric_words = words
            self.lyric_text.delete(1.0, tk.END)
            lines = [f"[{w[1]:.2f}-{w[2]:.2f}] {w[0]}" for w in words]
            self.lyric_text.insert(tk.END, "\n".join(lines))
            self.status_var.set("歌词识别完成")
        except Exception as e:
            messagebox.showerror("失败", str(e))

    # ---------- 播放 ----------
    def play_all(self):
        mix = self._create_mix()
        if mix is None:
            messagebox.showinfo("提示", "没有可播放的音轨")
            return
        self.status_var.set("播放中...")
        threading.Thread(target=self._play_mix_thread, args=(mix,), daemon=True).start()

    def _play_mix_thread(self, mix):
        sd.play(mix, SAMPLE_RATE)
        sd.wait()
        self.root.after(0, lambda: self.status_var.set("播放完毕"))

    def stop_audio(self):
        sd.stop()
        self.status_var.set("已停止")

    def _create_mix(self):
        solo_tracks = [t for t in self.tracks if t.get('solo')]
        if solo_tracks:
            active_tracks = solo_tracks
        else:
            active_tracks = [t for t in self.tracks if not t.get('mute')]

        if not active_tracks:
            return None

        # 计算最大时长
        max_dur = 0
        for t in active_tracks:
            try:
                dur = librosa.get_duration(path=t['file'])
                end_time = t.get('offset', 0) + dur
                if end_time > max_dur:
                    max_dur = end_time
            except:
                continue
        if max_dur == 0:
            return None

        mix = AudioSegment.silent(duration=max_dur*1000, frame_rate=SAMPLE_RATE)
        for t in active_tracks:
            try:
                seg = AudioSegment.from_file(t['file'])
                vol_factor = t.get('vol', 1.0)
                if vol_factor != 1.0:
                    seg = seg + (20 * np.log10(vol_factor))  # dB 调整
                offset_ms = int(t.get('offset', 0) * 1000)
                mix = mix.overlay(seg, position=offset_ms)
            except Exception as e:
                print(f"跳过轨道 {t['name']}: {e}")

        samples = np.array(mix.get_array_of_samples(), dtype=np.float32)
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak * 0.9
        return samples.astype(np.float32)

    # ---------- 导出 ----------
    def export_audio(self):
        mix = self._create_mix()
        if mix is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3")])
        if path:
            sf.write(path, mix, SAMPLE_RATE)
            if path.endswith('.mp3'):
                AudioSegment.from_wav(path[:-4]+".wav").export(path, format="mp3")
            self.status_var.set(f"已导出: {path}")

    def export_mp4(self):
        if not self.lyric_words:
            messagebox.showerror("错误", "请先识别歌词")
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp4")
        if not path:
            return
        self.status_var.set("生成视频中...")
        threading.Thread(target=self._render_mp4, args=(path,), daemon=True).start()

    def _render_mp4(self, out_path):
        try:
            mix = self._create_mix()
            if mix is None:
                return
            tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(tmp_audio, mix, SAMPLE_RATE)

            srt = ""
            for i, (word, start, end) in enumerate(self.lyric_words):
                srt += f"{i+1}\n"
                srt += f"{self._format_time(start)} --> {self._format_time(end)}\n"
                srt += f"{word}\n\n"
            tmp_srt = tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w", encoding="utf-8")
            tmp_srt.write(srt)
            tmp_srt.close()

            audio_clip = AudioFileClip(tmp_audio)
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={audio_clip.duration}",
                    "-i", tmp_audio,
                    "-vf", f"subtitles={tmp_srt.name}:force_style='FontSize=24,Alignment=2'",
                    "-c:a", "aac", "-shortest", out_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            except FileNotFoundError:
                video_clip = ColorClip(size=(1280,720), color=(0,0,0), duration=audio_clip.duration)
                video_clip = video_clip.with_audio(audio_clip)
                video_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=24, verbose=False, logger=None)

            os.unlink(tmp_audio)
            os.unlink(tmp_srt.name)
            self.status_var.set(f"视频已导出: {out_path}")
            messagebox.showinfo("成功", "MP4 视频已生成！")
        except Exception as e:
            self.status_var.set("导出失败")
            messagebox.showerror("失败", str(e))

    def _format_time(self, seconds):
        ms = int(seconds * 1000)
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        mil = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{mil:03d}"

# ---------- 启动 ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = Hum2MusicPro(root)
    root.mainloop()
