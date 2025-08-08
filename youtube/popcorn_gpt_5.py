import math
import threading
import time
import requests
import numpy as np
import pygame
import mido
import sounddevice as sd

MIDI_URL = "https://bitmidi.com/uploads/85996.mid"
MIDI_PATH = "song.mid"

# -------- Synth/audio config --------
SAMPLE_RATE = 44100
BLOCK_SIZE  = 512
MAX_POLY    = 64
MASTER_GAIN = 0.22

# -------- Window (resizable) --------
WIN_W, WIN_H = 1400, 720

# -------- Colors --------
WHITE = (245, 245, 245)
GRAY  = (60, 60, 60)
BG    = (18, 18, 24)
RED   = (235, 60, 60)
DARK  = (30, 30, 40)
INK   = (230, 230, 240)
CYAN  = (120, 220, 255)

# -------- Piano range --------
LOW_NOTE  = 21   # A0
HIGH_NOTE = 108  # C8

# -------- Visuals --------
FPS          = 60
NOTE_SPEED   = 220.0
SPAWN_OFFSET = 8
FADE_TIME    = 0.25

# Scope settings
SCOPE_SECONDS = 0.5  # how much past audio to show

def midi_to_freq(n): return 440.0 * (2.0 ** ((n - 69) / 12.0))
def is_black(n): return (n % 12) in {1, 3, 6, 8, 10}

def download_midi(url, path):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

# ====================== Soft Synth ======================
class ADSR:
    def __init__(self, a=0.004, d=0.07, s=0.70, r=0.18, sr=SAMPLE_RATE):
        self.a = max(1, int(a * sr)); self.d = max(1, int(d * sr))
        self.s = float(s); self.r = max(1, int(r * sr))
        self.level = 0.0; self.state = "idle"; self.count = 0
    def note_on(self): self.state, self.count = "attack", 0
    def note_off(self):
        if self.state != "idle":
            self.state, self.count, self.start_level = "release", 0, self.level
    def render(self, n):
        env = np.empty(n, np.float32)
        for i in range(n):
            if self.state == "attack":
                self.level = (self.count + 1) / self.a
                if self.level >= 1.0 or self.count >= self.a: self.level, self.state, self.count = 1.0, "decay", 0
                else: self.count += 1
            elif self.state == "decay":
                t = (self.count + 1) / self.d
                self.level = 1.0 + (self.s - 1.0) * t
                if self.count >= self.d: self.level, self.state = self.s, "sustain"
                else: self.count += 1
            elif self.state == "sustain":
                self.level = self.s
            elif self.state == "release":
                t = (self.count + 1) / self.r
                self.level = self.start_level * max(0.0, 1.0 - t)
                if self.count >= self.r: self.level, self.state = 0.0, "idle"
                else: self.count += 1
            else:
                self.level = 0.0
            env[i] = self.level
        return env
    def done(self): return self.state == "idle" and self.level <= 1e-5

class Voice:
    def __init__(self, note, velocity):
        self.note = note; self.freq = midi_to_freq(note); self.vel = velocity / 127.0
        self.phase = 0.0; self.phase2 = 0.0; self.env = ADSR(); self.env.note_on()
        self.z1 = 0.0
        cutoff = min(1800.0 + 2200.0 * self.vel, 5000.0)
        x = math.exp(-2.0 * math.pi * cutoff / SAMPLE_RATE)
        self.lp_a = 1.0 - x; self.lp_b = x
    def note_off(self): self.env.note_off()
    def render(self, frames):
        t1 = (2*math.pi*self.freq)/SAMPLE_RATE
        t2 = (2*math.pi*(self.freq*1.005))/SAMPLE_RATE
        out = np.empty(frames, np.float32); env = self.env.render(frames)
        for i in range(frames):
            s1 = math.sin(self.phase)
            saw = math.tanh(0.9*s1 + 0.25*math.sin(3*self.phase))
            s2 = math.sin(self.phase2)
            raw = 0.7*saw + 0.3*s2
            self.z1 = self.lp_a*raw + self.lp_b*self.z1
            out[i] = self.z1 * env[i] * self.vel
            self.phase += t1; self.phase2 += t2
            if self.phase > 1e9: self.phase -= 1e9; self.phase2 -= 1e9
        return out
    def finished(self): return self.env.done()

class SoftSynth:
    def __init__(self): self.voices = []; self.lock = threading.Lock()
    def note_on(self, note, velocity):
        if velocity == 0: return self.note_off(note)
        with self.lock:
            if len(self.voices) >= MAX_POLY: self.voices.pop(0)
            self.voices.append(Voice(note, velocity))
    def note_off(self, note):
        with self.lock:
            for v in self.voices:
                if v.note == note: v.note_off()
    def render_mix(self, frames):
        mix = np.zeros(frames, np.float32)
        with self.lock:
            for v in list(self.voices): mix += v.render(frames)
            self.voices[:] = [v for v in self.voices if not v.finished()]
        peak = np.max(np.abs(mix))
        if peak > 0.98: mix *= 0.98/peak
        return mix * MASTER_GAIN

# ====================== Oscilloscope Buffer ======================
class OsciBuffer:
    def __init__(self, seconds=3.0, sr=SAMPLE_RATE):
        self.size = int(seconds * sr)
        self.buf = np.zeros(self.size, np.float32)
        self.idx = 0
        self.total = 0
        self.lock = threading.Lock()
    def write(self, data: np.ndarray):
        n = len(data)
        with self.lock:
            end = self.idx + n
            if end < self.size:
                self.buf[self.idx:end] = data
            else:
                k = self.size - self.idx
                self.buf[self.idx:] = data[:k]
                self.buf[:end % self.size] = data[k:]
            self.idx = end % self.size
            self.total += n
    def snapshot_past(self, seconds: float):
        with self.lock:
            n_samples = min(int(seconds * SAMPLE_RATE), self.size)
            if self.total < n_samples: return None
            end = self.idx
            start = (end - n_samples) % self.size
            if start < end: return self.buf[start:end].copy()
            return np.concatenate([self.buf[start:], self.buf[:end]])

# ====================== MIDI → absolute events (for staff) ======================
class NoteEvent:
    __slots__ = ("note","start","end","vel")
    def __init__(self, note, start, end, vel):
        self.note = note; self.start = start; self.end = end; self.vel = vel

def build_note_events(mid: mido.MidiFile):
    """Return (events, bpm). Times in seconds."""
    tpq = mid.ticks_per_beat
    tempo = 500000  # default 120 bpm
    time_sec = 0.0
    on_stack = {}  # note -> (start_time, velocity)
    events = []
    for msg in mido.merge_tracks(mid.tracks):
        if msg.time:
            time_sec += mido.tick2second(msg.time, tpq, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            on_stack[msg.note] = (time_sec, msg.velocity)
        elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in on_stack:
                st, vel = on_stack.pop(msg.note)
                events.append(NoteEvent(msg.note, st, time_sec, vel))
    for n,(st,vel) in on_stack.items():
        events.append(NoteEvent(n, st, time_sec, vel))
    bpm = mido.tempo2bpm(tempo)
    return events, bpm

# ====================== Responsive Layout ======================
class Layout:
    """Computes rectangles and scale-dependent sizes from window size."""
    def __init__(self, w, h):
        self.w = w; self.h = h
        self.margin = max(12, int(0.02 * h))
        self.gap    = max(6, int(0.012 * h))

        # Height fractions (sum well under 1.0 to allow margins)
        # Keys (top) -> Staff (middle) -> Scope (bottom)
        self.keys_h_frac  = 0.20
        self.staff_h_frac = 0.14
        self.scope_h_frac = 0.26

        keys_h  = max(90, int(self.keys_h_frac  * h))
        staff_h = max(90, int(self.staff_h_frac * h))
        scope_h = max(120, int(self.scope_h_frac * h))

        # Place scope at bottom
        self.scope_rect = pygame.Rect(self.margin,
                                      h - scope_h - self.margin,
                                      w - 2*self.margin,
                                      scope_h)
        # Staff above scope
        self.staff_rect = pygame.Rect(self.margin,
                                      self.scope_rect.y - staff_h - self.gap,
                                      w - 2*self.margin,
                                      staff_h)
        # Keys above staff
        self.keys_y = self.staff_rect.y - keys_h - self.gap
        self.keys_rect = pygame.Rect(self.margin, self.keys_y,
                                     w - 2*self.margin, keys_h)

        # Staff metrics scaled from staff height
        self.staff_line_gap = max(6, int(self.staff_rect.h / 18))  # 5 lines ≈ 4 gaps; 18 keeps it compact
        self.note_w = max(9, int(self.staff_line_gap * 1.1))
        self.note_h = max(7, int(self.staff_line_gap * 0.9))
        self.staff_window_sec = 4.0  # time span shown across staff width
        self.playhead_x_ratio = 0.5   # keep "now" centered

        # Font sizes scale lightly with height
        self.font_sm = max(12, int(h * 0.018))

# ====================== Staff Renderer (responsive) ======================
class StaffRenderer:
    def __init__(self, layout: Layout):
        self.set_layout(layout)

    def set_layout(self, layout: Layout):
        self.rect = layout.staff_rect.copy()
        gap_between_systems = max(8, layout.staff_line_gap + 4)
        h2 = (self.rect.h - gap_between_systems) // 2
        self.treble_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.w, h2)
        self.bass_rect   = pygame.Rect(self.rect.x, self.rect.y + h2 + gap_between_systems, self.rect.w, h2)
        self.playhead_x  = self.rect.x + int(self.rect.w * layout.playhead_x_ratio)
        self.line_gap = layout.staff_line_gap
        self.step_px = self.line_gap / 2.0
        self.note_w = layout.note_w
        self.note_h = layout.note_h
        self.window_sec = layout.staff_window_sec
        self.font = pygame.font.SysFont("Arial", layout.font_sm, bold=True)

        # reference notes (approx mappings)
        self.treble_ref_note = 64  # E4 bottom line
        self.bass_ref_note   = 43  # G2 bottom line

    def note_y(self, note, treble=True):
        ref = self.treble_ref_note if treble else self.bass_ref_note
        steps = (note - ref) * (7.0/12.0)
        base_y = (self.treble_rect.y if treble else self.bass_rect.y) + self.line_gap*4
        return int(base_y - steps * self.step_px)

    def draw_staff_lines(self, surf, treble=True):
        r = self.treble_rect if treble else self.bass_rect
        color = (52, 56, 70)
        for i in range(5):
            y = r.y + i * self.line_gap
            pygame.draw.line(surf, color, (r.x+8, y), (r.x+r.w-8, y), 1)
        label = "Treble" if treble else "Bass"
        surf.blit(self.font.render(label, True, (170, 175, 190)), (r.x+8, r.y - 18))

    def draw_measure_grid(self, surf, bpm):
        beat_sec = 60.0 / max(1e-6, bpm)
        sec_per_px = self.window_sec / self.rect.w
        px_per_beat = max(16, int(beat_sec / sec_per_px))
        grid = (40, 40, 56)
        x = self.playhead_x
        for k in range(0, self.rect.w, px_per_beat):
            xx = x - k
            if self.rect.x <= xx <= self.rect.x + self.rect.w:
                pygame.draw.line(surf, grid, (xx, self.rect.y+2), (xx, self.rect.y+self.rect.h-2), 1)

    def draw_playhead(self, surf):
        pygame.draw.line(surf, RED, (self.playhead_x, self.rect.y+2), (self.playhead_x, self.rect.y+self.rect.h-2), 2)

    def draw_notes(self, surf, events, now):
        t0 = max(0.0, now - self.window_sec)
        sec_per_px = self.window_sec / self.rect.w

        for ev in events:
            if ev.end < t0 or ev.start > now: 
                continue
            st = max(ev.start, t0)
            ed = min(ev.end,   now)
            x1 = int(self.playhead_x - (now - st) / sec_per_px)
            x2 = int(self.playhead_x - (now - ed) / sec_per_px)
            if x2 == x1: x2 = x1 + 2
            width = x2 - x1

            treble = (ev.note >= 60)
            y = self.note_y(ev.note, treble)
            bar_rect = pygame.Rect(x1, y - self.note_h//2 + 3, width, max(3, self.note_h//3))
            pygame.draw.rect(surf, (170, 200, 230), bar_rect, border_radius=2)
            head_rect = pygame.Rect(x2 - self.note_w//2, y - self.note_h//2, self.note_w, self.note_h)
            pygame.draw.ellipse(surf, INK, head_rect)
            pygame.draw.ellipse(surf, (30, 35, 45), head_rect, 1)

            # simple ledgers if far out of range
            r = self.treble_rect if treble else self.bass_rect
            top_y = r.y; bot_y = r.y + self.line_gap*4
            if y < top_y - self.line_gap:
                for ly in range(y, top_y - self.line_gap*2, self.line_gap):
                    pygame.draw.line(surf, (80, 85, 100), (x2-12, ly), (x2+12, ly), 2)
            if y > bot_y + self.line_gap:
                for ly in range(bot_y + self.line_gap, y + 1, self.line_gap):
                    pygame.draw.line(surf, (80, 85, 100), (x2-12, ly), (x2+12, ly), 2)

    def draw(self, surf, events, now, bpm):
        pygame.draw.rect(surf, (14,14,20), self.rect, border_radius=10)
        pygame.draw.rect(surf, (40,40,56), self.rect, 1, border_radius=10)
        self.draw_staff_lines(surf, True)
        self.draw_staff_lines(surf, False)
        self.draw_measure_grid(surf, bpm)
        self.draw_notes(surf, events, now)
        self.draw_playhead(surf)

# ====================== Piano and Bars ======================
class PianoLayout:
    def __init__(self, rect: pygame.Rect, low=LOW_NOTE, high=HIGH_NOTE, black_ratio=0.6):
        self.x, self.y, self.w, self.h = rect.x, rect.y, rect.w, rect.h
        whites, blacks = [], []
        for n in range(low, high+1):
            (blacks if is_black(n) else whites).append(n)
        self.white_notes, self.black_notes = whites, blacks
        self.white_w = rect.w / len(self.white_notes)
        self.white_x = {}; x_cursor = rect.x
        for n in self.white_notes: self.white_x[n] = x_cursor; x_cursor += self.white_w
        self.black_w = self.white_w * black_ratio; self.black_h = rect.h * 0.62
        self.black_x = self._compute_black_positions()
    def _compute_black_positions(self):
        bx = {}
        for n in self.black_notes:
            lw = next((w for w in [n-1, n-2] if w in self.white_x), None)
            rw = next((w for w in [n+1, n+2] if w in self.white_x), None)
            if lw is not None and rw is not None:
                cx = (self.white_x[lw]+self.white_w/2 + self.white_x[rw]+self.white_w/2)/2
                bx[n] = cx - self.black_w/2
        return bx
    def key_rect(self, note):
        if is_black(note):
            return pygame.Rect(self.black_x[note], self.y, self.black_w, self.black_h)
        return pygame.Rect(self.white_x[note], self.y, self.white_w-1, self.h)
    def note_to_x_center(self, note):
        return (self.black_x[note]+self.black_w/2) if is_black(note) else (self.white_x[note]+self.white_w/2)

class VisualNote:
    def __init__(self, note, velocity, x_center, keyboard_top, color):
        self.note = note; self.velocity = velocity; self.x_center = x_center
        self.base_y = keyboard_top - SPAWN_OFFSET; self.y = self.base_y
        self.length_px = 6; self.color = color; self.released = False; self.alive = True
    def release(self): self.released = True
    def update(self, dt):
        self.y -= NOTE_SPEED * dt
        if not self.released: self.length_px += NOTE_SPEED * dt
        if self.y + self.length_px < -50: self.alive = False
    def draw(self, surf, width=14):
        rect = pygame.Rect(int(self.x_center - width/2), int(self.y), width, int(self.length_px))
        pygame.draw.rect(surf, self.color, rect, border_radius=4)

# ====================== Oscilloscope Drawing ======================
def draw_oscilloscope_scrolling(surf, rect, oscbuf: OsciBuffer):
    samples = oscbuf.snapshot_past(SCOPE_SECONDS)
    pygame.draw.rect(surf, (14, 14, 20), rect, border_radius=10)
    pygame.draw.rect(surf, (40, 40, 56), rect, 1, border_radius=10)
    grid = (40, 40, 56)
    for i in range(1, 6):
        y = rect.y + int(i * rect.h / 6)
        pygame.draw.line(surf, grid, (rect.x+6, y), (rect.x+rect.w-6, y), 1)
    for i in range(1, 9):
        x = rect.x + int(i * rect.w / 9)
        pygame.draw.line(surf, grid, (x, rect.y+6), (x, rect.y+rect.h-6), 1)
    mid_y = rect.y + rect.h // 2
    scale_y = (rect.h // 2) * 0.9
    if samples is None or len(samples) < 4:
        pygame.draw.line(surf, (60, 80, 100), (rect.x+4, mid_y), (rect.x+rect.w-4, mid_y), 1)
        return
    if len(samples) != rect.w:
        xi = np.linspace(0, len(samples)-1, rect.w)
        samples = np.interp(xi, np.arange(len(samples)), samples)
    samples = samples - np.mean(samples)
    amp = np.max(np.abs(samples)) + 1e-6
    samples = samples / (amp * 1.2)
    pts = [(rect.x + i, int(mid_y - samples[i] * scale_y)) for i in range(rect.w)]
    pygame.draw.aalines(surf, CYAN, False, pts)
    pygame.draw.line(surf, (60, 80, 100), (rect.x+4, mid_y), (rect.x+rect.w-4, mid_y), 1)

# ====================== Main ======================
def main():
    download_midi(MIDI_URL, MIDI_PATH)
    mid = mido.MidiFile(MIDI_PATH)
    events, bpm = build_note_events(mid)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("SoftSynth Piano Hero — Responsive Layout")
    clock = pygame.time.Clock()

    layout = Layout(WIN_W, WIN_H)
    staff = StaffRenderer(layout)
    kb = PianoLayout(layout.keys_rect)

    active_notes = {}
    visual_notes = []
    vn_lock = threading.Lock()

    palette = [
        ( 80,190,255), ( 80,255,200), (120,255,120), (255,220,100),
        (255,170, 80), (255,100,100), (255, 80,160), (200,100,255),
        (120,120,255), ( 80,160,255), ( 80,220,255), ( 80,255,240)
    ]

    synth = SoftSynth()
    oscbuf = OsciBuffer(seconds=3.0, sr=SAMPLE_RATE)

    # Track play time for staff "now"
    song_start_time = None
    play_lock = threading.Lock()

    def now_time():
        with play_lock:
            if song_start_time is None: return 0.0
            return max(0.0, time.time() - song_start_time)

    def audio_cb(outdata, frames, time_info, status):
        mix = synth.render_mix(frames)
        oscbuf.write(mix)
        outdata[:] = mix.reshape(-1, 1)

    def on_note_on(note, velocity):
        synth.note_on(note, velocity)
        if LOW_NOTE <= note <= HIGH_NOTE:
            active_notes[note] = {"vel": velocity, "state": "down", "t": time.time()}
            x = kb.note_to_x_center(note)
            with vn_lock:
                visual_notes.append(VisualNote(note, velocity, x, kb.y, palette[note % 12]))

    def on_note_off(note):
        synth.note_off(note)
        if note in active_notes:
            active_notes[note]["state"] = "fade"
            active_notes[note]["t"] = time.time()
        with vn_lock:
            for vn in reversed(visual_notes):
                if vn.note == note and not vn.released:
                    vn.release(); break

    def midi_driver():
        nonlocal song_start_time
        with play_lock: song_start_time = time.time()
        for msg in mid.play():
            if msg.type == 'note_on' and msg.velocity > 0:
                on_note_on(msg.note, msg.velocity)
            elif (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
                on_note_off(msg.note)

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                             channels=1, dtype='float32', callback=audio_cb)
    stream.start()
    threading.Thread(target=midi_driver, daemon=True).start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.VIDEORESIZE:
                # Recompute responsive layout
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                layout = Layout(event.w, event.h)
                staff.set_layout(layout)
                kb = PianoLayout(layout.keys_rect)

        screen.fill(BG)

        # ----- Keys (pressed = red, then fade) -----
        now = time.time()
        to_remove = []
        # white keys
        for n in kb.white_notes:
            rect = kb.key_rect(n)
            if n in active_notes:
                st = active_notes[n]["state"]
                if st == "down": color = RED
                else:
                    elapsed = now - active_notes[n]["t"]
                    if elapsed >= FADE_TIME: to_remove.append(n); color = WHITE
                    else:
                        a = 1.0 - (elapsed / FADE_TIME)
                        color = tuple(int(WHITE[i]*(1-a) + RED[i]*a) for i in range(3))
                pygame.draw.rect(screen, color, rect, border_radius=4)
            else:
                pygame.draw.rect(screen, WHITE, rect, border_radius=4)
            pygame.draw.rect(screen, GRAY, rect, 1, border_radius=4)
        # black keys
        for n in kb.black_notes:
            rect = kb.key_rect(n)
            if n in active_notes:
                st = active_notes[n]["state"]
                if st == "down":
                    pygame.draw.rect(screen, RED, rect, border_radius=6)
                    pygame.draw.rect(screen, (255,255,255), rect, 1, border_radius=6)
                else:
                    elapsed = now - active_notes[n]["t"]
                    if elapsed >= FADE_TIME:
                        to_remove.append(n); col = DARK
                    else:
                        a = 1.0 - (elapsed / FADE_TIME)
                        col = tuple(int(DARK[i]*(1-a) + RED[i]*a) for i in range(3))
                    pygame.draw.rect(screen, col, rect, border_radius=6)
            else:
                pygame.draw.rect(screen, DARK, rect, border_radius=6)

        for n in to_remove: active_notes.pop(n, None)

        # spawn edge above keys
        pygame.draw.line(screen, (180, 180, 195), (kb.x, kb.y-1), (kb.x+kb.w, kb.y-1), 1)

        # ----- Bars -----
        with vn_lock:
            for vn in visual_notes: vn.update(1.0/FPS)
            visual_notes[:] = [v for v in visual_notes if v.alive]
            for vn in visual_notes: vn.draw(screen)

        # ----- Staff (responsive size) -----
        staff.draw(screen, events, now_time(), bpm)

        # ----- Scope (scrolling, newest at right) -----
        draw_oscilloscope_scrolling(screen, layout.scope_rect, oscbuf)

        pygame.display.flip()
        clock.tick(FPS)

    stream.stop(); stream.close(); pygame.quit()

if __name__ == "__main__":
    main()

