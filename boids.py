"""
Boids – symulacja stada (wersja 0.1)
Projekt: Algorytmy inteligentne inspirowane naturą

Zaimplementowane:
  * trzy reguły Reynoldsa: separacja, wyrównanie, spójność
  * kąt i zasięg widzenia, maksymalna prędkość i siła sterująca
  * świat toroidalny (boidy wychodzące za krawędź pojawiają się po drugiej stronie)
  * panel z suwakami, przełącznikami i presetami (ptaki / ryby / drony)
  * tryb podążania za myszką
  * tryb debugowania: stożek widzenia i sąsiedzi wybranego boida
  * metryka uporządkowania stada (polaryzacja)

Do zrobienia na kolejne etapy (TODO):
  * przeszkody (losowo generowane + omijanie)
  * algorytm genetyczny do doboru parametrów
  * siatka przestrzenna (spatial hashing) dla dużej liczby boidów
  * rozszerzenia: drapieżnik, pożywienie, wykresy metryk

Sterowanie:
  Spacja – pauza     R – reset     M – podążanie za myszką
  V – debug widzenia 1/2/3 – presety Esc – wyjście
"""

import math
import sys

import numpy as np
import pygame

# --------------------------------------------------------------------------
# Konfiguracja
# --------------------------------------------------------------------------
SIM_W, SIM_H = 900, 750          # obszar symulacji
PANEL_W = 300                    # panel sterowania po prawej
WIDTH, HEIGHT = SIM_W + PANEL_W, SIM_H
FPS = 60

PANEL_BG = (34, 36, 44)
TEXT = (225, 228, 235)
MUTED = (140, 145, 160)
ACCENT = (90, 160, 255)

PRESETS = {
    "Ptaki": {
        "params": dict(count=120, sep_w=1.5, ali_w=1.0, coh_w=1.0,
                       view_radius=70, sep_radius=25, view_angle=270,
                       max_speed=4.0, max_force=0.08),
        "bg": (196, 222, 244), "color": (35, 38, 48),
    },
    "Ryby": {
        "params": dict(count=200, sep_w=1.8, ali_w=1.6, coh_w=1.4,
                       view_radius=50, sep_radius=16, view_angle=300,
                       max_speed=3.0, max_force=0.12),
        "bg": (12, 42, 82), "color": (255, 170, 60),
    },
    "Drony": {
        "params": dict(count=60, sep_w=2.5, ali_w=1.2, coh_w=0.6,
                       view_radius=110, sep_radius=45, view_angle=360,
                       max_speed=2.5, max_force=0.05),
        "bg": (24, 24, 30), "color": (80, 230, 230),
    },
}


# --------------------------------------------------------------------------
# Funkcje pomocnicze (wektorowo, NumPy)
# --------------------------------------------------------------------------
def limit(v, max_len):
    """Ogranicza długość każdego wiersza macierzy v do max_len."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, max_len / np.maximum(n, 1e-9))


def steer(desired, vel, max_speed, max_force):
    """Siła sterująca Reynoldsa: (pożądana prędkość - aktualna), ograniczona."""
    n = np.linalg.norm(desired, axis=1, keepdims=True)
    valid = n[:, 0] > 1e-9
    out = np.zeros_like(desired)
    d = desired[valid] / n[valid] * max_speed
    out[valid] = limit(d - vel[valid], max_force)
    return out


# --------------------------------------------------------------------------
# Model stada
# --------------------------------------------------------------------------
class Flock:
    def __init__(self, n, w, h, seed=None):
        self.w, self.h = w, h
        self.rng = np.random.default_rng(seed)
        self.pos = np.empty((0, 2))
        self.vel = np.empty((0, 2))
        self.last_visible = None     # macierz sąsiedztwa z ostatniego kroku
        self.set_count(n)

    def _random_boids(self, n, speed=2.0):
        pos = self.rng.random((n, 2)) * [self.w, self.h]
        ang = self.rng.random(n) * 2 * np.pi
        vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * speed
        return pos, vel

    def set_count(self, n):
        cur = len(self.pos)
        if n > cur:
            p, v = self._random_boids(n - cur)
            self.pos = np.vstack([self.pos, p])
            self.vel = np.vstack([self.vel, v])
        elif n < cur:
            self.pos, self.vel = self.pos[:n], self.vel[:n]

    def reset(self):
        self.pos, self.vel = self._random_boids(len(self.pos))

    def wrap(self, d):
        """Najkrótszy wektor różnicy w świecie toroidalnym."""
        d[..., 0] -= self.w * np.round(d[..., 0] / self.w)
        d[..., 1] -= self.h * np.round(d[..., 1] / self.h)
        return d

    def neighbours(self, p):
        """Zwraca: diff[i,j] = pos_j - pos_i, dystanse oraz maskę widoczności."""
        diff = self.wrap(self.pos[None, :, :] - self.pos[:, None, :])
        dist = np.hypot(diff[..., 0], diff[..., 1])
        np.fill_diagonal(dist, np.inf)

        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        heading = self.vel / np.maximum(speed, 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_a = np.einsum("ijk,ik->ij", diff, heading) / dist

        half = math.radians(p["view_angle"]) / 2
        visible = (dist < p["view_radius"]) & (cos_a >= math.cos(half) - 1e-9)
        return diff, dist, visible

    def update(self, p, target=None, target_w=1.0):
        if len(self.pos) == 0:
            return
        ms, mf = p["max_speed"], p["max_force"]
        diff, dist, vis = self.neighbours(p)
        self.last_visible = vis
        vis_f = vis.astype(float)
        has = vis.any(axis=1)
        acc = np.zeros_like(self.vel)

        # 1) Wyrównanie – dopasuj kierunek do średniej prędkości sąsiadów
        avg_vel = vis_f @ self.vel
        acc[has] += p["ali_w"] * steer(avg_vel[has], self.vel[has], ms, mf)

        # 2) Spójność – leć w stronę środka masy sąsiadów
        to_center = np.einsum("ij,ijk->ik", vis_f, diff)
        acc[has] += p["coh_w"] * steer(to_center[has], self.vel[has], ms, mf)

        # 3) Separacja – odpychanie od zbyt bliskich sąsiadów (siła ~ 1/d)
        close = vis & (dist < p["sep_radius"])
        has_close = close.any(axis=1)
        weights = close / np.maximum(dist, 1e-3) ** 2
        away = -np.einsum("ij,ijk->ik", weights, diff)
        acc[has_close] += p["sep_w"] * steer(away[has_close], self.vel[has_close], ms, mf)

        # Opcjonalny cel (np. kursor myszy)
        if target is not None:
            to_target = self.wrap(np.asarray(target, float)[None, :] - self.pos)
            acc += target_w * steer(to_target, self.vel, ms, mf)

        # TODO: omijanie przeszkód – kolejna siła dodawana do acc

        # Integracja ruchu
        self.vel += acc
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        min_speed = 0.4 * ms
        self.vel *= np.clip(speed, min_speed, ms) / np.maximum(speed, 1e-9)
        self.pos = (self.pos + self.vel) % [self.w, self.h]

    def polarization(self):
        """Miara uporządkowania: 1 = wszyscy lecą w tę samą stronę, ~0 = chaos."""
        if len(self.vel) == 0:
            return 0.0
        u = self.vel / np.maximum(np.linalg.norm(self.vel, axis=1, keepdims=True), 1e-9)
        return float(np.linalg.norm(u.mean(axis=0)))

    def mean_neighbours(self):
        if self.last_visible is None or len(self.last_visible) == 0:
            return 0.0
        return float(self.last_visible.sum(axis=1).mean())


# --------------------------------------------------------------------------
# Widgety GUI
# --------------------------------------------------------------------------
class Slider:
    def __init__(self, key, label, vmin, vmax, value, x, y, w=250, step=None, fmt="{:.2f}"):
        self.key, self.label = key, label
        self.vmin, self.vmax, self.value = vmin, vmax, value
        self.step, self.fmt = step, fmt
        self.rect = pygame.Rect(x, y + 20, w, 6)
        self.dragging = False

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
                and self.rect.inflate(16, 18).collidepoint(ev.pos):
            self.dragging = True
            self._set_from_x(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(ev.pos[0])

    def _set_from_x(self, x):
        t = min(max((x - self.rect.x) / self.rect.w, 0.0), 1.0)
        v = self.vmin + t * (self.vmax - self.vmin)
        if self.step:
            v = round(v / self.step) * self.step
        self.value = v

    def draw(self, surf, font):
        surf.blit(font.render(self.label, True, TEXT), (self.rect.x, self.rect.y - 20))
        val = font.render(self.fmt.format(self.value), True, ACCENT)
        surf.blit(val, (self.rect.right - val.get_width(), self.rect.y - 20))
        pygame.draw.rect(surf, (70, 74, 88), self.rect, border_radius=3)
        t = (self.value - self.vmin) / (self.vmax - self.vmin)
        filled = self.rect.copy()
        filled.w = int(self.rect.w * t)
        pygame.draw.rect(surf, ACCENT, filled, border_radius=3)
        pygame.draw.circle(surf, TEXT, (self.rect.x + filled.w, self.rect.centery), 7)


class Button:
    def __init__(self, label, rect, toggle=False, active=False):
        self.label, self.rect = label, pygame.Rect(rect)
        self.toggle, self.active = toggle, active

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos):
            if self.toggle:
                self.active = not self.active
            return True
        return False

    def draw(self, surf, font):
        bg = ACCENT if self.active else (60, 64, 78)
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        txt = font.render(self.label, True, TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


# --------------------------------------------------------------------------
# Rysowanie
# --------------------------------------------------------------------------
def draw_flock(surf, flock, color, size=6):
    ang = np.arctan2(flock.vel[:, 1], flock.vel[:, 0])
    cos, sin = np.cos(ang), np.sin(ang)
    for (x, y), c, s in zip(flock.pos, cos, sin):
        tip = (x + c * size * 1.6, y + s * size * 1.6)
        left = (x - c * size - s * size * 0.6, y - s * size + c * size * 0.6)
        right = (x - c * size + s * size * 0.6, y - s * size - c * size * 0.6)
        pygame.draw.polygon(surf, color, (tip, left, right))


def draw_debug(surf, flock, p, idx=0):
    """Stożek widzenia, promień separacji i sąsiedzi boida o indeksie idx."""
    if len(flock.pos) == 0 or flock.last_visible is None:
        return
    overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    x, y = flock.pos[idx]
    heading = math.atan2(flock.vel[idx, 1], flock.vel[idx, 0])
    half = math.radians(p["view_angle"]) / 2
    r = p["view_radius"]
    pts = [(x, y)] + [(x + r * math.cos(heading + t), y + r * math.sin(heading + t))
                      for t in np.linspace(-half, half, 40)]
    pygame.draw.polygon(overlay, (255, 230, 80, 55), pts)
    pygame.draw.circle(overlay, (255, 80, 80, 120), (int(x), int(y)), int(p["sep_radius"]), 1)

    diff0 = flock.wrap(flock.pos - flock.pos[idx])
    for j in np.nonzero(flock.last_visible[idx])[0]:
        end = (x + diff0[j, 0], y + diff0[j, 1])
        pygame.draw.line(overlay, (255, 255, 255, 160), (x, y), end, 1)
    surf.blit(overlay, (0, 0))
    pygame.draw.circle(surf, (255, 60, 60), (int(x), int(y)), 4)


# --------------------------------------------------------------------------
# Aplikacja
# --------------------------------------------------------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Boids – symulacja stada v0.1")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusans,arial", 14)
        self.font_big = pygame.font.SysFont("dejavusans,arial", 20, bold=True)

        x0 = SIM_W + 25
        specs = [
            ("count", "Liczba boidów", 10, 300, 1, "{:.0f}"),
            ("sep_w", "Waga separacji", 0, 5, 0.05, "{:.2f}"),
            ("ali_w", "Waga wyrównania", 0, 5, 0.05, "{:.2f}"),
            ("coh_w", "Waga spójności", 0, 5, 0.05, "{:.2f}"),
            ("view_radius", "Zasięg widzenia", 10, 200, 1, "{:.0f} px"),
            ("sep_radius", "Promień separacji", 5, 100, 1, "{:.0f} px"),
            ("view_angle", "Kąt widzenia", 30, 360, 5, "{:.0f}°"),
            ("max_speed", "Prędkość maks.", 0.5, 8, 0.1, "{:.1f}"),
            ("max_force", "Siła sterująca maks.", 0.01, 0.4, 0.01, "{:.2f}"),
        ]
        self.sliders = [Slider(k, lab, lo, hi, lo, x0, 55 + i * 44, step=st, fmt=f)
                        for i, (k, lab, lo, hi, st, f) in enumerate(specs)]

        y = 55 + len(specs) * 44 + 15
        self.btn_mouse = Button("Myszka", (x0, y, 80, 30), toggle=True)
        self.btn_debug = Button("Debug", (x0 + 85, y, 80, 30), toggle=True)
        self.btn_pause = Button("Pauza", (x0 + 170, y, 80, 30), toggle=True)
        y += 65
        self.preset_buttons = {name: Button(name, (x0 + i * 85, y, 80, 30))
                               for i, name in enumerate(PRESETS)}
        self.info_y = y + 50

        self.flock = Flock(0, SIM_W, SIM_H)
        self.apply_preset("Ptaki")

    @property
    def params(self):
        return {s.key: s.value for s in self.sliders}

    def apply_preset(self, name):
        preset = PRESETS[name]
        for s in self.sliders:
            s.value = preset["params"][s.key]
        self.bg, self.color = preset["bg"], preset["color"]
        self.preset_name = name
        for n, b in self.preset_buttons.items():
            b.active = (n == name)
        self.flock.set_count(int(preset["params"]["count"]))

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                elif ev.key == pygame.K_SPACE:
                    self.btn_pause.active = not self.btn_pause.active
                elif ev.key == pygame.K_r:
                    self.flock.reset()
                elif ev.key == pygame.K_m:
                    self.btn_mouse.active = not self.btn_mouse.active
                elif ev.key == pygame.K_v:
                    self.btn_debug.active = not self.btn_debug.active
                elif pygame.K_1 <= ev.key <= pygame.K_3:
                    self.apply_preset(list(PRESETS)[ev.key - pygame.K_1])
            for s in self.sliders:
                s.handle(ev)
            for b in (self.btn_mouse, self.btn_debug, self.btn_pause):
                b.handle(ev)
            for name, b in self.preset_buttons.items():
                if b.handle(ev):
                    self.apply_preset(name)
            # TODO: kliknięcie w obszar symulacji -> dodanie przeszkody
        return True

    def draw_panel(self):
        panel = pygame.Rect(SIM_W, 0, PANEL_W, HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        self.screen.blit(self.font_big.render("Parametry stada", True, TEXT), (SIM_W + 25, 15))
        for s in self.sliders:
            s.draw(self.screen, self.font)
        for b in (self.btn_mouse, self.btn_debug, self.btn_pause):
            b.draw(self.screen, self.font)
        self.screen.blit(self.font.render("Presety:", True, MUTED),
                         (SIM_W + 25, self.preset_buttons["Ptaki"].rect.y - 20))
        for b in self.preset_buttons.values():
            b.draw(self.screen, self.font)

        lines = [
            f"FPS: {self.clock.get_fps():.0f}",
            f"Boidy: {len(self.flock.pos)}",
            f"Polaryzacja: {self.flock.polarization():.2f}",
            f"Śr. liczba sąsiadów: {self.flock.mean_neighbours():.1f}",
            "",
            "Spacja – pauza   R – reset",
            "M – myszka   V – debug",
            "1/2/3 – presety   Esc – wyjście",
        ]
        for i, line in enumerate(lines):
            col = TEXT if i < 4 else MUTED
            self.screen.blit(self.font.render(line, True, col), (SIM_W + 25, self.info_y + i * 20))

    def run(self, max_frames=None):
        frame = 0
        running = True
        while running:
            running = self.handle_events()
            p = self.params
            self.flock.set_count(int(p["count"]))

            target = None
            mx, my = pygame.mouse.get_pos()
            if self.btn_mouse.active and mx < SIM_W:
                target = (mx, my)

            if not self.btn_pause.active:
                self.flock.update(p, target)

            self.screen.fill(self.bg)
            if self.btn_debug.active:
                draw_debug(self.screen, self.flock, p)
            draw_flock(self.screen, self.flock, self.color)
            if target is not None:
                pygame.draw.circle(self.screen, (255, 80, 80), target, 8, 2)
            self.draw_panel()

            pygame.display.flip()
            self.clock.tick(FPS)
            frame += 1
            if max_frames and frame >= max_frames:
                break
        pygame.quit()


if __name__ == "__main__":
    App().run()
    sys.exit()
