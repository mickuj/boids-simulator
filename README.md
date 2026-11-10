# 🐦 Boids – symulacja stada

Agentowa symulacja ruchu stada oparta na algorytmie Boids Craiga Reynoldsa. Projekt realizowany w ramach przedmiotu **Algorytmy inteligentne inspirowane naturą**.

Każdy boid widzi tylko sąsiadów w swoim polu widzenia i na tej podstawie w każdej klatce koryguje kurs. Nie ma lidera ani centralnego sterowania. Zorganizowany ruch całej grupy wyłania się z trzech prostych reguł lokalnych.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green)
![NumPy](https://img.shields.io/badge/NumPy-vectorized-orange)
![Status](https://img.shields.io/badge/status-v0.1-yellow)

---

## O projekcie

Chcemy sprawdzić, jak z prostych zasad lokalnych powstaje zachowanie całego stada bez centralnego sterowania. Algorytmy tego typu znajdują zastosowanie w grafice komputerowej, w modelowaniu ławic ryb i w koordynacji roju dronów.

---

## Funkcje

- Trzy reguły Reynoldsa: separacja, wyrównanie, spójność
- Kąt i zasięg widzenia (od 30° do 360°)
- Maksymalna prędkość i siła sterująca oraz prędkość minimalna
- Świat toroidalny: boidy wychodzące za krawędź pojawiają się po drugiej stronie
- Panel z 9 suwakami i przełącznikami
- Presety: ptaki, ryby, drony
- Tryb podążania za kursorem myszy
- Tryb debug: stożek widzenia i sąsiedzi wybranego boida
- Metryki na żywo: polaryzacja i średnia liczba sąsiadów
- Obliczenia wektorowe w NumPy, bez pętli po boidach

---

## Instalacja i uruchomienie

```bash
git clone <adres-repozytorium>
cd boids
pip install -r requirements.txt
python boids.py
```

`requirements.txt`:

```
pygame>=2.5
numpy>=1.24
```

---

## Sterowanie

| Klawisz / akcja | Działanie |
| --- | --- |
| `Spacja` | Pauza i wznowienie |
| `R` | Reset pozycji boidów |
| `M` | Włączenie lub wyłączenie podążania za myszką |
| `V` | Tryb debug: pole widzenia i sąsiedzi boida |
| `1` / `2` / `3` | Preset: Ptaki / Ryby / Drony |
| `Esc` | Wyjście |
| Suwaki w panelu | Zmiana parametrów na żywo |

---

## Presety

| Parametr | Ptaki | Ryby | Drony |
| --- | ---: | ---: | ---: |
| Liczba boidów | 120 | 200 | 60 |
| Waga separacji | 1.5 | 1.8 | 2.5 |
| Waga wyrównania | 1.0 | 1.6 | 1.2 |
| Waga spójności | 1.0 | 1.4 | 0.6 |
| Zasięg widzenia [px] | 70 | 50 | 110 |
| Promień separacji [px] | 25 | 16 | 45 |
| Kąt widzenia [°] | 270 | 300 | 360 |
| Prędkość maks. | 4.0 | 3.0 | 2.5 |
| Siła sterująca maks. | 0.08 | 0.12 | 0.05 |

Drony mają duży promień separacji i słabą spójność, dzięki czemu trzymają bezpieczne odstępy. Ryby tworzą gęste, silnie wyrównane ławice.

---

## Model

Na każdy boid działają siły sterujące liczone wyłącznie z sąsiadów, którzy mieszczą się jednocześnie w **zasięgu** i w **kącie widzenia**:

| Reguła | Działanie |
| --- | --- |
| **Separacja** | Odpychanie od zbyt bliskich sąsiadów, rosnące wraz ze zbliżeniem |
| **Wyrównanie** | Dopasowanie kierunku do średniej prędkości sąsiadów |
| **Spójność** | Ruch w stronę środka masy sąsiadów |

Każda siła ma postać siły sterującej Reynoldsa: *pożądana prędkość − aktualna prędkość*, z długością ograniczoną do `max_force`. Siły są ważone i sumowane:

```
acc = w_sep · separacja + w_ali · wyrównanie + w_coh · spójność (+ cel)
vel = clip(vel + acc, v_min, v_max)
pos = (pos + vel) mod rozmiar_świata
```

---

## Metryki

| Metryka | Co mierzy | Pożądany kierunek |
| --- | --- | --- |
| **Polaryzacja Φ** | Uporządkowanie kierunku ruchu (od 0 do 1) | ↑ |
| Średnia liczba sąsiadów | Gęstość lokalna | informacyjnie |

Polaryzacja:

$$\Phi = \frac{1}{N} \left\| \sum_{i=1}^{N} \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \right\|$$

Φ = 1 oznacza, że wszystkie boidy lecą w tę samą stronę, a Φ ≈ 0 oznacza chaos.

---

## Literatura

- Craig W. Reynolds, *Flocks, Herds, and Schools: A Distributed Behavioral Model*, Computer Graphics (SIGGRAPH '87), 21(4), 1987, s. 25–34.
