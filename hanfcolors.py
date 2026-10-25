"""
Hand effects avec OpenCV + MediaPipe
- 21 landmarks par main
- Effet sur chaque main : squelette colore, noeuds qui pulsent, particules
- Toile (cross web) entre la main gauche et la main droite

Usage :
    python hanfcolors.py
Touches : q = quitter
"""
import colorsys
import math
import random

import cv2
import mediapipe as mp
import numpy as np

CFG = {
    "lineWidth": 3,
    "nodeRadius": 5,
    "hueSpeed": 1.5,
    "pulseSpeed": 0.12,
    "smoothing": 0.6,       # 0 = pas de lissage, 0.9 = tres lisse
    "particlesPerTip": 2,
    "maxParticles": 400,
    "glow": True,
}

mp_hands = mp.solutions.hands
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
FINGERTIPS = (4, 8, 12, 16, 20)

# etat global
hue = 0.0
pulse = 0.0
frame_count = 0
smoothed_hands = {}   # "Left" / "Right" -> SmoothedHand
particles = []


# ---------- outils ----------
class _Pt:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = x, y


class SmoothedHand:
    """Main avec 21 points lissés (meme interface que MediaPipe : .landmark[i].x/.y)."""

    def __init__(self, raw_landmarks):
        self.landmark = [_Pt(p.x, p.y) for p in raw_landmarks.landmark]

    def update(self, raw_landmarks, alpha):
        for pt, new in zip(self.landmark, raw_landmarks.landmark):
            pt.x = alpha * pt.x + (1 - alpha) * new.x
            pt.y = alpha * pt.y + (1 - alpha) * new.y


def hsl_to_bgr(h, s, l):
    """h en degres, s et l en % -> tuple BGR (0-255)."""
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, l / 100.0, s / 100.0)
    return int(b * 255), int(g * 255), int(r * 255)


def lm(hand, i, w, h):
    """Landmark i -> coordonnees pixel."""
    p = hand.landmark[i]
    return int(p.x * w), int(p.y * h)


# ---------- effets ----------
def draw_particles(overlay):
    """Met a jour et dessine les particules."""
    global particles
    alive = []
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.05
        p["life"] -= 1
        if p["life"] > 0:
            r = max(1, int(p["life"] / 10))
            col = hsl_to_bgr(p["hue"], 100, 60)
            cv2.circle(overlay, (int(p["x"]), int(p["y"])), r, col, -1, cv2.LINE_AA)
            alive.append(p)
    particles = alive[-CFG["maxParticles"]:]


def spawn_particles(hand, w, h, base_hue):
    for tip in FINGERTIPS:
        x, y = lm(hand, tip, w, h)
        for _ in range(CFG["particlesPerTip"]):
            particles.append({
                "x": x, "y": y,
                "vx": random.uniform(-1.5, 1.5),
                "vy": random.uniform(-2.5, -0.5),
                "life": random.randint(25, 55),
                "hue": base_hue + random.uniform(-30, 30),
            })


def draw_hand_skeleton(overlay, hand, w, h, base_hue):
    for a, b in CONNECTIONS:
        pa = lm(hand, a, w, h)
        pb = lm(hand, b, w, h)
        col = hsl_to_bgr(base_hue + a * 4, 100, 55)
        cv2.line(overlay, pa, pb, col, CFG["lineWidth"], cv2.LINE_AA)


def draw_nodes(overlay, hand, w, h, base_hue, pulse_val):
    for i in range(21):
        p = lm(hand, i, w, h)
        col = hsl_to_bgr(base_hue + i * 8, 100, 60)
        r = int(CFG["nodeRadius"] * (1 + 0.6 * pulse_val))
        if i in FINGERTIPS:
            r += 2
        cv2.circle(overlay, p, r, col, -1, cv2.LINE_AA)
        cv2.circle(overlay, p, r + 2, (255, 255, 255), 1, cv2.LINE_AA)


def draw_cross_web(overlay, hand1, hand2, w, h, base_hue):
    """Relie chaque point de la main 1 au meme point de la main 2."""
    for i in range(21):
        p1 = lm(hand1, i, w, h)
        p2 = lm(hand2, i, w, h)
        col = hsl_to_bgr(base_hue + i * 8, 100, 50)
        cv2.line(overlay, p1, p2, col, 2, cv2.LINE_AA)


# ---------- main ----------
CAMERA_INDEX = 0   # essaie 1 si la mauvaise camera s'ouvre


def main():
    global hue, pulse, frame_count, smoothed_hands

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW = ouverture rapide sur Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la camera. Change CAMERA_INDEX (0, 1, 2).")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # effet miroir

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hue = (hue + CFG["hueSpeed"]) % 360
        pulse = (math.sin(frame_count * CFG["pulseSpeed"]) + 1) / 2
        frame_count += 1

        overlay = np.zeros_like(frame)
        seen = {}

        if results.multi_hand_landmarks and results.multi_handedness:
            for raw, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handed.classification[0].label  # "Left" / "Right"
                if label in seen:
                    label = "Right" if label == "Left" else "Left"
                if label not in smoothed_hands:
                    smoothed_hands[label] = SmoothedHand(raw)
                else:
                    smoothed_hands[label].update(raw, CFG["smoothing"])
                seen[label] = smoothed_hands[label]

        for label in list(smoothed_hands):
            if label not in seen:
                del smoothed_hands[label]

        # effet sur chaque main
        for label, hand in seen.items():
            base = hue if label == "Left" else hue + 180
            draw_hand_skeleton(overlay, hand, w, h, base)
            draw_nodes(overlay, hand, w, h, base, pulse)
            spawn_particles(hand, w, h, base)

        # connexion main gauche <-> main droite
        if "Left" in seen and "Right" in seen:
            draw_cross_web(overlay, seen["Left"], seen["Right"], w, h, hue)

        draw_particles(overlay)

        if CFG["glow"]:
            glow = cv2.GaussianBlur(overlay, (0, 0), 9)
            overlay = cv2.addWeighted(overlay, 1.0, glow, 0.9, 0)

        out = cv2.addWeighted(frame, 0.55, overlay, 1.0, 0)
        out = cv2.add(out, overlay)

        cv2.imshow("Hand Effects - q pour quitter", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
