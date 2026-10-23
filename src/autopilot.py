r"""
Імпровізований автопілот: реактивний вибір напрямку за покадровою картою
зайнятості. Жодного навчання, жодної моделі, лише геометрія і арифметика.

Вхід, те саме, що віддає ймовірнісний шар на кожен кадр:
    conf        (H, W) вага довіри, нечисло означає відкинутий піксель
    depth       (H, W) відкалібрована глибина в метрах
    K, T_wc     камера

Вихід на кожен кадр:
    direction   одиничний вектор напрямку руху у СВІТОВИХ координатах
    target      точка у світі, куди летіти, тобто положення плюс напрямок на
                відстань випередження
    speed       швидкість у метрах за секунду
    stop        True, якщо попереду немає жодного безпечного напрямку

===========================================================================
ЯК ВІН ВИРІШУЄ
===========================================================================

1. Пікселі кадру стають точками в системі камери. Кожна точка несе свою
   ЙМОВІРНІСТЬ ЗАЙНЯТОСТІ, а не просто факт існування:

       P = 1 / (1 + exp(-l_occ * w))

   Саме тут ймовірнісний шар і починає впливати на поведінку. Піксель, якому
   конвеєр не довіряє, гальмує апарат слабше, ніж той, у якому він упевнений.

2. Будується віяло кандидатів: напрямки в межах поля зору, по азимуту і куту
   місця. Для кожного рахується, що стоїть у коридорі завширшки з апарат.

3. Кожен кандидат дістає дві оцінки:

       запас     найближча точка в коридорі, яка певна настільки, що її
                 не можна ігнорувати. Звідси береться швидкість
       ризик     сума ймовірностей зайнятості уздовж коридору, де ближчі
                 точки важать більше. Звідси береться вибір напрямку

4. Перемагає кандидат з найменшою вартістю:

       вартість = ризик + відхилення від бажаного курсу

5. Швидкість пропорційна запасу. Якщо запас менший за гальмівний шлях,
   виходить нуль і ознака зупинки.

===========================================================================
ОБЕРЕЖНІСТЬ ДО НЕВІДОМОГО
===========================================================================

Відкинуті пікселі, тобто ті, де конвеєр сказав «не знаю», за замовчуванням
вважаються ПОРОЖНІМИ. Це оптимістично і годиться там, де відкидається небо.

`unknown_is_risk` перемикає на протилежне: невідоме вважається можливою
перешкодою з імовірністю `unknown_p`. Для шару безпеки над незнайомою
місцевістю це розумніше, хоч апарат і стане обережнішим до повільності.

Використання:

    from autopilot import ReactiveAutopilot
    ap = ReactiveAutopilot(drone_radius=0.5, max_speed=3.0)
    cmd = ap.step(conf, depth, K, T_wc)
    print(cmd["speed"], cmd["direction"], cmd["target"])
"""
import numpy as np

L_OCC = 0.85          # той самий приріст, що й у воксельній сітці

# Плаский вигляд команди: 11 чисел, фіксований порядок. Зручно, коли
# приймальний бік не хоче розбирати словник і просто читає масив.
COMMAND_FIELDS = (
    "timestamp",      # 0   мітка часу кадру, секунди
    "dir_x",          # 1   напрямок руху, одиничний вектор
    "dir_y",          # 2     у ТІЙ САМІЙ світовій системі, що й надіслана поза
    "dir_z",          # 3
    "target_x",       # 4   куди летіти, абсолютна точка у світі, метри
    "target_y",       # 5
    "target_z",       # 6
    "speed",          # 7   бажана швидкість, м/с. Нуль означає стояти
    "clearance_m",    # 8   вільна відстань попереду в обраному напрямку, м
    "azimuth_deg",    # 9   відхилення від курсу: плюс праворуч, мінус ліворуч
    "stop",           # 10  1 означає стоп, безпечного напрямку немає
)


def command_to_vector(cmd, timestamp=0.0):
    """Словник команди -> масив з 11 чисел у порядку COMMAND_FIELDS."""
    d = np.asarray(cmd["direction"], dtype=np.float64).reshape(3)
    t = np.asarray(cmd["target"], dtype=np.float64).reshape(3)
    return np.array([timestamp, d[0], d[1], d[2], t[0], t[1], t[2],
                     float(cmd["speed"]), float(cmd.get("clearance_m", 0.0)),
                     float(cmd.get("azimuth_deg", 0.0)),
                     1.0 if cmd.get("stop") else 0.0], dtype=np.float64)


def vector_to_command(v):
    """Зворотне перетворення, щоб приймальний бік не плутав порядок."""
    v = np.asarray(v, dtype=np.float64).reshape(11)
    return {"timestamp": v[0], "direction": v[1:4], "target": v[4:7],
            "speed": v[7], "clearance_m": v[8], "azimuth_deg": v[9],
            "stop": bool(v[10] >= 0.5)}


def velocity_vector(cmd):
    """Якщо керуванню зручніше одразу вектор швидкості в м/с, а не
    напрямок і модуль окремо."""
    return np.asarray(cmd["direction"], dtype=np.float64).reshape(3) * float(cmd["speed"])


class ReactiveAutopilot:
    def __init__(self,
                 drone_radius=0.5,
                 max_speed=3.0,
                 min_speed=0.0,
                 stop_distance=1.5,
                 slow_distance=8.0,
                 lookahead=5.0,
                 n_azimuth=21,
                 n_elevation=5,
                 max_azimuth_deg=50.0,
                 max_elevation_deg=20.0,
                 goal_weight=1.0,
                 risk_weight=2.5,
                 risk_decay=6.0,
                 p_ignore=0.55,
                 unknown_is_risk=False,
                 unknown_p=0.6,
                 max_points=6000,
                 min_depth=0.3,
                 max_range=30.0,
                 smooth=0.4,
                 seed=0):
        """drone_radius   півширина коридору, тобто габарит апарата з запасом
        stop_distance     ближче за це напрямок вважається заблокованим
        slow_distance     далі за це дозволена повна швидкість
        lookahead         на скільки метрів наперед ставиться цільова точка
        p_ignore          точки з імовірністю зайнятості нижче цієї не
                          блокують коридор. Нижче порога зчитування карти
                          0,5 опускатись немає сенсу
        risk_decay        на скільки метрів спадає вага ризику. Ближча
                          перешкода важливіша за далеку
        smooth            згладжування напрямку між кадрами, від 0 до 1.
                          Без нього команда смикається щокадру
        """
        self.drone_radius = drone_radius
        self.max_speed, self.min_speed = max_speed, min_speed
        self.stop_distance, self.slow_distance = stop_distance, slow_distance
        self.lookahead = lookahead
        self.goal_weight, self.risk_weight = goal_weight, risk_weight
        self.risk_decay = risk_decay
        self.p_ignore = p_ignore
        self.unknown_is_risk, self.unknown_p = unknown_is_risk, unknown_p
        self.max_points = max_points
        self.min_depth, self.max_range = min_depth, max_range
        self.smooth = smooth
        self.rng = np.random.default_rng(seed)
        self.last_direction_cam = None
        self.n_steps = 0

        # Віяло кандидатів у системі камери: X праворуч, Y ВНИЗ, Z уперед.
        # Оскільки Y вниз, додатний кут місця дає ВІД'ЄМНИЙ y. Без мінуса
        # звітний elevation_deg означав би протилежне загальноприйнятому, і
        # той, хто читає команду, розвернув би апарат не туди.
        az = np.deg2rad(np.linspace(-max_azimuth_deg, max_azimuth_deg, n_azimuth))
        el = np.deg2rad(np.linspace(-max_elevation_deg, max_elevation_deg, n_elevation))
        A, E = np.meshgrid(az, el, indexing="ij")
        A, E = A.ravel(), E.ravel()
        self.cand = np.stack([np.sin(A) * np.cos(E), -np.sin(E), np.cos(A) * np.cos(E)], axis=1)
        self.cand /= np.linalg.norm(self.cand, axis=1, keepdims=True)
        self.cand_az, self.cand_el = A, E

    # ------------------------------------------------------------------ вхід

    def _points_with_risk(self, conf, depth, K):
        """Пікселі кадру -> точки в системі камери плюс їхня ймовірність
        зайнятості. Саме тут покадрова карта довіри стає геометрією."""
        h, w = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        finite_d = np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_range)
        known = np.isfinite(conf)

        p_occ = np.zeros_like(depth, dtype=np.float64)
        p_occ[known] = 1.0 / (1.0 + np.exp(-L_OCC * conf[known]))
        if self.unknown_is_risk:
            # «не знаю» трактуємо як можливу перешкоду
            p_occ[~known & finite_d] = self.unknown_p

        take = finite_d & (p_occ > self.p_ignore)
        vs, us = np.nonzero(take)
        if len(us) == 0:
            return np.zeros((0, 3)), np.zeros(0)
        if len(us) > self.max_points:
            sel = self.rng.choice(len(us), self.max_points, replace=False)
            vs, us = vs[sel], us[sel]
        z = depth[vs, us].astype(np.float64)
        pts = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=1)
        return pts, p_occ[vs, us]

    # ------------------------------------------------------------- оцінювання

    def _score(self, pts, risk, goal_cam):
        """Для кожного кандидата: запас у метрах і накопичений ризик."""
        n = len(self.cand)
        if len(pts) == 0:
            return np.full(n, self.max_range), np.zeros(n)

        along = pts @ self.cand.T                       # (точки, кандидати)
        # відстань від точки до осі коридору
        perp_sq = (np.sum(pts ** 2, axis=1)[:, None] - along ** 2)
        np.maximum(perp_sq, 0.0, out=perp_sq)
        perp = np.sqrt(perp_sq)

        ahead = along > 0.0
        inside = ahead & (perp < self.drone_radius)

        # запас: найближча точка, що реально стоїть у коридорі
        blocked = np.where(inside, along, np.inf)
        clearance = blocked.min(axis=0)
        clearance[~np.isfinite(clearance)] = self.max_range

        # ризик: ближче важить більше, а за межами коридору вплив спадає
        soft = np.exp(-np.maximum(perp - self.drone_radius, 0.0) ** 2
                      / (2.0 * self.drone_radius ** 2))
        near = np.exp(-np.maximum(along, 0.0) / self.risk_decay)
        contrib = np.where(ahead, risk[:, None] * soft * near, 0.0)
        risk_sum = contrib.sum(axis=0) / max(len(pts), 1)

        return clearance, risk_sum

    # --------------------------------------------------------------- один крок

    def step(self, conf, depth, K, T_wc, goal_world=None, dt=None):
        T_wc = np.asarray(T_wc, dtype=np.float64)
        R_wc, pos = T_wc[:3, :3], T_wc[:3, 3]

        # бажаний курс у системі камери: або на ціль, або просто вперед
        if goal_world is not None:
            g = np.asarray(goal_world, dtype=np.float64) - pos
            n = np.linalg.norm(g)
            goal_cam = (R_wc.T @ g / n) if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        else:
            goal_cam = np.array([0.0, 0.0, 1.0])

        pts, risk = self._points_with_risk(np.asarray(conf, float),
                                            np.asarray(depth, float), np.asarray(K, float))
        clearance, risk_sum = self._score(pts, risk, goal_cam)

        # вартість: ризик плюс відхилення від бажаного курсу
        misalign = 1.0 - (self.cand @ goal_cam)
        blocked = clearance < self.stop_distance
        cost = self.risk_weight * risk_sum + self.goal_weight * misalign
        cost = np.where(blocked, cost + 1e3, cost)

        best = int(np.argmin(cost))
        all_blocked = bool(blocked.all())
        d_cam = self.cand[best]

        # згладжування, щоб команда не смикалась щокадру
        if self.last_direction_cam is not None and self.smooth > 0 and not all_blocked:
            d_cam = (1.0 - self.smooth) * d_cam + self.smooth * self.last_direction_cam
            d_cam /= max(np.linalg.norm(d_cam), 1e-9)
        if not all_blocked:
            self.last_direction_cam = d_cam

        free = float(clearance[best])
        if all_blocked or free <= self.stop_distance:
            speed = 0.0
        else:
            t = (free - self.stop_distance) / max(self.slow_distance - self.stop_distance, 1e-6)
            speed = self.min_speed + (self.max_speed - self.min_speed) * float(np.clip(t, 0.0, 1.0))

        d_world = R_wc @ d_cam
        d_world /= max(np.linalg.norm(d_world), 1e-9)
        reach = max(0.0, min(self.lookahead, free - self.stop_distance))
        target = pos + d_world * reach

        self.n_steps += 1
        return {
            "direction": d_world,
            "target": target,
            "speed": speed,
            "stop": all_blocked or speed <= 0.0,
            "clearance_m": free,
            "risk": float(risk_sum[best]),
            "azimuth_deg": float(np.rad2deg(self.cand_az[best])),
            "elevation_deg": float(np.rad2deg(self.cand_el[best])),
            "n_obstacle_points": int(len(pts)),
            "blocked_fraction": float(blocked.mean()),
            "position": pos,
            "_cand_clearance": clearance,
            "_cand_cost": cost,
            "_best": best,
        }

    def reason(self, cmd):
        """Короткий людський опис, чому саме така команда."""
        if cmd["stop"]:
            if cmd["blocked_fraction"] >= 1.0:
                return "усі напрямки заблоковані, стоп"
            return f"запас {cmd['clearance_m']:.1f} м менший за гальмівний, стоп"
        side = ("прямо" if abs(cmd["azimuth_deg"]) < 5
                else f"{'праворуч' if cmd['azimuth_deg'] > 0 else 'ліворуч'} "
                     f"на {abs(cmd['azimuth_deg']):.0f}°")
        return (f"{side}, запас {cmd['clearance_m']:.1f} м, "
                f"{cmd['speed']:.1f} м/с, заблоковано "
                f"{100*cmd['blocked_fraction']:.0f}% напрямків")
