"""
Bipedal T-SLIP running simulation (Torso + Spring-Loaded Inverted Pendulum)
===========================================================================
Two massless spring legs, one trunk carrying all of the mass, hip pitch
control in stance, Raibert-style foot placement.

Configured for V_TARGET = 19.5 m/s (m = 70 kg, k = 31 kN/m).

Outputs (written to the current working directory, tagged with the target speed):
    tslip_trajectories_<tag>.png   height / speed / pitch / ground-reaction-force plots
    tslip_strobe_<tag>.png         stroboscopic picture of the gait with force vectors
    tslip_running_<tag>.gif        animation of all 20 steps (10 gait cycles) with force vectors

Usage:
    python tslip_biped_running_19p5.py            # everything
    python tslip_biped_running_19p5.py --no-gif   # skip the (slow) animation
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ==========================================================================
# 1. SIMULATION & ROBOT PARAMETERS
# ==========================================================================
V_TARGET = 19.5           # target forward speed [m/s]
NUM_STEPS = 20            # 20 steps = 10 full gait cycles (left+right)

# Physical properties
MASS = 70.0               # trunk (= total, legs are massless) mass [kg]
J_COM = 4.0               # trunk inertia about its CoM [kg m^2]
D_COM = 0.10              # CoM distance above the hip joint, along the trunk [m]
J_HIP = J_COM + MASS * D_COM**2
GRAVITY = 9.81            # [m/s^2]
K_SPRING = 31000.0        # leg spring stiffness [N/m]
DAMPING = 0.0             # leg damping [N s/m]
LEG_L0 = 1.0              # rest leg length [m]

# Gait design
APEX_HEIGHT = 1.00        # hip height at the first flight apex [m] (0.96-1.04 all work at 19.5 m/s)
THETA_REF = np.radians(5.0)   # reference trunk pitch (+ = leaning forward)

# Controller
KP_THETA = 800.0          # hip pitch stiffness [N m/rad]  (raised from 500: needed at high speed)
KD_THETA = 400.0          # hip pitch damping   [N m s/rad]  (raised from 240)
K_V = 0.10                # Raibert speed gain  [rad per m/s]

# Numerics
RTOL, ATOL = 1e-9, 1e-10

# Visualization
TRUNK_LEN = 0.50          # drawn trunk length [m] (visual only)
SWING_RETRACT = 0.30      # visual leg shortening during swing (clearance)
FPS = 25
RATE_FLIGHT = 0.50        # playback speed relative to real time while airborne
RATE_STANCE = 0.12        # playback speed relative to real time during ground contact
FIGSIZE, DPI = (11.0, 6.2), 80

C_LEG = ["#e4572e", "#2e86ab"]          # leg 1 (left, red) / leg 2 (right, blue)
C_TRUNK, C_WEIGHT, C_TRAIL = "#2b2d42", "#6c757d", "#8d99ae"
BODY_WEIGHT = MASS * GRAVITY
TAG = f"{V_TARGET:g}ms".replace(".", "p")     # used in output file names


# ==========================================================================
# 2. CONTINUOUS DYNAMICS   state q = [x, y, theta, dx, dy, dtheta] (hip frame)
# ==========================================================================
def flight_dynamics(t, q):
    """No external force except gravity: CoM is ballistic, trunk spins freely."""
    x, y, th, dx, dy, dth = q
    d2x = D_COM * dth**2 * np.sin(th)
    d2y = -GRAVITY + D_COM * dth**2 * np.cos(th)
    return [dx, dy, dth, d2x, d2y, 0.0]


def leg_forces(q, x_f):
    """Spring force, hip torque and the resulting ground-reaction force (GRF).

    The GRF acting at the foot equals the force the (massless) leg exerts on the
    hip.  Because the hip torque is reacted by the leg, the GRF is not exactly
    along the leg axis:
        Fx = Fs*rx/r - tau*ry/r^2 ,   Fy = Fs*ry/r + tau*rx/r^2
    """
    x, y, th, dx, dy, dth = q
    rx, ry = x - x_f, y
    r = np.hypot(rx, ry)
    dr = (rx * dx + ry * dy) / r
    Fs = max(0.0, K_SPRING * (LEG_L0 - r) - DAMPING * dr)
    tau = -KP_THETA * (th - THETA_REF) - KD_THETA * dth
    Fx = Fs * rx / r - tau * ry / r**2
    Fy = Fs * ry / r + tau * rx / r**2
    return Fs, tau, Fx, Fy


def stance_dynamics(t, q, x_f):
    """Equations of motion (Lagrange) while one foot is pinned to the ground at x_f."""
    x, y, th, dx, dy, dth = q
    _, tau, Qx, Qy = leg_forces(q, x_f)
    md = MASS * D_COM
    M = np.array([
        [MASS, 0.0, md * np.cos(th)],
        [0.0, MASS, -md * np.sin(th)],
        [md * np.cos(th), -md * np.sin(th), J_HIP],
    ])
    rhs = np.array([
        Qx + md * dth**2 * np.sin(th),
        Qy - MASS * GRAVITY + md * dth**2 * np.cos(th),
        tau + md * GRAVITY * np.sin(th),
    ])
    acc = np.linalg.solve(M, rhs)
    return [dx, dy, dth, acc[0], acc[1], acc[2]]


# ==========================================================================
# 3. PHASE INTEGRATORS (each ends on an event)
# ==========================================================================
def integrate_flight(q0, t0, alpha_td, dense=False):
    """Fly until the hip is low enough for the foot (angle alpha_td) to touch down."""
    y_td = LEG_L0 * np.cos(alpha_td)

    def touchdown(t, s):
        return s[1] - y_td if s[4] <= 0 else 1.0     # only while descending
    touchdown.terminal, touchdown.direction = True, -1
    sol = solve_ivp(flight_dynamics, [t0, t0 + 3.0], q0, events=touchdown,
                    rtol=RTOL, atol=ATOL, dense_output=dense)
    return sol if sol.t_events[0].size else None


def integrate_stance(q0, t0, x_f, dense=False):
    """Stance until the leg has decompressed back to its rest length."""
    def liftoff(t, s):
        return np.hypot(s[0] - x_f, s[1]) - LEG_L0
    liftoff.terminal, liftoff.direction = True, 1
    sol = solve_ivp(lambda t, s: stance_dynamics(t, s, x_f), [t0, t0 + 1.0], q0,
                    events=liftoff, rtol=RTOL, atol=ATOL, dense_output=dense)
    return sol if sol.t_events[0].size else None


def integrate_to_apex(q0, t0, dense=False):
    def apex(t, s):
        return s[4]
    apex.terminal, apex.direction = True, -1
    sol = solve_ivp(flight_dynamics, [t0, t0 + 3.0], q0, events=apex,
                    rtol=RTOL, atol=ATOL, dense_output=dense)
    return sol if sol.t_events[0].size else None


# ==========================================================================
# 4. NOMINAL ATTACK ANGLE  (apex-to-apex return map, solved numerically)
# ==========================================================================
def initial_state():
    return np.array([0.0, APEX_HEIGHT, THETA_REF, V_TARGET, 0.0, 0.0])


def apex_return_error(alpha):
    """Apex height after one step minus the starting apex height."""
    q0 = initial_state()
    f = integrate_flight(q0, 0.0, alpha)
    if f is None:
        return np.nan
    q = f.y[:, -1]
    s = integrate_stance(q, f.t[-1], q[0] + LEG_L0 * np.sin(alpha))
    if s is None:
        return np.nan
    a = integrate_to_apex(s.y[:, -1], s.t[-1])
    return np.nan if a is None else a.y[1, -1] - APEX_HEIGHT


def find_nominal_attack_angle():
    """Smallest attack angle that gives a periodic apex height at V_TARGET."""
    alphas = np.radians(np.arange(8.0, 50.0, 1.0))
    err = np.array([apex_return_error(a) for a in alphas])
    for i in range(len(alphas) - 1):
        if np.isfinite(err[i]) and np.isfinite(err[i + 1]) and err[i] * err[i + 1] < 0:
            return brentq(apex_return_error, alphas[i], alphas[i + 1], xtol=1e-9)
    raise RuntimeError(
        "No periodic running gait found for these parameters. Try a different "
        "APEX_HEIGHT, V_TARGET or K_SPRING.")


# ==========================================================================
# 5. HYBRID GAIT SIMULATION
# ==========================================================================
def run_simulation(alpha_nominal):
    q, t = initial_state(), 0.0
    segs = []
    print(f"Simulating {NUM_STEPS} steps at {V_TARGET:g} m/s "
          f"(m={MASS:g} kg, k={K_SPRING/1e3:g} kN/m, nominal attack angle "
          f"{np.degrees(alpha_nominal):.2f} deg) ...")
    completed = 0
    for step in range(NUM_STEPS):
        leg = step % 2
        v_com = q[3] + D_COM * np.cos(q[2]) * q[5]
        alpha = alpha_nominal + K_V * (v_com - V_TARGET)        # Raibert foot placement

        f = integrate_flight(q, t, alpha, dense=True)
        if f is None:
            print(f"  ! no touchdown on step {step + 1}; stopping early.")
            break
        q, t = f.y[:, -1], f.t[-1]
        x_f = q[0] + LEG_L0 * np.sin(alpha)
        s = integrate_stance(q, t, x_f, dense=True)
        if s is None:
            print(f"  ! stance never ended on step {step + 1}; stopping early.")
            break
        segs.append(dict(kind="flight", step=step, leg=None, sol=f.sol, t0=f.t[0], t1=f.t[-1]))
        segs.append(dict(kind="stance", step=step, leg=leg, sol=s.sol, t0=s.t[0], t1=s.t[-1],
                         x_f=x_f, alpha_td=alpha))
        q, t = s.y[:, -1], s.t[-1]
        completed += 1

    if completed:                                    # finish at the next apex
        a = integrate_to_apex(q, t, dense=True)
        if a is not None:
            segs.append(dict(kind="flight", step=completed - 1, leg=None, sol=a.sol,
                             t0=a.t[0], t1=a.t[-1]))
    print(f"  completed {completed}/{NUM_STEPS} steps in {segs[-1]['t1']:.2f} s")
    return segs


# ==========================================================================
# 6. SAMPLING  (uniform-in-time states, forces and leg poses)
# ==========================================================================
def build_leg_model(segs):
    """Collect stance intervals per leg so the swing leg can be animated."""
    stances = {0: [], 1: []}
    for sg in segs:
        if sg["kind"] != "stance":
            continue
        x_lo, y_lo = sg["sol"](sg["t1"])[:2]
        stances[sg["leg"]].append(dict(
            t0=sg["t0"], t1=sg["t1"], x_f=sg["x_f"], phi_td=sg["alpha_td"],
            phi_lo=np.arctan2(sg["x_f"] - x_lo, y_lo)))
    t0s = np.array([sg["t0"] for sg in segs if sg["kind"] == "stance"])
    durs = np.array([sg["t1"] - sg["t0"] for sg in segs if sg["kind"] == "stance"])
    step_t = np.mean(np.diff(t0s)) if len(t0s) > 1 else 0.4
    alphas = np.array([sg["alpha_td"] for sg in segs if sg["kind"] == "stance"])
    return dict(stances=stances, swing_T=2 * step_t - durs.mean(), mean_alpha=alphas.mean())


def leg_foot(model, leg, t, hx, hy):
    """Foot position of `leg` at time t (stance: pinned; swing: visual interpolation)."""
    prev = nxt = None
    for st in model["stances"][leg]:
        if st["t0"] <= t < st["t1"]:
            return np.array([st["x_f"], 0.0]), True
        if st["t1"] <= t:
            prev = st
        elif nxt is None:
            nxt = st
    T = model["swing_T"]
    if prev is not None and nxt is not None:
        tp, php, tn, phn = prev["t1"], prev["phi_lo"], nxt["t0"], nxt["phi_td"]
    elif prev is not None:
        tp, php = prev["t1"], prev["phi_lo"]
        tn, phn = tp + T, model["mean_alpha"]
    else:
        tn, phn = nxt["t0"], nxt["phi_td"]
        tp, php = max(0.0, tn - T), -phn
    s = float(np.clip((t - tp) / max(tn - tp, 1e-9), 0.0, 1.0))
    sm = s * s * (3 - 2 * s)
    phi = php + (phn - php) * sm
    length = LEG_L0 * (1.0 - SWING_RETRACT * np.sin(np.pi * s))
    return np.array([hx + length * np.sin(phi), max(0.02, hy - length * np.cos(phi))]), False


def sample(segs, model, t, with_legs=True):
    n = len(t)
    S = dict(t=np.asarray(t, float), q=np.full((6, n), np.nan), contact=-np.ones(n, int),
             x_foot=np.full(n, np.nan), Fs=np.zeros(n), tau=np.zeros(n), Fx=np.zeros(n),
             Fy=np.zeros(n), r=np.full(n, np.nan), step=np.zeros(n, int))
    for sg in segs:
        last = sg is segs[-1]
        m = (S["t"] >= sg["t0"]) & ((S["t"] < sg["t1"]) | (last & (S["t"] <= sg["t1"] + 1e-12)))
        if not m.any():
            continue
        S["q"][:, m] = sg["sol"](S["t"][m])
        S["step"][m] = sg["step"]
        if sg["kind"] == "stance":
            S["contact"][m] = sg["leg"]
            S["x_foot"][m] = sg["x_f"]
            for i in np.where(m)[0]:
                S["Fs"][i], S["tau"][i], S["Fx"][i], S["Fy"][i] = leg_forces(S["q"][:, i], sg["x_f"])
                S["r"][i] = np.hypot(S["q"][0, i] - sg["x_f"], S["q"][1, i])
    q = S["q"]
    th = q[2]
    S["x_com"] = q[0] + D_COM * np.sin(th)
    S["y_com"] = q[1] + D_COM * np.cos(th)
    S["v_com"] = q[3] + D_COM * np.cos(th) * q[5]
    S["F"] = np.hypot(S["Fx"], S["Fy"])
    if with_legs:
        S["foot"] = np.full((2, 2, n), np.nan)
        S["touch"] = np.zeros((2, n), bool)
        for leg in (0, 1):
            for i in range(n):
                S["foot"][leg, :, i], S["touch"][leg, i] = leg_foot(model, leg, S["t"][i], q[0, i], q[1, i])
    return S


def stance_intervals(segs):
    return [(sg["t0"], sg["t1"], sg["leg"]) for sg in segs if sg["kind"] == "stance"]


def build_frame_times(segs):
    """Slow-motion playback: very slow in ground contact, gentler in flight (smooth blend)."""
    iv = np.array([(a, b) for a, b, _ in stance_intervals(segs)])
    t_end = segs[-1]["t1"]
    times, rates, t = [], [], 0.0
    while t < t_end:
        dist = np.min(np.where(iv[:, 0] <= t, np.where(t <= iv[:, 1], 0.0, t - iv[:, 1]), iv[:, 0] - t))
        u = np.clip(dist / 0.05, 0, 1)
        rate = RATE_STANCE + (RATE_FLIGHT - RATE_STANCE) * u * u * (3 - 2 * u)
        times.append(t)
        rates.append(rate)
        t += rate / FPS
    times.append(t_end)
    rates.append(rates[-1])
    return np.array(times), np.array(rates)


# ==========================================================================
# 7. SUMMARY
# ==========================================================================
def print_summary(segs, model):
    S = sample(segs, model, np.arange(0, segs[-1]["t1"], 0.001), with_legs=False)
    T, dist = segs[-1]["t1"], S["x_com"][-1]
    print("\n step leg  attack[deg]  stance[ms]  flight[ms]  peak GRF[N]  [BW]  compression[%]  v_com[m/s]")
    stances = [sg for sg in segs if sg["kind"] == "stance"]
    flights = [sg for sg in segs if sg["kind"] == "flight"]
    for sg in stances:
        k = sg["step"]
        m = S["step"] == k
        m_st = m & (S["contact"] >= 0)
        fl = flights[k]
        print(f" {k + 1:4d}  {sg['leg'] + 1}   {np.degrees(sg['alpha_td']):9.2f}  "
              f"{(sg['t1'] - sg['t0']) * 1e3:10.1f}  {(fl['t1'] - fl['t0']) * 1e3:10.1f}  "
              f"{S['F'][m_st].max():11.0f}  {S['F'][m_st].max() / BODY_WEIGHT:4.1f}  "
              f"{100 * (LEG_L0 - np.nanmin(S['r'][m_st])) / LEG_L0:14.1f}  {S['v_com'][m_st][0]:10.2f}")
    print(f"\n Total time {T:.2f} s | distance {dist:.1f} m | average speed {dist / T:.2f} m/s")
    print(f" Pitch range {np.degrees(S['q'][2].min()):+.1f} .. {np.degrees(S['q'][2].max()):+.1f} deg | "
          f"peak GRF {S['F'].max():.0f} N ({S['F'].max() / BODY_WEIGHT:.1f} BW)\n")


# ==========================================================================
# 8. DRAWING HELPERS
# ==========================================================================
def spring_xy(p0, p1, coils=7, amp=0.05, frac=(0.16, 0.84)):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-9:
        return np.array([p0[0], p1[0]]), np.array([p0[1], p1[1]])
    u, nrm = d / L, np.array([-d[1], d[0]]) / L
    pts = [p0, p0 + u * L * frac[0]]
    for j in range(2 * coils):
        s = frac[0] + (frac[1] - frac[0]) * (j + 0.5) / (2 * coils)
        pts.append(p0 + u * L * s + nrm * amp * (1 if j % 2 == 0 else -1))
    pts += [p0 + u * L * frac[1], p1]
    pts = np.array(pts)
    return pts[:, 0], pts[:, 1]


def pose(S, i):
    """Geometry of the runner at sample i (all in data coordinates)."""
    hx, hy, th = S["q"][0, i], S["q"][1, i], S["q"][2, i]
    u = np.array([np.sin(th), np.cos(th)])
    return dict(
        hip=np.array([hx, hy]), top=np.array([hx, hy]) + TRUNK_LEN * u,
        head=np.array([hx, hy]) + (TRUNK_LEN + 0.09) * u,
        com=np.array([S["x_com"][i], S["y_com"][i]]),
        foot=[S["foot"][0, :, i], S["foot"][1, :, i]], touch=[S["touch"][0, i], S["touch"][1, i]])


def draw_leg(ax, hip, foot, touching, leg, alpha=1.0, **kw):
    col = C_LEG[leg]
    if touching:
        x, y = spring_xy(hip, foot)
        return ax.plot(x, y, color=col, lw=2.4, alpha=alpha, solid_capstyle="round", **kw)[0]
    return ax.plot([hip[0], foot[0]], [hip[1], foot[1]], color=col, lw=3.0, alpha=0.55 * alpha,
                   solid_capstyle="round", **kw)[0]


MIN_ARROW_FORCE = 0.25 * BODY_WEIGHT     # hide force arrows smaller than this (head would dominate)


def make_grf_arrow(posA, posB, color, alpha=0.55, zorder=3, scale=22):
    """Wide translucent block arrow (drawn behind the legs so the spring stays visible)."""
    return FancyArrowPatch(posA, posB, arrowstyle="simple,head_length=1.0,head_width=1.5,tail_width=0.55",
                           mutation_scale=scale, fc=color, ec="white", lw=0.8, alpha=alpha,
                           zorder=zorder, shrinkA=0, shrinkB=0)


def style_world_axes(ax, ylim=(-0.25, 2.05)):
    ax.set_ylim(*ylim)
    ax.axhspan(ylim[0], 0, color="#8d6e63", alpha=0.28, lw=0)
    ax.axhline(0, color="#5d4037", lw=3)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylabel("height [m]")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.grid(axis="x", which="both", alpha=0.12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def shade_stances(ax, segs, alpha=0.13):
    for a, b, leg in stance_intervals(segs):
        ax.axvspan(a, b, color=C_LEG[leg], alpha=alpha, lw=0)


# ==========================================================================
# 9. STATIC FIGURES
# ==========================================================================
def plot_results(segs, model):
    S = sample(segs, model, np.arange(0, segs[-1]["t1"], 0.001), with_legs=False)
    t, q = S["t"], S["q"]
    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axs[0].plot(t, q[1], color="navy", lw=1.8, label=r"hip height $y_{hip}$")
    axs[0].plot(t, S["y_com"], color="crimson", ls="--", lw=1.5, label=r"CoM height $y_{com}$")
    axs[0].set_ylabel("height [m]")
    axs[0].set_title(f"Bipedal T-SLIP running at {V_TARGET:g} m/s   "
                     f"(m = {MASS:g} kg, k = {K_SPRING / 1e3:g} kN/m)")

    axs[1].plot(t, S["v_com"], color="forestgreen", lw=1.8, label=r"CoM forward speed $\dot{x}_{com}$")
    axs[1].axhline(V_TARGET, color="black", ls=":", label=f"target ({V_TARGET:g} m/s)")
    axs[1].set_ylabel("speed [m/s]")

    axs[2].plot(t, np.degrees(q[2]), color="darkorange", lw=1.8, label=r"trunk pitch $\theta$")
    axs[2].axhline(np.degrees(THETA_REF), color="black", ls=":", label=r"reference $\theta_{ref}$")
    axs[2].set_ylabel("pitch [deg]")

    axs[3].plot(t, S["Fy"] / BODY_WEIGHT, color="#222", lw=1.8, label=r"vertical GRF $F_y$")
    axs[3].plot(t, S["Fx"] / BODY_WEIGHT, color="#888", lw=1.6, ls="--", label=r"horizontal GRF $F_x$")
    axs[3].axhline(0, color="k", lw=0.6)
    axs[3].set_ylabel("GRF [body weights]")
    axs[3].set_xlabel("time [s]")

    for ax in axs:
        shade_stances(ax, segs)
        ax.grid(True, alpha=0.25)
        ax.margins(y=0.32)
        ax.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.9)
    axs[3].text(0.01, 0.04, "shaded: stance of leg 1 (red) / leg 2 (blue)", transform=axs[3].transAxes,
                fontsize=8.5, color="#444")
    fig.tight_layout()
    fig.savefig(f"tslip_trajectories_{TAG}.png", dpi=170)
    plt.close(fig)
    print(f"Saved 'tslip_trajectories_{TAG}.png'")


def plot_strobe(segs, model, fscale):
    """Stroboscopic view of one gait cycle (two steps) taken from mid-run."""
    k0 = min(8, NUM_STEPS - 3)
    st = [sg for sg in segs if sg["kind"] == "stance"]
    fl = [sg for sg in segs if sg["kind"] == "flight"]
    # At high speed one step is already ~9 m long, so show a single step (stance -> flight -> stance)
    n_steps = 2 if V_TARGET <= 12 else 1
    times, body = [], []
    for k in range(k0, k0 + n_steps):
        a, b = st[k]["t0"], st[k]["t1"]
        times.append(0.5 * (a + b))                    # mid-stance: peak force
        body.append(True)
        c0, c1 = b, fl[k + 1]["t1"] if k + 1 < len(fl) else b + 0.2
        for f in (0.2, 0.4, 0.6, 0.8):                 # flight poses
            times.append(c0 + (c1 - c0) * f)
            body.append(True)
    if n_steps == 1:                                   # close with the next stance
        a, b = st[k0 + 1]["t0"], st[k0 + 1]["t1"]
        times.append(0.5 * (a + b))
        body.append(True)
    times = np.array(times)
    S = sample(segs, model, times)
    fig, ax = plt.subplots(figsize=(14, 4.4))
    n = len(times)
    for i in range(n):
        P = pose(S, i)
        al = 0.35 + 0.65 * i / (n - 1)
        if body[i]:
            for leg in (1, 0):
                draw_leg(ax, P["hip"], P["foot"][leg], P["touch"][leg], leg, alpha=al)
            ax.plot(*zip(P["hip"], P["top"]), color=C_TRUNK, lw=6, alpha=al, solid_capstyle="round")
            ax.plot(*P["head"], "o", color=C_TRUNK, ms=13, alpha=al)
            ax.plot(*P["hip"], "o", mfc="white", mec=C_TRUNK, ms=6, alpha=al)
            ax.plot(*P["com"], "o", color="crimson", mec="white", ms=8, alpha=al, zorder=5)
            ax.add_patch(make_grf_arrow(P["com"], P["com"] + [0, -fscale * BODY_WEIGHT], C_WEIGHT,
                                        alpha=0.9 * al, zorder=5, scale=9))
        leg = S["contact"][i]
        if leg >= 0 and S["F"][i] > MIN_ARROW_FORCE:
            foot = np.array([S["x_foot"][i], 0.0])
            ax.add_patch(make_grf_arrow(foot, foot + fscale * np.array([S["Fx"][i], S["Fy"][i]]),
                                        C_LEG[leg], alpha=0.55 * al))
    ax.set_aspect("equal")
    style_world_axes(ax, (-0.25, 1.85))
    xs = S["q"][0]
    ax.set_xlim(xs.min() - 1.2, xs.max() + 1.8)
    ax.set_xlabel("distance [m]")
    what = "One gait cycle (left + right step)" if n_steps == 2 else "One step (stance -> flight -> stance)"
    ax.set_title(f"{what}: ground reaction forces (leg colours) and body weight (grey)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"tslip_strobe_{TAG}.png", dpi=170)
    plt.close(fig)
    print(f"Saved 'tslip_strobe_{TAG}.png'")


# ==========================================================================
# 10. ANIMATION
# ==========================================================================
def animate_running(segs, model, fscale):
    ft, rates = build_frame_times(segs)
    S = sample(segs, model, ft)
    Sp = sample(segs, model, np.arange(0, segs[-1]["t1"], 0.001), with_legs=False)
    nf = len(ft)
    print(f"Rendering {nf} frames ({nf / FPS:.1f} s of animation) ...")

    fig = plt.figure(figsize=FIGSIZE, facecolor="#fbfbfd")
    rect = [0.065, 0.335, 0.925, 0.60]
    ax = fig.add_axes(rect)
    axf = fig.add_axes([0.065, 0.085, 0.925, 0.17])

    # --- world view ----------------------------------------------------------
    ylim = (-0.25, 2.05)
    style_world_axes(ax, ylim)
    ax.set_xlabel("distance [m]")
    w_in, h_in = FIGSIZE[0] * rect[2], FIGSIZE[1] * rect[3]
    x_width = (ylim[1] - ylim[0]) * w_in / h_in
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Bipedal T-SLIP running at {V_TARGET:g} m/s  -  m = {MASS:g} kg,  "
                 f"k = {K_SPRING / 1e3:g} kN/m,  20 steps (10 gait cycles)", fontsize=12, pad=8)

    foot_x = [sg["x_f"] for sg in segs if sg["kind"] == "stance"]
    foot_leg = [sg["leg"] for sg in segs if sg["kind"] == "stance"]
    for xf, lg in zip(foot_x, foot_leg):
        ax.plot(xf, 0.0, "^", color=C_LEG[lg], ms=7, mec="white", zorder=3, clip_on=True)

    trail, = ax.plot([], [], color=C_TRAIL, lw=2, alpha=0.9, zorder=2)
    leg_lines = [ax.plot([], [], color=C_LEG[l], lw=3, solid_capstyle="round", zorder=4)[0] for l in (0, 1)]
    trunk, = ax.plot([], [], color=C_TRUNK, lw=7, solid_capstyle="round", zorder=6)
    head, = ax.plot([], [], "o", color=C_TRUNK, ms=15, zorder=6)
    hip_dot, = ax.plot([], [], "o", mfc="white", mec=C_TRUNK, ms=6, zorder=7)
    com_dot, = ax.plot([], [], "o", color="crimson", mec="white", mew=1.2, ms=10, zorder=8)
    grf_arrow = make_grf_arrow((0, 0), (0, 0), C_LEG[0], alpha=0.6, zorder=3)
    wt_arrow = make_grf_arrow((0, 0), (0, 0), C_WEIGHT, alpha=0.9, zorder=5, scale=9)
    ax.add_patch(grf_arrow)
    ax.add_patch(wt_arrow)
    hud = ax.text(0.012, 0.965, "", transform=ax.transAxes, va="top", family="monospace", fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cfcfd6", alpha=0.9))
    ax.legend(handles=[
        Line2D([], [], color=C_LEG[0], lw=3, label="GRF, leg 1 (left)"),
        Line2D([], [], color=C_LEG[1], lw=3, label="GRF, leg 2 (right)"),
        Line2D([], [], color=C_WEIGHT, lw=2, label="weight m*g"),
        Line2D([], [], color=C_TRAIL, lw=2, label="CoM path")],
        loc="upper right", fontsize=9, framealpha=0.9, ncol=2)

    # --- force-vs-time panel -----------------------------------------------------
    tp = Sp["t"]
    for a, b, lg in stance_intervals(segs):
        m = (tp >= a) & (tp <= b)
        axf.fill_between(tp[m], 0, Sp["Fy"][m] / BODY_WEIGHT, color=C_LEG[lg], alpha=0.65, lw=0)
    axf.plot(tp, Sp["Fx"] / BODY_WEIGHT, color="#333", lw=1.2, ls="--", label=r"horizontal GRF $F_x$")
    axf.plot([], [], color="#777", lw=6, alpha=0.65, label=r"vertical GRF $F_y$ (colour = leg)")
    axf.axhline(0, color="k", lw=0.6)
    axf.set_xlim(0, tp[-1])
    axf.set_ylabel("GRF [BW]")
    axf.set_xlabel("time [s]")
    axf.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)
    axf.grid(alpha=0.2)
    for sp in ("top", "right"):
        axf.spines[sp].set_visible(False)
    cursor = axf.axvline(0, color="k", lw=1.3)

    def update(i):
        P = pose(S, i)
        xc = P["com"][0] + 1.6
        ax.set_xlim(xc - x_width / 2, xc + x_width / 2)

        for leg in (0, 1):
            hip, foot, touch = P["hip"], P["foot"][leg], P["touch"][leg]
            if touch:
                x, y = spring_xy(hip, foot)
                leg_lines[leg].set_data(x, y)
                leg_lines[leg].set_alpha(1.0)
                leg_lines[leg].set_linewidth(2.4)
                leg_lines[leg].set_zorder(5)
            else:
                leg_lines[leg].set_data([hip[0], foot[0]], [hip[1], foot[1]])
                leg_lines[leg].set_alpha(0.55)
                leg_lines[leg].set_linewidth(3.0)
                leg_lines[leg].set_zorder(4)
        trunk.set_data([P["hip"][0], P["top"][0]], [P["hip"][1], P["top"][1]])
        head.set_data([P["head"][0]], [P["head"][1]])
        hip_dot.set_data([P["hip"][0]], [P["hip"][1]])
        com_dot.set_data([P["com"][0]], [P["com"][1]])

        j0 = max(0, i - 40)
        trail.set_data(S["x_com"][j0:i + 1], S["y_com"][j0:i + 1])

        wt_arrow.set_positions(P["com"], P["com"] + [0, -fscale * BODY_WEIGHT])
        leg = S["contact"][i]
        if leg >= 0:
            foot = np.array([S["x_foot"][i], 0.0])
            grf_arrow.set_positions(foot, foot + fscale * np.array([S["Fx"][i], S["Fy"][i]]))
            grf_arrow.set_facecolor(C_LEG[leg])
            grf_arrow.set_visible(bool(S["F"][i] > MIN_ARROW_FORCE))
            lines = (f"GRF = {S['F'][i]:6.0f} N ({S['F'][i] / BODY_WEIGHT:4.1f} BW)  "
                     f"leg {S['r'][i]:.2f} m  hip torque {S['tau'][i]:+5.0f} Nm")
        else:
            grf_arrow.set_visible(False)
            lines = "flight phase - no ground contact"
        hud.set_text(f"t = {S['t'][i]:5.2f} s   step {S['step'][i] + 1:2d}/{NUM_STEPS}   "
                     f"v = {S['v_com'][i]:5.2f} m/s   pitch = {np.degrees(S['q'][2, i]):+5.1f} deg\n"
                     f"{lines}\nplayback {rates[i]:.2f}x real time")
        cursor.set_xdata([S["t"][i], S["t"][i]])
        if i % 100 == 0:
            print(f"  frame {i}/{nf}")
        return ()

    anim = FuncAnimation(fig, update, frames=nf, interval=1000 / FPS, blit=False)
    anim.save(f"tslip_running_{TAG}.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"Saved 'tslip_running_{TAG}.gif'")


# ==========================================================================
# 11. MAIN
# ==========================================================================
if __name__ == "__main__":
    alpha_nom = find_nominal_attack_angle()
    segs = run_simulation(alpha_nom)
    model = build_leg_model(segs)
    print_summary(segs, model)

    Sall = sample(segs, model, np.arange(0, segs[-1]["t1"], 0.001), with_legs=False)
    FORCE_SCALE = 1.1 / Sall["F"].max()          # longest GRF arrow = 1.1 m on screen

    plot_results(segs, model)
    plot_strobe(segs, model, FORCE_SCALE)
    if "--no-gif" not in sys.argv:
        animate_running(segs, model, FORCE_SCALE)
