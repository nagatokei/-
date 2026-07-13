"""
Bell Test - Spinor Spherical Wave Simulation
==========================================
スピノル構造（クォータニオン）の振動が生み出す球状波をモデル化したシミュレーション。
AとBの検出器に対して、伝播した波を観測する手法として
「決定的（Deterministic）」と「確率的（Probabilistic）」の2通りを比較できます。
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
from datetime import datetime

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# --- CHSH角度設定 ---
CHSH_CONFIGS = {
    "Standard":  (0, 90, 45, 135),
    "Rotated30": (30, 120, 75, 165),
    "Rotated45": (45, 135, 90, 180),
}

# --- パラメータ ---
PARAMS = {
    "AMP_W":          {"val": 1.0, "min": 0.0, "max": 2.0},
    "AMP_X":          {"val": 1.0, "min": 0.0, "max": 2.0},
    "AMP_Y":          {"val": 1.0, "min": 0.0, "max": 2.0},
    "AMP_Z":          {"val": 1.0, "min": 0.0, "max": 2.0},
    "CONE_SHARPNESS": {"val": 1.5, "min": 0.0, "max": 4.0}, # kの値 (0で全検出)
}

def generate_particles(n_total, params):
    """
    スピノル（クォータニオン）の初期状態をランダムに生成。
    4次元単位球面上のランダムベクトルとして作成する。
    """
    v = np.random.normal(0, 1, (n_total, 4))
    
    # 軸ごとの波の幅を変調
    v[:, 0] *= params["AMP_W"]["val"]
    v[:, 1] *= params["AMP_X"]["val"]
    v[:, 2] *= params["AMP_Y"]["val"]
    v[:, 3] *= params["AMP_Z"]["val"]
    
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    q = np.zeros_like(v)
    mask_norm = norms.flatten() > 1e-8
    q[mask_norm] = v[mask_norm] / norms[mask_norm]
    
    # 3次元成分（x, y, z）を抽出して正規化（ウネリの代表ベクトル）
    particle_axes = q[:, 1:4]
    ax_norms = np.linalg.norm(particle_axes, axis=1, keepdims=True)
    
    # 0割りを防ぐ
    mask = ax_norms.flatten() > 1e-8
    particle_axes[mask] = particle_axes[mask] / ax_norms[mask]
    particle_axes[~mask] = 0.0
    
    return particle_axes

def measure(particle_axes, angle_deg, is_particle_b=False, method="probabilistic", params=None):
    """
    波の観測（測定）処理
    particle_axes: N x 3 のベクトル配列
    angle_deg: 検出器の角度
    is_particle_b: B側の観測ならTrue（反相関を再現するために判定を逆転させる）
    method: "deterministic" または "probabilistic"
    """
    rad = np.radians(angle_deg)
    # 検出器の軸（Z軸周りの回転を想定し、X-Y平面上に配置）
    det_x = np.cos(rad)
    det_y = np.sin(rad)
    
    # 粒子（波）の軸と検出器軸の内積を計算
    dot_products = particle_axes[:, 0] * det_x + particle_axes[:, 1] * det_y
    
    # B側の波はA側に対して「反対」の観測結果を生む（角運動量保存/反相関）
    if is_particle_b:
        dot_products = -dot_products
        
    if method == "deterministic":
        # 内積が0以上なら+1、未満なら-1
        results = np.where(dot_products >= 0, 1, -1)
    elif method == "probabilistic":
        # 内積から+1になる確率を計算し、ランダムに判定
        prob_plus = (1 + dot_products) / 2
        rand_vals = np.random.rand(len(dot_products))
        results = np.where(rand_vals < prob_plus, 1, -1)
    elif method == "mach_cone":
        # マッハコーン（観測の抜け穴）: 連続的な確率フィルター
        k = params["CONE_SHARPNESS"]["val"] if params else 1.0
        intensity = np.abs(dot_products)
        
        # 検出される確率は波の強さ（干渉）のべき乗
        p_detect = np.power(intensity, k)
        rand_vals = np.random.rand(len(dot_products))
        detected = rand_vals < p_detect
        
        # 検出されたものは決定論的に判定（波の芯と一致すれば必ず相関する）
        results = np.where(detected, np.where(dot_products >= 0, 1, -1), 0)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return results

def run_experimental_protocol(n_total, a, a2, b, b2, method, params):
    """
    実験プロトコル：
    各粒子ペアに対して4つの設定から1つを選び、1回だけ観測する。
    """
    # ランダムに設定を割り当て (0,1,2,3)
    choices = np.random.randint(0, 4, n_total)
    
    # 隠れ変数（スピノル状態）を生成
    particle_axes = generate_particles(n_total, params)
    
    # 各設定のインデックス
    idx_ab   = choices == 0
    idx_ab2  = choices == 1
    idx_a2b  = choices == 2
    idx_a2b2 = choices == 3
    
    # 各グループで測定と未検出(0)の除外
    def get_valid_products(idx, ang_a, ang_b):
        if not np.any(idx):
            return np.array([]), 0
        rA = measure(particle_axes[idx], ang_a, is_particle_b=False, method=method, params=params)
        rB = measure(particle_axes[idx], ang_b, is_particle_b=True, method=method, params=params)
        prod = rA * rB
        valid = prod != 0 # どちらかが未検出(0)なら除外
        if np.sum(valid) == 0:
            return np.array([0]), 0
        return prod[valid], np.sum(valid)

    p_ab, c_ab = get_valid_products(idx_ab, a, b)
    p_ab2, c_ab2 = get_valid_products(idx_ab2, a, b2)
    p_a2b, c_a2b = get_valid_products(idx_a2b, a2, b)
    p_a2b2, c_a2b2 = get_valid_products(idx_a2b2, a2, b2)
    
    results = {
        "ab": {"products": p_ab, "count": c_ab},
        "ab2": {"products": p_ab2, "count": c_ab2},
        "a2b": {"products": p_a2b, "count": c_a2b},
        "a2b2": {"products": p_a2b2, "count": c_a2b2}
    }
    
    E_ab   = np.mean(results["ab"]["products"])
    E_ab2  = np.mean(results["ab2"]["products"])
    E_a2b  = np.mean(results["a2b"]["products"])
    E_a2b2 = np.mean(results["a2b2"]["products"])
    
    S = E_ab - E_ab2 + E_a2b + E_a2b2
    
    return {
        "S": S, "abs": abs(S),
        "E_ab": E_ab, "E_ab2": E_ab2, "E_a2b": E_a2b, "E_a2b2": E_a2b2,
        "counts": {k: results[k]["count"] for k in results},
        "angles": f"a={a} a'={a2} b={b} b'={b2}",
    }

def run_correlation_curve(n_pairs, method, params):
    """E(0, theta) の相関カーブを計算"""
    particle_axes = generate_particles(n_pairs, params)
    angles = np.arange(0, 181, 5)
    correlations = []
    
    # 一定のA側(0度)に対する測定結果を先に計算
    rA = measure(particle_axes, 0, is_particle_b=False, method=method, params=params)
    
    for theta in angles:
        rB = measure(particle_axes, theta, is_particle_b=True, method=method, params=params)
        prod = rA * rB
        valid = prod != 0
        if np.sum(valid) > 0:
            correlations.append(np.mean(prod[valid]))
        else:
            correlations.append(0.0)
        
    return angles, np.array(correlations)


# === GUI ===
def main():
    root = tk.Tk()
    root.title("Bell Test - Spinor Spherical Wave")
    root.geometry("600x750")
    root.attributes('-topmost', True)
    root.configure(bg='#1a1a2e')
    
    # --- ヘッダー ---
    header = tk.Label(root, text="Spinor Spherical Wave Simulation", 
                      bg='#1a1a2e', fg='#4ecdc4', font=('Consolas', 14, 'bold'))
    header.pack(pady=10)
    
    # --- パラメータスライダー ---
    slider_frame = tk.Frame(root, bg='#1a1a2e')
    slider_frame.pack(fill=tk.X, padx=10, pady=5)
    for key, data in PARAMS.items():
        frame = tk.Frame(slider_frame, bg='#1a1a2e')
        frame.pack(fill=tk.X, pady=1)
        label_var = tk.StringVar(value=f"{key}: {data['val']:.1f}")
        tk.Label(frame, textvariable=label_var, bg='#1a1a2e', fg='#ccc',
                 font=('Consolas', 9), width=12, anchor='w').pack(side=tk.LEFT)
        def make_cmd(k, lv):
            def on_change(val):
                PARAMS[k]["val"] = float(val)
                lv.set(f"{k}: {float(val):.1f}")
            return on_change
        slider = ttk.Scale(frame, from_=data["min"], to=data["max"],
                         orient=tk.HORIZONTAL, command=make_cmd(key, label_var))
        slider.set(data["val"])
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    # --- 観測手法の選択 ---
    method_var = tk.StringVar(value="probabilistic")
    
    frame_method = tk.Frame(root, bg='#1a1a2e')
    frame_method.pack(pady=5)
    
    tk.Label(frame_method, text="Observation Method:", bg='#1a1a2e', fg='#ccc', 
             font=('Consolas', 10)).pack(side=tk.LEFT, padx=10)
             
    rb_det = tk.Radiobutton(frame_method, text="Deterministic", 
                            variable=method_var, value="deterministic",
                            bg='#1a1a2e', fg='#ffd93d', selectcolor='#333',
                            font=('Consolas', 9))
    rb_det.pack(side=tk.LEFT, padx=5)
    
    rb_prob = tk.Radiobutton(frame_method, text="Probabilistic", 
                             variable=method_var, value="probabilistic",
                             bg='#1a1a2e', fg='#ffd93d', selectcolor='#333',
                             font=('Consolas', 9))
    rb_prob.pack(side=tk.LEFT, padx=5)
    
    rb_mach = tk.Radiobutton(frame_method, text="Mach Cone (Loophole)", 
                             variable=method_var, value="mach_cone",
                             bg='#1a1a2e', fg='#ff6b6b', selectcolor='#333',
                             font=('Consolas', 9, 'bold'))
    rb_mach.pack(side=tk.LEFT, padx=5)

    # --- グラフ ---
    fig, (ax_curve, ax_bars) = plt.subplots(1, 2, figsize=(7.0, 3.5), dpi=90,
                                             gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor('#1a1a2e')
    
    # 左: 相関カーブ
    ax_curve.set_facecolor('#16213e')
    ax_curve.set_xlabel('Angle (deg)', color='#aaa', fontsize=9)
    ax_curve.set_ylabel('E(theta)', color='#aaa', fontsize=9)
    ax_curve.tick_params(colors='#888', labelsize=8)
    ax_curve.set_xlim(0, 180)
    ax_curve.set_ylim(-1.1, 1.1)
    ax_curve.grid(True, alpha=0.2, color='#555')
    ax_curve.axhline(y=0, color='#555', linewidth=0.5)
    for spine in ax_curve.spines.values():
        spine.set_color('#333')
    
    qm_angles = np.linspace(0, 180, 200)
    ax_curve.plot(qm_angles, -np.cos(np.radians(qm_angles)), '--', color='#ff6b6b',
                 linewidth=1.5, label='QM (-cos)', alpha=0.7)
    ax_curve.plot(qm_angles, -1 + 2*qm_angles/180, ':', color='#ffd93d',
                 linewidth=1, label='Classical Linear', alpha=0.5)
    sim_line, = ax_curve.plot([], [], 'o-', color='#4ecdc4', linewidth=1.5,
                              markersize=3, label='Wave Model', alpha=0.9)
    ax_curve.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e',
                   edgecolor='#333', labelcolor='#ccc')
    
    # 右: CHSH比較バー
    ax_bars.set_facecolor('#16213e')
    ax_bars.set_ylabel('|S|', color='#aaa', fontsize=9)
    ax_bars.tick_params(colors='#888', labelsize=8)
    ax_bars.set_ylim(0, 3.0)
    ax_bars.axhline(y=2, color='#ffd93d', linewidth=2, linestyle='--', alpha=0.8, label='Bell Limit')
    ax_bars.axhline(y=2*np.sqrt(2), color='#ff6b6b', linewidth=1, linestyle=':', alpha=0.5, label='QM Limit')
    for spine in ax_bars.spines.values():
        spine.set_color('#333')
    ax_bars.legend(loc='upper right', fontsize=6, facecolor='#1a1a2e',
                  edgecolor='#333', labelcolor='#ccc')
    
    fig.tight_layout()
    
    canvas_fig = FigureCanvasTkAgg(fig, master=root)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # --- 結果テキスト ---
    result_text = tk.Text(root, height=10, bg='#0a0a1a', fg='#ccc',
                         font=('Consolas', 9), relief='flat', padx=5, pady=5)
    result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    status_var = tk.StringVar(value="Ready")
    
    def on_run():
        run_btn.config(state='disabled', text="Running...")
        result_text.delete('1.0', tk.END)
        method = method_var.get()
        status_var.set(f"Running with {method} method...")
        root.update()
        
        def worker():
            try:
                lines = []
                lines.append("=" * 50)
                lines.append(f"  SPINOR SPHERICAL WAVE SIMULATION")
                lines.append(f"  Method: {method.upper()}")
                lines.append("  N = 400,000 pairs")
                lines.append("=" * 50)
                
                n_total = 400000
                
                # --- 相関カーブ ---
                root.after(0, lambda: status_var.set("Computing correlation curve..."))
                angles, corrs = run_correlation_curve(100000, method, PARAMS)
                
                def update_curve():
                    sim_line.set_data(angles, corrs)
                    canvas_fig.draw()
                root.after(0, update_curve)
                
                bar_vals = []
                bar_labels = []
                
                for config_name, (a, a2, b, b2) in CHSH_CONFIGS.items():
                    root.after(0, lambda cn=config_name: status_var.set(f"Testing {cn}..."))
                    
                    lines.append(f"\n--- {config_name}: a={a} a'={a2} b={b} b'={b2} ---")
                    
                    exp = run_experimental_protocol(n_total, a, a2, b, b2, method, PARAMS)
                    lines.append(f"  E(a,b)={exp['E_ab']:.4f}  E(a,b')={exp['E_ab2']:.4f}")
                    lines.append(f"  E(a',b)={exp['E_a2b']:.4f}  E(a',b')={exp['E_a2b2']:.4f}")
                    lines.append(f"  S = {exp['S']:.6f}  |S| = {exp['abs']:.6f}")
                    
                    if exp['abs'] > 2.0:
                        lines.append(f"  >>> |S| > 2 (Bell Limit Exceeded) ! <<<")
                        
                    bar_vals.append(exp['abs'])
                    bar_labels.append(config_name[:8])
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SCREENSHOT_DIR, f"spinor_wave_{method}_{timestamp}.txt")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                def update_final():
                    result_text.delete('1.0', tk.END)
                    result_text.insert('1.0', '\n'.join(lines))
                    
                    # バーグラフ更新
                    ax_bars.clear()
                    ax_bars.set_facecolor('#16213e')
                    ax_bars.set_ylabel('|S|', color='#aaa', fontsize=9)
                    ax_bars.tick_params(colors='#888', labelsize=8)
                    ax_bars.set_ylim(0, 3.0)
                    ax_bars.axhline(y=2, color='#ffd93d', linewidth=2, linestyle='--', alpha=0.8)
                    ax_bars.axhline(y=2*np.sqrt(2), color='#ff6b6b', linewidth=1, linestyle=':', alpha=0.5)
                    for spine in ax_bars.spines.values():
                        spine.set_color('#333')
                    
                    x_pos = np.arange(len(bar_labels))
                    w = 0.5
                    colors = ['#4ecdc4' if val <= 2 else '#ff6b6b' for val in bar_vals]
                    ax_bars.bar(x_pos, bar_vals, w, color=colors, alpha=0.8)
                    ax_bars.set_xticks(x_pos)
                    ax_bars.set_xticklabels(bar_labels, fontsize=7, color='#aaa')
                    ax_bars.set_title('|S| Value', color='#aaa', fontsize=9)
                    
                    fig.tight_layout()
                    canvas_fig.draw()
                    
                    fig.savefig(os.path.join(SCREENSHOT_DIR, f"spinor_wave_{method}_{timestamp}.png"),
                               facecolor=fig.get_facecolor(), dpi=150)
                    
                    status_var.set(f"Done! Saved. |S|={bar_vals[0]:.4f}")
                    run_btn.config(state='normal', text="Run Simulation")
                
                root.after(0, update_final)
                
            except Exception as e:
                root.after(0, lambda: status_var.set(f"Error: {e}"))
                root.after(0, lambda: run_btn.config(state='normal', text="Run Simulation"))
        
        threading.Thread(target=worker, daemon=True).start()
    
    btn_frame = tk.Frame(root, bg='#1a1a2e')
    btn_frame.pack(fill=tk.X, padx=10, pady=5)
    
    run_btn = tk.Button(btn_frame, text="Run Simulation",
                       command=on_run, bg='#2d6a4f', fg='white',
                       font=('Consolas', 11, 'bold'), relief='flat', pady=5)
    run_btn.pack(fill=tk.X)
    
    tk.Label(root, textvariable=status_var, bg='#1a1a2e', fg='#888',
            font=('Consolas', 9), anchor='w').pack(fill=tk.X, padx=10, pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()
