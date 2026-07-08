"""
四元数の振動可視化 + ベル実験グラフ
「回転は振動の位相差から生まれる」を確認し、
その構造でベルの不等式を検証する。

操作方法:
  - スライダーで位相差・周波数比・測定角度を調整
  - 左: 各軸の振動（個別に見ると行って戻るだけ）
  - 右: 合成された3D軌跡（回転に見えるか？）
  - グラフ窓: 相関 vs 角度（-cosθ と比較）
  - [Run Bell Test] ボタンで現在パラメータのベル実験を実行
"""

import numpy as np
from vispy import app, scene
from vispy.scene import visuals
import threading
import tkinter as tk
from tkinter import ttk
import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

app.use_app('glfw')

# --- 最新のベル実験結果を保持 ---
last_bell_result = {"angles": None, "correlations": None, "match_rates": None}

# --- スクリーンショット保存先 ---
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# --- UIパラメータ ---
UI_PARAMS = {
    "PHASE_YZ": {"val": 90.0, "min": 0.0, "max": 360.0},      # Y-Z軸間の位相差（度）
    "PHASE_WX": {"val": 90.0, "min": 0.0, "max": 360.0},      # W-X軸間の位相差（度）
    "FREQ_RATIO": {"val": 1.0, "min": 0.5, "max": 3.0},       # 周波数比（2:1カップリング）
    "AMPLITUDE_W": {"val": 0.5, "min": 0.0, "max": 1.0},      # W成分（cos(φ/2)）の振幅
    "AMPLITUDE_X": {"val": 1.0, "min": 0.0, "max": 1.0},      # X軸振動の振幅
    "AMPLITUDE_Y": {"val": 1.0, "min": 0.0, "max": 1.0},      # Y軸振動の振幅
    "AMPLITUDE_Z": {"val": 1.0, "min": 0.0, "max": 1.0},      # Z軸振動の振幅
    "MEASURE_ANGLE": {"val": 0.0, "min": 0.0, "max": 180.0},  # 測定軸の角度（度）
    "TRAIL_LENGTH": {"val": 200, "min": 50, "max": 500},       # 軌跡の長さ
    "SPEED": {"val": 1.0, "min": 0.1, "max": 5.0},            # 時間の進む速さ
}

# --- ベル実験のシミュレーション ---
def run_bell_test(n_pairs=10000):
    """
    現在のUIパラメータで四元数粒子ペアを生成し、
    角度差ごとの相関を計算する。
    【numpy完全ベクトル化版】
    """
    phase_yz = np.radians(UI_PARAMS["PHASE_YZ"]["val"])
    phase_wx = np.radians(UI_PARAMS["PHASE_WX"]["val"])
    freq_ratio = UI_PARAMS["FREQ_RATIO"]["val"]
    amp_w = UI_PARAMS["AMPLITUDE_W"]["val"]
    amp_x = UI_PARAMS["AMPLITUDE_X"]["val"]
    amp_y = UI_PARAMS["AMPLITUDE_Y"]["val"]
    amp_z = UI_PARAMS["AMPLITUDE_Z"]["val"]
    base_freq = 2.0
    
    angles = np.arange(0, 181, 5)  # 0°〜180°、5°刻み
    correlations = []
    match_rates = []
    
    # 全粒子を一括生成
    t0 = np.random.uniform(0, 2 * np.pi / base_freq, n_pairs)
    
    w_vals = amp_w * np.cos(base_freq * freq_ratio * t0 + phase_wx)
    x_vals = amp_x * np.sin(base_freq * t0)
    y_vals = amp_y * np.sin(base_freq * t0 + phase_yz)
    z_vals = amp_z * np.sin(base_freq * freq_ratio * t0)
    
    norm = np.sqrt(w_vals**2 + x_vals**2 + y_vals**2 + z_vals**2)
    mask = norm > 1e-8
    x_n = np.where(mask, x_vals / norm, 0)
    y_n = np.where(mask, y_vals / norm, 0)
    z_n = np.where(mask, z_vals / norm, 0)
    
    for theta_deg in angles:
        theta = np.radians(theta_deg)
        
        det_a = np.array([1.0, 0.0, 0.0])
        det_b = np.array([np.cos(theta), np.sin(theta), 0.0])
        
        # 粒子Aの射影
        proj_a = x_n * det_a[0] + y_n * det_a[1] + z_n * det_a[2]
        # 粒子Bは反転（シングレット状態）
        proj_b = -(x_n * det_b[0] + y_n * det_b[1] + z_n * det_b[2])
        
        result_a = np.where(proj_a >= 0, 1, -1)
        result_b = np.where(proj_b >= 0, 1, -1)
        
        products = result_a * result_b
        correlations.append(np.mean(products))
        match_rates.append(np.mean(products == 1))
    
    return angles, np.array(correlations), np.array(match_rates)

def measure_pair_correlation(angle_a_deg, angle_b_deg, n_pairs=10000):
    """
    検出器Aを角度a°、検出器Bを角度b°に設定し、
    n_pairs個の粒子ペアで相関 E(a,b) を直接測定する。
    角度差の仮定を使わない、正しい測定。
    【numpy完全ベクトル化版】
    """
    phase_yz = np.radians(UI_PARAMS["PHASE_YZ"]["val"])
    phase_wx = np.radians(UI_PARAMS["PHASE_WX"]["val"])
    freq_ratio = UI_PARAMS["FREQ_RATIO"]["val"]
    amp_w = UI_PARAMS["AMPLITUDE_W"]["val"]
    amp_x = UI_PARAMS["AMPLITUDE_X"]["val"]
    amp_y = UI_PARAMS["AMPLITUDE_Y"]["val"]
    amp_z = UI_PARAMS["AMPLITUDE_Z"]["val"]
    base_freq = 2.0
    
    # 検出器の軸ベクトル（絶対角度）
    rad_a = np.radians(angle_a_deg)
    rad_b = np.radians(angle_b_deg)
    det_a = np.array([np.cos(rad_a), np.sin(rad_a), 0.0])
    det_b = np.array([np.cos(rad_b), np.sin(rad_b), 0.0])
    
    # 全粒子を一括生成（ベクトル化）
    t0 = np.random.uniform(0, 2 * np.pi / base_freq, n_pairs)
    
    w_vals = amp_w * np.cos(base_freq * freq_ratio * t0 + phase_wx)
    x_vals = amp_x * np.sin(base_freq * t0)
    y_vals = amp_y * np.sin(base_freq * t0 + phase_yz)
    z_vals = amp_z * np.sin(base_freq * freq_ratio * t0)
    
    # 正規化
    norm = np.sqrt(w_vals**2 + x_vals**2 + y_vals**2 + z_vals**2)
    mask = norm > 1e-8
    x_n = np.where(mask, x_vals / norm, 0)
    y_n = np.where(mask, y_vals / norm, 0)
    z_n = np.where(mask, z_vals / norm, 0)
    
    # 粒子Aの射影（内積を一括計算）
    proj_a = x_n * det_a[0] + y_n * det_a[1] + z_n * det_a[2]
    # 粒子Bは反転（シングレット状態）
    proj_b = -(x_n * det_b[0] + y_n * det_b[1] + z_n * det_b[2])
    
    # 2値化
    result_a = np.where(proj_a >= 0, 1, -1)
    result_b = np.where(proj_b >= 0, 1, -1)
    
    return np.mean(result_a * result_b)


def compute_chsh_proper(n_pairs=10000, progress_callback=None):
    """
    正しいCHSH検証：各(a,b)ペアを独立に測定する。
    角度差の仮定を一切使わない。
    
    S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    局所実在: |S| ≤ 2
    量子力学: |S| = 2√2 ≈ 2.83
    """
    results = {}
    
    def report(msg):
        if progress_callback:
            progress_callback(msg)
    
    # --- 標準設定3つ ---
    configs = [
        ("S1", 0, 90, 45, 135,  "最大QM違反角度"),
        ("S2", 0, 45, 25, 70,   "近似22.5°設定"),
        ("S3", 0, 60, 30, 120,  "対称設定"),
    ]
    
    for label, a, a2, b, b2, desc in configs:
        report(f"Measuring {label}: {desc}...")
        E_ab   = measure_pair_correlation(a, b, n_pairs)
        E_ab2  = measure_pair_correlation(a, b2, n_pairs)
        E_a2b  = measure_pair_correlation(a2, b, n_pairs)
        E_a2b2 = measure_pair_correlation(a2, b2, n_pairs)
        S = E_ab - E_ab2 + E_a2b + E_a2b2
        results[label] = {
            "S": S, "abs": abs(S),
            "angles": f"a={a}° a'={a2}° b={b}° b'={b2}°",
            "E_ab": E_ab, "E_ab2": E_ab2, "E_a2b": E_a2b, "E_a2b2": E_a2b2
        }
    
    # --- 周辺探索: 標準設定1の近傍で最適角度を探す ---
    report("Searching nearby angles for max |S|...")
    S_max = max(results[k]["abs"] for k in ["S1", "S2", "S3"])
    S_max_detail = max([results[k] for k in ["S1", "S2", "S3"]], key=lambda x: x["abs"])
    
    # 標準設定1の周辺 ±20°を5°刻みで探索（効率的）
    for da in range(-20, 25, 10):
        for da2 in range(-20, 25, 10):
            for db in range(-20, 25, 10):
                for db2 in range(-20, 25, 10):
                    a, a2, b, b2 = da, 90+da2, 45+db, 135+db2
                    if a == a2 or b == b2:
                        continue
                    e1 = measure_pair_correlation(a, b, n_pairs=20000)
                    e2 = measure_pair_correlation(a, b2, n_pairs=20000)
                    e3 = measure_pair_correlation(a2, b, n_pairs=20000)
                    e4 = measure_pair_correlation(a2, b2, n_pairs=20000)
                    S_test = abs(e1 - e2 + e3 + e4)
                    if S_test > S_max:
                        S_max = S_test
                        S_max_detail = {
                            "S": e1 - e2 + e3 + e4, "abs": S_test,
                            "angles": f"a={a}° a'={a2}° b={b}° b'={b2}°",
                            "E_ab": e1, "E_ab2": e2, "E_a2b": e3, "E_a2b2": e4
                        }
    
    # 最大値を精密再測定
    if S_max_detail and S_max_detail.get("abs", 0) > max(results["S1"]["abs"], results["S2"]["abs"]):
        report(f"Refining best: {S_max_detail['angles']}...")
        parts = S_max_detail["angles"].replace("°", "").split()
        a_val = int(parts[0].split("=")[1])
        a2_val = int(parts[1].split("=")[1])
        b_val = int(parts[2].split("=")[1])
        b2_val = int(parts[3].split("=")[1])
        
        e1 = measure_pair_correlation(a_val, b_val, n_pairs * 2)
        e2 = measure_pair_correlation(a_val, b2_val, n_pairs * 2)
        e3 = measure_pair_correlation(a2_val, b_val, n_pairs * 2)
        e4 = measure_pair_correlation(a2_val, b2_val, n_pairs * 2)
        S_refined = e1 - e2 + e3 + e4
        S_max_detail = {
            "S": S_refined, "abs": abs(S_refined),
            "angles": f"a={a_val}° a'={a2_val}° b={b_val}° b'={b2_val}°",
            "E_ab": e1, "E_ab2": e2, "E_a2b": e3, "E_a2b2": e4
        }
        S_max = abs(S_refined)
    
    results["S_max"] = S_max_detail
    results["bell_limit"] = 2.0
    results["qm_value"] = 2 * np.sqrt(2)
    
    return results


# --- コントロールパネル + グラフ窓 ---
def start_control_panel():
    def run_ui():
        root = tk.Tk()
        root.title("Quaternion Oscillation Visualizer")
        root.geometry("420x750")
        root.attributes('-topmost', True)

        # --- スライダー群 ---
        for key, data in UI_PARAMS.items():
            frame = tk.Frame(root)
            frame.pack(fill=tk.X, padx=10, pady=3)
            label_var = tk.StringVar(value=f"{key}: {data['val']:.1f}")
            label = tk.Label(frame, textvariable=label_var, width=22, anchor='w', justify='left', font=('Consolas', 9))
            label.pack(side=tk.LEFT)
            def make_cmd(k, l_var):
                def on_change(val):
                    UI_PARAMS[k]["val"] = float(val)
                    l_var.set(f"{k}: {UI_PARAMS[k]['val']:.1f}")
                return on_change
            slider = ttk.Scale(frame, from_=data["min"], to=data["max"],
                             orient=tk.HORIZONTAL, command=make_cmd(key, label_var))
            slider.set(data["val"])
            slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # --- ベル実験グラフ ---
        sep = ttk.Separator(root, orient='horizontal')
        sep.pack(fill=tk.X, padx=10, pady=8)
        
        graph_label = tk.Label(root, text="Bell Test: E(θ) vs -cos(θ)", font=('Consolas', 11, 'bold'))
        graph_label.pack()
        
        # matplotlib figure をtkinterに埋め込む
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2), dpi=90)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.set_xlabel('Angle θ (degrees)', color='#aaa', fontsize=9)
        ax.set_ylabel('Correlation E(θ)', color='#aaa', fontsize=9)
        ax.tick_params(colors='#888', labelsize=8)
        ax.set_xlim(0, 180)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.2, color='#555')
        ax.axhline(y=0, color='#555', linewidth=0.5)  # ゼロ線
        for spine in ax.spines.values():
            spine.set_color('#333')
        
        # QM予測線（-cosθ）: 0°〜180°
        qm_angles = np.linspace(0, 180, 200)
        ax.plot(qm_angles, -np.cos(np.radians(qm_angles)), '--', color='#ff6b6b', 
                linewidth=2, label='-cos(θ) [QM]', alpha=0.8)
        
        # 線形予測（ベルの限界）
        ax.plot(qm_angles, -1 + 2*qm_angles/180, ':', color='#ffd93d',
                linewidth=1.5, label='Linear [Bell limit]', alpha=0.6)
        
        # シミュレーション結果（初期は空）
        sim_line, = ax.plot([], [], 'o-', color='#4ecdc4', linewidth=2, 
                           markersize=3, label='Your model', alpha=0.9)
        
        # CHSH値の表示テキスト
        chsh_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, fontsize=9,
                           verticalalignment='bottom', horizontalalignment='right',
                           fontweight='bold', fontfamily='monospace')
        
        ax.legend(loc='upper left', fontsize=6, facecolor='#1a1a2e', 
                 edgecolor='#333', labelcolor='#ccc')
        fig.tight_layout()
        
        canvas_fig = FigureCanvasTkAgg(fig, master=root)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # --- 実行ボタン ---
        status_var = tk.StringVar(value="Ready")
        
        def on_run_bell():
            run_btn.config(state='disabled', text="⏳ Running...")
            status_var.set("Step 1/2: Measuring correlation curve...")
            root.update()
            
            def bell_worker():
                try:
                    # Step 1: 相関カーブを計算
                    angles, correlations, match_rates = run_bell_test(n_pairs=100000)
                    
                    last_bell_result["angles"] = angles
                    last_bell_result["correlations"] = correlations
                    last_bell_result["match_rates"] = match_rates
                    
                    # UIスレッドでグラフ更新
                    def update_curve():
                        sim_line.set_data(angles, correlations)
                        canvas_fig.draw()
                        status_var.set("Step 2/2: CHSH verification...")
                    root.after(0, update_curve)
                    
                    # Step 2: 正しいCHSH検証
                    def progress_cb(msg):
                        root.after(0, lambda: status_var.set(msg))
                    
                    chsh = compute_chsh_proper(n_pairs=100000, progress_callback=progress_cb)
                    last_bell_result["chsh"] = chsh
                    
                    # UIスレッドで最終更新
                    def update_final():
                        S1_abs = chsh["S1"]["abs"]
                        S_max_abs = chsh["S_max"]["abs"]
                        S_display = max(S1_abs, S_max_abs)
                        
                        if S_display > 2.0:
                            chsh_color = '#ff4444'
                            chsh_label = f'|S|={S_display:.3f} > 2  BELL VIOLATION!'
                        else:
                            chsh_color = '#ffd93d'
                            chsh_label = f'|S|={S_display:.3f} ≤ 2  Within Bell limit'
                        chsh_text.set_text(chsh_label)
                        chsh_text.set_color(chsh_color)
                        
                        s1_info = chsh["S1"]
                        s_max_info = chsh["S_max"]
                        status_text = (f"S1: |S|={s1_info['abs']:.3f}  "
                                      f"Max: |S|={s_max_info['abs']:.3f}")
                        status_var.set(status_text)
                        canvas_fig.draw()
                        
                        # 保存
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        graph_path = os.path.join(SCREENSHOT_DIR, f"bell_graph_{timestamp}.png")
                        fig.savefig(graph_path, facecolor=fig.get_facecolor(), dpi=150)
                        
                        detail_path = os.path.join(SCREENSHOT_DIR, f"bell_chsh_{timestamp}.txt")
                        with open(detail_path, 'w', encoding='utf-8') as f_out:
                            f_out.write(f"=== CHSH Proper Verification ==={os.linesep}")
                            f_out.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{os.linesep}")
                            f_out.write(f"Method: Each (a,b) pair measured independently{os.linesep}")
                            f_out.write(f"{os.linesep}--- Parameters ---{os.linesep}")
                            for key, data in UI_PARAMS.items():
                                f_out.write(f"{key}: {data['val']:.4f}{os.linesep}")
                            f_out.write(f"{os.linesep}--- CHSH Results ---{os.linesep}")
                            f_out.write(f"Bell limit: |S| <= 2.000{os.linesep}")
                            f_out.write(f"QM value:   |S| = {2*np.sqrt(2):.3f}{os.linesep}")
                            for label in ["S1", "S2", "S3"]:
                                info = chsh[label]
                                f_out.write(f"{os.linesep}[{label}] {info['angles']}{os.linesep}")
                                f_out.write(f"  E(a,b)={info['E_ab']:.4f}  E(a,b')={info['E_ab2']:.4f}{os.linesep}")
                                f_out.write(f"  E(a',b)={info['E_a2b']:.4f}  E(a',b')={info['E_a2b2']:.4f}{os.linesep}")
                                f_out.write(f"  S = {info['S']:.4f}  |S| = {info['abs']:.4f}")
                                if info['abs'] > 2.0:
                                    f_out.write(f"  *** VIOLATION ***")
                                f_out.write(os.linesep)
                            info = chsh["S_max"]
                            f_out.write(f"{os.linesep}[S_max] {info['angles']}{os.linesep}")
                            f_out.write(f"  E(a,b)={info['E_ab']:.4f}  E(a,b')={info['E_ab2']:.4f}{os.linesep}")
                            f_out.write(f"  E(a',b)={info['E_a2b']:.4f}  E(a',b')={info['E_a2b2']:.4f}{os.linesep}")
                            f_out.write(f"  S = {info['S']:.4f}  |S| = {info['abs']:.4f}")
                            if info['abs'] > 2.0:
                                f_out.write(f"  *** VIOLATION ***")
                            f_out.write(os.linesep)
                        
                        status_var.set(status_text + "  [Saved]")
                        run_btn.config(state='normal', text="▶ Run Bell Test (10,000 pairs)")
                    
                    root.after(0, update_final)
                    
                except Exception as e:
                    root.after(0, lambda: status_var.set(f"Error: {e}"))
                    root.after(0, lambda: run_btn.config(state='normal', text="▶ Run Bell Test (10,000 pairs)"))
            
            worker = threading.Thread(target=bell_worker, daemon=True)
            worker.start()
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        run_btn = tk.Button(btn_frame, text="▶ Run Bell Test (100,000 pairs)", 
                           command=on_run_bell, bg='#2d6a4f', fg='white',
                           font=('Consolas', 10, 'bold'), relief='flat', pady=4)
        run_btn.pack(fill=tk.X)
        
        status_label = tk.Label(root, textvariable=status_var, fg='#888', 
                               font=('Consolas', 8), anchor='w')
        status_label.pack(fill=tk.X, padx=10)
        
        # 説明
        info = tk.Label(root, text="赤破線: -cos(θ) = 量子力学の予測\n"
                                   "黄点線: 直線 = 局所隠れ変数の限界\n"
                                   "青緑丸: あなたのモデルの結果",
                       justify='left', fg='#666', font=('Consolas', 8))
        info.pack(pady=5)
        
        root.mainloop()
    
    ui_thread = threading.Thread(target=run_ui, daemon=True)
    ui_thread.start()

# --- メイン ---
if __name__ == "__main__":
    start_control_panel()
    
    # --- VisPy セットアップ ---
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='#111111',
                               size=(1200, 700), title='四元数の振動 → 回転の可視化')
    
    # 左右に2つのビューを作成
    grid = canvas.central_widget.add_grid()
    
    # 左ビュー: 各軸の振動（個別）
    view_left = grid.add_view(row=0, col=0, camera='turntable')
    view_left.camera.fov = 45
    view_left.camera.distance = 8
    view_left.camera.center = (0, 0, 0)
    
    # 右ビュー: 合成3D軌跡
    view_right = grid.add_view(row=0, col=1, camera='turntable')
    view_right.camera.fov = 45
    view_right.camera.distance = 5
    view_right.camera.center = (0, 0, 0)
    
    # 描画オブジェクト
    # 左: 各軸の振動を表す線
    axis_x_line = scene.visuals.Line(parent=view_left.scene, color='#FF6666', width=2)
    axis_y_line = scene.visuals.Line(parent=view_left.scene, color='#66FF66', width=2)
    axis_z_line = scene.visuals.Line(parent=view_left.scene, color='#6666FF', width=2)
    axis_w_line = scene.visuals.Line(parent=view_left.scene, color='#FFFF66', width=2)
    
    # 各軸のラベル（現在の振動位置を示すマーカー）
    marker_x = scene.visuals.Markers(parent=view_left.scene)
    marker_y = scene.visuals.Markers(parent=view_left.scene)
    marker_z = scene.visuals.Markers(parent=view_left.scene)
    marker_w = scene.visuals.Markers(parent=view_left.scene)
    
    # 左: 軸の方向を示す補助線
    for axis_data, color in [
        (np.array([[0,0,0],[3,0,0]]), '#FF666644'),   # X軸
        (np.array([[0,0,0],[0,3,0]]), '#66FF6644'),   # Y軸  
        (np.array([[0,0,0],[0,0,3]]), '#6666FF44'),   # Z軸
    ]:
        scene.visuals.Line(parent=view_left.scene, pos=axis_data.astype(np.float32),
                          color=color, width=1)
    
    # 右: 合成軌跡
    trail_line = scene.visuals.Line(parent=view_right.scene, width=2, method='gl')
    trail_marker = scene.visuals.Markers(parent=view_right.scene)
    
    # 右: 測定軸を示す線
    measure_axis_line = scene.visuals.Line(parent=view_right.scene, color='#FF00FF88', width=3)
    
    # 右: 座標軸の補助線
    for axis_data, color in [
        (np.array([[-2,0,0],[2,0,0]]), '#FF666622'),
        (np.array([[0,-2,0],[0,2,0]]), '#66FF6622'),
        (np.array([[0,0,-2],[0,0,2]]), '#6666FF22'),
    ]:
        scene.visuals.Line(parent=view_right.scene, pos=axis_data.astype(np.float32),
                          color=color, width=1)
    
    # 射影結果の表示
    projection_marker = scene.visuals.Markers(parent=view_right.scene)
    
    # 時間
    t_global = [0.0]
    
    def on_timer(event):
        speed = UI_PARAMS["SPEED"]["val"]
        t_global[0] += 0.033 * speed
        t = t_global[0]
        
        # パラメータ取得
        phase_yz = np.radians(UI_PARAMS["PHASE_YZ"]["val"])
        phase_wx = np.radians(UI_PARAMS["PHASE_WX"]["val"])
        freq_ratio = UI_PARAMS["FREQ_RATIO"]["val"]
        amp_w = UI_PARAMS["AMPLITUDE_W"]["val"]
        amp_x = UI_PARAMS["AMPLITUDE_X"]["val"]
        amp_y = UI_PARAMS["AMPLITUDE_Y"]["val"]
        amp_z = UI_PARAMS["AMPLITUDE_Z"]["val"]
        trail_len = int(UI_PARAMS["TRAIL_LENGTH"]["val"])
        measure_angle = np.radians(UI_PARAMS["MEASURE_ANGLE"]["val"])
        
        # --- 各軸の振動を計算 ---
        base_freq = 2.0
        
        # 軌跡のための時間配列
        t_trail = np.linspace(t - trail_len * 0.033, t, trail_len)
        
        # 4つの振動成分（各軸は独立に振動しているだけ）
        w_osc = amp_w * np.cos(base_freq * freq_ratio * t_trail + phase_wx)
        x_osc = amp_x * np.sin(base_freq * t_trail)
        y_osc = amp_y * np.sin(base_freq * t_trail + phase_yz)
        z_osc = amp_z * np.sin(base_freq * freq_ratio * t_trail)
        
        # --- 四元数として正規化（単位球面上に射影）---
        norm = np.sqrt(w_osc**2 + x_osc**2 + y_osc**2 + z_osc**2) + 1e-8
        w_n = w_osc / norm
        x_n = x_osc / norm
        y_n = y_osc / norm
        z_n = z_osc / norm
        
        # --- 左ビュー: 各軸の振動を個別表示 ---
        offset = 2.0
        
        current_x = x_n[-1]
        current_y = y_n[-1]
        current_z = z_n[-1]
        current_w = w_n[-1]
        
        axis_x_line.set_data(pos=np.array([[0, offset, 0], [current_x * 2, offset, 0]], dtype=np.float32))
        axis_y_line.set_data(pos=np.array([[0, 0, 0], [0, current_y * 2, 0]], dtype=np.float32))
        axis_z_line.set_data(pos=np.array([[0, -offset, 0], [0, -offset, current_z * 2]], dtype=np.float32))
        axis_w_line.set_data(pos=np.array([[-offset, -offset, 0], [-offset + current_w * 2, -offset, 0]], dtype=np.float32))
        
        marker_x.set_data(pos=np.array([[current_x * 2, offset, 0]], dtype=np.float32),
                         face_color='#FF6666', size=12, edge_width=0)
        marker_y.set_data(pos=np.array([[0, current_y * 2, 0]], dtype=np.float32),
                         face_color='#66FF66', size=12, edge_width=0)
        marker_z.set_data(pos=np.array([[0, -offset, current_z * 2]], dtype=np.float32),
                         face_color='#6666FF', size=12, edge_width=0)
        marker_w.set_data(pos=np.array([[-offset + current_w * 2, -offset, 0]], dtype=np.float32),
                         face_color='#FFFF66', size=12, edge_width=0)
        
        # --- 右ビュー: 3D合成軌跡 ---
        trail_pos = np.column_stack((x_n, y_n, z_n)).astype(np.float32)
        
        trail_colors = np.zeros((trail_len, 4), dtype=np.float32)
        trail_colors[:, 0] = 0.3 + 0.7 * np.abs(w_n)
        trail_colors[:, 1] = 0.8
        trail_colors[:, 2] = 1.0 - 0.5 * np.abs(w_n)
        trail_colors[:, 3] = np.linspace(0.05, 1.0, trail_len)
        
        trail_line.set_data(pos=trail_pos, color=trail_colors)
        
        trail_marker.set_data(
            pos=np.array([[x_n[-1], y_n[-1], z_n[-1]]], dtype=np.float32),
            face_color='white', size=15, edge_width=0
        )
        
        # --- 測定軸の表示 ---
        mx = np.cos(measure_angle)
        my = np.sin(measure_angle)
        measure_axis_line.set_data(
            pos=np.array([[-mx*2, -my*2, 0], [mx*2, my*2, 0]], dtype=np.float32)
        )
        
        # --- 射影結果（測定軸への投影）---
        detector = np.array([mx, my, 0])
        particle_vec = np.array([x_n[-1], y_n[-1], z_n[-1]])
        proj = np.dot(particle_vec, detector)
        
        proj_pos = proj * detector
        proj_color = '#FF00FF' if proj >= 0 else '#00FFFF'
        projection_marker.set_data(
            pos=np.array([[proj_pos[0], proj_pos[1], proj_pos[2]]], dtype=np.float32),
            face_color=proj_color, size=20, edge_width=0
        )
    
    # --- Sキーでスクリーンショット + パラメータ保存 ---
    def on_key_press(event):
        if event.key == 'S' or event.key == 's':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # パラメータからファイル名を生成
            ph_yz = UI_PARAMS["PHASE_YZ"]["val"]
            ph_wx = UI_PARAMS["PHASE_WX"]["val"]
            fr = UI_PARAMS["FREQ_RATIO"]["val"]
            name_tag = f"PYZ{ph_yz:.0f}_PWX{ph_wx:.0f}_FR{fr:.1f}"
            
            base_name = f"quat_{name_tag}_{timestamp}"
            
            # VisPy画面をスクリーンショット保存
            img_path = os.path.join(SCREENSHOT_DIR, f"{base_name}.png")
            img = canvas.render()
            from vispy.io import write_png
            write_png(img_path, img)
            
            # パラメータをテキストファイルに保存
            txt_path = os.path.join(SCREENSHOT_DIR, f"{base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Quaternion Visualizer Screenshot ==={os.linesep}")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{os.linesep}")
                f.write(f"{os.linesep}--- Parameters ---{os.linesep}")
                for key, data in UI_PARAMS.items():
                    f.write(f"{key}: {data['val']:.4f}{os.linesep}")
                
                # ベル実験結果があれば追記
                if last_bell_result["correlations"] is not None:
                    f.write(f"{os.linesep}--- Bell Test Results ---{os.linesep}")
                    f.write(f"{'Angle':>6s}  {'E(θ)':>8s}  {'Match%':>8s}  {'-cos(θ)':>8s}{os.linesep}")
                    for i, ang in enumerate(last_bell_result["angles"]):
                        e_val = last_bell_result["correlations"][i]
                        m_val = last_bell_result["match_rates"][i]
                        qm_val = -np.cos(np.radians(ang))
                        f.write(f"{ang:6.1f}° {e_val:8.4f}  {m_val:8.4f}  {qm_val:8.4f}{os.linesep}")
            
            print(f"[SAVED] {img_path}")
            print(f"[SAVED] {txt_path}")
    
    canvas.events.key_press.connect(on_key_press)
    
    timer = app.Timer(interval=1.0/30.0, connect=on_timer, start=True)
    app.run()
