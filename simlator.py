import os

import numpy as np
import cupy as cp
import cupyx
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates
from vispy import app, scene

import threading
import tkinter as tk
from tkinter import ttk

# Qt6のDPIバグを完全に回避するため、描画バックエンドをゲーム用の'glfw'に変更する
app.use_app('glfw')

# --- 流体宇宙の哲学とタイムスケール ---
# 私たちが夜空に見る「安定した銀河の形」は、人間の短い寿命が切り取った一瞬のスナップショットに過ぎない。
# 実際には、宇宙は水や空気と同じ「流体」として、凄まじいダイナミズムで常にその形を激しく変え続けていると考えます。
# 人為的な辻褄合わせ（質量引力や暗黒物質）を完全に排し、
# 純粋な「流体の圧力勾配と渦の曲率」のみを第一原理として、その真理をデジタル空間に描き出す。
# 【宇宙の膨張と赤方偏移について】
# 宇宙は海であり、どの方向を見ても同じである。全体が風船のように膨張している(ビッグバン)という概念は不要。
# 空間の伸縮(流体の張力と速度勾配)が存在している以上、引き伸ばされた空間を伝わる光はそれに従って波長が伸びる。

# --- 1. 空間と環境変数の初期化 ---
GRID_SIZE = 60
NUM_PARTICLES = 0
MAX_PARTICLES = 90000  # 星の数の上限（これ以上増えると古い・小さいものから消えるか、生成を止める）

# --- バリオン音響振動（流体の初期疎密波）の再現 ---
# 単一のノイズではなく、異なる波長の音波（圧力波）が重なり合ったフラクタルな揺らぎを作る
# 箱庭が狭くても「広大なボイド（虚無）」ができるように、波紋のスケール（sigma）を大きくする
noise_fine = gaussian_filter(cp.random.rand(GRID_SIZE, GRID_SIZE, GRID_SIZE), sigma=5.0, mode='wrap')
noise_mid = gaussian_filter(cp.random.rand(GRID_SIZE, GRID_SIZE, GRID_SIZE), sigma=10.0, mode='wrap')
noise_large = gaussian_filter(cp.random.rand(GRID_SIZE, GRID_SIZE, GRID_SIZE), sigma=20.0, mode='wrap')

# 大きなうねりをベースに、中・小の波紋を足し合わせる（宇宙の初期ゆらぎのパワースペクトルに似せる）
density = noise_large * 0.5 + noise_mid * 0.3 + noise_fine * 0.2

# 0.0〜1.0に正規化（ここで浮動小数点の誤差がない「滑らかな地形」を確定させる）
density = (density - cp.min(density)) / (cp.max(density) - cp.min(density))

# うねり（濃淡の差）を小さく穏やかにするための調整
# 以前は 0.8倍+0.5 だったため、初期から密度のムラが激しすぎる（0.5〜1.3）不自然な状態でした。
# コメントの意図通り、振幅を0.1倍に抑え、0.45〜0.55のほぼ均一（一定に近い）で平坦な空間にします。
density = density * 0.1 + 0.45

# 総質量を記録（この質量を永遠に保つ）
INITIAL_TOTAL_MASS = float(cp.sum(density))

SPEED_LIMIT = 300.0    # これを超える速度になると空間に還元（消滅）
SIZE_LIMIT = 300.0     # これを超えるサイズに育つと超新星爆発（空間に還元）
DIFFUSION_RATE = 0.3# ラプラシアン拡散の強さ（小さいほど構造が保持される）
SHEAR_FACTOR = 0.5     # せん断によるフィラメント形成の強さ

# --- UIからリアルタイム調整可能なパラメータ群 ---
# スライダーで動かした数値が即座に物理法則に反映されます
UI_PARAMS = {
    "PRESSURE_FACTOR": {"val": 0.0, "min": 0.0, "max": 10.0},     # 圧力の強さ（風の強さ）
    "DAMPING": {"val": 0.9, "min": 0.9, "max": 1.0},              # 空間の摩擦（1.0で摩擦ゼロ）
    "CURVATURE_FACTOR": {"val": 0.0, "min": 0.0, "max": 5.0},     # 渦による空間の歪み（見かけの引力）
    "TENSION_FACTOR": {"val": 0.0, "min": 0.0, "max": 5.0},       # 空間の張力（まとまる力）
    "THERMAL_FACTOR": {"val": 0.0, "min": 0.0, "max": 5.0},       # 熱膨張（広がる力）
    "FLATTENING_RATE": {"val": 0.0, "min": 0.0, "max": 1.0},      # 円盤化（平坦化）の強さ
    "BULGE_DISP": {"val": 0.0, "min": 0.0, "max": 2.0},           # バルジの熱運動（ジェットの強さ）
    "ABSORPTION": {"val": 0.0, "min": 0.0, "max": 0.5},           # 星がガスを吸い込む速度
    "SPIRAL_FACTOR": {"val": 0.0, "min": 0.0, "max": 2.0},        # 渦巻きを作る横向きの力
    "TURBULENCE": {"val": 0.0, "min": 0.0, "max": 2.0},           # 空間の基本揺らぎ（サブグリッド乱流）
    "SUPERNOVA_POWER": {"val": 0.0, "min": 0.0, "max": 20.0},     # 超新星爆発時の暴風の強さ
    "RECON_POWER": {"val": 0.0, "min": 0.0, "max": 2.0},          # 磁気リコネクションの爆発力
    "TIME_SCALE": {"val": 0.5, "min": 0.01, "max": 2.0},          # 時間の進み方（dt）
}

# 粒子データ（CuPyの2次元配列 N x 7 で一括管理する。x,y,z, mass, vx,vy,vz）
particles = cp.empty((0, 7))

# 空間のグリッド座標をあらかじめ生成（毎フレーム作ると重いため）
GRID_Z, GRID_Y, GRID_X = cp.meshgrid(cp.arange(GRID_SIZE), cp.arange(GRID_SIZE), cp.arange(GRID_SIZE), indexing='ij')

# シミュレーションのステップ数をカウント
sim_step = 0

# 重い計算をキャッシュするためのグローバル変数
cached_local_mass_drop = None
cached_grad_m_z = None
cached_grad_m_y = None
cached_grad_m_x = None
cached_pressure_noise = None

# --------------------------------------------------
# 超高速なぼかし関数（重いgaussian_filterの代わり）
def fast_blur(f, iterations=1):
    res = f
    for _ in range(iterations):
        # X, Y, Z軸の順番に1次元ずつぼかす（分離可能フィルタ）ことで、
        # 処理の軽さを保ったまま、斜め方向にも均等に丸く滲むようにする
        res = 0.25 * cp.roll(res, 1, axis=0) + 0.5 * res + 0.25 * cp.roll(res, -1, axis=0)
        res = 0.25 * cp.roll(res, 1, axis=1) + 0.5 * res + 0.25 * cp.roll(res, -1, axis=1)
        res = 0.25 * cp.roll(res, 1, axis=2) + 0.5 * res + 0.25 * cp.roll(res, -1, axis=2)
    return res

# 空間がループしている（周期境界）前提での勾配計算関数
def periodic_gradient(f):
    grad_z = 0.5 * (cp.roll(f, -1, axis=0) - cp.roll(f, 1, axis=0))
    grad_y = 0.5 * (cp.roll(f, -1, axis=1) - cp.roll(f, 1, axis=1))
    grad_x = 0.5 * (cp.roll(f, -1, axis=2) - cp.roll(f, 1, axis=2))
    return grad_z, grad_y, grad_x

# CuPyのmap_coordinatesのwrapバグ（境界での補間失敗）を回避するための完全な周期補間関数
def periodic_advect(f, bz, by, bx, grid_size):
    z0 = cp.floor(bz).astype(cp.int32) % grid_size
    y0 = cp.floor(by).astype(cp.int32) % grid_size
    x0 = cp.floor(bx).astype(cp.int32) % grid_size
    z1 = (z0 + 1) % grid_size
    y1 = (y0 + 1) % grid_size
    x1 = (x0 + 1) % grid_size
    dz = bz - cp.floor(bz)
    dy = by - cp.floor(by)
    dx = bx - cp.floor(bx)
    
    c00 = f[z0, y0, x0] * (1 - dx) + f[z0, y0, x1] * dx
    c01 = f[z0, y1, x0] * (1 - dx) + f[z0, y1, x1] * dx
    c10 = f[z1, y0, x0] * (1 - dx) + f[z1, y0, x1] * dx
    c11 = f[z1, y1, x0] * (1 - dx) + f[z1, y1, x1] * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    return c0 * (1 - dz) + c1 * dz

# --- コントロールパネルの起動関数 ---
def start_control_panel():
    def run_ui():
        root = tk.Tk()
        root.title("Space Engine Control Panel")
        root.geometry("400x600")
        root.attributes('-topmost', True) # メインの宇宙ウィンドウの後ろに隠れないようにする

        for key, data in UI_PARAMS.items():
            frame = tk.Frame(root)
            frame.pack(fill=tk.X, padx=10, pady=10)
            label_var = tk.StringVar(value=f"{key}:\n{data['val']:.3f}")
            label = tk.Label(frame, textvariable=label_var, width=18, anchor='w', justify='left')
            label.pack(side=tk.LEFT)
            def make_cmd(k, l_var):
                def on_change(val):
                    UI_PARAMS[k]["val"] = float(val)
                    l_var.set(f"{k}:\n{UI_PARAMS[k]['val']:.3f}")
                return on_change
            slider = ttk.Scale(frame, from_=data["min"], to=data["max"], orient=tk.HORIZONTAL, command=make_cmd(key, label_var))
            slider.set(data["val"])
            slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        root.mainloop()
    ui_thread = threading.Thread(target=run_ui, daemon=True)
    ui_thread.start()

def update_simulation_step(density, vx, vy, vz, dt=0.5):
    global particles, sim_step
    global cached_local_mass_drop, cached_grad_m_z, cached_grad_m_y, cached_grad_m_x
    global cached_pressure_noise
    
    # UIで設定された最新のパラメータを取得
    P_PRESSURE = UI_PARAMS["PRESSURE_FACTOR"]["val"]
    P_DAMPING = UI_PARAMS["DAMPING"]["val"]
    P_CURVATURE = UI_PARAMS["CURVATURE_FACTOR"]["val"]
    P_TENSION = UI_PARAMS["TENSION_FACTOR"]["val"]
    P_THERMAL = UI_PARAMS["THERMAL_FACTOR"]["val"]
    P_FLATTEN = UI_PARAMS["FLATTENING_RATE"]["val"]
    P_BULGE = UI_PARAMS["BULGE_DISP"]["val"]
    P_ABSORB = UI_PARAMS["ABSORPTION"]["val"]
    P_SPIRAL = UI_PARAMS["SPIRAL_FACTOR"]["val"]
    P_TURBULENCE = UI_PARAMS["TURBULENCE"]["val"]
    P_SUPERNOVA = UI_PARAMS["SUPERNOVA_POWER"]["val"]
    P_RECON = UI_PARAMS["RECON_POWER"]["val"]

    grid_size = density.shape[0]
    sim_step += 1
    
    # --------------------------------------------------
    # 仕様⓪：密度の勾配から圧力（速度場）を生み出し更新する
    # --------------------------------------------------
    # 1. 浮力：無効化（0にする）
    
    # 0. 密度の移流（流速に合わせて密度自身を移動させる）
    # 現在の座標から、風（速度）の分だけ少し戻った場所の密度をサンプリングする
    back_z = (GRID_Z - vz * dt) % grid_size
    back_y = (GRID_Y - vy * dt) % grid_size
    back_x = (GRID_X - vx * dt) % grid_size
    
    # カスタム関数で完全にシームレスな移流を行う（map_coordinatesの壁バグを回避）
    density[:] = periodic_advect(density, back_z, back_y, back_x, grid_size)

    # --- 流速（風）の移流を追加 ---
    vx[:] = periodic_advect(vx, back_z, back_y, back_x, grid_size)
    vy[:] = periodic_advect(vy, back_z, back_y, back_x, grid_size)
    vz[:] = periodic_advect(vz, back_z, back_y, back_x, grid_size)

    # --- 空間全体の速度勾配と渦度（曲率）の計算 ---
    # 「空間の歪みは渦の曲率に比例する」という哲学に基づき、ここで渦度を計算する
    dvx_dz, dvx_dy, dvx_dx = periodic_gradient(vx)
    dvy_dz, dvy_dy, dvy_dx = periodic_gradient(vy)
    dvz_dz, dvz_dy, dvz_dx = periodic_gradient(vz)
    
    # --- ガスの圧縮と膨張（圧縮性流体の再現） ---
    # 風が吹き込む場所（収束）ではガスが圧縮されて密度が濃くなり、吹き出す場所（発散）では真空になる
    # どんなに強い暴風でも、1フレームで無限に圧縮・膨張することはないため、圧縮限界を設ける
    divergence = cp.clip(dvx_dx + dvy_dy + dvz_dz, -2.0, 2.0)
    density -= density * divergence * dt

    # ガスの自然拡散（等方的な拡散）
    # 毎フレーム完全にぼかすと「ぽわぽわと膨らみ続ける星雲」になってしまうため、
    # 元の密度を強く保持(0.9)しつつ、微小な拡散(0.1)に留めることで構造を維持します。
    density[:] = density * 0.9 + fast_blur(density, iterations=1) * 0.1

    # 2. 圧力：ベルヌーイの定理（動圧）
    # 【不自然な加速の原因①：動圧の自己加速ループを撤廃】
    # 「風が吹く → 動圧で負圧になる → さらに風を吸い込む → 無限に風が加速する」という暴走ループが起きていました。
    # ベルヌーイの吸い込み効果は「空間の歪み（渦の曲率）」ですでに十分表現できているため、
    # 直線的な速度による気圧低下（単なる風の暴走）は無効化し、純粋な渦の力に引き込みを委ねます。
    dynamic_pressure = 0.0

    # 3次元の渦度ベクトル（∇×v）の各成分を計算（特定の軸に依存しない自然な空間のねじれ）
    omega_x = dvz_dy - dvy_dz
    omega_y = dvx_dz - dvz_dx
    omega_z = dvy_dx - dvx_dy
    
    # 局所的な微小な渦度（水面のさざ波）
    local_vorticity = cp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    # 大海の揺らぎ（マクロなうねり）：周囲の空間も連動して渦巻いているかを評価する
    raw_vorticity = fast_blur(local_vorticity, iterations=1) # ぼかしを弱めて、局所的な渦の引力ピークを残す

    # 【物理法則の追加】プラズマの剪断限界（渦度の暴走防止）
    # 空間のねじれ（速度差）が極端になった場所で渦度が無限大に向かい、
    # 磁気圧や空間の歪みが2乗で暴走して宇宙を吹き飛ばす（構造ができる前の爆発）のを完全に防ぎます。
    # プラズマはある程度以上ねじれると「すべり（破断）」が生じるため、渦度には物理的な上限が存在します。
    vorticity = cp.clip(raw_vorticity, 0.0, 40.0)

    # --- 圧力勾配による宇宙の大規模構造と重力の再現 ---
    # 質量が存在するだけで崩壊が起こる質量＝引力（プラスの力で引っ張る）という概念は使いません。
    # さらに極め、「エネルギー(物質)が存在するだけで空間がへこむ」というアインシュタイン的な概念すらも排し、
    # 純粋な「流速と渦」が生み出す圧力勾配のみで宇宙の構造を形作ります。
    
    # 有効圧力 = ガスの熱圧力(高密度ほど高気圧) - 流速による動圧 - 空間の歪み(渦の曲率)
    # 渦巻いている（空間が歪んでいる）時は負圧が勝って引き込むが、
    # 爆発によって超高密度に還元されたガスが自重に潰されず、本来の低いところへ吹き出せるように、
    # 密度が高いほど急激に反発力が強くなる非線形な熱圧力（高温ガスの放射圧）を計算する。
    # 【爆発防止の物理法則】エディントン限界：
    # 超新星爆発などで密度が極端に跳ね上がった際、熱圧力が2乗で暴走して銀河中心を吹き飛ばすの防ぎます。
    # 密度が高すぎる場所では光（放射圧）が外に逃げられず、圧力が頭打ちになる現象を再現します。
    limited_density = cp.clip(density, 0.0, 30.0)
    # 広げる力（熱膨張）が少し強すぎたため、係数を下げてマイルドにします。
    raw_thermal_pressure = limited_density * P_THERMAL + (limited_density ** 2) * 0.01
    
    # --- 放射冷却（プラズマの冷却による星形成のトリガー） ---
    # 宇宙空間は真空であるため対流や伝導で熱は逃げないが、高温高密度のプラズマは「光（電磁波）」を放って自らエネルギーを失う。
    # エネルギー（熱）を失うと反発する圧力（高気圧）がなくなり、空間の張力や周囲の圧力に押し潰されて急激に凝縮する。
    # これが、輝く星雲が自ら冷えて暗黒星雲となり、重力崩壊を起こして新しい星を生むメカニズムである。
    # 超高密度（星の残骸）ではほぼ完全に冷え切る（暗黒星雲化する）ように最大効率を引き上げます。
    cooling_efficiency = cp.clip(density * 0.02, 0.0, 0.98)
    thermal_pressure = raw_thermal_pressure * (1.0 - cooling_efficiency)
    
    # 磁気圧（Magnetic Pressure）の導入：バルジの形成
    # 渦が集まって極めて強くなる中心部では、磁力線同士の反発（磁気圧）が生まれる。
    # この反発力が上下からの押し潰しに抵抗するため、中心部はペチャンコにならず「バルジ（球状の膨らみ）」ができる。
    magnetic_pressure = (vorticity ** 2) * 0.05
    
    # --- 空間の歪み（曲率）とニュートンの「見かけの引力」の再現 ---
    # ニュートン自身は「質量が引力を生む」とは言わず、現象を数式化しただけでした。
    # 流体力学において、渦の中心に向かって低下する圧力の勾配は、遠くの物質を中心へ引き寄せるポテンシャル場として働きます。
    # 質量＝引力とした場合未観測の様々な物の配置が避けられない為これを「万有引力」にします。
    # 渦度の2乗による深い歪みを、ポテンシャル場のように空間へなだらかに伝播させることで、遠距離まで届く引力を再現します。
    # 渦度自体に剪断限界（上限40.0）を設けたため、ここでは安全に2乗の計算を行えます。
    # 宇宙全体をダイナミックに巻き込む大きなポテンシャル（引力）として機能させるため、係数を少し上げます。
    # 【自然な負圧の再現（ポテンシャルの裾野）】
    # 単純なぼかし(Gaussian)では遠距離で急減衰してしまい、重力のような長距離引力(1/r)になりません。
    # 異なるスケールのぼかしを合成（マルチスケール化）することで、近くでは深く、遠くまでなだらかに届く自然な負圧場を作ります。
    raw_curvature = (vorticity ** 2) * P_CURVATURE
    blur1 = fast_blur(raw_curvature, iterations=1)
    blur2 = fast_blur(blur1, iterations=2)
    blur4 = fast_blur(blur2, iterations=4)
    space_curvature = blur1 * 0.6 + blur2 * 0.3 + blur4 * 0.1
    
    # --- 空間の張力の再現 ---
    # 純粋な流体力学だけでは、渦は分裂を繰り返してカオス（乱流）へと崩壊してしまいます。
    # 銀河が形を保つためには、空間そのものが持つ「千切れにくさ（張力）」が必要です。
    # 密度のラプラシアン（周囲との差）を計算することで、水滴の表面張力のように構造を内側にまとめる力を生み出します。
    laplacian_density = fast_blur(density, iterations=1) - density
    spatial_tension = laplacian_density * P_TENSION
    
    # --- 空間の微小な圧力揺らぎ（サブグリッド乱流の源） ---
    # 速度ベクトルに直接揺らぎを足すと、同じ方向に力が加わり続けて無限加速（暴走）を引き起こしてしまいます。
    # 物理的に正しく「気圧の揺らぎ（スカラー）」として空間に足すことで、自然な圧力勾配（全方位への湧き出しと吸い込み）の風を生ませます。
    if cached_pressure_noise is None:
        cached_pressure_noise = cp.zeros((grid_size, grid_size, grid_size), dtype=cp.float32)
        
    new_noise = fast_blur(cp.random.rand(grid_size, grid_size, grid_size) - 0.5, iterations=1)
    
    # 毎フレーム完全ランダムだと「ホワイトノイズの爆音」になり、風が育つ前に破壊されて（弾かれて）しまいます。
    # 現在の地形に10%ずつ新しいノイズを混ぜることで、数秒かけてゆっくりウネウネと変わる自然な気圧のうねりにします。
    cached_pressure_noise = cached_pressure_noise * 0.9 + new_noise * 0.1
    cached_pressure_noise -= cp.mean(cached_pressure_noise) # 全体の気圧変動の平均をゼロにする

    turbulence_pressure = cached_pressure_noise * P_TURBULENCE * 2.0
    
    effective_pressure = thermal_pressure + magnetic_pressure - dynamic_pressure - space_curvature + spatial_tension + turbulence_pressure
    
    # --- 数値振動（チェッカーボード不安定性）の防止 ---
    # 「タービュランスを0にしても揺れが止まらない」原因は、デジタル空間特有のバグ（高周波ノイズの自己増幅）です。
    # 圧力が隣り合うマスで「高・低・高・低」と市松模様に振動し始めると、永久に止まらなくなってしまいます。
    # 風（勾配）を計算する直前に、圧力の微細なトゲ（ノイズ）だけを平滑化して連鎖を断ち切ります。
    effective_pressure = fast_blur(effective_pressure, iterations=1)
    
    # --- 磁気リコネクション（プラズマ流体のショートとエネルギー解放） ---
    # 異なる向きの渦（磁場）が衝突した時、プラズマ宇宙論では「磁力線の繋ぎ変え（リコネクション）」が起き、
    # 溜まっていたストレスが莫大な熱と運動エネルギー（太陽フレアや宇宙ジェットの源）として解放されます。
    lap_omega_x = fast_blur(omega_x, iterations=1) - omega_x
    lap_omega_y = fast_blur(omega_y, iterations=1) - omega_y
    lap_omega_z = fast_blur(omega_z, iterations=1) - omega_z
    
    reconnection_energy = cp.sqrt(lap_omega_x**2 + lap_omega_y**2 + lap_omega_z**2)
    
    # 宇宙全体の中で、激突エネルギーが特に大きい上位の場所だけでバーストを起こす（少し条件を厳しくする）
    RECON_THRESHOLD = cp.mean(reconnection_energy) + cp.std(reconnection_energy) * 4.0
    recon_mask = reconnection_energy > RECON_THRESHOLD
    if cp.any(recon_mask):
        # ランダムな速度（花火）を足すのではなく、局所的な「圧力の急増」として処理する。
        # 以前の * 20.0 は威力が強すぎて星が吹き飛び消滅してしまったため、スケールを大幅に下げて上限（リミッター）を設けます。
        # 【復活】宇宙を壊さない「微小な風のタネ」としてマイルドに復活させます。
        burst_pressure = cp.clip(reconnection_energy[recon_mask] * P_RECON, 0.0, 5.0)
        effective_pressure[recon_mask] += burst_pressure

    
    grad_z, grad_y, grad_x = periodic_gradient(effective_pressure)
    
    # --- 速度の壁（圧力波の伝播限界による安全装置） ---
    # 宇宙空間（流体）において、どんなに極端な特異点が生まれても、衝撃波の加速度には媒質による限界があります。
    # キャビテーションなどで無限大に近い勾配が生じた際、一瞬で光速を超えて星が消滅するのを完全に防ぎます。
    grad_x = cp.clip(grad_x, -50.0, 50.0)
    grad_y = cp.clip(grad_y, -50.0, 50.0)
    grad_z = cp.clip(grad_z, -50.0, 50.0)
    
    # 1. 圧力勾配による直線的な風（引力）
    vx -= grad_x * P_PRESSURE * dt
    vy -= grad_y * P_PRESSURE * dt
    vz -= grad_z * P_PRESSURE * dt
    
    # 2. スパイラル力（地衡風と渦の引き伸ばし）
    # 流体力学（地衡風）の法則に従い、気圧勾配（-grad）と局所的な渦の回転軸（omega）の外積を取ることで、
    # 中心へ向かう風を強制的に横へ曲げ、美しい台風や銀河の腕（スパイラルアーム）のような渦巻きの気流を生み出します。
    spiral_x = grad_y * omega_z - grad_z * omega_y
    spiral_y = grad_z * omega_x - grad_x * omega_z
    spiral_z = grad_x * omega_y - grad_y * omega_x
    
    vx -= spiral_x * P_PRESSURE * P_SPIRAL * dt
    vy -= spiral_y * P_PRESSURE * P_SPIRAL * dt
    vz -= spiral_z * P_PRESSURE * P_SPIRAL * dt
    
    # --- 渦による磁場の発生と両極ジェット（電磁流体力学の再現） ---
    # 渦が巻くことでダイナモ効果が働き、回転軸（右ねじの方向）に沿って磁気軸が生まれる（観測事実）。
    # この磁気軸は星を強制的に止めるのではなく、極方向への「磁気ジェット（プラズマの噴出）」を引き起こす。
    epsilon = 1e-8
    nx = omega_x / (local_vorticity + epsilon)
    ny = omega_y / (local_vorticity + epsilon)
    nz = omega_z / (local_vorticity + epsilon)
    
      # 各座標の速度ベクトルが、磁場方向（北・南）にどれくらい向かっているかを計算
    v_dot_n = vx * nx + vy * ny + vz * nz
    v_para_x = v_dot_n * nx
    v_para_y = v_dot_n * ny
    v_para_z = v_dot_n * nz
    
    # --- 渦による平坦化（円盤の形成） ---
    grad_para = grad_x * nx + grad_y * ny + grad_z * nz
    vx -= nx * grad_para * P_PRESSURE * P_FLATTEN * dt
    vy -= ny * grad_para * P_PRESSURE * P_FLATTEN * dt
    vz -= nz * grad_para * P_PRESSURE * P_FLATTEN * dt
    
    # --- 磁気圧勾配とアルヴェン波によるバルジ・ジェットの形成 ---
    # ここは磁場軸（nx, ny, nz）に沿った「プラズマの熱振動」なので、毎フレーム完全にランダムな振動（ホワイトノイズ）を与えます。
    # 時間的に滑らかな波にしてしまうと、数十フレーム同じ方向へ加速し続ける「ロケットエンジン（暴走）」になってしまいます。
    jet_noise = cp.random.rand(grid_size, grid_size, grid_size) - 0.5
    bulge_dispersion = vorticity * P_BULGE
    vx += nx * jet_noise * bulge_dispersion * dt
    vy += ny * jet_noise * bulge_dispersion * dt
    vz += nz * jet_noise * bulge_dispersion * dt
    
    # --- 宇宙のマクロな粘性（エーテルの非線形抵抗と光速の壁） ---
    # 「絶対に超えてはならない速度の壁」をご都合主義的なルール（外からのクリップ）で強制するのではなく、
    # 速度が上がるほど空間（流体）からの摩擦が非線形に増大し、自然にそれ以上加速できなくなる現象として記述する。
    # 複数の力がうまく纏まるパラメータの均衡点、それこそが「物理法則」の正体である。
    speed_sq = vx**2 + vy**2 + vz**2
    # 限界速度（SPEED_LIMIT）に近づくほど、DAMPING（減衰率）が急激に低下し、強いブレーキがかかる
    dynamic_damping = P_DAMPING * (1.0 - cp.clip(speed_sq / (SPEED_LIMIT ** 2), 0.0, 0.99))
    
    vx *= dynamic_damping
    vy *= dynamic_damping
    vz *= dynamic_damping
    
    # 流体（風）の粘性：速度場を全方位に少しだけ馴染ませることで、四角い風のブロック化を防ぎ丸い渦を作る
    # 粘性拡散（コルモゴロフスケール）：細かくなりすぎた乱流エネルギーを熱として散逸させ、構造の崩壊（ブラウン運動化）を防ぐ物理法則。
    # 【レイノルズ数の調整】25%のブレンドは粘性が強すぎたため、「水中の絵の具」のように細長く伸びてしまっていました。
    # ブレンド率を下げてサラサラな流体（高レイノルズ数）にすることで、風が一直線に伸びず、丸く巻き取られて大きな渦になりやすくなります。
    vx[:] = vx * 0.92 + fast_blur(vx, iterations=1) * 0.08
    vy[:] = vy * 0.92 + fast_blur(vy, iterations=1) * 0.08
    vz[:] = vz * 0.92 + fast_blur(vz, iterations=1) * 0.08
    
    # --- 宇宙全体のドリフト（海流）の防止（運動量保存則） ---
    # 【不自然な加速の原因②：平均値の引き算による逆走バグの撤廃】
    # 局所的な爆発（超新星など）で一部の風速が跳ね上がった時、全体の平均値がズレてしまい、
    # 全く関係ない遠くの無風の空間までが「逆方向へ強制的に加速させられる」という不自然な現象が起きていました。
    # 揺らぎの平均値は既にゼロにしているため、強引な引き算は削除して自然な摩擦（Damping）に任せます。
    
    # 数値計算上の完全な破綻（無限大発散）を防ぐための最終安全装置としてのみクリップを残す
    cp.clip(vx, -SPEED_LIMIT, SPEED_LIMIT, out=vx)
    cp.clip(vy, -SPEED_LIMIT, SPEED_LIMIT, out=vy)
    cp.clip(vz, -SPEED_LIMIT, SPEED_LIMIT, out=vz)
    
    # --------------------------------------------------
    # 仕様①：渦による星の生成と円盤化
    # --------------------------------------------------
    
    # 空間全体の渦度の「平均＋標準偏差×2（上位数％の強い渦）」を動的な閾値とする
    # これにより、風が穏やかでも激しくても、常に相対的に一番渦巻いている場所から星が生まれる
    current_vor_threshold = cp.mean(vorticity) + cp.std(vorticity) * 2.0
    spawn_mask = (vorticity > current_vor_threshold) & (cp.random.rand(*vorticity.shape) < 0.05)
    spawn_z, spawn_y, spawn_x = cp.where(spawn_mask)
    
    # パーティクル数が上限に達していない時だけ生成を許可する
    if len(spawn_z) > 0 and len(particles) < MAX_PARTICLES:
        # 無から質量を作らないよう、生成場所の空間密度をそのまま星の質量（サイズ）として奪う
        spawn_density = density[spawn_z, spawn_y, spawn_x]
        
        # 密度が薄すぎる場所からは星が生まれないようにする（閾値を下げて星の枯渇を防ぐ）
        valid_mask = spawn_density > 0.05
        if cp.any(valid_mask):
            sz, sy, sx = spawn_z[valid_mask], spawn_y[valid_mask], spawn_x[valid_mask]
            # 【不自然な真空の防止】星が生まれる時、周囲のガスを半分だけ吸収し、残りは緩やかに残るようにする
            initial_sizes = spawn_density[valid_mask] * 0.5
            
            new_particles = cp.column_stack((
                (sx + cp.random.rand(len(sx))) % grid_size,
                (sy + cp.random.rand(len(sy))) % grid_size,
                (sz + cp.random.rand(len(sz))) % grid_size,
                initial_sizes,
                vx[sz, sy, sx], # 生成された場所の流速を初期速度とする
                vy[sz, sy, sx],
                vz[sz, sy, sx]
            ))
            particles = cp.vstack((particles, new_particles))
            
            # 星になった分、空間の密度は半分になる（完全な真空(0.0)ができるとノイズ爆発の元になるため）
            density[sz, sy, sx] *= 0.5
            
            # もし上限を超えてしまったら、配列を削ってリミットを守る
            if len(particles) > MAX_PARTICLES:
                particles = particles[-MAX_PARTICLES:]

    # --------------------------------------------------
    # 仕様②：移動と、限界速度による「空間への消滅」
    # --------------------------------------------------
    if len(particles) > 0:
        # 星の座標から周囲8マスの風（速度）を滑らかに合成して受け取る（アナログな三線形補間）
        x_p, y_p, z_p = particles[:, 0], particles[:, 1], particles[:, 2]
        x0 = cp.floor(x_p).astype(cp.int32) % grid_size
        y0 = cp.floor(y_p).astype(cp.int32) % grid_size
        z0 = cp.floor(z_p).astype(cp.int32) % grid_size
        x1, y1, z1 = (x0 + 1) % grid_size, (y0 + 1) % grid_size, (z0 + 1) % grid_size
        
        dx_p, dy_p, dz_p = x_p - cp.floor(x_p), y_p - cp.floor(y_p), z_p - cp.floor(z_p)

        def sample_field(f):
            c00 = f[z0, y0, x0] * (1 - dx_p) + f[z0, y0, x1] * dx_p
            c01 = f[z0, y1, x0] * (1 - dx_p) + f[z0, y1, x1] * dx_p
            c10 = f[z1, y0, x0] * (1 - dx_p) + f[z1, y0, x1] * dx_p
            c11 = f[z1, y1, x0] * (1 - dx_p) + f[z1, y1, x1] * dx_p
            c0 = c00 * (1 - dy_p) + c01 * dy_p
            c1 = c10 * (1 - dy_p) + c11 * dy_p
            return c0 * (1 - dz_p) + c1 * dz_p

        p_vx = sample_field(vx)
        p_vy = sample_field(vy)
        p_vz = sample_field(vz)
        p_vorticity = sample_field(vorticity)
        
        # --- 等価原理（ガリレオ・アインシュタインの真理）への回帰 ---
        # 以前は「空気中のチリ」の法則を使い、重い星ほど慣性が強く風を無視して直進するようにしていました。
        # しかし宇宙では、重い星も軽い星も、空間の歪み（風）に対しては全く同じ軌道（測地線）を描きます（等価原理）。
        # 質量によって慣性を変える処理を廃止し、すべての星が等しく空間の流線に素直に乗るようにします。
        # これにより、重い星だけが曲がりきれずに「弧を描いて外へ飛んでいく」不自然な現象が解決します。
        # 【修正】0.85では慣性が強すぎて、急カーブ（渦）を曲がりきれずにコースアウトして直進していました。
        # 風のカーブにしっかりとグリップできるように、0.5に下げます。
        inertia = 0.5
        
        particles[:, 4] = particles[:, 4] * inertia + p_vx * (1.0 - inertia)
        particles[:, 5] = particles[:, 5] * inertia + p_vy * (1.0 - inertia)
        particles[:, 6] = particles[:, 6] * inertia + p_vz * (1.0 - inertia)
        
        # --- 星全体のドリフト（海流）の防止 ---
        # 【不自然な加速の原因②：同上】
        # 一部の星が爆風で飛んだ時、他の静止している星が逆走するのを防ぐため、全体平均の引き算を削除。
        
        # 速度の暴走を防ぐため、速度オーバーで消滅させるのではなく「限界速度（光速）で頭打ち」にする
        speed_factor = SPEED_LIMIT / (cp.sqrt(particles[:, 4]**2 + particles[:, 5]**2 + particles[:, 6]**2) + 1e-8)
        clip_mask = speed_factor < 1.0
        particles[clip_mask, 4] *= speed_factor[clip_mask]
        particles[clip_mask, 5] *= speed_factor[clip_mask]
        particles[clip_mask, 6] *= speed_factor[clip_mask]
        
        # --- 渦による星の閉じ込め（ピンチ効果：通称「宇宙ころころ説」） ---
        # 星が自重（引力）で形を保つのではなく、「周囲の空間の渦によって外側から転がされ、押し固められている」ため分解しない。
        # 渦（押し固める力）が強い場所にある星ほど、限界を超えた巨大なエネルギー（サイズ）でも形状を維持できる。
        # 【連鎖爆発の防止】以前の 100.0 は係数が強すぎたため、中心部の渦が少し揺らいだ瞬間に巨大星が一斉に超新星爆発を起こしていました。
        local_size_limit = SIZE_LIMIT + p_vorticity * 15.0
        surviving_mask = (particles[:, 3] <= local_size_limit)
        
        # 死亡した星は、持っていた質量（サイズ）を空間の密度（ガス）として還元する
        dead_particles = particles[~surviving_mask]
        if len(dead_particles) > 0:
            dx_p, dy_p, dz_p = dead_particles[:, 0], dead_particles[:, 1], dead_particles[:, 2]
            dm = dead_particles[:, 3]
            
            x0 = cp.floor(dx_p).astype(cp.int32) % grid_size
            y0 = cp.floor(dy_p).astype(cp.int32) % grid_size
            z0 = cp.floor(dz_p).astype(cp.int32) % grid_size
            x1, y1, z1 = (x0 + 1) % grid_size, (y0 + 1) % grid_size, (z0 + 1) % grid_size
            ddx, ddy, ddz = dx_p - cp.floor(dx_p), dy_p - cp.floor(dy_p), dz_p - cp.floor(dz_p)
            
            # 還元時もカーネルを使って周囲にフワッと広がるように返す
            cupyx.scatter_add(density, (z0, y0, x0), dm * (1-ddx) * (1-ddy) * (1-ddz))
            cupyx.scatter_add(density, (z0, y0, x1), dm * ddx * (1-ddy) * (1-ddz))
            cupyx.scatter_add(density, (z0, y1, x0), dm * (1-ddx) * ddy * (1-ddz))
            cupyx.scatter_add(density, (z0, y1, x1), dm * ddx * ddy * (1-ddz))
            cupyx.scatter_add(density, (z1, y0, x0), dm * (1-ddx) * (1-ddy) * ddz)
            cupyx.scatter_add(density, (z1, y0, x1), dm * ddx * (1-ddy) * ddz)
            cupyx.scatter_add(density, (z1, y1, x0), dm * (1-ddx) * ddy * ddz)
            cupyx.scatter_add(density, (z1, y1, x1), dm * ddx * ddy * ddz)
            
            # 【爆発原因の解消】以前はここで宇宙全体の密度(density)にぼかしをかけていたため、
            # 星が寿命を迎えるたびに宇宙全域の構造が崩壊（爆発消滅）するというデジタルな連鎖崩壊が起きていました。
            # 宇宙の破壊を止め、超新星爆発の物理法則として「死んだ星が周囲の流体を吹き飛ばす爆風」に置き換えます。
            # ※一斉爆発時の暴風を防ぐため、風の強さをマイルド（60 -> 15）に下げます。
            # 【復活】完全に消すと宇宙がかき混ぜられず渦ができないため、そよ風(3.0)として復活させます。
            cupyx.scatter_add(vx, (z0, y0, x0), (cp.random.rand(len(dm)) - 0.5) * P_SUPERNOVA)
            cupyx.scatter_add(vy, (z0, y0, x0), (cp.random.rand(len(dm)) - 0.5) * P_SUPERNOVA)
            cupyx.scatter_add(vz, (z0, y0, x0), (cp.random.rand(len(dm)) - 0.5) * P_SUPERNOVA)

        # 生存しているパーティクルだけを抽出し一括で移動させる
        surviving_particles = particles[surviving_mask]
        
        # 完全に1点に集積（太陽系化）するのを防ぐため、移動に微小な拡散（ブラウン運動）を加える
        # 宇宙スケールの滑らかな軌道を再現するため、人工的なブラウン運動を限りなくゼロにする
        diffusion = 0.0 * dt
        surviving_particles[:, 0] = (surviving_particles[:, 0] + surviving_particles[:, 4] * dt + (cp.random.rand(len(surviving_particles)) - 0.5) * diffusion) % grid_size
        surviving_particles[:, 1] = (surviving_particles[:, 1] + surviving_particles[:, 5] * dt + (cp.random.rand(len(surviving_particles)) - 0.5) * diffusion) % grid_size
        surviving_particles[:, 2] = (surviving_particles[:, 2] + surviving_particles[:, 6] * dt + (cp.random.rand(len(surviving_particles)) - 0.5) * diffusion) % grid_size
        
        particles = surviving_particles

        # 星が移動先にある空間のガス（密度）を吸収して成長する
        # 切り捨て(astype)による空間的な偏り（常に左下へ力が蓄積するバグ）を防ぐため、四捨五入(round)を使用します。
        new_idx_x = cp.round(particles[:, 0]).astype(cp.int32) % grid_size
        new_idx_y = cp.round(particles[:, 1]).astype(cp.int32) % grid_size
        new_idx_z = cp.round(particles[:, 2]).astype(cp.int32) % grid_size
        
        local_density = density[new_idx_z, new_idx_y, new_idx_x]
        
        # 錬金術防止：同じ座標にいる星の数を数え、その場所のガスを「山分け」する
        star_counts = cp.zeros_like(density)
        cupyx.scatter_add(star_counts, (new_idx_z, new_idx_y, new_idx_x), cp.ones(len(particles), dtype=cp.float32))
        local_star_counts = star_counts[new_idx_z, new_idx_y, new_idx_x]
        
        absorption = (local_density * P_ABSORB) / local_star_counts
        particles[:, 3] += absorption
        
        # 吸収した分、空間の密度を減らす（マイナス加算）
        cupyx.scatter_add(density, (new_idx_z, new_idx_y, new_idx_x), -absorption)
        
        # 密度がマイナスにならないようにする（上限は外して超新星爆発のガスが消滅するのを防ぐ）
        cp.clip(density, 0.0, None, out=density)

        # --- エーテルの引きずり効果と作用・反作用の法則 ---
        # 「物質が動くとき、周囲の空間（エーテル）も一緒に引きずられて動く」という現象の再現。
        # 以前は星の絶対速度をそのまま風に足していたため、星と風がお互いを押し合って「無限加速（ロケット暴走）」していました。
        # これが、時間を進めると直線運動が画面端をループして無限に加速していく真犯人でした。
        # 物理法則に従い、「星と風の相対速度（速度差）」の分だけを空間に伝達し、運動量保存則を守ります。
        drag_factor = 0.05 * dt
        momentum_transfer_x = (particles[:, 4] - p_vx) * drag_factor
        momentum_transfer_y = (particles[:, 5] - p_vy) * drag_factor
        momentum_transfer_z = (particles[:, 6] - p_vz) * drag_factor
        
        cupyx.scatter_add(vx, (new_idx_z, new_idx_y, new_idx_x), momentum_transfer_x)
        cupyx.scatter_add(vy, (new_idx_z, new_idx_y, new_idx_x), momentum_transfer_y)
        cupyx.scatter_add(vz, (new_idx_z, new_idx_y, new_idx_x), momentum_transfer_z)

        # 統合処理が重いため、一度コメントアウトします
        # if sim_step % 30 == 0:
        #     # 小数点以下1桁の精度で同じ位置にいるものを一意（ユニーク）にまとめる
        #     rounded_coords = cp.round(particles[:, :3], decimals=1)
        #     _, unique_indices, inverse_indices = cp.unique(rounded_coords, axis=0, return_index=True, return_inverse=True)
        #     # 重なった星たちのサイズを合算する
        #     new_sizes = cp.bincount(inverse_indices, weights=particles[:, 3])
        #     particles = particles[unique_indices]
        #     particles[:, 3] = new_sizes # 合計サイズで更新する
        # --------------------------------------------------
        # 星の衝突・合体（軽量化＆運動量保存版）
        # --------------------------------------------------
        # 自然はアナログであるという哲学に基づき、「グリッド（デジタルの箱）に入ったら強制合体」という
        # デジタル特有の不自然なルールを完全に排除します。星は合体せず、すれ違って星団を作ります。
        # if sim_step % 1 == 0 and len(particles) > 0:
        #     # 重い cp.unique を廃止し、超高速な 1次元配列の bincount を使ったグリッド集約に変更
        #     idx_x = particles[:, 0].astype(cp.int32) % grid_size
        #     idx_y = particles[:, 1].astype(cp.int32) % grid_size
        #     idx_z = particles[:, 2].astype(cp.int32) % grid_size
        #     
        #     # 3次元座標を1次元のインデックスに変換
        #     flat_idx = idx_z * (grid_size ** 2) + idx_y * grid_size + idx_x
        #     max_idx = grid_size ** 3
        #     
        #     # 各マスの総エネルギー（規模）を超高速で合算
        #     new_sizes = cp.bincount(flat_idx, weights=particles[:, 3], minlength=max_idx)
        #     active_mask = new_sizes > 0
        #     
        #     if cp.sum(active_mask) < len(particles):
        #         active_flat_idx = cp.where(active_mask)[0]
        #         
        #         new_particles = cp.zeros((len(active_flat_idx), 7), dtype=cp.float32)
        #         new_particles[:, 0] = cp.bincount(flat_idx, weights=particles[:, 0] * particles[:, 3], minlength=max_idx)[active_mask] / new_particles[:, 3]
        #         new_particles[:, 1] = cp.bincount(flat_idx, weights=particles[:, 1] * particles[:, 3], minlength=max_idx)[active_mask] / new_particles[:, 3]
        #         new_particles[:, 2] = cp.bincount(flat_idx, weights=particles[:, 2] * particles[:, 3], minlength=max_idx)[active_mask] / new_particles[:, 3]
        #         new_particles[:, 3] = new_sizes[active_mask]
        #         
        #         new_particles[:, 4] = cp.bincount(flat_idx, weights=particles[:, 3] * particles[:, 4], minlength=max_idx)[active_mask] / new_particles[:, 3]
        #         new_particles[:, 5] = cp.bincount(flat_idx, weights=particles[:, 3] * particles[:, 5], minlength=max_idx)[active_mask] / new_particles[:, 3]
        #         new_particles[:, 6] = cp.bincount(flat_idx, weights=particles[:, 3] * particles[:, 6], minlength=max_idx)[active_mask] / new_particles[:, 3]
        #         
        #         particles = new_particles

    # --- 質量保存（構造の進化を阻害しない穏やかな補正） ---
    # 一律スケーリングはボイドの深化を妨げるため、1フレーム最大0.1%の微小補正に制限
    current_star_mass = float(cp.sum(particles[:, 3])) if len(particles) > 0 else 0.0
    current_gas_mass = float(cp.sum(density))
    target_gas_mass = INITIAL_TOTAL_MASS - current_star_mass
    if target_gas_mass > 0 and current_gas_mass > 0:
        ratio = target_gas_mass / current_gas_mass
        # 大きな補正を制限し、構造の自然な進化を許す
        gentle_ratio = 1.0 + max(-0.001, min(0.001, ratio - 1.0))
        density *= gentle_ratio

# --- 実行と描画のメインループ ---
if __name__ == "__main__":
    # コントロールパネルのUIを別スレッドで起動
    start_control_panel()

    # 関数に渡す速度場(vx, vy, vz)の初期化（最初は無風状態）
    vx = cp.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    vy = cp.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    vz = cp.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    # --- VisPy (GPU描画) のセットアップ ---
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(800, 800))
    view = canvas.central_widget.add_view()
    
    # カメラの設定 (マウスでドラッグして視点変更可能)
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = GRID_SIZE * 1.5
    view.camera.center = (GRID_SIZE/2, GRID_SIZE/2, GRID_SIZE/2)
    
    # 星雲と星の描画用オブジェクト (GPUに直接送られる)
    scatter_density = scene.visuals.Markers(parent=view.scene)
    scatter_particles = scene.visuals.Markers(parent=view.scene)
    particle_trails = scene.visuals.Line(parent=view.scene, connect='segments', method='gl', antialias=False)
    
    # フレームごとの更新処理（タイマーから毎フレーム自動で呼ばれる）
    def on_timer(event):
        global density, vx, vy, vz, particles
        
        # UIから最新のタイムスケールを取得
        current_dt = UI_PARAMS["TIME_SCALE"]["val"]
        
        # シミュレーションを1ステップ進める
        update_simulation_step(density, vx, vy, vz, dt=current_dt) # dtを小さくして時間の進み方をゆっくりにする
        
        # --- 密度の描画（星雲）---
        # 密度可視化が点滅して見づらく、処理も重いため完全に非表示（真っ暗）にする
        scatter_density.visible = False
        
        # --- 星の描画 ---
        if len(particles) > 0:
            # 座標とサイズもGPU側で計算・型変換してから持ってくる
            pos_particles = particles[:, :3].astype(cp.float32)
            vel_particles = particles[:, 4:7].astype(cp.float32)
            raw_sizes = particles[:, 3]
            N = len(particles)
            
            # 速度ベクトルを使って「星の軌跡（尻尾）」を計算する
            tail_length = 0.1 # 軌跡の長さ
            pos_tails = pos_particles - vel_particles * tail_length
            
            # 線分として描画するために、[頭1, 尻尾1, 頭2, 尻尾2...] と並んだ配列を作る
            segments = cp.empty((N * 2, 3), dtype=cp.float32)
            segments[0::2] = pos_particles
            segments[1::2] = pos_tails
            
            fixed_size = 1.0 # パーティクルの見た目のサイズ（お好みの小ささに調整してください）
            
            # 星のエネルギー規模（サイズ）に応じて色を変える（青色巨星 → 白 → 黄色 → 赤色超巨星/爆発間近）
            mass_ratio = cp.clip(raw_sizes / SIZE_LIMIT, 0.0, 1.0)
            inv_mass_ratio = 1.0 - mass_ratio # 割合を反転させる
            p_colors = cp.ones((N, 4), dtype=cp.float32)
            
            p_colors[:, 0] = cp.clip(1.0 - (inv_mass_ratio - 0.5) * 0.5, 0.5, 1.0) # 赤
            p_colors[:, 1] = cp.clip(0.5 + inv_mass_ratio, 0.0, 1.0)               # 緑
            p_colors[:, 2] = cp.clip(inv_mass_ratio * 1.5, 0.0, 1.0)               # 青
            
            # --- 空間の伸縮による「光の波長の引き伸ばし（赤方偏移）」の再現 ---
            # 宇宙全体の膨張（ビッグバン）という概念を排し、流体空間の局所的な伸縮によって光が赤くズレる現象を記述する。
            # 観測者（空間の中心付近）から見て遠ざかる速度成分（空間が引き伸ばされている方向）を持つ星ほど、
            # 波長が伸びて赤方偏移（Redshift）し、近づく星は青方偏移（Blueshift）する。
            center_pos = pos_particles - cp.array([GRID_SIZE/2.0, GRID_SIZE/2.0, GRID_SIZE/2.0], dtype=cp.float32)
            dist = cp.sqrt(cp.sum(center_pos**2, axis=1)) + 1e-8
            radial_dir = center_pos / dist[:, None] # 中心から外へ向かうベクトル
            
            # 視線方向（空間の引き伸ばし方向）の速度成分
            radial_vel = cp.sum(vel_particles * radial_dir, axis=1)
            
            # 限界速度（SPEED_LIMIT）に対する割合で波長の伸び縮みを計算
            redshift = cp.clip(radial_vel / (SPEED_LIMIT * 0.3), -1.0, 1.0)
            
            # 光の波長のシフトを色に適用（遠ざかる＝赤が強まる、近づく＝青が強まる）
            p_colors[:, 0] = cp.clip(p_colors[:, 0] + redshift * 0.5, 0.0, 1.0)
            p_colors[:, 2] = cp.clip(p_colors[:, 2] - redshift * 0.5, 0.0, 1.0)

            # 実際の宇宙空間（真空）では大気の揺らぎがないため、星は瞬きません。
            # シミュレータの利点を活かし、空気の邪魔を排除した純粋で鋭い輝きにします。
            p_colors[:, 3] = 1.0
            
            # 軌跡用の色（頭はそのままの色、尻尾の先は透明になるようにグラデーションさせる）
            segment_colors = cp.empty((N * 2, 4), dtype=cp.float32)
            segment_colors[0::2] = p_colors
            tail_colors = p_colors.copy()
            tail_colors[:, 3] = 0.05 # 尻尾の先はうっすら透明
            segment_colors[1::2] = tail_colors
            
            # CPUに転送して描画
            scatter_particles.set_data(pos_particles.get(), face_color=p_colors.get(), size=fixed_size, edge_width=0)
            particle_trails.set_data(pos=segments.get(), color=segment_colors.get())
            scatter_particles.visible = True
            particle_trails.visible = True
        else:
            scatter_particles.visible = False
            particle_trails.visible = False

    # タイマーをセットして描画ループを開始 (約30FPSに制限してGUIのフリーズを防ぐ)
    timer = app.Timer(interval=1.0 / 30.0, connect=on_timer, start=True)
    
    # VisPyのウィンドウを表示して実行
    app.run()
