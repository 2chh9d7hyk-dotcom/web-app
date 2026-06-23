import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ================================================================
# SECTION 0: ページ設定（必ず最初のst.*コール）
# ================================================================
st.set_page_config(
    page_title="ミッション03: AI育成 | AI Inquiry Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# SECTION 1: セキュリティガード
# ================================================================
if "user_role" not in st.session_state:
    st.session_state.user_role = "viewer"

if st.query_params.to_dict():
    st.error("不正なアクセスを検知しました。URLを直接書き換えないでください。")
    st.stop()

st.markdown("""
<script>
    if (window.top !== window.self) { window.top.location = window.self.location; }
    document.querySelectorAll('a').forEach(link => { link.setAttribute('rel', 'noopener noreferrer'); });
</script>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 2: 定数・パス設定
# ================================================================
SCRIPT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PARENT_DIR, "assets")
CSS_FILE   = os.path.join(ASSETS_DIR, "style.css")

DATASETS = {
    "🔌 ANDゲート":    "and",
    "⚡ ORゲート":     "or",
    "↗ 線形分類":      "linear",
    "✖ XOR（不可能）": "xor",
}

ACTIVATION_FUNS = {
    "シグモイド σ(z)": "sigmoid",
    "ReLU max(0,z)":  "relu",
    "tanh(z)":        "tanh",
}

# ================================================================
# SECTION 3: ユーティリティ関数
# ================================================================
def load_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def activate(x: np.ndarray, fn: str) -> np.ndarray:
    return {"sigmoid": sigmoid, "relu": relu, "tanh": tanh_act}[fn](x)

# ================================================================
# SECTION 4: シミュレーション関数群
# ================================================================
def generate_dataset(key: str, n: int = 120, seed: int = 42) -> tuple:
    np.random.seed(seed)
    if key == "and":
        base   = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        labels = np.array([0, 0, 0, 1], dtype=float)
        X = np.tile(base, (n // 4, 1)) + np.random.normal(0, 0.06, (n, 2))
        y = np.tile(labels, n // 4)
    elif key == "or":
        base   = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        labels = np.array([0, 1, 1, 1], dtype=float)
        X = np.tile(base, (n // 4, 1)) + np.random.normal(0, 0.06, (n, 2))
        y = np.tile(labels, n // 4)
    elif key == "linear":
        X = np.random.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
    else:
        X = np.random.randn(n, 2) * 0.8
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    return X, y

def train_perceptron(X, y, lr, epochs, act_fn="sigmoid"):
    np.random.seed(0)
    w = np.random.randn(X.shape[1]) * 0.1
    b = 0.0
    losses, accuracies = [], []
    for _ in range(epochs):
        z    = X @ w + b
        pred = activate(z, act_fn)
        eps  = 1e-8
        loss = -np.mean(y * np.log(np.clip(pred, eps, 1)) + (1 - y) * np.log(np.clip(1 - pred, eps, 1)))
        losses.append(float(loss))
        dz = pred - y
        w -= lr * (X.T @ dz / len(y))
        b -= lr * float(np.mean(dz))
        accuracies.append(float(np.mean((pred > 0.5) == y)) * 100)
    return w, b, losses, accuracies

def plot_activation_curve(z_val: float, fn_name: str):
    x   = np.linspace(-6, 6, 300)
    y   = activate(x, fn_name)
    out = float(activate(np.array([z_val]), fn_name)[0])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    for sp in ax.spines.values():
        sp.set_color("#3d5a8a")
    ax.tick_params(colors="#e6edf3")
    ax.plot(x, y, color="#00d4ff", lw=2.5, label=fn_name)
    ax.axhline(0, color="#3d5a8a", lw=0.8, ls="--")
    ax.axvline(0, color="#3d5a8a", lw=0.8, ls="--")
    ax.scatter([z_val], [out], color="#ffd700", s=120, zorder=5)
    ax.vlines(z_val, min(y.min(), out) - 0.05, out, color="#ffd700", ls=":", alpha=0.7)
    ax.hlines(out, -6, z_val, color="#ffd700", ls=":", alpha=0.7)
    ax.set_xlabel("z（加重和）", color="#e6edf3", fontsize=9)
    ax.set_ylabel("出力", color="#e6edf3", fontsize=9)
    ax.set_title(f"活性化関数：{fn_name}", color="#e6edf3", fontsize=10)
    ax.legend(labelcolor="#e6edf3", facecolor="#1e2a3a", edgecolor="#3d5a8a", fontsize=8)
    ax.grid(True, alpha=0.15, color="#3d5a8a")
    plt.tight_layout(pad=0.5)
    return fig, out

def plot_loss_acc(losses, accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor("#1e2a3a")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1e2a3a")
        ax.tick_params(colors="#e6edf3")
        for sp in ax.spines.values():
            sp.set_color("#3d5a8a")
        ax.grid(True, alpha=0.15, color="#3d5a8a")
    ep = np.arange(1, len(losses) + 1)
    ax1.plot(ep, losses, color="#ff71ce", lw=2)
    ax1.set_title("損失 (Loss)", color="#e6edf3")
    ax1.set_xlabel("Epoch", color="#e6edf3")
    ax1.set_ylabel("Loss", color="#e6edf3")
    ax2.plot(ep, accuracies, color="#05ffa1", lw=2)
    ax2.set_ylim(0, 105)
    ax2.set_title("精度 (Accuracy)", color="#e6edf3")
    ax2.set_xlabel("Epoch", color="#e6edf3")
    ax2.set_ylabel("Accuracy (%)", color="#e6edf3")
    plt.suptitle("学習曲線", color="#ffd700", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_decision_boundary(X, y, w, b, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    for sp in ax.spines.values():
        sp.set_color("#3d5a8a")
    ax.tick_params(colors="#e6edf3")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
    z_grid = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)
    ax.contourf(xx, yy, z_grid, levels=50, cmap="RdYlGn", alpha=0.3)
    ax.contour(xx, yy, z_grid, levels=[0.5], colors="#ffd700", linewidths=2)
    ax.scatter(X[y==1, 0], X[y==1, 1], c="#05ffa1", s=30, label="クラス1", edgecolors="white", lw=0.4, zorder=3)
    ax.scatter(X[y==0, 0], X[y==0, 1], c="#ff71ce", s=30, label="クラス0", edgecolors="white", lw=0.4, zorder=3)
    ax.legend(labelcolor="#e6edf3", facecolor="#1e2a3a", edgecolor="#3d5a8a", fontsize=8)
    ax.set_title(f"{dataset_name} の分類境界", color="#e6edf3", fontsize=10)
    plt.tight_layout(pad=0.5)
    return fig

def plot_overfitting(degree, noise, seed=7):
    np.random.seed(seed)
    n = 18
    x_tr   = np.sort(np.random.uniform(-1, 1, n))
    y_tr   = np.sin(np.pi * x_tr) + np.random.normal(0, noise, n)
    x_te   = np.linspace(-1.3, 1.3, 200)
    y_true = np.sin(np.pi * x_te)
    degree = max(1, min(degree, 12))
    try:
        coeffs     = np.polyfit(x_tr, y_tr, degree)
        poly       = np.poly1d(coeffs)
        y_fit      = np.clip(poly(x_te), -5, 5)
        train_mse  = float(np.mean((y_tr - poly(x_tr)) ** 2))
        test_mse   = float(np.mean((y_true - poly(x_te)) ** 2))
    except Exception:
        y_fit = np.zeros_like(x_te)
        train_mse = test_mse = 0.0
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    for sp in ax.spines.values():
        sp.set_color("#3d5a8a")
    ax.tick_params(colors="#e6edf3")
    ax.scatter(x_tr, y_tr, c="#05ffa1", s=60, zorder=5, label="訓練データ", edgecolors="white", lw=0.5)
    ax.plot(x_te, y_true, color="#00d4ff", lw=2, ls="--", label="真の関数（正解）")
    ax.plot(x_te, y_fit,  color="#ff71ce", lw=2, label=f"モデル（次数{degree}）")
    ax.set_ylim(-4, 4)
    ax.set_xlim(-1.5, 1.5)
    ax.grid(True, alpha=0.15, color="#3d5a8a")
    ax.legend(labelcolor="#e6edf3", facecolor="#1e2a3a", edgecolor="#3d5a8a", fontsize=8)
    ax.set_title(f"多項式フィット（次数 {degree}）", color="#e6edf3", fontsize=11)
    plt.tight_layout()
    return fig, train_mse, test_mse

def plot_nn_diagram(layer_sizes):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n_layers    = len(layer_sizes)
    xs          = np.linspace(0.1, 0.9, n_layers)
    layer_cols  = ["#00d4ff", "#05ffa1", "#ffd700", "#ff71ce", "#00d4ff"]
    all_pos     = []
    max_n       = max(layer_sizes)
    for li, (x, n) in enumerate(zip(xs, layer_sizes)):
        col  = layer_cols[li % len(layer_cols)]
        gap  = min(0.65 / max_n, 0.12)
        ys   = [0.5 + (j - (n - 1) / 2) * gap for j in range(n)]
        all_pos.append([(x, y) for y in ys])
        if li > 0:
            for px, py in all_pos[li - 1]:
                for cx, cy in all_pos[li]:
                    ax.plot([px, cx], [py, cy], color="#3d5a8a", lw=0.6, alpha=0.5, zorder=1)
        for nx, ny in all_pos[li]:
            circle = plt.Circle((nx, ny), 0.022, color=col, ec="white", lw=0.8, zorder=3)
            ax.add_patch(circle)
        lbl = "入力層" if li == 0 else ("出力層" if li == n_layers - 1 else f"隠れ層{li}")
        ax.text(x, 0.07, lbl,        ha="center", color=col,       fontsize=8, fontweight="bold")
        ax.text(x, 0.02, f"({n})",   ha="center", color="#8b9ab0", fontsize=7)
    ax.set_title("ネットワーク構造", color="#ffd700", fontsize=13, pad=5)
    plt.tight_layout()
    return fig

# ================================================================
# SECTION 5: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)

st.markdown("""
<div class="main-title-container">
    <h1 class="main-title-text">🧬 AI育成</h1>
    <p class="sub-title-text">MISSION 03 — AIを生み出し、鍛え、成長を見守る体験</p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 6: セッション状態の初期化
# ================================================================
if "train_result_3"   not in st.session_state: st.session_state["train_result_3"]   = None
if "best_accuracy_3"  not in st.session_state: st.session_state["best_accuracy_3"]  = 0.0
if "train_attempts_3" not in st.session_state: st.session_state["train_attempts_3"] = 0

# ================================================================
# SECTION 7: サイドバー
# ================================================================
with st.sidebar:
    st.markdown("""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#ffd700; font-weight:bold; font-size:0.9rem;">🧬 TRAINING MODE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### ⚡ 学習パラメータ")
    lr_sidebar     = st.slider("学習率 (η)", 0.001, 1.0, 0.1, 0.005, format="%.3f", key="lr_sidebar")
    epochs_sidebar = st.slider("エポック数",  10,   500, 100, 10,               key="ep_sidebar")

    st.divider()
    st.markdown("### 🧭 Navigation")
    st.page_link("main_app.py",          label="司令室 (Home)",         icon="🏠")
    st.page_link("pages/1_AIの目.py",    label="ミッション01: AIの目",  icon="👁️")
    st.page_link("pages/2_AI騙し.py",   label="ミッション02: AI騙し",  icon="🎭")
    st.page_link("pages/3_AI育成.py",   label="ミッション03: AI育成",  icon="🧬")
    st.divider()
    st.success("🛡️ SHIELD: ONLINE")

# ================================================================
# SECTION 8: ヒーロー紹介
# ================================================================
st.markdown("""
<div class="explanation-box">
<h3>🎓 AIを「育てる」とはどういうことか？</h3>
ChatGPTも自動運転AIも、最初は何も知らない「白紙の状態」から始まります。<br>
大量のデータを与え、正解と間違いを繰り返し教えることで、徐々に賢くなっていく——<br>
これが<b>機械学習（Machine Learning）</b>の本質です。<br><br>
このミッションでは、あなた自身がAIの「先生」となり、
<b>ニューロンの誕生 → 学習の成功と失敗 → ネットワーク構築 → 過学習の克服</b>まで、
AI育成の全プロセスを手を動かしながら体験します。
</div>
""", unsafe_allow_html=True)

st.header("🔬 STEP 2：タブを選んでAI育成の全プロセスを体験しよう！")

tab1, tab2, tab3, tab4 = st.tabs([
    "① ニューロンを育てる",
    "② ミニAIを鍛える",
    "③ ネットワークを組む",
    "④ 過学習の罠",
])

# ================================================================
# SECTION 9: タブコンテンツ
# ================================================================

# ---------------------------------------------------------------
# TAB 1: ニューロンの仕組み
# ---------------------------------------------------------------
with tab1:
    st.header("⚡ AIの最小単位「ニューロン」を動かせ！")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_text1, col_text2 = st.columns([3, 2])

        with col_text1:
            st.markdown("""
            <div class="explanation-box">
            <h3>🧠 ニューロンとは何か？</h3>
            人間の脳には約<b>860億個</b>の神経細胞（ニューロン）があります。
            AIのニューラルネットワークは、この生物ニューロンを<b>数式でモデル化</b>したものです。<br><br>
            <b>人工ニューロン（パーセプトロン）の動作：</b>
            <ol>
                <li>複数の「入力（x）」を受け取る</li>
                <li>それぞれに「重み（w）」を掛けて足し合わせる</li>
                <li>「バイアス（b）」を加える → これが <b>加重和 z</b></li>
                <li>「活性化関数」で 0〜1 の信号に変換する → これが <b>出力</b></li>
            </ol>
            下のスライダーを動かして、ニューロンが「発火」する瞬間を体感しよう！
            </div>
            """, unsafe_allow_html=True)

        with col_text2:
            st.markdown("""
            <div style="background:#0d1117; border:2px solid #ffd700; padding:16px;
                        font-family:monospace; border-left:6px solid #00d4ff;">
                <div style="color:#ffd700; font-weight:bold; margin-bottom:10px;">▶ 数式</div>
                <div style="color:#00d4ff; font-size:1.05rem;">z = x₁·w₁ + x₂·w₂ + b</div>
                <div style="color:#8b9ab0; margin:6px 0;">（加重和）</div>
                <div style="color:#05ffa1; font-size:1.05rem;">output = σ(z)</div>
                <div style="color:#8b9ab0; margin:6px 0;">（活性化）</div>
                <br>
                <div style="color:#ff71ce; font-size:0.8rem;">
                ● z が大きい → 強く発火 (→1)<br>
                ● z が小さい → 発火しない (→0)<br>
                ● z ≈ 0　　 → 境界（確率 50%）
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🕹️ インタラクティブ・ニューロン")

    n_col1, n_col2, n_col3 = st.columns([1, 1, 2])

    with n_col1:
        st.markdown("**📥 入力値**")
        x1 = st.slider("入力 x₁", -3.0, 3.0,  1.0, 0.1, key="n_x1")
        x2 = st.slider("入力 x₂", -3.0, 3.0,  0.5, 0.1, key="n_x2")

    with n_col2:
        st.markdown("**⚖️ 重み・バイアス**")
        w1   = st.slider("重み w₁",    -3.0, 3.0,  0.8,  0.1, key="n_w1")
        w2   = st.slider("重み w₂",    -3.0, 3.0,  0.6,  0.1, key="n_w2")
        bias = st.slider("バイアス b", -3.0, 3.0, -0.5,  0.1, key="n_bias")
        act_choice = st.selectbox("活性化関数", list(ACTIVATION_FUNS.keys()), key="n_act")
        act_fn     = ACTIVATION_FUNS[act_choice]

    z_val   = x1 * w1 + x2 * w2 + bias
    out_val = float(activate(np.array([z_val]), act_fn)[0])
    fired   = out_val > 0.5

    with n_col3:
        fig_act, _ = plot_activation_curve(z_val, act_fn)
        st.pyplot(fig_act)
        plt.close(fig_act)

    m1, m2, m3 = st.columns(3)
    m1.metric("加重和 z",    f"{z_val:.3f}")
    m2.metric("出力（確率）", f"{out_val:.3f}")
    m3.metric("ニューロン状態", "🔥 発火中！（→クラス1）" if fired else "😴 静止（→クラス0）")

    st.markdown(f"""
    <div style="background:#0d1117; border:1px solid #3d5a8a; padding:14px;
                font-family:monospace; font-size:0.95rem; margin-top:8px;">
        <span style="color:#00d4ff;">z</span> =
        (<span style="color:#e6edf3;">{x1:.1f}</span> × <span style="color:#ffd700;">{w1:.1f}</span>) +
        (<span style="color:#e6edf3;">{x2:.1f}</span> × <span style="color:#ffd700;">{w2:.1f}</span>) +
        <span style="color:#ff71ce;">{bias:.1f}</span> =
        <span style="color:#05ffa1; font-weight:bold;">{z_val:.3f}</span>
        &nbsp;&nbsp;→&nbsp;&nbsp;
        output = {act_choice}(<span style="color:#05ffa1;">{z_val:.3f}</span>) =
        <span style="color:#ffd700; font-weight:bold;">{out_val:.3f}</span>
    </div>
    """, unsafe_allow_html=True)

    if fired:
        st.success(f"✅ このニューロンは「発火」しました！ 出力 {out_val:.3f} > 0.5 → クラス 1 として判定")
    else:
        st.info(f"💤 このニューロンは「静止」しています。 出力 {out_val:.3f} ≤ 0.5 → クラス 0 として判定")

    st.markdown("""
    <div class="explanation-box" style="margin-top:16px;">
    <h3>💡 重みとバイアスの役割</h3>
    <ul>
    <li><b>重み（w）</b>：その入力が「どれくらい重要か」を決める係数。大きいほど影響力が強い。</li>
    <li><b>バイアス（b）</b>：「どれくらい活性化しやすいか」のオフセット。発火のしきい値を制御する。</li>
    <li><b>学習とは</b>：大量のデータを見ながら、w と b を少しずつ調整していくプロセスです。</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 2: 学習シミュレーター
# ---------------------------------------------------------------
with tab2:
    st.header("🏋️ ミニAIを鍛えあげろ！")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation-box">
        <h3>📚 機械学習の3ステップ</h3>
        AIが学習するプロセスは、3つのステップの繰り返しです：<br>
        <b>① 予測（Forward Pass）→ ② 誤差計算（Loss）→ ③ 重みの更新（Backward Pass）</b><br>
        これを何度もくり返すことで、AIは徐々に賢くなります。
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🎯 データセットとパラメータを設定しよう")

    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)

    with cfg_c1:
        dataset_name = st.selectbox("データセット",    list(DATASETS.keys()), key="ds_select")
        dataset_key  = DATASETS[dataset_name]
        n_samples    = st.slider("サンプル数", 40, 200, 100, 20, key="n_samples_3")

    with cfg_c2:
        lr_t2     = st.slider("学習率 η",  0.001, 1.0, 0.1, 0.005, format="%.3f", key="lr_tab2")
        epochs_t2 = st.slider("エポック数", 10,   500, 100, 10,               key="ep_tab2")

    with cfg_c3:
        act_ch2 = st.selectbox("活性化関数", list(ACTIVATION_FUNS.keys()), key="act_tab2")
        act_fn2 = ACTIVATION_FUNS[act_ch2]
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 学習スタート！", use_container_width=True, key="train_btn"):
            X_tr, y_tr = generate_dataset(dataset_key, n_samples)
            w_f, b_f, losses, accs = train_perceptron(X_tr, y_tr, lr_t2, epochs_t2, act_fn2)
            st.session_state["train_result_3"] = {
                "X": X_tr, "y": y_tr, "w": w_f, "b": b_f,
                "losses": losses, "accuracies": accs,
                "dataset_name": dataset_name, "dataset_key": dataset_key,
            }
            best_acc = max(accs)
            if best_acc > st.session_state["best_accuracy_3"]:
                st.session_state["best_accuracy_3"] = best_acc
            st.session_state["train_attempts_3"] += 1

    if dataset_key == "xor":
        st.warning("⚠️ XOR 問題は単層パーセプトロンでは解けません！どんな設定でも精度が 50% 付近に留まります。これが「ディープラーニング」が必要な理由です！")

    result = st.session_state.get("train_result_3")

    if result:
        st.markdown("### 📊 学習結果レポート")

        final_loss = result["losses"][-1]
        final_acc  = result["accuracies"][-1]
        best_acc   = max(result["accuracies"])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("最終 Loss",      f"{final_loss:.4f}")
        r2.metric("最終精度",        f"{final_acc:.1f}%")
        r3.metric("最高精度",        f"{best_acc:.1f}%")
        r4.metric("トレーニング回数", f"{st.session_state['train_attempts_3']} 回")

        res_c1, res_c2 = st.columns([3, 2])
        with res_c1:
            fig_lc = plot_loss_acc(result["losses"], result["accuracies"])
            st.pyplot(fig_lc)
            plt.close(fig_lc)
        with res_c2:
            fig_db = plot_decision_boundary(result["X"], result["y"], result["w"], result["b"], result["dataset_name"])
            st.pyplot(fig_db)
            plt.close(fig_db)

        w = result["w"]; b_v = result["b"]
        st.markdown(f"""
        <div style="background:#0d1117; border:1px solid #3d5a8a; padding:12px;
                    font-family:monospace; font-size:0.9rem;">
            <span style="color:#ffd700;">最終的な重みパラメータ:</span><br>
            w₁ = <span style="color:#05ffa1;">{w[0]:.4f}</span> &nbsp;|&nbsp;
            w₂ = <span style="color:#05ffa1;">{w[1]:.4f}</span> &nbsp;|&nbsp;
            b  = <span style="color:#ff71ce;">{b_v:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

        if best_acc >= 95:
            st.success(f"🏆 エクセレント！ 精度 {best_acc:.1f}% 達成！AI の学習が成功しました！")
            st.balloons()
        elif best_acc >= 80:
            st.success(f"✅ グッジョブ！ 精度 {best_acc:.1f}% — 学習率やエポック数を調整してさらに改善できるか試してみよう！")
        elif result["dataset_key"] != "xor":
            st.warning(f"⚠️ 精度 {best_acc:.1f}% — まだ改善の余地あり。学習率やエポック数を変えてみよう！")

        st.info(f"🥇 今セッション最高精度：**{st.session_state['best_accuracy_3']:.1f}%**")

        st.markdown("""
        <div class="explanation-box">
        <h3>📖 学習曲線の読み方</h3>
        <ul>
        <li><b>Loss が下がる</b>：AI が正解に近づいている証拠</li>
        <li><b>Accuracy が上がる</b>：正解できる割合が増えている</li>
        <li><b>学習率が大きすぎる</b>：Loss が乱高下して収束しない</li>
        <li><b>学習率が小さすぎる</b>：ゆっくり収束するが時間がかかる</li>
        </ul>
        最適な学習率を探す作業を「<b>ハイパーパラメータ調整</b>」と言います。これは AI エンジニアの核心スキルです！
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⬆️「学習スタート！」ボタンを押して AI を鍛えよう！")

# ---------------------------------------------------------------
# TAB 3: ネットワーク構造ビジュアライザー
# ---------------------------------------------------------------
with tab3:
    st.header("🕸️ ニューラルネットワークを自分で組もう！")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation-box">
        <h3>🏗️ 「深さ」が生み出す表現力</h3>
        単一ニューロンでは直線でしか分類できません（TAB 2 の XOR でそれを体験しましたね）。<br>
        ニューロンを複数の「層（レイヤー）」に積み重ねることで、
        AI は非常に複雑なパターンを学習できるようになります。<br>
        これが「<b>Deep Learning（深層学習）</b>」と呼ばれる理由です。
        </div>
        """, unsafe_allow_html=True)

    st.subheader("⚙️ アーキテクチャを設計しよう")

    arch_c1, arch_c2, arch_c3 = st.columns(3)
    with arch_c1:
        n_input   = st.slider("入力ノード数",     1, 8, 2, key="arch_in")
    with arch_c2:
        n_hidden1 = st.slider("隠れ層1 ノード数", 1, 8, 4, key="arch_h1")
        n_hidden2 = st.slider("隠れ層2 ノード数", 0, 8, 3, key="arch_h2")
    with arch_c3:
        n_output  = st.slider("出力ノード数",     1, 4, 1, key="arch_out")

    layer_sizes = [n_input, n_hidden1]
    if n_hidden2 > 0:
        layer_sizes.append(n_hidden2)
    layer_sizes.append(n_output)

    total_params = sum(
        layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        for i in range(len(layer_sizes) - 1)
    )

    fig_nn = plot_nn_diagram(layer_sizes)
    st.pyplot(fig_nn)
    plt.close(fig_nn)

    stat_c1, stat_c2, stat_c3 = st.columns(3)
    stat_c1.metric("総レイヤー数",        f"{len(layer_sizes)}")
    stat_c2.metric("総ノード数",          f"{sum(layer_sizes)}")
    stat_c3.metric("学習可能パラメータ数", f"{total_params:,}")

    st.subheader("📊 有名なネットワークとの比較")
    cmp_cols = st.columns(4)
    famous_nets = [
        ("あなたのネット",  sum(layer_sizes), f"{total_params:,}",    "#ffd700"),
        ("AlexNet (2012)", 8,                "60,000,000",            "#05ffa1"),
        ("GPT-3 (2020)",  96,                "175,000,000,000",       "#00d4ff"),
        ("GPT-4 (2023)",  "???",             "〜1.7兆（推定）",        "#ff71ce"),
    ]
    for col, (name, layers, params, color) in zip(cmp_cols, famous_nets):
        col.markdown(f"""
        <div style="background:#1e2a3a; border:2px solid {color}; border-left:6px solid {color};
                    padding:12px; text-align:center;">
            <div style="color:{color}; font-weight:bold; font-size:0.9rem;">{name}</div>
            <div style="color:#e6edf3; font-size:0.8rem; margin-top:6px;">層数: {layers}</div>
            <div style="color:#8b9ab0; font-size:0.75rem;">パラメータ: {params}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation-box" style="margin-top:16px;">
    <h3>💡 なぜ「深い」ネットワークが強いのか？</h3>
    <ul>
    <li><b>層 1</b>：シンプルな特徴（エッジ・色の変化）を学ぶ</li>
    <li><b>層 2</b>：複合的な特徴（目・鼻・テクスチャ）を学ぶ</li>
    <li><b>層 3 以降</b>：抽象的な概念（「これは犬の顔だ」）を学ぶ</li>
    </ul>
    階層的に学ぶことで、少ないデータでも複雑なパターンを汎化できます。
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔮 フォワードパスを体験しよう")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.caption("入力値を設定すると、最初の隠れ層で値がどう変わるかを計算します（重みはランダム初期化の例）。")

        fp_cols_list = st.columns(min(n_input, 4))
        x_inputs = []
        for i, col in enumerate(fp_cols_list[:n_input]):
            val = col.slider(f"x{i+1}", -2.0, 2.0, float(i) * 0.5 - 0.5, 0.1, key=f"fp_x{i}")
            x_inputs.append(val)

        np.random.seed(42)
        x_vec = np.array(x_inputs)
        W1 = np.random.randn(n_hidden1, n_input) * 0.5
        h1 = np.tanh(W1 @ x_vec + np.zeros(n_hidden1))

        fp_c1, fp_c2 = st.columns(2)
        with fp_c1:
            in_str = " | ".join([f'<span style="color:#00d4ff;">x{i+1}={v:.2f}</span>' for i, v in enumerate(x_inputs)])
            st.markdown(f"""
            <div style="background:#0d1117; border:1px solid #3d5a8a; padding:12px;
                        font-family:monospace; font-size:0.85rem;">
            <span style="color:#ffd700;">入力層</span><br>{in_str}
            </div>
            """, unsafe_allow_html=True)
        with fp_c2:
            h1_str = " | ".join([f'<span style="color:#05ffa1; font-weight:bold;">{v:.2f}</span>' for v in h1])
            st.markdown(f"""
            <div style="background:#0d1117; border:1px solid #3d5a8a; padding:12px;
                        font-family:monospace; font-size:0.85rem;">
            <span style="color:#ffd700;">隠れ層1（tanh 活性化後）</span><br>{h1_str}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 4: 過学習の罠
# ---------------------------------------------------------------
with tab4:
    st.header("🪤 過学習の罠に落ちるな！")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        ov_text, ov_box = st.columns([3, 2])

        with ov_text:
            st.markdown("""
            <div class="explanation-box">
            <h3>😈 過学習（Overfitting）とは？</h3>
            AI が「訓練データを丸暗記」してしまい、初めて見るデータには全く対応できなくなる現象です。<br><br>
            テスト前に過去問だけを暗記した学生が、少し違う問題が出ると解けなくなるのと同じです。<br><br>
            <b>過学習の症状：</b>
            <ul>
            <li>訓練データの精度：超高い（≒ 100%）</li>
            <li>テストデータの精度：低い（汎化できていない）</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with ov_box:
            st.markdown("""
            <div style="background:#0d1117; border:2px solid #ff4555; padding:18px;
                        font-family:monospace; border-left:6px solid #ffd700;">
                <div style="color:#ffd700; font-weight:bold; margin-bottom:10px;">▶ 過学習の比較</div>
                <div style="color:#05ffa1;">良いモデル</div>
                <div style="color:#e6edf3; font-size:0.82rem;">訓練精度: 92% / テスト精度: 90%</div>
                <div style="color:#8b9ab0; font-size:0.78rem; margin-bottom:10px;">→ バランスよく汎化できている</div>
                <div style="color:#ff4555;">過学習モデル</div>
                <div style="color:#e6edf3; font-size:0.82rem;">訓練精度: 99% / テスト精度: 55%</div>
                <div style="color:#8b9ab0; font-size:0.78rem;">→ 訓練データを丸暗記している</div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("🔬 過学習シミュレーター")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        ov_c1, ov_c2 = st.columns(2)
        with ov_c1:
            poly_degree = st.slider("モデルの複雑さ（多項式の次数）", 1, 12, 3, key="poly_deg")
            noise_level = st.slider("データのノイズ量",              0.05, 1.0, 0.3, 0.05, key="noise_lv")
        with ov_c2:
            st.markdown("""
            <div style="background:#1e2a3a; border:1px solid #3d5a8a; padding:14px; font-size:0.88rem;">
            <span style="color:#ffd700;">🎓 実験方法：</span>
            <ul style="color:#e6edf3; margin:8px 0;">
            <li>次数 1〜2：直線/2次曲線。シンプルすぎる（<b>未学習</b>）</li>
            <li>次数 3〜5：適度な複雑さ。<b>ちょうど良い</b></li>
            <li>次数 8〜12：過度な複雑さ。訓練点を全部通ろうとして<b>過学習</b></li>
            </ul>
            ノイズを増やしながら次数を上げると、過学習がより顕著になります！
            </div>
            """, unsafe_allow_html=True)

        fig_ov, train_mse, test_mse = plot_overfitting(poly_degree, noise_level)
        st.pyplot(fig_ov)
        plt.close(fig_ov)

        ov_m1, ov_m2, ov_m3 = st.columns(3)
        ov_m1.metric("訓練 MSE（低いほど良）",  f"{train_mse:.4f}")
        ov_m2.metric("テスト MSE（低いほど良）", f"{test_mse:.4f}",
                     f"{test_mse - train_mse:+.4f}" if train_mse > 0 else "")
        overfit_ratio = test_mse / max(train_mse, 1e-6)
        ov_m3.metric("過学習スコア（1 に近いほど良）", f"{overfit_ratio:.2f}")

        if overfit_ratio > 5:
            st.error(f"❌ 重篤な過学習！ テスト MSE が訓練 MSE の {overfit_ratio:.1f} 倍です。次数を下げてください！")
        elif overfit_ratio > 2:
            st.warning(f"⚠️ 過学習の兆候。次数 {poly_degree} はこのデータには複雑すぎるかもしれません。")
        elif poly_degree <= 2 and test_mse > 0.3:
            st.warning("💡 未学習（Underfitting）状態です。モデルが単純すぎて真の関数を捉えられていません。")
        else:
            st.success("✅ バランスの良いフィット！ 訓練とテストの MSE が近い値です。")

        st.markdown("""
        <div class="explanation-box" style="margin-top:16px;">
        <h3>🛡️ 過学習を防ぐ主な方法</h3>
        <table style="color:#e6edf3; width:100%; font-size:0.88rem; border-collapse:collapse;">
        <tr style="border-bottom:1px solid #3d5a8a;">
            <th style="text-align:left; padding:6px; color:#ffd700;">手法</th>
            <th style="text-align:left; padding:6px; color:#ffd700;">概要</th>
        </tr>
        <tr style="border-bottom:1px solid #3d5a8a;">
            <td style="padding:6px; color:#05ffa1;">📊 データ拡張</td>
            <td style="padding:6px;">画像の反転・回転などで訓練データを人工的に増やす</td>
        </tr>
        <tr style="border-bottom:1px solid #3d5a8a;">
            <td style="padding:6px; color:#05ffa1;">💧 ドロップアウト</td>
            <td style="padding:6px;">学習中にランダムでニューロンを無効化し、依存関係を壊す</td>
        </tr>
        <tr style="border-bottom:1px solid #3d5a8a;">
            <td style="padding:6px; color:#05ffa1;">⚖️ 正則化（L1/L2）</td>
            <td style="padding:6px;">重みが大きくなりすぎないようにペナルティを課す</td>
        </tr>
        <tr>
            <td style="padding:6px; color:#05ffa1;">🛑 早期終了</td>
            <td style="padding:6px;">検証データの精度が下がり始めたら学習を止める</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🎓 学習のまとめ：AI 育成マスターへの道")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        summary_cards = [
            ("🧠", "ニューロン",    "AI の最小単位。重み × 入力の合計を活性化関数に通して発火/静止を決める",              "#00d4ff"),
            ("📉", "勾配降下法",   "損失関数の坂を下る方向に重みを少しずつ修正する最適化アルゴリズム",                    "#05ffa1"),
            ("🕸️", "深層学習",     "多くの層を重ねることで非線形な複雑パターンを学習できるようになる",                    "#ffd700"),
            ("⚠️", "過学習",       "訓練データへの過剰適合。防ぐには正則化・ドロップアウト・データ拡張などが有効",         "#ff71ce"),
        ]
        summ_cols = st.columns(4)
        for col, (icon, title, desc, color) in zip(summ_cols, summary_cards):
            col.markdown(f"""
            <div style="background:#1e2a3a; border:2px solid {color}; border-left:6px solid {color};
                        padding:14px; min-height:160px;">
                <div style="font-size:1.8rem; text-align:center;">{icon}</div>
                <div style="color:{color}; font-weight:bold; text-align:center; margin:6px 0;">{title}</div>
                <div style="color:#e6edf3; font-size:0.82rem; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ================================================================
# SECTION 10: フッター
# ================================================================
st.markdown("""
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。掌握する。育てる。</p>
    <p>MISSION 03: AI育成 — ニューロンの誕生から深層学習まで</p>
</div>
""", unsafe_allow_html=True)
