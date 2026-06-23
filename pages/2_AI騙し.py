import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

try:
    import torch
    import torchvision.models as tv_models
    from torchvision.models import MobileNet_V2_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ================================================================
# SECTION 0: ページ設定（必ず最初のst.*コール）
# ================================================================
st.set_page_config(
    page_title="ミッション02: AI騙し | AI Inquiry Lab",
    page_icon="🎭",
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
DATA_DIR   = os.path.join(PARENT_DIR, "data")
ASSETS_DIR = os.path.join(PARENT_DIR, "assets")
CSS_FILE   = os.path.join(ASSETS_DIR, "style.css")

SAMPLE_IMAGES = {
    "🐦 とり":  "bird.jpg",
    "🐈 ねこ":  "cat.jpg",
    "🚦 標識":  "hyousiki.jpg",
    "🏢 建物":  "building.jpg",
    "🏔 山":    "mountain.jpg",
    "⚽ ボール": "soccerball.jpg",
}

ATTACK_INFO = {
    "🌪️ ガウスノイズ": {
        "key": "gaussian",
        "desc": "ランダムなノイズを全ピクセルに加算。自然界の撮影ノイズに似た形状で、シンプルかつ汎用的な攻撃手法です。",
    },
    "🧂 塩コショウ": {
        "key": "salt_pepper",
        "desc": "ランダムに白（塩）と黒（胡椒）の点を散りばめます。デジタル信号の欠損ノイズを再現した古典的な手法です。",
    },
    "⚡ 疑似FGSM": {
        "key": "pseudo_fgsm",
        "desc": "画像の「エッジ（境目）」の方向に沿った構造的なノイズ。本物のFGSMが利用する勾配方向を近似した、最も賢い攻撃です。",
    },
    "🌈 カラーシフト": {
        "key": "color_shift",
        "desc": "赤・緑・青の各色成分を微妙にずらします。人間の目には色が同じに見えても、AIの数値判断を混乱させます。",
    },
}

# ================================================================
# SECTION 3: ユーティリティ関数
# ================================================================
def load_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSSファイルが見つかりません: {path}")

def get_image_as_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ================================================================
# SECTION 4: AIモデル（推論エンジン）
# ================================================================
if TORCH_AVAILABLE:
    @st.cache_resource
    def load_model():
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = tv_models.mobilenet_v2(weights=weights)
        model.eval()
        return model, weights.meta["categories"], weights.transforms()
    _model, _categories, _preprocess = load_model()
else:
    _model, _categories, _preprocess = None, [], None

def predict_image(img_array: np.ndarray) -> tuple:
    """画像をAIに推論させ (ラベル, 確信度, Top5リスト) を返す"""
    if not TORCH_AVAILABLE or _model is None:
        return "AI利用不可", 0.0, []
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    input_tensor = _preprocess(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = _model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_probs, top5_idx = torch.topk(probs, 5)
    top_label = _categories[top5_idx[0].item()]
    top_conf  = float(top5_probs[0].item() * 100)
    top5 = [
        (_categories[top5_idx[i].item()], float(top5_probs[i].item() * 100))
        for i in range(5)
    ]
    return top_label, top_conf, top5

# ================================================================
# SECTION 5: 画像攻撃関数群
# ================================================================
def attack_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def attack_salt_pepper(img: np.ndarray, density: float) -> np.ndarray:
    result = img.copy()
    h, w = img.shape[:2]
    mask_salt   = np.random.random((h, w)) < density / 2
    mask_pepper = np.random.random((h, w)) < density / 2
    result[mask_salt]   = 255
    result[mask_pepper] = 0
    return result

def attack_pseudo_fgsm(img: np.ndarray, epsilon: float) -> np.ndarray:
    """FGSMを近似した、エッジ勾配方向への構造的ノイズ"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_sign = np.sign(gx + gy)
    perturbed = np.stack([
        img[:, :, 0].astype(np.float32) + epsilon * gradient_sign,
        img[:, :, 1].astype(np.float32) + epsilon * gradient_sign * 0.85,
        img[:, :, 2].astype(np.float32) + epsilon * gradient_sign * 1.1,
    ], axis=2)
    return np.clip(perturbed, 0, 255).astype(np.uint8)

def attack_color_shift(img: np.ndarray, strength: float) -> np.ndarray:
    """色チャンネルをランダム方向にシフトして混乱させる"""
    result = img.astype(np.float32).copy()
    shifts = np.random.choice([-strength, strength], size=(3,))
    for ch, shift in enumerate(shifts):
        result[:, :, ch] = np.clip(result[:, :, ch] + shift, 0, 255)
    return result.astype(np.uint8)

def apply_attack(img: np.ndarray, key: str, strength: float) -> np.ndarray:
    if key == "gaussian":
        return attack_gaussian(img, sigma=strength * 1.5)
    elif key == "salt_pepper":
        return attack_salt_pepper(img, density=strength / 500)
    elif key == "pseudo_fgsm":
        return attack_pseudo_fgsm(img, epsilon=strength)
    elif key == "color_shift":
        return attack_color_shift(img, strength=strength)
    return img

def compute_noise_visibility(original: np.ndarray, perturbed: np.ndarray) -> float:
    diff = np.abs(original.astype(np.float32) - perturbed.astype(np.float32))
    return float(np.mean(diff) / 255 * 100)

def judge_deception(
    orig_conf: float, pert_conf: float,
    orig_label: str, pert_label: str
) -> tuple:
    """騙しの成否を判定してメッセージ・カラーを返す"""
    label_changed = orig_label != pert_label
    conf_drop     = orig_conf - pert_conf

    if label_changed and conf_drop > 30:
        return "🎉 大成功！AIを完全に騙した！",      "#05ffa1"
    elif label_changed:
        return "✅ 成功！AIが別の物を見ている！",    "#05ffa1"
    elif conf_drop > 20:
        return "⚠️ 部分成功！AIが迷い始めている！",  "#ffd700"
    elif conf_drop > 5:
        return "🔄 惜しい！AIが揺れている",          "#00d4ff"
    else:
        return "🛡️ 失敗…AIは騙せなかった",           "#ff4555"

def draw_confidence_bar(top5: list, highlight_label: str, bar_color: str) -> plt.Figure:
    """Top-5確信度の横棒グラフを生成して返す"""
    labels = [f"{l[:18]}..." if len(l) > 18 else l for l, _ in top5][::-1]
    values = [v for _, v in top5][::-1]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#1e2a3a")
    ax.set_facecolor("#1e2a3a")
    bars = ax.barh(labels, values, color=[
        "#ff71ce" if (labels[::-1][i] == highlight_label or
                      (labels[::-1][i] + "...") == highlight_label[:18] + "...")
        else bar_color
        for i in range(len(labels))
    ][::-1])
    ax.set_xlim(0, 100)
    ax.tick_params(colors="#e6edf3", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3d5a8a")
    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 95), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#e6edf3", fontsize=7)
    plt.tight_layout(pad=0.5)
    return fig

# ================================================================
# SECTION 6: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)

st.markdown("""
<div class="main-title-container">
    <h1 class="main-title-text">🎭 AI騙し</h1>
    <p class="sub-title-text">MISSION 02 — 微小な「嘘」でAIの認識を崩壊させろ</p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 7: セッション状態の初期化
# ================================================================
if "target_img_path" not in st.session_state:
    for fn in SAMPLE_IMAGES.values():
        fp = os.path.join(DATA_DIR, fn)
        if os.path.exists(fp):
            st.session_state["target_img_path"] = fp
            break

if "use_upload_2"  not in st.session_state:
    st.session_state["use_upload_2"] = False
if "best_score_2"  not in st.session_state:
    st.session_state["best_score_2"] = 0

# ================================================================
# SECTION 8: サイドバー（攻撃設定センター）
# ================================================================
with st.sidebar:
    st.markdown("""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#ff71ce; font-weight:bold; font-size:0.9rem;">🎭 AI HACKING MODE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected_attack_name = st.selectbox(
        "🗡️ 攻撃タイプを選択",
        list(ATTACK_INFO.keys()),
        key="attack_select"
    )
    selected_attack_key  = ATTACK_INFO[selected_attack_name]["key"]

    attack_strength = st.slider("💪 攻撃強度", min_value=1, max_value=100, value=20)

    st.divider()
    st.info(f"💡 **{selected_attack_name}**\n\n{ATTACK_INFO[selected_attack_name]['desc']}")

    st.divider()
    st.markdown("### 🧭 Navigation")
    st.page_link("main_app.py",            label="司令室 (Home)",         icon="🏠")
    st.page_link("pages/1_AIの目.py",      label="ミッション01: AIの目",  icon="👁️")
    st.page_link("pages/2_AI騙し.py",     label="ミッション02: AI騙し",  icon="🎭")
    st.divider()
    st.success("🛡️ SHIELD: ONLINE")

# ================================================================
# SECTION 9: STEP 1 — 作戦対象の画像選択
# ================================================================
st.header("📸 STEP 1：作戦対象を選択せよ")

col_select, col_preview = st.columns([2, 1])

with col_select:
    tab_sample, tab_upload = st.tabs(["📂 サンプルから選ぶ", "📤 自分の画像を使う"])

    with tab_sample:
        btn_cols = st.columns(3)
        for idx, (label, filename) in enumerate(SAMPLE_IMAGES.items()):
            filepath = os.path.join(DATA_DIR, filename)
            if btn_cols[idx % 3].button(label, key=f"img_btn_{idx}", use_container_width=True):
                st.session_state["target_img_path"] = filepath
                st.session_state["use_upload_2"]    = False
                st.rerun()

    with tab_upload:
        uploaded = st.file_uploader(
            "画像をアップロード", type=["jpg", "png", "jpeg"], key="file_up_2"
        )
        if uploaded:
            st.session_state["uploaded_img_2"] = uploaded
            st.session_state["use_upload_2"]   = True
            st.rerun()

# 画像の確定
if st.session_state["use_upload_2"] and st.session_state.get("uploaded_img_2"):
    target_image = Image.open(st.session_state["uploaded_img_2"]).convert("RGB")
elif "target_img_path" in st.session_state and os.path.exists(st.session_state["target_img_path"]):
    target_image = Image.open(st.session_state["target_img_path"]).convert("RGB")
else:
    target_image = None

with col_preview:
    if target_image:
        st.image(target_image, caption="選択中の画像", use_container_width=True)
    else:
        st.warning("有効な画像が見つかりません")

# ================================================================
# SECTION 10: STEP 2 — メインコンテンツ（4タブ構成）
# ================================================================
st.header("🧪 STEP 2：ミッション開始")

tab1, tab2, tab3, tab4 = st.tabs([
    "① AIの弱点を知る",
    "② 騙しシミュレーター",
    "③ 攻撃の仕組み",
    "④ 防御と未来",
])

# ---------------------------------------------------------------
# TAB 1: AIの弱点を知る
# ---------------------------------------------------------------
with tab1:
    st.header("🧠 なぜAIは騙されるのか？")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        col_text, col_box = st.columns([3, 2])
        with col_text:
            st.markdown("""
            <div class="explanation-box">
            <h3>🎭 AIと人間の「見え方」の根本的な違い</h3>
            人間は物体の<b>「形（シルエット）」</b>を見て直感的に判断します。
            パンダを見るとき、黒と白のパターン・丸い体形・耳の形を無意識に総合して判断しています。<br><br>
            しかし現代のAI（畳み込みニューラルネットワーク）は、<b>「テクスチャ（模様・質感）」</b>に
            強く依存して判断する傾向があります。

            <h3>💥 ほんの少しの変化で騙せる理由</h3>
            AIが判断に使っている数値は、人間の目には見えないほど微小な変化に<b>極めて敏感</b>です。<br>
            画像全体の変化量がわずか<b>0.007%</b>のノイズを加えるだけで、AIの内部計算が狂い、
            全く別の物体として認識してしまうことがあります。
            </div>
            """, unsafe_allow_html=True)

        with col_box:
            st.markdown("""
            <div style="background:#0d1117; border:2px solid #00d4ff; padding:20px;
                        font-family:monospace; border-left:6px solid #ff71ce;">
                <div style="color:#ffd700; font-weight:bold; margin-bottom:12px;">
                    ▶ ADVERSARIAL DEMO
                </div>
                <div style="color:#05ffa1;">元画像: 🐼 パンダ</div>
                <div style="color:#e6edf3; margin:4px 0;">確信度: 99.3%</div>
                <br>
                <div style="color:#ff4555; font-size:1.3rem;">＋ ε ノイズ</div>
                <div style="color:#8b9ab0; font-size:0.8rem;">（人間には見えない変化量）</div>
                <br>
                <div style="color:#05ffa1;">↓ AIの判定が変わる</div>
                <br>
                <div style="color:#ff71ce; font-weight:bold;">認識結果: テナガザル 🦧</div>
                <div style="color:#e6edf3; margin:4px 0;">確信度: 94.1%</div>
                <br>
                <div style="border-top:1px solid #3d5a8a; padding-top:10px;
                            color:#8b9ab0; font-size:0.78rem;">
                    ← Goodfellow et al., 2014<br>
                    最初に発見された対抗的サンプル
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.header("📊 AIが騙される3つの根本理由")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="guide-box">
                <h4>🎨 テクスチャ偏重</h4>
                <p>CNNは模様・質感を過度に重視します。
                「ゾウの皮膚の質感」でゾウと判断するため、
                質感を少し変えるだけで騙せます。</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="guide-box">
                <h4>🔢 高次元の脆弱性</h4>
                <p>画像は数百万次元の空間で表現されます。
                人間には見えない方向へほんの少し動かすだけで
                AIの「判断境界線」を越えられます。</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="guide-box">
                <h4>⚡ 勾配の逆用</h4>
                <p>AIは「どの方向に変化させると最も迷うか」を
                計算できます。攻撃者はこれを逆用して
                最小限の変化で最大の混乱を引き起こします。</p>
            </div>
            """, unsafe_allow_html=True)

    st.header("🧩 人間 vs AI — 認識の仕組み比較")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        comp_cols = st.columns(2)
        with comp_cols[0]:
            st.markdown("""
            <div style="background:#1e2a3a; border:2px solid #05ffa1;
                        border-left:6px solid #05ffa1; padding:18px;">
                <h3 style="color:#05ffa1; background:transparent !important;
                           border:none !important; padding:0 !important;
                           box-shadow:none !important; margin:0 0 12px 0;">
                    🧑 人間の認識
                </h3>
                <ul style="color:#e6edf3; line-height:2;">
                    <li>形・シルエットを重視</li>
                    <li>少々ノイズがあっても正しく認識</li>
                    <li>文脈・前後関係から推測</li>
                    <li>「これは何？」を直感的に判断</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with comp_cols[1]:
            st.markdown("""
            <div style="background:#1e2a3a; border:2px solid #ff71ce;
                        border-left:6px solid #ff71ce; padding:18px;">
                <h3 style="color:#ff71ce; background:transparent !important;
                           border:none !important; padding:0 !important;
                           box-shadow:none !important; margin:0 0 12px 0;">
                    🤖 AIの認識
                </h3>
                <ul style="color:#e6edf3; line-height:2;">
                    <li>テクスチャ・模様を重視</li>
                    <li>微小な数値変化に過敏</li>
                    <li>文脈を持たず数値のみで判断</li>
                    <li>「確率が最大の答え」を出力</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    st.header("🔬 AIの弱点を実際に体験してみよう！")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        if target_image is None:
            st.info("⬆️ STEP 1 で画像を選択すると、AIの弱点をインタラクティブに体験できます！")
        else:
            img_arr_weak = np.array(target_image.resize((400, 400)))

            experiment = st.radio(
                "🔬 試してみる弱点を選択：",
                ["🌪️ ノイズ耐性", "🔄 回転耐性", "🌗 コントラスト依存"],
                horizontal=True,
                key="weakness_exp"
            )

            test_img = img_arr_weak.copy()

            if "🌪️" in experiment:
                noise_level = st.slider("ノイズ強度（大きいほど荒くなる）", 0, 128, 20, key="noise_slider_w")
                if noise_level > 0:
                    noise    = np.random.normal(0, noise_level, img_arr_weak.shape)
                    test_img = np.clip(img_arr_weak.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                st.caption("💡 高周波ノイズはCNNの特徴抽出を破壊します。")

            elif "🔄" in experiment:
                angle = st.slider("回転角度（度）", 0, 180, 45, key="rotate_slider_w")
                if angle > 0:
                    h, w     = img_arr_weak.shape[:2]
                    M        = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    test_img = cv2.warpAffine(img_arr_weak, M, (w, h))
                st.caption("💡 多くのCNNは回転不変性を完全には持ちません。")

            else:
                contrast = st.slider("コントラスト倍率", 0.1, 2.0, 0.3, step=0.05, key="contrast_slider_w")
                test_img = cv2.convertScaleAbs(img_arr_weak, alpha=contrast, beta=0)
                st.caption("💡 低コントラストはエッジ情報を弱め、AIを迷わせます。")

            col_orig_w, col_test_w = st.columns(2)

            with col_orig_w:
                st.markdown("""
                <div class="img-frame">
                    <span class="step-badge">ORIGINAL</span>
                    <div class="caption-text">元の画像</div>
                </div>""", unsafe_allow_html=True)
                st.image(img_arr_weak, use_container_width=True)
                if TORCH_AVAILABLE:
                    orig_lbl_w, orig_pct_w = predict_image(img_arr_weak)[:2]
                    st.metric("AIの判定", orig_lbl_w[:24] if len(orig_lbl_w) <= 24 else orig_lbl_w[:21] + "…")
                    st.metric("確信度",   f"{orig_pct_w:.1f}%")

            with col_test_w:
                st.markdown("""
                <div class="img-frame frame-phase2">
                    <span class="step-badge">MODIFIED</span>
                    <div class="caption-text">変化後の画像</div>
                </div>""", unsafe_allow_html=True)
                st.image(test_img, use_container_width=True)
                if TORCH_AVAILABLE:
                    test_lbl_w, test_pct_w = predict_image(test_img)[:2]
                    delta_w = test_pct_w - orig_pct_w
                    st.metric("AIの判定", test_lbl_w[:24] if len(test_lbl_w) <= 24 else test_lbl_w[:21] + "…")
                    st.metric("確信度",   f"{test_pct_w:.1f}%", f"{delta_w:+.1f}%")

            if TORCH_AVAILABLE:
                if orig_lbl_w != test_lbl_w:
                    st.error(f"⚠️ AIが「{orig_lbl_w}」→「{test_lbl_w}」へ誤認識しました！")
                elif abs(orig_pct_w - test_pct_w) > 10:
                    st.warning(f"⚠️ AIの確信度が{abs(orig_pct_w - test_pct_w):.1f}%も変化しました！")
                else:
                    st.success("✅ AIはまだ正確に認識しています。スライダーを動かして攻撃を強めてみよう！")
            else:
                st.info("💡 ② 騙しシミュレーター タブでAIへの実際の影響を体験できます！")

# ---------------------------------------------------------------
# TAB 2: 騙しシミュレーター
# ---------------------------------------------------------------
with tab2:
    st.header("🎮 AI騙しチャレンジ！")

    if target_image is None:
        st.error("先に STEP 1 で画像を選択してください！")
        st.stop()

    img_array = np.array(target_image.resize((400, 400)))

    # --- 攻撃を実行 ---
    perturbed_array = apply_attack(img_array, selected_attack_key, attack_strength)

    # --- ノイズ可視化（10倍増幅）---
    diff = np.abs(img_array.astype(np.float32) - perturbed_array.astype(np.float32))
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

    # --- AI推論 ---
    visibility    = compute_noise_visibility(img_array, perturbed_array)
    orig_label, orig_conf, orig_top5 = predict_image(img_array)
    pert_label, pert_conf, pert_top5 = predict_image(perturbed_array)
    conf_drop     = max(0.0, orig_conf - pert_conf)
    result_msg, result_color = judge_deception(orig_conf, pert_conf, orig_label, pert_label)

    # ===== 画像3枚の比較表示 =====
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        col_orig, col_pert, col_diff = st.columns(3)

        with col_orig:
            st.markdown("""
            <div class="img-frame">
                <span class="step-badge">ORIGINAL</span>
                <div class="caption-text">元の画像</div>
            </div>""", unsafe_allow_html=True)
            st.image(img_array, use_container_width=True)
            if TORCH_AVAILABLE:
                st.metric("AIの判定", orig_label[:22] if len(orig_label) <= 22 else orig_label[:19] + "…")
                st.metric("確信度", f"{orig_conf:.1f}%")
            else:
                st.info("（AI推論なし）")

        with col_pert:
            st.markdown("""
            <div class="img-frame frame-phase2">
                <span class="step-badge">ATTACKED</span>
                <div class="caption-text">攻撃後の画像</div>
            </div>""", unsafe_allow_html=True)
            st.image(perturbed_array, use_container_width=True)
            if TORCH_AVAILABLE:
                st.metric("AIの判定", pert_label[:22] if len(pert_label) <= 22 else pert_label[:19] + "…")
                st.metric("確信度", f"{pert_conf:.1f}%", f"{pert_conf - orig_conf:+.1f}%")

        with col_diff:
            st.markdown("""
            <div class="img-frame frame-result">
                <span class="step-badge badge-result">NOISE ×10</span>
                <div class="caption-text" style="color:#c0392b;">ノイズ可視化（10倍）</div>
            </div>""", unsafe_allow_html=True)
            st.image(diff_amplified, use_container_width=True)
            st.metric("ノイズ視認性", f"{visibility:.3f}%")
            st.caption("明るいほどノイズが大きい場所")

    # ===== 騙し判定バナー =====
    st.markdown(f"""
    <div style="background:#1e2a3a; border:3px solid {result_color};
                border-left:10px solid {result_color}; padding:20px 24px;
                text-align:center; margin:20px 0; box-shadow:6px 6px 0px #000;">
        <span style="color:{result_color}; font-size:1.4rem; font-weight:bold;
                     font-family:'DotGothic16', sans-serif;">
            {result_msg}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ===== 確信度グラフ =====
    if TORCH_AVAILABLE and orig_top5 and pert_top5:
        st.header("📊 AIの確信度の変化")
        with st.container():
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

            chart_col_l, chart_col_r = st.columns(2)
            with chart_col_l:
                st.markdown("**元画像に対する判定 Top-5**")
                fig = draw_confidence_bar(orig_top5, orig_label, "#00d4ff")
                st.pyplot(fig)
                plt.close(fig)
            with chart_col_r:
                st.markdown("**攻撃後の画像に対する判定 Top-5**")
                fig = draw_confidence_bar(pert_top5, orig_label, "#ff4555")
                st.pyplot(fig)
                plt.close(fig)

    # ===== スコアシステム =====
    st.header("🏆 騙し効率スコア")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        if TORCH_AVAILABLE:
            efficiency    = (conf_drop / max(visibility, 0.001)) * 10
            label_bonus   = 200 if orig_label != pert_label else 0
            total_score   = int(efficiency + label_bonus)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("確信度ダメージ",           f"{conf_drop:.1f}%")
            sc2.metric("ノイズ視認性（低いほど優秀）", f"{visibility:.3f}%")
            sc3.metric("🏆 効率スコア",             f"{total_score} pts")

            if total_score > st.session_state["best_score_2"]:
                st.session_state["best_score_2"] = total_score
                if total_score > 10:
                    st.balloons()

            st.info(f"🥇 今回の最高スコア：**{st.session_state['best_score_2']} pts**")

            st.markdown("""
            <div class="explanation-box">
            <h3>💡 スコアの計算式</h3>
            <b>スコア = (確信度下落幅 ÷ ノイズ視認性) × 10 ＋ ラベル変更ボーナス（+200）</b><br><br>
            目標は「人間が気付かないほど小さいノイズで、AIに大きなダメージを与えること」。<br>
            これは本物の対抗的サンプル攻撃の目的と全く同じです。
            強度スライダーを小さくして、攻撃タイプを変えて試してみよう！
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("PyTorch が利用できないためAI確信度は計算できません。画像の変化を目で楽しんでください！")

# ---------------------------------------------------------------
# TAB 3: 攻撃の仕組み
# ---------------------------------------------------------------
with tab3:
    st.header("⚡ 敵対的攻撃の仕組み")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explanation-box">
        <h3>🔬 FGSM とは何か？（Fast Gradient Sign Method）</h3>
        FGSMは2014年にGoogleのGoodfellow氏らが発見した、AIを最も効率よく騙す攻撃手法です。<br>
        その数式はシンプルです：

        <div style="background:#0d1117; border:2px solid #ffd700; padding:16px; margin:14px 0;
                    font-family:monospace; color:#ffd700; font-size:1.1rem; text-align:center;
                    letter-spacing:0.05em;">
            x_adv = x + ε × sign( ∇ₓ J(θ, x, y) )
        </div>

        <ul>
        <li><b>x</b>：元の画像（数値の集合）</li>
        <li><b>ε（イプシロン）</b>：ノイズの強さ。人間の目に見えないほど小さい値</li>
        <li><b>∇ₓJ</b>：「この方向に変えると最も損失が増える」という損失関数の勾配</li>
        <li><b>sign()</b>：プラスかマイナスかだけを取り出す関数</li>
        </ul>
        つまり、<b>「AIが一番間違えやすい方向に、ほんの少しだけ画像を動かす」</b>のが FGSM の本質です。
        </div>
        """, unsafe_allow_html=True)

    st.header("🗺️ 攻撃手法の分類図鑑")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        attack_cards = [
            {
                "name": "🌪️ ランダムノイズ",
                "badge": "ブラックボックス攻撃",
                "desc": "モデルの内部情報が不要なシンプルな攻撃。誰でも実行できるが効率は低い。",
                "color": "#00d4ff",
            },
            {
                "name": "⚡ FGSM",
                "badge": "ホワイトボックス攻撃",
                "desc": "モデルの勾配情報を活用した高効率攻撃。少ないノイズで最大のダメージ。研究の基準手法。",
                "color": "#ff71ce",
            },
            {
                "name": "🎯 標的型攻撃（Targeted）",
                "badge": "高度ホワイトボックス",
                "desc": "「パンダ→テナガザル」のように特定の誤分類を狙う攻撃。より多くの計算が必要。",
                "color": "#ffd700",
            },
            {
                "name": "🖨️ 物理的攻撃",
                "badge": "リアルワールド攻撃",
                "desc": "プリントしたパターンをカメラに見せる攻撃。自動運転の標識認識や顔認証を物理的に騙せる。",
                "color": "#ff4555",
            },
        ]

        card_col_l, card_col_r = st.columns(2)
        for i, card in enumerate(attack_cards):
            target_col = card_col_l if i % 2 == 0 else card_col_r
            with target_col:
                st.markdown(f"""
                <div style="background:#1e2a3a; border:2px solid {card['color']};
                            border-left:6px solid {card['color']}; padding:16px;
                            margin-bottom:16px; box-shadow:4px 4px 0px #000;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="color:{card['color']}; font-weight:bold; font-size:1rem;">
                            {card['name']}
                        </span>
                        <span style="background:#263548; color:#8b9ab0; padding:2px 8px;
                                     font-size:0.72rem; font-family:monospace;">
                            {card['badge']}
                        </span>
                    </div>
                    <p style="color:#e6edf3; font-size:0.88rem; margin:0; line-height:1.7;">
                        {card['desc']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.header("🚨 現実世界への脅威")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        threat_c1, threat_c2, threat_c3 = st.columns(3)
        with threat_c1:
            st.markdown("""
            <div class="guide-box">
                <h4>🚗 自動運転</h4>
                <p>道路の「一時停止」標識に特殊なシールを貼ると、AIが「時速45マイル制限」と
                誤認識する実験が報告されています。高速での事故につながりかねない深刻な問題です。</p>
            </div>
            """, unsafe_allow_html=True)
        with threat_c2:
            st.markdown("""
            <div class="guide-box">
                <h4>🏦 顔認証</h4>
                <p>特殊なパターンを印刷した眼鏡をかけることで、スマートフォンやセキュリティカメラの
                顔認証システムを別人として突破できることが研究で示されています。</p>
            </div>
            """, unsafe_allow_html=True)
        with threat_c3:
            st.markdown("""
            <div class="guide-box">
                <h4>🩺 医療診断AI</h4>
                <p>X線画像に微小なノイズを加えることで、正常な肺をAIに「肺炎」と誤診断させる
                攻撃が理論的に可能です。医療現場でのAI活用には特に慎重な対策が必要です。</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 4: 防御と未来
# ---------------------------------------------------------------
with tab4:
    st.header("🛡️ AIを守るために — 防御手法と倫理")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        defenses = [
            {
                "name": "🏋️ 敵対的訓練（Adversarial Training）",
                "desc": "攻撃された画像もデータセットに含めてAIを再学習させる方法。最も効果的な防御の一つですが、"
                        "学習コストが高く、見たことのない新しい攻撃手法には対応できないことがあります。",
                "level": 82, "color": "#05ffa1",
            },
            {
                "name": "🔍 攻撃検知モデル",
                "desc": "推論の前に「この画像は攻撃されているか？」を別のモデルで判定する方法。"
                        "検知した場合に警告・拒否できますが、検知モデル自体が攻撃される可能性もあります。",
                "level": 68, "color": "#00d4ff",
            },
            {
                "name": "🌊 入力平滑化（Input Smoothing）",
                "desc": "入力画像にぼかし処理をかけてノイズを除去してから推論する方法。"
                        "シンプルで高速ですが、画像の細部も失われるため精度が低下することがあります。",
                "level": 53, "color": "#ffd700",
            },
            {
                "name": "🎲 ランダム化防御",
                "desc": "推論のたびにランダムな変換を加えることで、攻撃者が狙える方向を予測不能にする方法。"
                        "同じ画像でも毎回少し異なる推論結果になります。",
                "level": 62, "color": "#ff71ce",
            },
        ]

        for d in defenses:
            d_col_l, d_col_r = st.columns([4, 1])
            with d_col_l:
                st.markdown(f"""
                <div style="background:#1e2a3a; border:2px solid {d['color']};
                            border-left:6px solid {d['color']}; padding:14px; margin-bottom:6px;">
                    <strong style="color:{d['color']};">{d['name']}</strong>
                    <p style="color:#e6edf3; margin:8px 0 0 0; font-size:0.88rem;">{d['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
            with d_col_r:
                st.markdown(f"<br>**有効度: {d['level']}%**", unsafe_allow_html=True)
                st.progress(d["level"] / 100)

    st.header("🌍 あなたはどう考える？— 倫理クイズ")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explanation-box">
        <h3>AIセキュリティは社会全体の問題</h3>
        AI騙しは「ハッカーの遊び」ではありません。自動運転・医療診断・セキュリティカメラ…
        AIが社会インフラになった今、<b>AIの脆弱性を理解することは社会全体の安全に直結</b>します。
        </div>
        """, unsafe_allow_html=True)

        ethics_q = st.radio(
            "もし自動運転AIの重大な脆弱性を発見したら、あなたはどうしますか？",
            [
                "🔐 自動車メーカーに非公開で報告し、修正後に発表を許可する",
                "📢 すぐにSNSで公開して広く知らせる",
                "💰 脆弱性情報を買い取る業者に売る",
                "🔬 修正を待たずにすぐ学術論文として発表する",
            ],
            key="ethics_radio"
        )

        ethics_responses = {
            "🔐 自動車メーカーに非公開で報告し、修正後に発表を許可する": (
                "✅ 正解です！これが「Responsible Disclosure（責任ある開示）」と呼ばれる国際的に推奨されている倫理的アプローチです。"
                "メーカーが修正する時間を確保しながら、最終的に公開することで社会全体に貢献できます。"
                "多くの大企業は「バグバウンティプログラム」という報奨金制度を設けています。",
                "success"
            ),
            "📢 すぐにSNSで公開して広く知らせる": (
                "⚠️ 気持ちは分かりますが、即座の公開は悪意ある攻撃者にも情報が届くリスクがあります。"
                "修正が完了する前に実害が出る可能性があるため、まず開発者への通知が優先されます。",
                "warning"
            ),
            "💰 脆弱性情報を買い取る業者に売る": (
                "❌ 正規のバグバウンティプログラム以外での脆弱性売買は、法的・倫理的に問題があります。"
                "悪意ある攻撃者への販売は刑事責任につながる可能性があります。",
                "error"
            ),
            "🔬 修正を待たずにすぐ学術論文として発表する": (
                "💡 学術発表は重要ですが、一般的には事前に開発者へ通知し、修正完了後に発表する"
                "「Coordinated Disclosure（調整された開示）」が推奨されます。多くの学術誌もこの手順を求めています。",
                "info"
            ),
        }

        if ethics_q in ethics_responses:
            msg, level = ethics_responses[ethics_q]
            getattr(st, level)(msg)

    st.header("🚀 AIセキュリティの未来")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explanation-box">
        <h3>AIを「騙しにくく」する研究の最前線</h3>
        <ul>
        <li><b>形状バイアスの訓練</b>：テクスチャではなく「形」で判断するよう意図的に訓練する研究</li>
        <li><b>証明可能な堅牢性</b>：「εの範囲内では絶対に騙されない」と数学的に証明できるAIの開発</li>
        <li><b>人間と同じ知覚</b>：人間の視覚システムを真似た、よりロバストなアーキテクチャの研究</li>
        <li><b>量子機械学習</b>：量子計算を使った、古典的攻撃が通用しないAIシステムの理論研究</li>
        </ul>

        <h3>あなたが学べること</h3>
        AIの弱点を知ることは、AIを「正しく使う」ための第一歩です。<br>
        騙されるAIをただ批判するのではなく、<b>なぜ騙されるのか・どう防ぐのかを理解する人</b>こそ、
        これからのAI時代をリードできる人材です。
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# SECTION 11: フッター
# ================================================================
st.markdown("""
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。掌握する。</p>
    <p>MISSION 02: AI騙し — 対抗的サンプルの世界へようこそ</p>
</div>
""", unsafe_allow_html=True)
