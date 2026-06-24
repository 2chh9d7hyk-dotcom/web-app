import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import base64

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
    page_title="ミッション01: AIの目 | AI Inquiry Lab",
    page_icon="👁️",
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


# ================================================================
# SECTION 3: ユーティリティ関数
# ================================================================
def load_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_image_as_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def create_dummy_image(text: str, color: tuple) -> Image.Image:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = color
    cv2.putText(img, text, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return Image.fromarray(img)

# ================================================================
# SECTION 4: AIモデル（推論エンジン）
# ================================================================
if TORCH_AVAILABLE:
    @st.cache_resource
    def load_model():
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        m = tv_models.mobilenet_v2(weights=weights)
        m.eval()
        return m, weights.meta["categories"], weights.transforms()
    _model, _categories, _preprocess = load_model()
else:
    _model, _categories, _preprocess = None, [], None

def predict_image(img_array: np.ndarray) -> tuple:
    """画像を推論して (ラベル, 確信度%) を返す"""
    if not TORCH_AVAILABLE or _model is None:
        return "AI機能は現在利用できません", 0.0
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    input_tensor = _preprocess(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = _model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_idx = torch.max(probs, 0)
    return _categories[top_idx.item()], top_prob.item() * 100

# ================================================================
# SECTION 5: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)

st.markdown("""
<nav class="top-nav">
  <span class="nav-brand">🧠 AI LAB</span>
  <input type="checkbox" id="nav-toggle" class="nav-toggle-input">
  <label for="nav-toggle" class="hamburger-btn">☰ Menu</label>
  <div class="nav-links">
    <a href="javascript:void(0)" onclick="window.location.href='/'" class="nav-link">🏠 Home</a>
    <a href="javascript:void(0)" onclick="window.location.href='/AI%E3%81%AE%E7%9B%AE'" class="nav-link nav-active">👁️ M01: AIの目</a>
    <a href="javascript:void(0)" onclick="window.location.href='/AI%E9%A8%99%E3%81%97'" class="nav-link">🎭 M02: AI騙し</a>
    <a href="javascript:void(0)" onclick="window.location.href='/AI%E8%82%B2%E6%88%90'" class="nav-link">🧬 M03: AI育成</a>
  </div>
</nav>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title-container">
    <h1 class="main-title-text">👁️ AIの目</h1>
    <p class="sub-title-text">MISSION 01 — AIは画像をどうやって「見て」いるのか？</p>
</div>
""", unsafe_allow_html=True)


# ================================================================
# SECTION 7: サイドバー
# ================================================================
with st.sidebar:
    st.markdown("""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#3b82f6; font-weight:bold; font-size:0.9rem;">👁️ VISION MODE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.info("💡 **ヒント** AIはピクセルごとの「数値」を見ています。")
    st.divider()
    st.markdown("### 🧭 Navigation")
    st.page_link("main_app.py",          label="司令室 (Home)",        icon="🏠")
    st.page_link("pages/1_AIの目.py",    label="ミッション01: AIの目", icon="👁️")
    st.page_link("pages/2_AI騙し.py",   label="ミッション02: AI騙し", icon="🎭")
    st.page_link("pages/3_AI育成.py",   label="ミッション03: AI育成", icon="🧬")
    st.divider()
    st.success("🛡️ SHIELD: ONLINE")

# ================================================================
# SECTION 8: 画像ロード（固定: 鳥の画像を使用）
# ================================================================
bird_path = os.path.join(DATA_DIR, "bird.jpg")
image = Image.open(bird_path).convert("RGB")

# ================================================================
# SECTION 9: STEP 2 — タブコンテンツ
# ================================================================
st.header("🔬 タブを選んでAIの見ている世界を体験しよう！")

tab1, tab2, tab3 = st.tabs([
    "① 数字の世界 (RGB)",
    "② 輪郭の世界 (Edge)",
    "③ フィルタの世界 (CNN)",
])

image_resized = image.resize((600, int(600 * image.height / image.width)))
img_array     = np.array(image_resized)

# ---------------------------------------------------------------
# TAB 1: RGBの世界
# ---------------------------------------------------------------
with tab1:
    st.header("🧮 画像の正体は『数字の集まり』！")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_main1, col_main2 = st.columns([1, 2])

        with col_main1:
            rgb_b64 = get_image_as_base64(os.path.join(DATA_DIR, "rgb.jpg"))
            if rgb_b64:
                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:8px; overflow:hidden; background:white;">
                    <img src="data:image/jpeg;base64,{rgb_b64}" style="width:100%; display:block;">
                </div>
                <p style="text-align:center; color:#666; font-size:0.8em; margin-top:5px;">
                    【図：加法混色】R・G・Bの光の強さで色が決まる
                </p>
                """, unsafe_allow_html=True)
            else:
                st.info("rgb.jpg 図説なし")

        with col_main2:
            st.markdown("""
            <div class="explanation-box">
            <h3>1. 画像は「小さな点」の集合体</h3>
            スマホで見ている綺麗な写真は、実は「ピクセル」という超小さな点の集まりです。
            ひとつひとつの点は、<b>赤(R)・緑(G)・青(B)</b>という3つの光の強さで作られています。

            <h3>2. コンピュータが見ているのは「数字」</h3>
            コンピュータは「色」として画像を認識できません。
            その代わり、各ピクセルの光の強さを<b>0〜255の数字</b>として処理しています。<br>
            R・G・Bがそれぞれ256通りなので、組み合わせは
            <b>256×256×256 ＝ 約1,677万通り</b>！

            <h3>3. AIはどうやって物体を見つける？</h3>
            AIはこの膨大な数字の並びをスキャンして、
            「この数字のパターンは猫の耳だ！」と判断します。
            人間には単なる色に見えるものでも、AIには<b>計算可能なデータの塊</b>です。
            </div>
            """, unsafe_allow_html=True)

        st.divider()

    st.header("🔍 AIに見えているデータを覗いてみよう！")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_img, col_data = st.columns([3, 2])

        with col_img:
            st.subheader("RGBレイヤーを分解してみよう")
            col_orig_disp, col_rgb_disp = st.columns(2)
            with col_orig_disp:
                st.markdown("▼ 元画像")
                st.image(image_resized, caption="3色の組み合わせ画像", use_container_width=True)
            with col_rgb_disp:
                st.markdown("▼ 成分画像")
                target_vis_placeholder = st.empty()

        with col_data:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("AIはこの数字の変化（グラデーション）を計算して、物体の形を判断します。")

            channel = st.radio(
                "✂️ 分解する色を選択",
                ["Red", "Green", "Blue"],
                horizontal=True,
                key="rgb_selector_fixed"
            )

            st.markdown("##### 🔍 調査ポイントを動かす")
            col_x, col_y = st.columns(2)
            pick_x = col_x.slider("横の位置 (X)", 0, image_resized.width  - 1, int(image_resized.width  / 2), key="slider_x")
            pick_y = col_y.slider("縦の位置 (Y)", 0, image_resized.height - 1, int(image_resized.height / 2), key="slider_y")

            r_ch  = img_array[:, :, 0]
            g_ch  = img_array[:, :, 1]
            b_ch  = img_array[:, :, 2]
            zeros = np.zeros_like(r_ch)

            if channel == "Red":
                target_vis = np.stack([r_ch, zeros, zeros], axis=2)
                cmap_style = "Reds"
            elif channel == "Green":
                target_vis = np.stack([zeros, g_ch, zeros], axis=2)
                cmap_style = "Greens"
            else:
                target_vis = np.stack([zeros, zeros, b_ch], axis=2)
                cmap_style = "Blues"

            target_vis_placeholder.image(target_vis, caption=f"明るい場所＝{channel}が強い", use_container_width=True)

            pix_r, pix_g, pix_b = img_array[pick_y, pick_x, :3]
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 Red",   int(pix_r))
            m2.metric("🟢 Green", int(pix_g))
            m3.metric("🔵 Blue",  int(pix_b))

            st.markdown(f"**▼ 周辺の数値データ** ({channel}チャンネル)")

            zoom_radius = 4
            y_start = max(0, pick_y - zoom_radius)
            y_end   = min(target_vis.shape[0], pick_y + zoom_radius + 1)
            x_start = max(0, pick_x - zoom_radius)
            x_end   = min(target_vis.shape[1], pick_x + zoom_radius + 1)
            zoom_area = target_vis[y_start:y_end, x_start:x_end]
            zoom_disp = cv2.resize(zoom_area, (250, 250), interpolation=cv2.INTER_NEAREST)

        col_zoom1, col_zoom2 = st.columns([3, 2])
        zoom_area_orig = img_array[y_start:y_end, x_start:x_end]
        zoom_disp_orig = cv2.resize(zoom_area_orig, (250, 250), interpolation=cv2.INTER_NEAREST)

        with col_zoom1:
            c1, c2 = st.columns(2)
            c1.image(zoom_disp_orig, caption="選択範囲(元)",  use_container_width=True)
            c2.image(zoom_disp,      caption="選択範囲(成分)", use_container_width=True)

        with col_zoom2:
            ch_idx      = 0 if channel == "Red" else 1 if channel == "Green" else 2
            zoom_single = zoom_area[:, :, ch_idx].astype(np.int32)
            max_val     = int(np.max(zoom_single))
            max_pos_local = np.unravel_index(np.argmax(zoom_single), zoom_single.shape)
            col_labels  = list(range(x_start, x_end))
            row_labels  = list(range(y_start, y_end))
            max_x = col_labels[max_pos_local[1]]
            max_y = row_labels[max_pos_local[0]]
            df_subset = pd.DataFrame(zoom_single, columns=col_labels, index=row_labels)
            st.table(
                df_subset.style
                    .background_gradient(cmap=cmap_style, axis=None, vmin=0, vmax=255)
                    .highlight_max(axis=None, props="color:white; font-weight:bold; background-color:#FF4B4B;")
                    .format("{:d}")
            )
            st.write(f"🎯 **中心**:({pick_x},{pick_y}) | 🌟 **最大輝度**:({max_x},{max_y}) [値:{max_val}]")
            if max_val == 255:
                st.success("✨ ビンゴ！一番明るい点を発見！")

# ---------------------------------------------------------------
# TAB 2: エッジ検出
# ---------------------------------------------------------------
with tab2:
    st.header("📐 輪郭を取り出す（エッジ検出）")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation-box">
        <p>AIは物体の「輪郭（エッジ）」を、とても重要な手がかりとして使います。
        エッジとは、画像の中で色や明るさが急激に変わる境界線のことです。<br>
        AIはエッジを探す前に、細かいノイズを消すために画像を<b>「ぼかす」</b>処理をします。<br>
        2つの有名なアルゴリズム（CannyとLaplacian）を切り替えて、その違いを体験しましょう！</p>
        </div>
        """, unsafe_allow_html=True)

        edge_mode = st.radio("アルゴリズムを選択", ["Canny法", "ラプラシアンフィルタ"], key="edge_mode_radio")

        st.write("🔧 **AIフィルタ設定**")
        c_set1, c_set2, c_set3 = st.columns(3)
        blur_val = c_set1.slider("ガウシアンフィルタ（ぼかし）", 1, 15, 3, step=2)

        if edge_mode == "Canny法":
            th1 = c_set2.slider("感度:Min", 0, 255, 100)
            th2 = c_set3.slider("感度:Max", 0, 255, 200)
        else:
            lap_ksize = c_set2.slider("ラプラシアンフィルタ（エッジ検出）", 1, 7, 3, step=2)
            st.caption("※ラプラシアンは境界の『変化の大きさ』を計算します。")

        gray    = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)

        if edge_mode == "Canny法":
            processed_edges = cv2.Canny(blurred, th1, th2)
        else:
            lap_raw = cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_ksize)
            processed_edges = cv2.convertScaleAbs(lap_raw)

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.image(image_resized,   caption="1. 元画像",            use_container_width=True)
        with col_res2:
            st.image(blurred,         caption="2. ぼかし後",           use_container_width=True)
        with col_res3:
            st.image(processed_edges, caption=f"3. {edge_mode} 結果", use_container_width=True)

        st.markdown("""
        <div class="explanation-box">
        <h3>🔧 ガウシアンフィルタ（ぼかし）</h3>
        値を大きくすると画像がよりぼやけます。ぼかすことで細かいノイズが消え、
        物体の大きな輪郭だけが際立ちます。AIはこのぼかしで「重要な輪郭だけ」を見つけやすくします。
        </div>
        """, unsafe_allow_html=True)

        if edge_mode == "Canny法":
            st.markdown("""
            <div class="explanation-box">
            <h3>💡 Cannyエッジ検出の仕組み</h3>
            「感度」は色の変化をどれくらい厳しくチェックして「線」と認めるかの基準です。<br>
            理論上、<b>Min : Max = 1 : 2 または 1 : 3</b> の比率が最も綺麗な線を描けます。<br>
            <ul>
                <li>● <b>Max以上</b>：確実なエッジ → 無条件採用</li>
                <li>● <b>Min〜Maxの間</b>：迷いエッジ → 確実なエッジと繋がっていれば採用</li>
                <li>● <b>Min以下</b>：ノイズ → 無視</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="explanation-box">
            <h3>🔮 ラプラシアンフィルタの仕組み</h3>
            色の明るさが急激に切り替わる地点を数学の「2階微分」で捉えるフィルタです。<br>
            全方向360度の変化を一度に計算するため、輪郭の点・線を強調します。<br>
            ただしノイズに敏感なため、ぼかし処理とセットで使うことが多いです。
            </div>
            """, unsafe_allow_html=True)

    st.header("🖼️ ギャラリー：AIの視点プロセス完全図解")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        def show_row_arrow():
            st.markdown('<div class="row-arrow">➡</div>', unsafe_allow_html=True)

        steps_row1 = [
            {"file": os.path.join(DATA_DIR, "ori.png"),  "badge": "STEP 1", "title": "入力：元画像"},
            {"file": os.path.join(DATA_DIR, "mono.png"), "badge": "STEP 2", "title": "変換：モノクロ"},
            {"file": os.path.join(DATA_DIR, "blur.png"), "badge": "STEP 3", "title": "除去：ぼかし"},
        ]
        steps_row2 = [
            {"file": os.path.join(DATA_DIR, "edge.png"), "badge": "STEP 4", "title": "抽出：エッジ"},
            {"file": os.path.join(DATA_DIR, "ans.png"),  "badge": "TARGET", "title": "理想：正解データ"},
            {"file": os.path.join(DATA_DIR, "dif.png"),  "badge": "RESULT", "title": "判定：一致率"},
        ]

        st.subheader("Phase 1: 情報を削ぎ落とす")
        st.markdown("<br>", unsafe_allow_html=True)
        cols_1 = st.columns([10, 2, 10, 2, 10])
        for i, col_index in enumerate([0, 2, 4]):
            with cols_1[col_index]:
                item = steps_row1[i]
                st.markdown(f"""
                <div class="img-frame">
                    <span class="step-badge">{item['badge']}</span>
                    <div class="caption-text">{item['title']}</div>
                </div>""", unsafe_allow_html=True)
                if os.path.exists(item["file"]):
                    st.image(item["file"], use_container_width=True)
                else:
                    st.warning(f"画像なし: {os.path.basename(item['file'])}")
            if col_index != 4:
                with cols_1[col_index + 1]:
                    show_row_arrow()

        st.markdown("""
        <div style="margin-top:30px; display:flex; flex-direction:column;">
            <div style="text-align:right; padding-right:5%;">
                <svg width="60" height="60" viewBox="0 0 100 100">
                    <path d="M 20,10 Q 80,10 80,60" fill="none" stroke="#D32F2F" stroke-width="12" stroke-linecap="round"/>
                    <polygon points="65,55 80,85 95,55" fill="#D32F2F"/>
                </svg>
            </div>
            <div style="text-align:center; color:black; font-weight:bold; background-color:#c0c0c0;
                        padding:15px; border-radius:10px; margin:10px 0;">
                🌀 情報を整理したので、ここから「形」を取り出します 🌀
            </div>
            <div style="text-align:left; padding-left:5%;">
                <svg width="60" height="60" viewBox="0 0 100 100">
                    <path d="M 80,10 Q 20,10 20,60" fill="none" stroke="#D32F2F" stroke-width="12" stroke-linecap="round"/>
                    <polygon points="5,55 20,85 35,55" fill="#D32F2F"/>
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Phase 2: 形を見極める")
        st.markdown("<br>", unsafe_allow_html=True)
        cols_2 = st.columns([10, 2, 10, 2, 10])

        with cols_2[0]:
            item = steps_row2[0]
            st.markdown(f"""
            <div class="img-frame frame-phase2">
                <span class="step-badge badge-phase2">{item['badge']}</span>
                <div class="caption-text">{item['title']}</div>
            </div>""", unsafe_allow_html=True)
            if os.path.exists(item["file"]):
                st.image(item["file"], use_container_width=True)
            else:
                st.warning(f"画像なし: {os.path.basename(item['file'])}")
        with cols_2[1]: show_row_arrow()

        with cols_2[2]:
            item = steps_row2[1]
            st.markdown(f"""
            <div class="img-frame frame-phase2">
                <span class="step-badge badge-phase2">{item['badge']}</span>
                <div class="caption-text">{item['title']}</div>
            </div>""", unsafe_allow_html=True)
            if os.path.exists(item["file"]):
                st.image(item["file"], use_container_width=True)
            else:
                st.warning(f"画像なし: {os.path.basename(item['file'])}")
        with cols_2[3]: show_row_arrow()

        with cols_2[4]:
            item = steps_row2[2]
            st.markdown(f"""
            <div class="img-frame frame-result">
                <span class="step-badge badge-result">{item['badge']}</span>
                <div class="caption-text" style="color:#c0392b;">{item['title']}</div>
            </div>""", unsafe_allow_html=True)
            if os.path.exists(item["file"]):
                st.image(item["file"], use_container_width=True)
            else:
                st.warning(f"画像なし: {os.path.basename(item['file'])}")
            st.caption("※一致率が95%を超えると、AIは「完全な認識」として学習を完了します。")

    st.header("🦅 AIの認識精度を実際に測ってみよう")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        karasu_path = os.path.join(DATA_DIR, "karasu.png")
        pro_path    = os.path.join(DATA_DIR, "pro.jpg")

        if not os.path.exists(karasu_path) or not os.path.exists(pro_path):
            st.warning("サンプル画像（karasu.png / pro.jpg）が見つかりません。")
        else:
            target_color  = cv2.cvtColor(cv2.imread(karasu_path), cv2.COLOR_BGR2RGB)
            target_gray   = cv2.cvtColor(target_color, cv2.COLOR_RGB2GRAY)
            edge_insert   = cv2.imread(pro_path)
            height, width = edge_insert.shape[:2]
            target_color  = cv2.resize(target_color, (width, height))
            target_gray   = cv2.resize(target_gray,  (width, height))

            y_true_edges = cv2.Canny(target_gray, 200, 300)
            kernel       = np.ones((3, 3), np.uint8)
            y_true_zone  = cv2.dilate(y_true_edges, kernel, iterations=1)

            proc_gray         = cv2.cvtColor(edge_insert, cv2.COLOR_BGR2GRAY)
            _, y_pred_binary  = cv2.threshold(proc_gray, 100, 255, cv2.THRESH_BINARY)
            y_pred_vis        = cv2.dilate(y_pred_binary, kernel, iterations=1)
            y_true_vis        = cv2.dilate(y_true_edges,  kernel, iterations=1)

            vis_img           = np.zeros((height, width, 3), dtype=np.uint8)
            vis_img[:, :, 1]  = (y_true_zone > 0) * 255
            vis_img[:, :, 0]  = (y_pred_vis  > 0) * 255

            ideal_img         = np.zeros((height, width, 3), dtype=np.uint8)
            ideal_img[:, :, 1] = (y_true_zone > 0) * 255
            ideal_img[:, :, 0] = (y_true_vis  > 0) * 255

            intersection = np.logical_and(y_pred_binary > 0, y_true_zone > 0)
            union        = np.logical_or (y_pred_binary > 0, y_true_zone > 0)
            iou_score    = float(np.sum(intersection) / np.sum(union)) if np.sum(union) > 0 else 0.0

            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.markdown("**元画像**")
                st.image(karasu_path, caption="処理前のカラス", use_container_width=True)
            with col_a2:
                st.markdown("**🔍 重ね合わせ検証**")
                st.image(vis_img, caption="🔴エッジ検出 🟢正解領域 🟡的中", use_container_width=True)
            with col_a3:
                st.markdown("**🎯 理論上の100%**")
                st.image(ideal_img, caption="理想的な重なり", use_container_width=True)

            st.warning(
                f"参考：エッジ的中率（IoU）は約{iou_score*100:.1f}%でした。"
                "同じカラスに見えても、デジタルデータとしては全く別物だとわかります。"
            )
            st.markdown("""
            <div class="explanation-box">
            <h3>この実験が示すこと</h3>
            <p>人間には「同じカラス」に見えても、AIにとっては画像ごとに全く別のデータです。
            角度・背景・光の当たり方が少し違うだけで、エッジの出る位置がガラッと変わります。
            AIが安定して物体を認識するためには、非常に多くの「似た画像」で学習する必要があります。
            これが「AIには膨大なデータが必要」と言われる理由です。</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 3: 畳み込みフィルタ
# ---------------------------------------------------------------
with tab3:
    st.header("🕶️ 特徴を見つける眼鏡（畳み込みフィルタ）")

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        lasta_b64 = get_image_as_base64(os.path.join(DATA_DIR, "lasta.png"))
        col_view, col_text = st.columns([2, 3])

        with col_view:
            if lasta_b64:
                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:8px; overflow:hidden">
                    <img src="data:image/png;base64,{lasta_b64}" style="width:100%; display:block;">
                </div>
                <p style="text-align:center; color:#666; font-size:0.8em; margin-top:5px;">
                    【図：ラスタスキャン】左上から1マスずつ計算
                </p>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ 解説画像が見つかりません（lasta.png）。")

        with col_text:
            st.markdown("""
            <div class="explanation-box">
            <h3>🔍 AIはどうやって画像を見る？</h3>
            <p>AIは画像を「意味」として一瞬で理解しているわけではありません。<br>
            左の図のように、<b>「3×3マスの小さな窓（フィルタ）」</b>を
            左上から1マスずつスライドさせながら計算しています。</p>
            <p>各ピクセル周辺の「色の変化」を数値として測るこの方式を、
            専門用語で<b>「ラスタスキャン」</b>と呼びます。</p>
            <h3>💡 ここがポイント！</h3>
            <p>窓の中の9個の数字が、縦線・横線・角などの特徴に反応するよう設定されています。
            これを画像全体で<b>何万回も繰り返す</b>ことで、
            AIは「これは猫の耳だ！」と気づくことができるのです。</p>
            </div>
            """, unsafe_allow_html=True)

    st.header("1️⃣ 有名なフィルタ係数を見てみよう")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])

        with c1:
            filter_type = st.selectbox("かけてみるフィルタを選択", [
                "恒等 (何もしない)",
                "ぼかし (平均化/Mean)",
                "シャープ化 (Sharpen)",
                "輪郭抽出 (Laplacian)",
                "縦の輪郭 (Sobel X)",
                "横の輪郭 (Sobel Y)",
                "エンボス (Emboss)",
            ])

            kernel_map = {
                "恒等 (何もしない)":     (np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),       "中央が1で他が0。元の画素をそのまま出力します。", False),
                "ぼかし (平均化/Mean)":  (np.ones((3, 3), np.float32) / 9,                             "周囲9マスの平均をとります。ノイズ低減に有効です。", False),
                "シャープ化 (Sharpen)": (np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),    "中央を強く・周囲を引くことで輪郭をクッキリさせます。", False),
                "輪郭抽出 (Laplacian)": (np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),       "周囲との差分を計算してエッジを検出します。", True),
                "縦の輪郭 (Sobel X)":  (np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),    "左右の差を計算。縦線がある場所だけ光ります。", True),
                "横の輪郭 (Sobel Y)":  (np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),     "上下の差を計算。横線がある場所だけ光ります。", True),
                "エンボス (Emboss)":    (np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),    "斜めの光と影を作り出し、立体的に見せます。", False),
            }

            kernel_val, desc, is_edge = kernel_map[filter_type]
            if is_edge:
                processed_raw = cv2.filter2D(img_array, cv2.CV_64F, kernel_val)
                processed     = cv2.convertScaleAbs(processed_raw)
            else:
                processed = cv2.filter2D(img_array, -1, kernel_val)

            st.markdown("<h3>⚙️ カーネル（計算式）</h3>", unsafe_allow_html=True)
            df_kernel = pd.DataFrame(kernel_val)
            fmt = "{:.2f}" if filter_type == "ぼかし (平均化/Mean)" else "{:.0f}"
            st.write(df_kernel.style.format(fmt))
            st.info(f"💡 {desc}")

        with c2:
            _, sub_right = st.columns([0.2, 1])
            with sub_right:
                st.image(processed, caption=f"【変換後】{filter_type} の世界", use_container_width=True)

        st.divider()
        st.warning("""
        🎓 プロ豆知識：なぜ「左上」から？
        昔のWindows画像（BMP）は「左下から右上」に向かって走査していました（数学グラフの名残）。
        現在主流のJPEG・PNG・OpenCVは「左上から右下」（本を読む順序）に走査します。
        """)

    st.header("🧪 DIYラボ：自分だけのフィルタを作ろう！")
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_lab_input, col_lab_result = st.columns(2)

        with col_lab_input:
            st.markdown("<h3>⚙️ カーネル行列の入力</h3>", unsafe_allow_html=True)
            k = np.zeros((3, 3))
            r1c1, r1c2, r1c3 = st.columns(3)
            r2c1, r2c2, r2c3 = st.columns(3)
            r3c1, r3c2, r3c3 = st.columns(3)
            with r1c1: k[0,0] = st.number_input("0,0", value=0.0,  step=1.0, key="k00", label_visibility="collapsed")
            with r1c2: k[0,1] = st.number_input("0,1", value=-1.0, step=1.0, key="k01", label_visibility="collapsed")
            with r1c3: k[0,2] = st.number_input("0,2", value=0.0,  step=1.0, key="k02", label_visibility="collapsed")
            with r2c1: k[1,0] = st.number_input("1,0", value=-1.0, step=1.0, key="k10", label_visibility="collapsed")
            with r2c2: k[1,1] = st.number_input("1,1", value=5.0,  step=1.0, key="k11", label_visibility="collapsed")
            with r2c3: k[1,2] = st.number_input("1,2", value=-1.0, step=1.0, key="k12", label_visibility="collapsed")
            with r3c1: k[2,0] = st.number_input("2,0", value=0.0,  step=1.0, key="k20", label_visibility="collapsed")
            with r3c2: k[2,1] = st.number_input("2,1", value=-1.0, step=1.0, key="k21", label_visibility="collapsed")
            with r3c3: k[2,2] = st.number_input("2,2", value=0.0,  step=1.0, key="k22", label_visibility="collapsed")
            st.caption("☝️ この数字を変えてみてください！")
            use_abs = st.checkbox("結果を絶対値にする（境目を白く光らせる）", value=False)

        with col_lab_result:
            if use_abs:
                custom_raw = cv2.filter2D(img_array, cv2.CV_64F, k)
                custom_out = cv2.convertScaleAbs(custom_raw)
            else:
                custom_out = cv2.filter2D(img_array, -1, k)
            _, sub_right2 = st.columns([0.2, 1])
            with sub_right2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.image(custom_out, caption="あなたの実験結果", use_container_width=True)

        st.success("""
        💡 実験のヒント:
        ・中央だけ大きい値で他を0 → 元画像のまま（恒等変換）
        ・中央を大きく＋周囲を負 → コントラスト強化（シャープ）
        ・全マス同じ値 → 平均化（ぼかし）
        ・正と負の値を混ぜる → 差分計算（エッジ検出）
        """)

# ================================================================
# SECTION 10: フッター
# ================================================================
st.markdown("""
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
    <p>MISSION 01: AIの目 — 画像認識の仕組みを解き明かせ</p>
</div>
""", unsafe_allow_html=True)
