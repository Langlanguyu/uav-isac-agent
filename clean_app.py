#cd /Users/fanbllang/Documents/BUPTstudy/2026寒/uav_isac_demo
#source .venv/bin/activate
#python3 -m streamlit run clean_app.py

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from typing import List, Tuple, Dict, Optional
import heapq

from streamlit_drawable_canvas import st_canvas

GridPos = tuple[int, int]


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 800px at 20% 10%, rgba(124, 92, 255, 0.10), transparent 60%),
                        radial-gradient(900px 600px at 90% 20%, rgba(46, 212, 166, 0.10), transparent 55%),
                        linear-gradient(180deg, rgba(12, 14, 18, 0.02), rgba(12, 14, 18, 0.00) 60%);
        }
        header[data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 1.0rem; padding-bottom: 2.2rem; }

        h1, h2, h3 { letter-spacing: -0.02em; }

        .muted { color: rgba(20,20,20,0.60); font-size: 0.92rem; }

        .card {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            padding: 14px 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            backdrop-filter: blur(10px);
        }
        .card-tight { padding: 12px 12px; }

        .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.85rem;
            border: 1px solid rgba(0,0,0,0.08);
            background: rgba(255,255,255,0.72);
            line-height: 1;
            white-space: nowrap;
        }
        .badge.ok { border-color: rgba(33,150,83,0.25); }
        .badge.warn { border-color: rgba(242,153,74,0.35); }
        .badge.info { border-color: rgba(47,128,237,0.28); }
        .badge.soft { border-color: rgba(0,0,0,0.08); color: rgba(20,20,20,0.75); }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.72));
            border-right: 1px solid rgba(0,0,0,0.06);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(0,0,0,0.12);
            box-shadow: 0 8px 18px rgba(0,0,0,0.06);
            padding: 0.55rem 0.9rem;
        }
        .stButton > button:active { transform: scale(0.99); }
        .stDownloadButton > button { border-radius: 14px; }

        div[data-testid="stProgress"] > div > div { border-radius: 999px; }

        .canvas-wrap {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: 0 10px 26px rgba(0,0,0,0.06);
            background: rgba(255,255,255,0.65);
        }

        .kpi {
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            align-items:center;
        }
        .kpi .badge { background: rgba(255,255,255,0.86); }

        .small {
            font-size: 0.9rem;
            color: rgba(20,20,20,0.65);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_comm_map(
    H: int,
    W: int,
    bs: GridPos,
    decay: float,
    shadow_center: GridPos,
    shadow_radius: int,
    shadow_strength: float,
) -> np.ndarray:
    rr, cc = np.indices((H, W))
    dist = np.sqrt((rr - bs[0]) ** 2 + (cc - bs[1]) ** 2)
    dist_norm = dist / (dist.max() + 1e-9)

    comm = np.exp(-decay * dist_norm)
    comm = (comm - comm.min()) / (comm.max() - comm.min() + 1e-9)

    manhattan = np.abs(rr - shadow_center[0]) + np.abs(cc - shadow_center[1])
    mask = manhattan <= shadow_radius
    comm[mask] = comm[mask] * (1.0 - shadow_strength)

    return np.clip(comm, 0.0, 1.0)


def load_comm_map_from_csv(uploaded_file, H: int, W: int) -> np.ndarray:
    df = pd.read_csv(uploaded_file, header=None)
    arr = df.values.astype(float)

    if arr.shape != (H, W):
        raise ValueError(f"CSV 尺寸是 {arr.shape}，当前网格是 {(H, W)}")

    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros((H, W), dtype=float)
    arr = (arr - mn) / (mx - mn)
    return np.clip(arr, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def comm_map_to_image_cached(comm_bytes: bytes, H: int, W: int, bs: GridPos, out_px: int) -> Image.Image:
    comm_map = np.frombuffer(comm_bytes, dtype=np.float64).reshape((H, W))
    z = (comm_map * 255).astype(np.uint8)

    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(255 * np.clip(1.6 * (t - 0.55), 0, 1))
        g = int(255 * np.clip(1.8 * (t - 0.20), 0, 1))
        b = int(255 * np.clip(1.2 * (0.95 - t), 0, 1))
        lut[i] = [r, g, b]

    rgb = lut[z]
    img = Image.fromarray(rgb, mode="RGB").resize((out_px, out_px), resample=Image.NEAREST)

    draw = ImageDraw.Draw(img)
    cell = out_px / W
    cx = (bs[1] + 0.5) * cell
    cy = (bs[0] + 0.5) * cell
    R = max(6, int(cell * 0.25))
    draw.ellipse((cx - R, cy - R, cx + R, cy + R), fill=(255, 165, 0), outline=(0, 0, 0), width=2)

    return img


def pixel_to_grid(x_px: float, y_px: float, grid_size: int, canvas_px: int) -> GridPos | None:
    if x_px is None or y_px is None:
        return None
    if not (0 <= x_px < canvas_px and 0 <= y_px < canvas_px):
        return None
    cell = canvas_px / grid_size
    col = int(x_px // cell)
    row = int(y_px // cell)
    row = max(0, min(grid_size - 1, row))
    col = max(0, min(grid_size - 1, col))
    return (row, col)


def assign_tasks_nearest(starts: List[GridPos], tasks: List[GridPos]) -> List[List[GridPos]]:
    tasks_left = tasks.copy()
    plans = [[] for _ in starts]
    cur_pos = starts.copy()

    while tasks_left:
        for i in range(len(cur_pos)):
            if not tasks_left:
                break
            best_j, best_d = None, 1e18
            for j, t in enumerate(tasks_left):
                d = abs(cur_pos[i][0] - t[0]) + abs(cur_pos[i][1] - t[1])
                if d < best_d:
                    best_d, best_j = d, j
            t = tasks_left.pop(best_j)
            plans[i].append(t)
            cur_pos[i] = t
    return plans


def assign_tasks_comm_aware(
    starts: List[GridPos],
    tasks: List[GridPos],
    comm_map: np.ndarray,
    alpha_dist: float,
    lambda_comm: float,
) -> List[List[GridPos]]:
    tasks_left = tasks.copy()
    plans = [[] for _ in starts]
    cur_pos = starts.copy()

    while tasks_left:
        for i in range(len(cur_pos)):
            if not tasks_left:
                break
            best_j, best_cost = None, 1e18
            for j, t in enumerate(tasks_left):
                dist = abs(cur_pos[i][0] - t[0]) + abs(cur_pos[i][1] - t[1])
                comm_q = float(comm_map[t])
                cost = alpha_dist * dist + lambda_comm * (1.0 - comm_q)
                if cost < best_cost:
                    best_cost, best_j = cost, j
            t = tasks_left.pop(best_j)
            plans[i].append(t)
            cur_pos[i] = t
    return plans


def astar(
    grid_walkable: np.ndarray,
    start: GridPos,
    goal: GridPos,
    cell_cost: Optional[np.ndarray] = None,
) -> List[GridPos]:
    H, W = grid_walkable.shape
    if cell_cost is None:
        cell_cost = np.zeros((H, W), dtype=float)

    def in_bounds(p):
        r, c = p
        return 0 <= r < H and 0 <= c < W

    def neighbors(p):
        r, c = p
        cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        out = []
        for q in cand:
            if in_bounds(q) and grid_walkable[q]:
                out.append(q)
        return out

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap = []
    heapq.heappush(open_heap, (h(start), 0.0, start))
    came_from: Dict[GridPos, Optional[GridPos]] = {start: None}
    gscore: Dict[GridPos, float] = {start: 0.0}

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            p = cur
            while p is not None:
                path.append(p)
                p = came_from[p]
            path.reverse()
            return path

        for nb in neighbors(cur):
            step_cost = 1.0 + float(cell_cost[nb])
            ng = gscore[cur] + step_cost
            if nb not in gscore or ng < gscore[nb]:
                gscore[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))
    return []


def plan_paths(
    grid_size: int,
    starts: List[GridPos],
    task_plans: List[List[GridPos]],
    comm_map: np.ndarray,
    w_comm_risk: float,
) -> List[List[GridPos]]:
    walkable = np.ones((grid_size, grid_size), dtype=bool)
    cell_cost = w_comm_risk * (1.0 - comm_map)

    trajectories = []
    for i, s in enumerate(starts):
        cur = s
        traj = [cur]
        for t in task_plans[i]:
            path = astar(walkable, cur, t, cell_cost=cell_cost)
            if not path:
                break
            traj.extend(path[1:])
            cur = t
        trajectories.append(traj)
    return trajectories


def evaluate_paths(comm_map: np.ndarray, trajectories: List[List[GridPos]], weak_th: float) -> Dict[str, float]:
    total = 0
    weak = 0
    avg = 0.0
    for traj in trajectories:
        for p in traj:
            q = float(comm_map[p])
            avg += q
            total += 1
            if q < weak_th:
                weak += 1
    if total == 0:
        return {"avg_comm": 0.0, "weak_ratio": 1.0, "total_steps": 0}
    return {"avg_comm": avg / total, "weak_ratio": weak / total, "total_steps": float(total)}


def explain_plain(
    algo_name: str,
    comm_map: np.ndarray,
    task_plans: List[List[GridPos]],
    w_comm_risk: float,
    lambda_comm: float,
    weak_th: float,
) -> str:
    def q(p: GridPos) -> float:
        return float(comm_map[p])

    def manhattan(a: GridPos, b: GridPos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    all_tasks = [p for plan in task_plans for p in plan]
    if all_tasks:
        qs = np.array([q(p) for p in all_tasks], dtype=float)
        q_min, q_mean, q_max = float(qs.min()), float(qs.mean()), float(qs.max())
        weak_cnt = int((qs < weak_th).sum())
        weak_ratio = weak_cnt / max(1, len(all_tasks))
    else:
        q_min = q_mean = q_max = 0.0
        weak_cnt = 0
        weak_ratio = 0.0

    if "通信感知" in algo_name:
        if lambda_comm < 1.0:
            style_hint = "这次我更偏效率：大体按就近派单，信号只做轻度约束。"
        elif lambda_comm < 3.5:
            style_hint = "这次我会在“飞行距离”和“信号稳定”之间做平衡，不盲目追求最近。"
        else:
            style_hint = "这次我强烈偏通信稳定：宁愿多飞一点，也尽量避免把任务派到弱覆盖区域。"
        assign_rule = f"分配代价 ≈ 距离 + λ·(1-通信质量)，λ={lambda_comm:.2f}"
    else:
        style_hint = "这次是基线策略：主要按就近派单（距离最近优先）。"
        assign_rule = "分配代价 ≈ 距离（只看距离）"

    if w_comm_risk < 1.0:
        path_hint = "路线规划更像“走最短路”，信号只做很轻的参考。"
    elif w_comm_risk < 4.0:
        path_hint = "路线规划会明显绕开弱覆盖：如果穿过去容易掉线，我会倾向绕一点点。"
    else:
        path_hint = "路线规划非常重视稳定连接：会强烈避开弱覆盖，哪怕绕路更多。"

    lines: List[str] = []
    lines.append("【我做了什么】")
    lines.append("这次可以理解成两步：先把订单分给不同无人机（分工），再给每架无人机规划一条更稳的路线（走路）。")
    lines.append("")
    lines.append("【这次我怎么分工】")
    lines.append(f"- 使用策略：{algo_name}")
    lines.append(f"- 分配规则：{assign_rule}")
    lines.append(f"- 直觉解释：{style_hint}")
    if all_tasks:
        lines.append(
            f"- 订单点信号概况：范围 {q_min:.2f}~{q_max:.2f}，平均 {q_mean:.2f}；低于阈值 {weak_th:.2f} 的有 {weak_cnt} 个（约 {weak_ratio*100:.0f}%）"
        )
    lines.append("")
    lines.append("【分配结果（每架 UAV 的订单顺序）】")
    for i, plan in enumerate(task_plans):
        if plan:
            lines.append(f"- UAV{i+1}：{plan}")
        else:
            lines.append(f"- UAV{i+1}：未分到订单")
    lines.append("")
    if all_tasks:
        lines.append("【我为什么这么分（可解释证据）】")
        lines.append("下面每一行都在解释：这个订单对该 UAV 来说，距离成本+信号风险合在一起是否划算。")
        lines.append("（说明：dist 用的是该 UAV 当前任务链上“上一个点→这个点”的曼哈顿距离；q 是该点通信质量。）")
        lines.append("")
        for i, plan in enumerate(task_plans):
            if not plan:
                continue
            lines.append(f"UAV{i+1} 的任务链：")
            prev = None
            for k, p in enumerate(plan):
                dist = manhattan(prev, p) if prev is not None else 0
                qq = q(p)
                if "通信感知" in algo_name:
                    cost = dist + lambda_comm * (1.0 - qq)
                    lines.append(f"  - 第{k+1}单 {p} | dist={dist:>2d} | q={qq:.2f} | 代价≈{cost:.2f}")
                else:
                    lines.append(f"  - 第{k+1}单 {p} | dist={dist:>2d} | q={qq:.2f} | 代价≈{float(dist):.2f}")
                prev = p
            lines.append("")
    lines.append("【我怎么走路（路线规划）】")
    lines.append("分完工以后，我不会只追求最短路，而是把每个格子看成“走过去会不会容易掉线”。")
    lines.append(f"- 我把通信风险写进每一步的代价：每走一步都会额外加上 w·(1-通信质量)，w={w_comm_risk:.2f}")
    lines.append(f"- 弱覆盖阈值是 {weak_th:.2f}：如果一条路会长时间穿过低于这个值的区域，我会倾向换路线。")
    lines.append(f"- 直觉解释：{path_hint}")
    lines.append("")
    lines.append("【一句话总结】")
    if "通信感知" in algo_name:
        lines.append("我做的是“通信感知调度”：分配时不只看距离，路线也会绕开弱覆盖，让整体更稳、更像真实城市低空运行。")
    else:
        lines.append("我做的是“就近调度”：分配主要看距离，路线也更偏最短路，更像效率优先。")
    return "\n".join(lines)


def draw_overlay(
    base_img: Image.Image,
    grid_size: int,
    uavs: List[GridPos],
    tasks: List[GridPos],
    trajectories: Optional[List[List[GridPos]]] = None,
) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    Wpx, _ = img.size
    cell = Wpx / grid_size

    def center(p: GridPos):
        r, c = p
        return ((c + 0.5) * cell, (r + 0.5) * cell)

    colors = [
        (255, 255, 255),
        (255, 100, 100),
        (100, 255, 100),
        (100, 180, 255),
        (255, 255, 100),
        (255, 150, 255),
    ]

    if trajectories:
        for i, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            pts = [center(p) for p in traj]
            col = colors[i % len(colors)]
            draw.line(pts, fill=col, width=max(2, int(cell * 0.15)))

    for i, p in enumerate(uavs):
        cx, cy = center(p)
        R = max(5, int(cell * 0.25))
        col = colors[i % len(colors)]
        draw.rectangle((cx - R, cy - R, cx + R, cy + R), fill=col, outline=(0, 0, 0), width=2)

    for _, p in enumerate(tasks):
        cx, cy = center(p)
        R = max(6, int(cell * 0.30))
        draw.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(255, 255, 255), width=3)

    return img


def ensure_state():
    st.session_state.setdefault("uavs", [])
    st.session_state.setdefault("tasks", [])
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("prev_obj_len", 0)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("open_explain", False)


def reset_all():
    st.session_state.uavs = []
    st.session_state.tasks = []
    st.session_state.last_click = None
    st.session_state.prev_obj_len = 0
    st.session_state.result = None
    st.session_state.history = []
    st.session_state.open_explain = False


def current_auto_stage(num_uav: int, num_task: int) -> str:
    if len(st.session_state.uavs) < num_uav:
        return "UAV"
    if len(st.session_state.tasks) < num_task:
        return "TASK"
    return "DONE"


def stage_badge(stage: str) -> str:
    if stage == "UAV":
        return '<span class="badge warn">UAV</span>'
    if stage == "TASK":
        return '<span class="badge warn">TASK</span>'
    return '<span class="badge ok">READY</span>'


def build_task_id_map(tasks: List[GridPos]) -> Dict[GridPos, str]:
    # 全局任务ID：按输入顺序编号
    return {p: f"T{i}" for i, p in enumerate(tasks)}


def build_assignment_df(task_plans: List[List[GridPos]], tasks_global: List[GridPos]) -> pd.DataFrame:
    tid = build_task_id_map(tasks_global)

    rows = []
    for i, plan in enumerate(task_plans):
        if plan:
            seq = " → ".join([f"{tid.get(p, 'T?')}({p[0]},{p[1]})" for p in plan])
            first = plan[0]
            last = plan[-1]
            rows.append(
                {
                    "UAV": f"U{i+1}",
                    "Tasks": len(plan),
                    "Sequence (Global)": seq,
                    "First": f"{tid.get(first, 'T?')}({first[0]},{first[1]})",
                    "Last": f"{tid.get(last, 'T?')}({last[0]},{last[1]})",
                }
            )
        else:
            rows.append(
                {
                    "UAV": f"U{i+1}",
                    "Tasks": 0,
                    "Sequence (Global)": "-",
                    "First": "-",
                    "Last": "-",
                }
            )

    return pd.DataFrame(rows)


st.set_page_config(page_title="UAV ISAC Agent", layout="wide")
inject_css()
ensure_state()

st.markdown(
    """
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <div style="font-size:1.55rem; font-weight:900;">🛰️ ISAC UAV Scheduling Agent</div>
        <div class="kpi">
          <span class="badge info">Demo</span>
          <span class="badge soft">Canvas</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


with st.sidebar:
    st.markdown("### ⚙️")
    with st.expander("Grid", expanded=True):
        grid_size = st.slider("Size", 15, 60, 25)
        num_uav = st.slider("UAV", 1, 10, 3)
        num_task = st.slider("Task", 1, 30, 8)

    with st.expander("Comm Map", expanded=True):
        mode = st.radio("Source", ["B", "A"], index=0, horizontal=True)
        bs_r = st.slider("BS row", 0, grid_size - 1, grid_size // 2)
        bs_c = st.slider("BS col", 0, grid_size - 1, grid_size // 2)
        decay = st.slider("Decay", 0.5, 8.0, 3.0, 0.1)
        sh_r = st.slider("Shadow row", 0, grid_size - 1, int(grid_size * 0.75))
        sh_c = st.slider("Shadow col", 0, grid_size - 1, int(grid_size * 0.75))
        sh_rad = st.slider("Shadow r", 1, grid_size // 2, max(2, grid_size // 5))
        sh_strength = st.slider("Shadow s", 0.0, 0.95, 0.75, 0.05)

        uploaded = None
        if mode == "A":
            uploaded = st.file_uploader("CSV", type=["csv"])

    with st.expander("Policy", expanded=False):
        algo_name = st.selectbox("Assign", ["最近邻（基线）", "通信感知分配（创新）"])
        alpha_dist = st.slider("α", 0.5, 5.0, 1.0, 0.1)
        lambda_comm = st.slider("λ", 0.0, 10.0, 2.5, 0.1)
        w_comm_risk = st.slider("w", 0.0, 10.0, 3.0, 0.1)
        weak_th = st.slider("weak", 0.0, 1.0, 0.55, 0.01)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    undo = c1.button("↩︎", use_container_width=True)
    clear_u = c2.button("U", use_container_width=True)
    clear_t = c3.button("T", use_container_width=True)
    clear_all = st.button("✕", use_container_width=True)


if undo:
    if st.session_state.history:
        typ, p = st.session_state.history.pop()
        if typ == "UAV":
            if p in st.session_state.uavs:
                st.session_state.uavs.remove(p)
        else:
            if p in st.session_state.tasks:
                st.session_state.tasks.remove(p)
        st.session_state.result = None
        st.rerun()
    else:
        st.rerun()

if clear_u:
    st.session_state.uavs = []
    st.session_state.history = [(t, p) for (t, p) in st.session_state.history if t != "UAV"]
    st.session_state.result = None
    st.rerun()

if clear_t:
    st.session_state.tasks = []
    st.session_state.history = [(t, p) for (t, p) in st.session_state.history if t != "TASK"]
    st.session_state.result = None
    st.rerun()

if clear_all:
    reset_all()
    st.rerun()


bs = (bs_r, bs_c)
comm_map = generate_comm_map(
    grid_size, grid_size,
    bs=bs, decay=decay,
    shadow_center=(sh_r, sh_c),
    shadow_radius=sh_rad,
    shadow_strength=sh_strength,
)

if mode == "A" and uploaded is not None:
    try:
        comm_map = load_comm_map_from_csv(uploaded, grid_size, grid_size)
    except Exception as e:
        st.error(str(e))

st.session_state.uavs = st.session_state.uavs[:num_uav]
st.session_state.tasks = st.session_state.tasks[:num_task]

CANVAS_PX = 560
base = comm_map_to_image_cached(comm_map.astype(np.float64).tobytes(), grid_size, grid_size, bs, CANVAS_PX)

overlay_bg = draw_overlay(
    base_img=base,
    grid_size=grid_size,
    uavs=st.session_state.uavs,
    tasks=st.session_state.tasks,
    trajectories=(st.session_state.result["trajectories"] if st.session_state.result else None),
)

st.image(overlay_bg, caption="debug heatmap")

stage = current_auto_stage(num_uav, num_task)
u_cnt = len(st.session_state.uavs)
t_cnt = len(st.session_state.tasks)
progress = (u_cnt + t_cnt) / (num_uav + num_task)

top_left, top_right = st.columns([1.25, 0.75], gap="large")
with top_left:
    st.markdown(
        f"""
        <div class="card card-tight">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
            <div class="kpi">
              {stage_badge(stage)}
              <span class="badge soft">U {u_cnt}/{num_uav}</span>
              <span class="badge soft">T {t_cnt}/{num_task}</span>
              <span class="badge soft">{int(progress*100)}%</span>
            </div>
            <div class="kpi">
              <span class="badge info">Grid {grid_size}</span>
              <span class="badge soft">BS {bs}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, progress))

with top_right:
    if st.session_state.result is not None:
        m = st.session_state.result["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("q̄", f"{m['avg_comm']:.3f}")
        c2.metric("weak", f"{m['weak_ratio']:.3f}")
        c3.metric("steps", f"{int(m['total_steps'])}")
    else:
        st.markdown(
            """
            <div class="card card-tight">
              <div class="kpi">
                <span class="badge soft">q̄</span>
                <span class="badge soft">weak</span>
                <span class="badge soft">steps</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.markdown(
        """
        <div class="card card-tight">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div style="font-weight:900;">🗺️ Canvas</div>
            <div class="kpi"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown('<div class="canvas-wrap">', unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=1,
        stroke_color="rgba(0,0,0,0)",
        background_image=overlay_bg,
        update_streamlit=True,
        height=CANVAS_PX,
        width=CANVAS_PX,
        drawing_mode="point",
        point_display_radius=1,
        display_toolbar=False,
        key="click_canvas_input",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if canvas.json_data is not None:
        objs = canvas.json_data.get("objects", [])
        if len(objs) > st.session_state.prev_obj_len:
            new_objs = objs[st.session_state.prev_obj_len:]
            for obj in new_objs:
                x_px = float(obj.get("left", 0.0))
                y_px = float(obj.get("top", 0.0))
                p = pixel_to_grid(x_px, y_px, grid_size, CANVAS_PX)
                if p is None:
                    continue

                st.session_state.last_click = p
                st.session_state.result = None

                stage_now = current_auto_stage(num_uav, num_task)
                if stage_now == "UAV":
                    if len(st.session_state.uavs) < num_uav and p not in st.session_state.uavs:
                        st.session_state.uavs.append(p)
                        st.session_state.history.append(("UAV", p))
                elif stage_now == "TASK":
                    if len(st.session_state.tasks) < num_task and p not in st.session_state.tasks:
                        st.session_state.tasks.append(p)
                        st.session_state.history.append(("TASK", p))
                else:
                    pass

            st.session_state.prev_obj_len = len(objs)

with right:
    st.markdown(
        """
        <div class="card card-tight">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div style="font-weight:900;">🧭 Panel</div>
            <div class="kpi"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    tabs = st.tabs(["Overview", "Assignment", "Explanation", "Export"])

    with tabs[0]:
        st.markdown("#### Input")
        colA, colB = st.columns(2)
        with colA:
            st.markdown('<div class="card card-tight">', unsafe_allow_html=True)
            st.write(f"U ({u_cnt}/{num_uav})")
            st.write(st.session_state.uavs)
            st.markdown("</div>", unsafe_allow_html=True)
        with colB:
            st.markdown('<div class="card card-tight">', unsafe_allow_html=True)
            st.write(f"T ({t_cnt}/{num_task})")
            st.write(st.session_state.tasks)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.last_click is not None:
            r, c = st.session_state.last_click
            st.caption(f"{r},{c}")

        ready = (u_cnt == num_uav and t_cnt == num_task)
        run = st.button("🚀 Run", disabled=not ready, use_container_width=True)

        if run:
            starts = st.session_state.uavs
            tasks = st.session_state.tasks

            if algo_name.startswith("最近邻"):
                task_plans = assign_tasks_nearest(starts, tasks)
            else:
                task_plans = assign_tasks_comm_aware(
                    starts, tasks, comm_map,
                    alpha_dist=alpha_dist,
                    lambda_comm=lambda_comm,
                )

            trajectories = plan_paths(
                grid_size=grid_size,
                starts=starts,
                task_plans=task_plans,
                comm_map=comm_map,
                w_comm_risk=w_comm_risk,
            )

            metrics = evaluate_paths(comm_map, trajectories, weak_th=weak_th)
            explain = explain_plain(
                algo_name=algo_name,
                comm_map=comm_map,
                task_plans=task_plans,
                w_comm_risk=w_comm_risk,
                lambda_comm=lambda_comm,
                weak_th=weak_th,
            )

            st.session_state.result = {
                "task_plans": task_plans,
                "trajectories": trajectories,
                "metrics": metrics,
                "explain": explain,
            }
            st.rerun()

        if st.session_state.result is not None:
            m = st.session_state.result["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("q̄", f"{m['avg_comm']:.3f}")
            c2.metric("weak", f"{m['weak_ratio']:.3f}")
            c3.metric("steps", f"{int(m['total_steps'])}")

    with tabs[1]:
        if st.session_state.result is None:
            st.markdown('<span class="badge soft">—</span>', unsafe_allow_html=True)
        else:
            df = build_assignment_df(
                st.session_state.result["task_plans"],
                tasks_global=st.session_state.tasks,
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[2]:
        if st.session_state.result is None:
            st.markdown('<span class="badge soft">—</span>', unsafe_allow_html=True)
        else:
            explain_text = st.session_state.result["explain"]

            def _open_explain():
                st.session_state.open_explain = True

            st.markdown(
                """
                <div class="card">
                    <div style="font-weight:900; margin-bottom:6px;">🗣️ Decision Report</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.button("📄 Open full explanation", on_click=_open_explain, use_container_width=True)
            st.write("")
            st.button(
                "📋 Copy",
                on_click=lambda: st.toast("已生成解释文本，可直接复制使用"),
                use_container_width=True,
            )

            if st.session_state.open_explain:
                @st.dialog("🗣️ Decision Report (Full)")
                def _dlg():
                    st.markdown(
                        """
                        <div class="card" style="margin-bottom:10px;">
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(explain_text.replace("\n", "  \n"), unsafe_allow_html=False)
                    st.write("")
                    if st.button("Close", use_container_width=True):
                        st.session_state.open_explain = False
                        st.rerun()

                _dlg()

    with tabs[3]:
        if st.session_state.result is None:
            st.markdown('<span class="badge soft">—</span>', unsafe_allow_html=True)
        else:
            export = {
                "uavs": st.session_state.uavs,
                "tasks": st.session_state.tasks,
                "algo": algo_name,
                "task_plans": st.session_state.result["task_plans"],
                "metrics": st.session_state.result["metrics"],
            }
            st.download_button(
                "⬇︎ JSON",
                data=pd.Series(export).to_json(force_ascii=False, indent=2),
                file_name="uav_agent_result.json",
                mime="application/json",
                use_container_width=True,
            )