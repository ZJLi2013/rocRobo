# RocRobSim 定位 + Isaac Sim 拆解 + AMD/ROCm 落地策略（研究笔记）

> 📎 本文给 **RocRobSim 立项定位**，系统拆解对标对象 **NVIDIA Isaac Sim** 的组件 / 管线能力 / 成熟度，
> 并落到 **AMD GPU（ROCm）、NV-free 的当前策略**。方法论照搬运动模块的
> [`rocRobo/docs/curobo-study.md`](../rocRobo/docs/curobo-study.md)（cuRobo 拆解 → 开源替代 → ROCm 落地）。
>
> 一句话：**RocRobSim = AMD/ROCm 平台下的开源 Isaac Sim 等价物**，
> = **Genesis（仿真底座）+ rocRobo（运动模块，对标 cuRobo，已落地）+ 整合层**。

> 图例：✅ 纯 CPU/跨平台 ｜ 🟢 ROCm 直接跑（Genesis-amdgpu / JAX-ROCm / torch-ROCm）｜ 🟡 需适配 ｜ 🔴 NV/CUDA 专属，放弃

---

## 0. RocRobSim 定位与底座策略（**本节是核心决策**）

| 项 | 内容 |
|---|---|
| **目标** | 在 AMD GPU（ROCm）上、不依赖 NVIDIA 生态，做一个 **Isaac Sim 等价**的机器人仿真 / 合成数据 / 学习 / 运动 平台 |
| **对标** | NVIDIA Isaac Sim（Omniverse + OpenUSD + PhysX/Newton + RTX + Replicator + Isaac Lab + cuRobo）|
| **仿真底座（决策）** | ✅ **Genesis**（官方 [`ROCm/Genesis`](https://github.com/Genesis-Embodied-AI/genesis-world) fork，`gs.init(backend=gs.amdgpu)`）——一个平台同时给**物理 + 渲染 + 生成式数据 + 并行环境** |
| **运动模块（已落地）** | ✅ **rocRobo**：碰撞感知 IK + 避障轨迹优化（对标 cuRobo），JAX-ROCm + PyRoki，gfx950/gfx942 开箱可用 |
| **核心方法** | 不逐组件自写、不移植 NV 底座；**复用 Genesis 这个已经跨平台 + AMD 官方支持的开源仿真平台**，RocRobSim 的自有价值 = AMD enablement + 模块整合 + rocRobo 运动后端 |

### 明确的"不走"路线（关键认知）

- **不走 Warp / Newton 迁移**：Isaac Sim 6.0 的 Newton 建在 **NVIDIA Warp**（无 ROCm 后端 🔴），移植不划算。AMD 路线直接用 **Genesis** 这个独立的开源仿真平台，绕开整个 Omniverse 栈。
- **不以 OpenUSD / Fabric 为数据底座**：USD/Fabric 是 Omniverse 的资产/运行时层，**不在 RocRobSim 当前考虑范围**。Genesis 有自己的 entity/scene 模型，资产用标准 **MJCF / URDF / mesh** 加载即可。
- **不追实时 RTX 光追**：Genesis 自带光追渲染（LuisaRender，Vulkan 后端）+ PyRender 光栅快渲，够用。

> 为什么是 Genesis（而非 MJX/Brax 拼装）：MJX/Brax 只解决"物理"一块；**Genesis 是一个完整平台**——通用物理引擎（多解算器）+ 光追渲染 + 生成式数据引擎 + 并行环境，且 **AMD 官方维护 ROCm fork + 出了 ROCm 落地博客（Ryzen AI MAX 数字孪生）**。一个依赖覆盖 Isaac Sim 的 P2–P5 多块管线，摩擦最小。

### 与 RoboSmith 的分层（重要：很多块已在 AMD 上落地）

RocRobSim 不是从零起。AMD/ROCm 生态里**多数 Isaac Sim 等价能力已有现成块**，且已被
**RoboSmith**（具身数据基础设施）在 MI300X 上跑通：

```
┌─────────────────────────────────────────────────────────────┐
│  RoboSmith（应用层 / 具身数据基础设施 · 已落地）              │
│  资产生成(TRELLIS.2/Hunyuan3D) · CAP 任务/场景授权 ·          │
│  GraspGen 抓取 · 专家轨迹 · LeRobot 数据导出 · VLA bench      │
└───────────────▲───────────────────────────▲─────────────────┘
                │ 消费仿真                    │ 消费运动
┌───────────────┴───────────────┐ ┌──────────┴──────────────────┐
│  Genesis（仿真核 · ROCm/Genesis）│ │  rocRobo（运动核 · 已落地）   │
│  物理/并行env/渲染/传感器        │ │  collision-free IK/trajopt    │
└────────────────────────────────┘ └─────────────────────────────┘
        └──────────  RocRobSim = 仿真+运动平台层  ──────────┘
```

- **RocRobSim 拥有/整合**：仿真核（Genesis-ROCm）+ 运动核（rocRobo）+ 二者衔接，对外暴露统一的"sim + motion"平台 API。
- **RoboSmith 是消费方**：资产生成、任务/场景授权（CAP）、抓取（GraspGen）、数据导出（LeRobot）都在 RoboSmith，**已在 AMD 上运行**——这些属"数据基础设施"，不是 RocRobSim 重做的对象。
- 因此 RocRobSim 的真实工作量 = **把已有的 Genesis-ROCm 与 rocRobo 收敛成一个干净的平台层**，而非铺全新管线（见 §4）。

---

## 1. Isaac Sim 全局认知（架构 + 2026 现状）

**主线**：以 **OpenUSD** 为场景/数据骨架，**Omniverse Kit** 为应用框架，串起
**物理仿真 → 渲染/传感器 → 合成数据 → 机器人学习 → 运动规划 → 连接(ROS/图)** 的全栈管线，
卖点 = **GPU 并行 + 照片级渲染 + sim-to-real 资产一致性**。

```
   场景/数据底座: OpenUSD  (+ Fabric 运行时加速)            ← RocRobSim 不采用此层
        │
   应用框架: Omniverse Kit  🔴 ─────────────────────────────────────────────┐
        │                                                                   │
   ┌────┴─────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌─────┴────┐
   │ ① 物理   │ │ ② 渲染    │ │ ③ 合成数据 │ │ ④ 机器人  │ │ ⑤ 运动    │ │ ⑥ 连接   │
   │ PhysX/   │ │ RTX +     │ │ Replicator│ │ 学习      │ │ cuRobo   │ │ ROS2 /   │
   │ Newton   │ │ NuRec3DGS │ │ (DR+标注) │ │ Isaac Lab │ │          │ │ OmniGraph│
   └──────────┘ └──────────┘ └───────────┘ └──────────┘ └──────────┘ └──────────┘
```

**2026 现状（Isaac Sim 5.0 / 6.0，GTC'26 早期版）**：
- **模块化物理后端**：默认 **PhysX 5**，新增 **Newton**（开源，基于 **NVIDIA Warp** + **MuJoCo Warp**，NVIDIA/Google DeepMind/Disney 维护、Linux Foundation 托管）；运行时可切。
- **Core Experimental API**：旧 PyTorch Core API 弃用，新接口建在 **Warp** 上（engine-agnostic）。
- **渲染**：Kit 110 引入 **NuRec 3D Gaussian Splatting** + Fabric Scene Delegate；RTX 路径追踪仍为主。
- **导入器**：URDF/MJCF importer 重建于 USD Exchange SDK，支持多物理后端 schema。

> 对 RocRobSim 的意义：**Kit / RTX / PhysX-GPU / Warp / Newton 全是 CUDA/NV 专属 🔴**——这正是不移植本体、改用 Genesis 的理由。Isaac Sim 的价值在于"管线能力"的拆解（下表），而非具体实现。

---

## 1.5 Isaac Sim 解决哪些 tasks（用例 / 工作流视角）

> §2 是"用什么组件"，本节是"解决什么问题"。Isaac Sim 不是单一仿真器，而是围绕
> **机器人数据 → 学习 → 验证 → 部署** 的一组工作流。下表按 task 列出 AMD/ROCm 现状与 RocRobSim 覆盖范围。

| # | Task / 用例 | Isaac Sim 怎么做 | AMD/ROCm 现状 | RocRobSim / 生态定位 |
|---|---|---|---|---|
| **T1** | 机器人策略学习（RL/IL）| **Isaac Lab** 千级并行 env + Newton/PhysX，sim-to-real | 🟢 Genesis 并行 env / MJX | **待完善**：平台已支持，待固化基线 |
| **T2** | 合成操作数据生成（SDG）| Replicator + 域随机化 + GR00T-Gen | ✅ **RoboSmith 已落地**：资产→场景→专家轨迹→LeRobot | ✅ 已落地（含**资产级 real2sim**）|
| **T3** | **真实场景重建（real2sim）** | **NuRec**：手机/立体/lidar → 3DGS → USDZ + nvblox 碰撞 mesh | 资产级 ✅（RoboSmith TRELLIS.2/Hunyuan3D）；场景级 🟡（gsplat/COLMAP/Open3D 待 ROCm）| **待完善**：资产级已有，场景级为自然扩展 |
| **T4** | 感知模型训练数据 | RTX 渲染 + Replicator 标注（分割/bbox/深度/法线）| 🟢 Genesis 渲染 buffer + 自写标注导出 | 部分（RoboSmith 偏操作数据）|
| **T5** | sim-to-real 迁移 | 物理精度 + 域随机化(DR) + **sim-and-real co-training** | 🟡 方法/数据策略，跨平台 | **待完善**：数据策略，非平台特性 |
| **T6** | **世界模型驱动数据** | **Cosmos WFM + GR00T-Dreams**（单图+语言→合成轨迹）、**GR00T-N** 跨本体基础模型 | 🔴/🟡 Cosmos/GR00T 为 NV；开源 WFM 早期 | **待完善（前沿）**：可借开源 world model |
| **T7** | 数字孪生 / 测试验证（SIL/HIL）| Omniverse 数字孪生、ROS 在环测试、AV | 🟢 Genesis 数字孪生（AMD blog 已演示）| 平台可做（**优先级低**）|
| **T8** | 遥操作数据采集 | VR/teleop → 轨迹 → 训练/后训练 | ✅ 通用、跨平台 | 生态可加（**优先级低**）|

### 剩余工作（task 视角收敛）

T2（操作 SDG，**含资产级 real2sim**）已由 RoboSmith 在 AMD 落地；T4 感知数据部分覆盖；T7 数字孪生、T8 遥操作优先级低。
去掉这些后，**真正待完善集中在四块**：

| 待完善 | task | 内容 | 现状 |
|---|---|---|---|
| **real2sim（场景级）** | T3 | 真实数据 → 可仿真 3D 场景。RoboSmith 已有**资产级**（图/扫描 → 单物体 3D 资产），**场景级**为自然扩展 | 资产级 ✅；场景级 🟡 |
| **sim2real** | T5 | 物理精度 + 域随机化(DR) + sim-and-real co-training 数据策略 | 🟡 跨平台 |
| **世界模型数据** | T6 | 生成式 world model（单图+语言→合成轨迹）补 VLA 数据，绕开人工遥操作 | 🔴/🟡 开源 WFM 早期 |
| **RL** | T1 | 千级并行 env 训练策略；Genesis 并行 env 已支持 | 🟢 待固化基线 |

### real2sim 细节（T3 展开）

> 资产级 real2sim（图/扫描 → 单物体 3D 资产）RoboSmith 已用 TRELLIS.2/Hunyuan3D 落地；
> 下列为**场景级**（整场景重建）的 Isaac Sim 参考管线与 ROCm 替代，属自然扩展方向。

**Isaac Sim NuRec 管线**（CUDA/Omniverse 栈）：采集 → **NCore** 格式 → `3DGRUT`/Instant NuRec 重建出 **USDZ**（3DGS）→ **nvblox** 由深度生成碰撞 mesh → **gsplat** + gRPC 实时渲染；附 `Fixer` 补伪影、`Asset Harvester` 抽资产。

| 场景级零件 | 作用 | ROCm 开源替代 | 成熟度 |
|---|---|---|---|
| SfM / 位姿估计 | 多视图求相机位姿 | **COLMAP** | ✅/🟡 |
| 3DGS 重建 | 高斯泼溅训练 | **gsplat / nerfstudio**（CUDA kernel 重）| 🟡 待 ROCm 适配 |
| 3DGS 渲染 | 实时渲染高斯场景 | **gsplat** / Vulkan splat | 🟡 |
| 碰撞 mesh 生成 | 外观→可碰撞几何 | **Open3D TSDF** / poisson 重建 | 🟢/🟡 |

> **RocRobSim 定位**：资产级 real2sim 已有；场景级的核心瓶颈是 3DGS 重建/渲染仍依赖 CUDA kernel（gsplat/3DGRUT），ROCm 适配未成熟，列为中期重点扩展项。

---

## 2. Isaac Sim 组件 → RocRobSim 方案（Genesis / rocRobo）

> task 视角见 §1.5，管线汇总见 §3。本节一句话定方案：**Genesis 优先覆盖物理/渲染/数据/学习，运动用 rocRobo**。

| Isaac Sim 组件 | RocRobSim 方案 | 成熟度 |
|---|---|---|
| OpenUSD + Fabric + Omniverse Kit 🔴 | **不采用**；Genesis 内置 entity/scene + Python API，资产走 MJCF/URDF/mesh + Menagerie | 🟢 |
| 物理 PhysX 5 / Newton 🔴（刚体/软体/流体/并行/可微）| **Genesis 通用物理引擎**：多解算器 MPM/SPH/FEM/PBD + 并行 env + 可微，`gs.amdgpu`/ROCm-HIP | 🟢 |
| 渲染 RTX 光追 / NuRec 3DGS 🔴 | **Genesis 光追（LuisaRender·Vulkan）+ PyRender 快渲**；场景级 real2sim(3DGS) 见 §1.5 T3，列缺口 | 🟢/🟡 |
| 传感器 相机/深度/分割/Lidar/IMU/接触 🔴 | Genesis 渲染 buffer + 物理接触读数 + 光线投射 | 🟢 |
| 合成数据 Replicator 🔴 | **Genesis 生成式数据引擎** + 自写 DR/标注；编排走 RoboSmith | 🟢/🟡 |
| 机器人学习 Isaac Lab 🔴 | **Genesis 并行 env**（gym）+ CleanRL/SB3（torch-ROCm），备选 MJX/Brax（JAX-ROCm）| 🟢 |
| 运动 cuRobo 🔴 | ✅ **rocRobo**（JAX-ROCm + PyRoki：多种子 IK + 批量 trajopt + serve）| ✅ 已落地 |
| 连接 ROS2 Bridge / OmniGraph 🔴 | ROS2 原生 + Python/行为树 DAG | ✅ |

> rocRobo（JAX 栈）与 Genesis（torch/Taichi 栈）通过 rocRobo 已有的 `serve` 进程边界（JSON IPC）解耦，
> 与 RoboSmith 消费 rocRobo 的方式一致（见 [`rocRobo/docs/up_integration.md`](../rocRobo/docs/up_integration.md)）。

---

## 3. 主要管线能力 → RocRobSim 方案 + 成熟度（汇总）

| # | 管线能力 | Isaac Sim | RocRobSim 方案 | 成熟度 |
|---|---|---|---|---|
| P1 | 场景/资产 | OpenUSD + SimReady | **Genesis scene** + MJCF/URDF（不用 USD）| 🟢 |
| P2 | 物理 + 并行环境 | PhysX / Newton | **Genesis（gs.amdgpu / ROCm-HIP）** | 🟢 |
| P3 | 渲染 + 传感器 | RTX / NuRec 3DGS | **Genesis 光追(Vulkan) + PyRender 快渲** | 🟢/🟡 |
| P4 | 合成数据 (SDG) | Replicator | **RoboSmith data engine**（资产 TRELLIS.2/Hunyuan3D + 资产级 real2sim + CAP 任务 + GraspGen + LeRobot 导出，已落地）| ✅ RoboSmith |
| P5 | 机器人学习 | Isaac Lab | **Genesis 并行 env** + CleanRL/SB3（备选 MJX/Brax）| 🟢 |
| P6 | 运动生成 | cuRobo | ✅ **rocRobo**（已落地）| ✅ |
| P7 | 连接/编排 | ROS2 Bridge / OmniGraph | ROS2 原生 / Python DAG | ✅ |

**结论**：**Genesis（ROCm）一个平台覆盖 P1–P5**，rocRobo 提供 P6，ROS2/Python 提供 P7——
RocRobSim 不需要拼装多个物理/渲染/数据库，也不需要 USD/Warp。整合工作量集中在
"Genesis ↔ rocRobo 衔接 + 合成数据编排"，而非底座移植。

---

## 4. RocRobSim 核心组件 与 pipeline 定位

> RocRobSim 只聚焦**平台层的两个核心 + 一条衔接**：仿真核、运动核、sim×motion 衔接。
> 上层数据基础设施（资产/任务/抓取/导出）是 **RoboSmith 的域、已在 AMD 落地**，RocRobSim 不重做。

### 4.1 三个核心组件（谁拥有 / 现状 / 出处）

| 核心 | 组件 | 能力 | AMD/ROCm 现状 | RocRobSim 角色 |
|---|---|---|---|---|
| **C1 仿真核** | **Genesis** | 物理（rigid + MPM/SPH/FEM/PBD）+ 并行 env + 光追/光栅渲染 + 传感器 | 🟢 `ROCm/Genesis`（`gs.amdgpu`）；**RoboSmith 已用于 MI300X** | **整合**（不自写）|
| **C2 运动核** | **rocRobo** | collision-free 多种子 IK + 避障 trajopt + retiming + 编排 + serve | ✅ 已落地，gfx950/gfx942 验证 | **自有** |
| **C3 衔接** | **sim×motion bridge** | rocRobo(JAX) 出轨迹 → Genesis(torch) 执行/校验；jax↔torch 进程边界 | 🟡 rocRobo `serve` 已具备，待对接 | **自有（待建）** |

> 边界澄清：资产生成（TRELLIS.2/Hunyuan3D）、任务/场景授权（CAP）、抓取（GraspGen）、
> 数据导出（LeRobot）、VLA 评测 —— **全在 RoboSmith，已在 AMD 上跑**，是 RocRobSim 的**消费方/上层**，不在本平台层范围。

### 4.2 平台层提供的 pipeline 能力

RocRobSim 对消费方（如 RoboSmith）暴露三条管线能力：

| Pipeline | 输入 → 输出 | 组件 | 状态 |
|---|---|---|---|
| **PL1 仿真 rollout** | scene/robot/action → 物理状态 + 并行 env | C1 Genesis | 🟢 现成（RoboSmith 在用）|
| **PL2 plan→execute** | (start, goal/waypoints, world) → collision-free 轨迹 → 在 sim 中执行/校验 | C2 rocRobo + C3 → C1 | 🟡 rocRobo 侧已成，待衔接 Genesis |
| **PL3 render→obs** | 物理状态 → RGB-D / 分割 / 接触观测 | C1 Genesis（光追 Vulkan / PyRender）| 🟢 Genesis 自带 |

### 4.3 路线（基于"已落地块"而非从零）

| 步 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **S0** | rocRobo 运动核（IK/trajopt/serve）| — | ✅ 已落地+验证 |
| **S1** | `ROCm/Genesis` 跑通基线：`gs.amdgpu` 加载 Franka + 场景 + 并行 env（与 RoboSmith 用法对齐）| Genesis | 现成，待在本 repo 固化基线 |
| **S2** | **C3 衔接（核心增量）**：rocRobo `serve` 出轨迹 → Genesis 执行/碰撞校验，封装统一平台 API（PL2）| S0 + S1 | 下一步 |
| **S3** | 把 PL1/PL2/PL3 收敛成 RocRobSim 平台 API，供 RoboSmith 直接消费（替换其手写 IK/直线插值，见 rocRobo `up_integration` 路线 R0–R10）| S2 | 规划 |

**结论**：RocRobSim 的真正增量是 **C3（sim×motion 衔接）+ 平台 API 收敛**——
仿真核（Genesis-ROCm）与运动核（rocRobo）都已在 AMD 上跑通，缺的是把两者拼成一个干净、可被 RoboSmith 消费的平台层。
不需要从零铺物理/渲染/合成数据。

---

## reference

* Isaac Sim 生态：[Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/) ｜ [Isaac Lab](https://github.com/isaac-sim/IsaacLab) ｜ [Newton Physics](https://github.com/newton-physics/newton)（Warp+MuJoCo Warp，🔴 无 ROCm）｜ [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD)（RocRobSim 不采用）
* real2sim / 世界模型（T3/T6，缺口）：[Omniverse NuRec](https://developer.nvidia.com/omniverse/nurec)（3DGS real2sim）｜ [gsplat](https://github.com/nerfstudio-project/gsplat)（NuRec 渲染所用）｜ [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) ｜ [COLMAP](https://github.com/colmap/colmap) ｜ [Isaac GR00T](https://developer.nvidia.com/isaac/gr00t) ｜ Cosmos WFM / GR00T-Dreams（NV，🔴）
* **Genesis（RocRobSim 仿真底座）**：[ROCm/Genesis](https://github.com/ROCm/Genesis)（AMD 官方 fork）｜ [Genesis 上游](https://github.com/Genesis-Embodied-AI/Genesis) ｜ [文档](https://genesis-world.readthedocs.io/) ｜ [AMD ROCm 博客：Genesis 数字孪生 on Ryzen AI MAX](https://rocm.blogs.amd.com/artificial-intelligence/rocm-genesis/README.html)
* 物理备选（JAX-ROCm）：[MuJoCo / MJX](https://github.com/google-deepmind/mujoco) ｜ [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) ｜ [Brax](https://github.com/google/brax)
* 渲染：[LuisaCompute/LuisaRender](https://github.com/LuisaGroup/LuisaCompute) ｜ [pyrender](https://github.com/mmatl/pyrender)
* 学习：[CleanRL](https://github.com/vwxyzjn/cleanrl) ｜ [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
* 资产：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) ｜ [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py)
* 运动（已落地）：[rocRobo](../rocRobo/) ｜ [curobo-study](../rocRobo/docs/curobo-study.md)
* 底座：[JAX ROCm](https://github.com/ROCm/jax) ｜ [PyTorch ROCm](https://pytorch.org/) ｜ [Triton](https://github.com/triton-lang/triton)
