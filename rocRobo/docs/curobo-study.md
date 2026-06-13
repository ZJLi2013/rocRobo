# cuRobo 拆解 + 开源替代 + ROCm 落地（研究笔记）

> 📎 本文是 rocRobo 立项时的**调研/设计研究**，从 README 归档而来（README 现只保留模块定位 + API + 性能）。
> 跨 feature 的**实验结论速查**见 [findings.md](findings.md)；项目架构/backlog/选型见 [../design.md](../design.md)。

> 目标：梳理 NVIDIA cuRobo 的机器人学体系解法（collision-free / IK feasibility / motion plan），
> 找到每个模块的开源对应物，在 **AMD GPU（ROCm）** 平台、不依赖 NV 生态的前提下重做一个等价组件。
>
> 关键约束（决定技术选型）：
> - cuRobo 底座 = **CUDA C++ kernel + PyTorch(CUDA) + NVIDIA Warp**（v2 大量使用 Warp / CUDA-Python kernel）。
> - **Warp 无 ROCm 后端**，CUDA kernel 不能直接运行。自写 kernel 需改用 **Triton（官方支持 AMD）/ HIP / 纯 PyTorch**。
> - cuRobo 体系中需要 GPU 的部分仅为"批量并行"几块（FK/IK/碰撞/rollout 优化）；
>   规划、精确碰撞、动力学、retiming 多为 **CPU 库（FCL/OMPL/Pinocchio/Drake/Ruckig）**，跨平台、零移植。
>
> **rocm-robot 路线**：第一版做 **v0 = v1 最小子集**（跑通主干，见 §4.0），再补齐到 **curobo-v1 等价**（§1 的 5 大 feature），v2 增量（§2）留到后续阶段。

---

> 跨 feature 的**实战结论速查（C1–C13 + 环境踩坑）**已独立归档至 [findings.md](findings.md)。

---

## 0. 全局认知：cuRobo 的 baseline 框架

**主线**：经典机器人学算法（IK、CHOMP/STOMP 式轨迹优化、PRM、ESDF）全部 batch 化 + GPU 化，
以"几百~上千个种子并行求解 + 取最优"同时获得速度与成功率。
体系套路 = **球近似几何 → SDF 化碰撞 → 约束写成可微代价 → GPU 并行优化 → 取最优**。

**官方 baseline = 5 大 feature（来自 [curobo.org](https://curobo.org/)，对应 v1 / v0.7.x 用户 API）**：

```
            ┌────────────── (底座) 数值优化引擎：L-BFGS / MPPI ──────────────┐
            │                  （IK / trajopt / MPC 共用）                    │
   ┌────────┴────────┐   ┌──────────────────┐   ┌──────────────┐   ┌─────────┴────────┐
   │ ① IK            │   │ ② 碰撞检测        │   │ ③ 几何规划    │   │ ④ 轨迹优化        │
   │ (collision-free)│   │ (robot & world)  │   │ (PRM 出初值)  │   │ (平滑/jerk/避障)  │
   └────────┬────────┘   └────────┬─────────┘   └──────┬───────┘   └─────────┬────────┘
            └─────────────────────┴─────── ⑤ Motion Generation = ①→③→④→retiming ──────┘
```

- 5 项（IK / 碰撞 / 几何规划 / 轨迹优化 / motion generation）构成完整 v1 baseline。
  官方 Overview 另含一个底层项：**数值优化（梯度下降 / L-BFGS / MPPI）**，非独立功能，是上述各项共用的求解引擎。
- curobo.org 为 v1 文档，不含 v2 的两块增量：**GPU-native ESDF 感知** 与 **dynamics-aware（逆动力学 + B-spline）**，见 §2。
- rocm-robot 初版只复刻这 5 项 baseline（阶段 1）；感知/动力学属阶段 2/3。

> 图例（贯穿全文）：✅ 纯 CPU/跨平台，零移植 ｜ 🟢 PyTorch/JAX，ROCm 直接跑 ｜ 🟡 需移植(CUDA→Triton/HIP) ｜ 🔴 NV 专属，放弃

---

## 1. 五大 feature 逐个拆解（能力 + 内部零件 + 开源替代）

### ① IK（逆运动学，collision-free）

**能力**：给定末端位姿求关节角，要求解无碰撞。cuRobo 以几百随机种子并行优化，取收敛且 collision-free 的解。对应 **IK feasibility**。

| 内部零件 | 作用 | 开源替代（ROCm） |
|---|---|---|
| 运动学 FK/Jacobian | 关节角→末端位姿 + 梯度 | **pytorch_kinematics**🟢 / PyRoki(JAX)🟢 / Pinocchio✅(CPU解析) |
| 世界碰撞 + 自碰撞 | 排除撞墙/撞自身的解 | 见 ②（同一套碰撞代价） |
| 代价 rollout + 优化器 | 位姿误差 + 碰撞惩罚，并行优化 | torch + **torch.optim.LBFGS**🟢 / **theseus**🟢 |

### ② 碰撞检测（robot ↔ world，collision-free 核心）

**能力**：判断机器人与环境/自身是否碰撞，输出距离 + 梯度供优化使用。环境支持图元(cuboid)、网格(mesh)、深度图。

| 内部零件 | 作用 | 开源替代（ROCm） |
|---|---|---|
| 机器人 sphere 化 | link 近似为球集合，碰撞降为 sphere-vs-世界 | **foam**✅ + torch 自写 sphere-sphere🟢 |
| 自碰撞 | sphere-sphere 穿透惩罚（v2 用 map-reduce 扛高 DoF） | torch/Triton 自写🟢 |
| 世界图元/网格 SDF | 图元解析 SDF；网格用精确碰撞 | **coal/HPP-FCL**✅、**python-fcl**✅（精确）；**pytorch-volumetric**🟢（mesh→SDF）；图元 SDF torch 自写🟢 |
| (v1) ESDF 体素 | 体素化距离场 | 见 §2 感知 |

### ③ 几何规划（geometric planning）

**能力**：在关节空间求一条无碰撞全局路径，作为轨迹优化初值，规避优化的局部最优。cuRobo 用 GPU 版 PRM 风格图搜索。

| 内部零件 | 作用 | 开源替代（ROCm） |
|---|---|---|
| 图/采样规划 | RRT*/PRM/BIT* 求无碰撞路径 | **OMPL**✅（金标准,CPU）/ **VAMP**✅（SIMD,极快）/ Drake✅ |

> 该块负责出初值，非性能热点，rocm-robot 直接用 CPU 的 OMPL/VAMP，不做 GPU 化。

### ④ 轨迹优化（trajectory optimization）

**能力**：将初始路径优化为平滑、短、低 jerk、无碰撞的轨迹；v2 升级为 B-spline 控制点优化，内生平滑并支持力矩限制。motion plan 的优化分支。

| 内部零件 | 作用 | 开源替代（ROCm） |
|---|---|---|
| 代价 rollout | 平滑/jerk/加速度/边界/碰撞 代价 + 梯度 | 纯 torch 自写🟢 |
| 优化器 | L-BFGS（梯度）/ MPPI（粒子） | **torch.optim.LBFGS**🟢 / **pytorch_mppi**🟢 / theseus🟢 |
| 轨迹表示 | waypoint(v1) / B-spline(v2) | scipy 基函数 + torch 优化🟢 |
| 轨迹重定时 | 速度/加速度限内重分配时间 | **Ruckig**✅ / **TOPP-RA**✅ |
| 参考体系 | 整套轨迹优化方案 | **TrajOpt/sco**✅ / **Drake** kin-trajopt✅ / Crocoddyl✅ |

### ⑤ Motion Generation（编排，一站式 collision-free 运动）

**能力**：一次 `plan()` 串联 ①IK → ③几何规划 → ④轨迹优化 → 重定时，约 30ms 输出全局无碰撞运动。本身为编排器，非新算法。

| 内部零件 | 作用 | 开源替代（ROCm） |
|---|---|---|
| motion_gen 编排 | 串联 IK/规划/trajopt/retiming + 失败重试 | **自写编排层**（拼接 ①③④） |

### (底座) 数值优化引擎

①IK、④trajopt、MPC 调用同一套批量并行优化器，写一次复用三处，是 cuRobo 代码组织的核心。

| 内部零件 | 作用 | 开源替代（ROCm） |
|---|---|---|
| L-BFGS（牛顿系，带 GPU line-search） | 梯度法收敛 | **torch.optim.LBFGS**🟢 / **theseus**🟢 / jaxopt🟢 |
| MPPI / 粒子优化 | 采样法，无梯度/避局部最优 | **pytorch_mppi**🟢 / **storm**(NVlabs,纯torch)🟢 |
| 底层 kernel 库 | FK/几何/line-search/tensor-step | 自写 **Triton**🟡（替代 cuRobo 的 CUDA/Warp） |

---

## 2. v2 增量（baseline 之上，阶段 2/3）

curobo.org(v1) 不含以下两块，二者为 cuRoboV2（[arXiv 2603.05493](https://arxiv.org/abs/2603.05493)）的核心。

### 感知：GPU-native TSDF → 稠密 ESDF

**能力**：深度图→block-sparse TSDF→全工作空间稠密 ESDF（PBA+ 算法），O(1) 距离查询，比 nvblox 快 10×、省 8× 显存，毫秒级更新，支撑反应式避障。

| 开源替代 | 说明 | ROCm |
|---|---|---|
| **Open3D** | 深度→TSDF 积分、mesh 处理 | ✅/🟡 |
| **voxblox / FIESTA / EWOK** | CPU ESDF（ROS），算法参考 | ✅ 非实时 |
| **OpenVDB / NanoVDB** | 稀疏体素结构 | ✅ / 🟡 |
| **PBA**(Parallel Banding) | cuRobo 稠密 ESDF 算法，原版 CUDA，可移植 Triton/HIP | 🟡 自写 |
| nvblox | 对照对象 | 🔴 NV 专属 |

### 动力学感知：可微逆动力学（RNEA）+ 高 DoF

**能力**：计算关节力矩，使轨迹优化支持力矩限制（physically executable）；topology-aware 稀疏 Jacobian + map-reduce 自碰撞，扩展至 48-DoF 人形。

| 开源替代 | 说明 | ROCm |
|---|---|---|
| **Pinocchio** | RNEA/ABA + 解析导数，金标准（CPU 参考） | ✅ |
| **MuJoCo MJX** | JAX 批量动力学 | 🟢 |
| **Brax / jaxsim / torchkin** | 可微刚体动力学 | 🟢 |

---

## 3. GPU 计算底座选型

| 方案 | 说明 | ROCm |
|------|------|------|
| **PyTorch (ROCm build)** | 官方 ROCm wheel，张量/autograd 全可用，首选底座 | 🟢 成熟 |
| **OpenAI Triton** | 官方支持 AMD，自定义 kernel 首选（替代 Warp/CUDA） | 🟢 |
| **JAX (ROCm)** | 函数式 + jit + vmap，适合运动学/动力学；可借 PyRoki/MJX | 🟢 较成熟 |
| **HIP / hipify** | CUDA C++ 翻译为 HIP，适合移植已有 kernel | 🟡 |
| **Taichi** | 跨平台 kernel DSL，备选 | 🟡 |
| NVIDIA Warp | v2 重度依赖 | 🔴 无 ROCm |

---

## 4. rocm-robot 落地建议

### 4.0 第一版 = v0（v1 的最小子集，只跑通主干）

第一版不追完整 v1，先做能端到端闭环的最小主干：**FK → 多种子 IK → 直线初值 → 轨迹优化（平滑 + 碰撞）→ 输出轨迹**。
目的为验证核心回路"运动学 + 碰撞代价 + 优化器"在 ROCm 上跑通，而非覆盖全部场景。

| 维度 | v0 最小子集 | 暂不做（留给完整 v1 / v2） |
|---|---|---|
| 机器人 | 单臂（如 Franka/UR5），URDF + pytorch_kinematics | 双臂 / 人形 / mimic joint |
| ② 碰撞 | sphere-robot vs **图元(cuboid/sphere)** 世界 + 自碰撞 | 网格(mesh) / 深度图 / ESDF 体素 |
| ③ 几何规划 | **跳过**，用关节空间直线插值做初值 | OMPL/VAMP 全局规划 |
| ① IK | 多种子 + torch LBFGS + 碰撞代价 | 高精度(<1mm) / 约束轴规划 |
| ④ 轨迹优化 | waypoint + 平滑/碰撞代价 | jerk/加速度精调、B-spline、力矩限制 |
| 重定时 | 固定时长或简单缩放 | Ruckig / TOPP-RA |
| ⑤ motion gen | 极简编排：IK(目标) → 直线初值 → trajopt | 失败重试、多策略 |

判定 v0 完成：给定一个带几个 cuboid 障碍的场景，能稳定输出一条无碰撞、可执行的关节轨迹。

**社区已有可直接复用/参考的 v0 级开源项目**（v0 不必从零写）：

| 项目 | 内容 | 平台 | 对 v0 的用法 |
|---|---|---|---|
| **PyRoki** (Berkeley, IROS'25) | 可微 FK + IK + 轨迹优化 + 碰撞(球/胶囊)，LM 求解器，**显式对标 cuRobo** | JAX → CPU/GPU/TPU，**AMD 经 JAX-ROCm 可跑** | **最接近 v0，可直接当起点**；其 trajopt 示例即 v0 主干。局限：无采样规划、mesh 近似 capsule、JIT 对 shape 敏感 |
| **pyroboplan** (sea-bass, MIT) | IK + RRT/PRM + cubic 轨迹优化(含碰撞约束)，Pinocchio + coal，纯 Python 手写 | CPU | **最佳"代码蓝本"**：可读性极佳，照其结构用 torch 重写批量 GPU 版 |
| **mplib** (Sapien, ManiSkill 在用) | collision-free 轨迹 + IK + retiming，Pinocchio+FCL+OMPL | CPU (C++/pybind) | 轻量生产级，可做对照基线 / 快速验证 |
| **pymp** (motion-planning) | RRT-Connect + IK + 时间参数化，Pinocchio+hpp-fcl+toppra | CPU | CPU 组合 pipeline 参考 |
| **storm** (NVlabs) | torch MPC/MPPI（在线反应控制） | torch → ROCm | 阶段 2 反应式控制参考 |
| Drake / MoveIt+OMPL / Tesseract | 完整重型 pipeline | C++/ROS | 体系参考，不轻量 |

→ v0 两条捷径：**(a) JAX 路线** 以 PyRoki 为起点在 JAX-ROCm 扩展；**(b) PyTorch 路线** 以 pyroboplan 为结构蓝本用 torch 重写批量 GPU 版。

**AMD 平台推荐（明确结论）**：

- **主线起点 = PyRoki**：唯一现成、跨平台设计、对标 cuRobo 的可微+批量 pipeline，JAX 官方支持 ROCm，最快拿到 GPU 加速 v0。
- **蓝本 + 兜底 = pyroboplan**：纯 CPU，AMD 零摩擦；当 PyTorch 路线的结构蓝本，并作 PyRoki 输出的正确性对照 oracle。
- **快速基线 = mplib**：pip 即装、CPU，先搭"能跑的 demo"做对照。
- **不建议** storm 起步做 v0（在线 MPC，非离线规划，留阶段 2）。

**决策闸门（feature1 已通过 ✅）**：在 gfx950 上 `jax[rocm]`（AMD 官方 `rocm/jax` docker）+ PyRoki trajopt 示例已跑通 GPU → **确定走 JAX/PyRoki 路线**。PyTorch + pyroboplan 仅留作兜底蓝本与对照 oracle，不再作为主线。

### 4.1 总策略（KISS，feature1 后已定）

> **路线已定（feature1 验证）**：底座走 **JAX-ROCm + PyRoki**，而非本文最初设想的 torch+Triton 从零自写重写。
> rocRobo **不追求"每个组件都自写"——能直接复用 PyRoki 的就复用**（其可微 FK/IK/碰撞代价/trajopt 求解器现成且显式对标 cuRobo）。充分利用 PyRoki 是更优解。
> rocRobo 的自有价值 = ① AMD enablement（JAX-ROCm 跑通 PyRoki）｜② 多种子 IK 选择｜③ 极简编排层 `plan()`｜④ GPU 批量拐点认知；后续按需把单个组件原生化（如 feature3 图元 SDF）。

1. 底座 = **JAX-ROCm**（官方 `rocm/jax` docker），**PyRoki 为 v0 主干起点**；需自定义 kernel 时退到 PyTorch-ROCm + Triton（非主线）。
2. 精确碰撞(coal/FCL)、几何规划(OMPL/VAMP)、动力学(Pinocchio)、重定时(Ruckig/TOPP-RA) 仍用 CPU 库，跨平台零移植，按阶段接入。
3. GPU 加速范围 = "批量并行那一圈"（FK/IK/碰撞 rollout/轨迹优化），由 PyRoki + JAX `vmap/jit` 直接拿到，无需自写 CUDA/Triton kernel。

### 4.2 推荐技术栈（按 5 大 feature）

| feature | cuRobo 用 | rocRobo 实际选（JAX/PyRoki 路线，feature1 后） |
|----|----------|--------------|
| 底座 | CUDA + Warp + torch | **JAX-ROCm**（`rocm/jax` docker）；需自定义 kernel 时退 PyTorch-ROCm + Triton |
| ① IK | 自写 CUDA + 并行种子 | **PyRoki IK + 多种子选择（rocRobo 自有）**，JAX `vmap` 批量 |
| ② 碰撞 | sphere + WorldCollision | **PyRoki 图元碰撞**(球/胶囊/HalfSpace/Box)；精确/SDF 留 coal/FCL · pytorch-volumetric（feature3+ 原生化） |
| ③ 几何规划 | GPU PRM | **OMPL / VAMP**（CPU 出初值，不 GPU 化） |
| ④ 轨迹优化 | TrajOpt/B-spline + CUDA | **PyRoki trajopt 代价 + LM 求解器**；retiming 留 **Ruckig**（阶段 1） |
| ⑤ motion gen | motion_gen | **自写编排层 `plan()`（rocRobo 自有）** 串 ①②④ |
| 数值优化 | L-BFGS+MPPI(CUDA) | **PyRoki / jaxls LM**；在线 MPC 留 pytorch_mppi / storm（阶段 2） |
| 感知(v2) | TSDF+PBA(CUDA) | **Open3D TSDF** + Triton PBA（阶段 2） |
| 动力学(v2) | RNEA(CUDA) | **Pinocchio**(CPU) → JAX 批量化（阶段 3） |

### 4.3 分阶段路线（每阶段独立验证）

- **阶段 0｜底座验证（feature1 ✅）**：AMD(gfx950) 装 **JAX-ROCm**，跑通 **PyRoki trajopt 示例**，验证可微 FK/IK/trajopt + GPU 批量（autograd/`vmap`/`jit`）。
- **阶段 v0｜最小主干（feature2 ✅ 功能达成）**：单臂 + 图元碰撞(②) + 多种子 IK(①) + 直线初值 + waypoint 轨迹优化(④) + 极简编排(⑤)；跳过几何规划与 retiming。`plan()` 在障碍场景输出无碰撞越墙轨迹。其中 ①选择+⑤编排为 rocRobo 自有，②④复用 PyRoki。**feature3 已把编排做厚**（成功判定+端点收紧到<1mm+失败重试 42%→75%）。
- **阶段 1｜补齐完整 v1**：在 v0 上加 OMPL/VAMP 几何规划(③) + 网格/深度碰撞(②) + Ruckig/TOPP-RA retiming + jerk 精调，达到 curobo-v1 等价。
- **阶段 2｜感知避障**：Open3D TSDF + Triton 稠密 ESDF，接入 ② 的碰撞代价做反应式避障。
- **阶段 3｜dynamics-aware（对齐 v2）**：Pinocchio/批量 RNEA 加力矩限制 + B-spline 轨迹 + 高 DoF map-reduce 自碰撞。


---

## reference

* [curobo](https://github.com/NVlabs/curobo) ｜ [curobo doc(v1)](https://curobo.org/) ｜ [cuRoboV2 paper (arXiv 2603.05493)](https://arxiv.org/abs/2603.05493)
* 运动学/动力学：[pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) ｜ [PyRoki](https://github.com/chungmin99/pyroki) ｜ [theseus](https://github.com/facebookresearch/theseus) ｜ [Pinocchio](https://github.com/stack-of-tasks/pinocchio) ｜ [MuJoCo MJX](https://github.com/google-deepmind/mujoco)
* 碰撞/SDF/感知：[coal(HPP-FCL)](https://github.com/coal-library/coal) ｜ [python-fcl](https://github.com/BerkeleyAutomation/python-fcl) ｜ [pytorch-volumetric](https://github.com/UM-ARM-Lab/pytorch_volumetric) ｜ [Open3D](https://github.com/isl-org/Open3D) ｜ [voxblox](https://github.com/ethz-asl/voxblox) ｜ [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb)
* 规划/优化/重定时：[OMPL](https://github.com/ompl/ompl) ｜ [Drake](https://github.com/RobotLocomotion/drake) ｜ [VAMP](https://github.com/KavrakiLab/vamp) ｜ [TOPP-RA](https://github.com/hungpham2511/toppra) ｜ [Ruckig](https://github.com/pantor/ruckig) ｜ [Crocoddyl](https://github.com/loco-3d/crocoddyl)
* v0 级整合 pipeline：[PyRoki](https://github.com/chungmin99/pyroki)(JAX,对标cuRobo) ｜ [pyroboplan](https://github.com/sea-bass/pyroboplan)(纯Python教学) ｜ [mplib](https://github.com/haosulab/MPlib)(Pinocchio+FCL+OMPL) ｜ [pymp/motion-planning](https://pypi.org/project/motion-planning/)
* MPC/MPPI：[storm](https://github.com/NVlabs/storm) ｜ [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi)
* 底座：[PyTorch ROCm](https://pytorch.org/) ｜ [Triton](https://github.com/triton-lang/triton) ｜ [JAX ROCm](https://github.com/ROCm/jax)
* 几何体碰撞检测预处理方法：[知乎](https://zhuanlan.zhihu.com/p/2047993856789067468)
    * [CoACD](https://github.com/SarahWeiii/CoACD) ｜ [visacd](https://github.com/3dlg-hcvc/visacd) ｜ [learning convex decomposition](https://github.com/nv-tlabs/learning-convex-decomposition)
