# RocRobSim

**AMD GPU（ROCm）上、NV-free 的开源 Isaac Sim 等价物**——机器人仿真 + 运动平台层。

> 定位与 Isaac Sim 全量拆解见 [`docs/isaacsim_study.md`](docs/isaacsim_study.md)。

## Dependencies

| Component | Path |
|:---|:---|
| PyRoki | [`pyroki/`](pyroki/) — `git submodule` ([upstream](https://github.com/chungmin99/pyroki)) |
| rocRobo | [`rocRobo/core/`](rocRobo/core/) — Python package `rocrobo` |

```bash
git submodule update --init --recursive
```

在已有 JAX-ROCm 环境（或 §0 镜像 + 挂载本仓库）下，可编辑安装：

```bash
pip install -e pyroki
pip install -e rocRobo/core
export ROCROBO_ASSETS="$PWD/pyroki/examples/assets"
```

## 主线条

```
RoboSmith（应用层 / 具身数据基础设施 · 已在 AMD 落地）
   资产生成 · CAP 任务/场景 · GraspGen · 专家轨迹 · LeRobot 导出
        ▲ 消费仿真 + 运动
RocRobSim（平台层）= 仿真核 + 运动核 + 衔接
   ├─ C1 仿真核：Genesis（ROCm/Genesis，gs.amdgpu）—— 物理 / 并行 env / 渲染 / 传感器
   ├─ C2 运动核：rocRobo（JAX-ROCm + PyRoki）—— collision-free IK / trajopt / serve  ✅ 已落地
   └─ C3 衔接：rocRobo(JAX) 出轨迹 → Genesis(torch) 执行/校验（serve 进程边界）
```

- **不走** Warp/Newton 迁移、不以 USD/Fabric 为底座、不追实时 RTX 光追；用 **Genesis** 一个平台覆盖物理/渲染/数据/学习。
- **已落地**：rocRobo 运动核（gfx950/gfx942 验证）；Genesis-ROCm 与 RoboSmith 数据管线已在 MI300X 跑通。
- **平台对外能力**：仿真 rollout（Genesis）/ plan→execute（rocRobo→Genesis）/ render→obs（Genesis）。

## Task 覆盖（按 isaacsim_study.md 任务视角）

| Task | 状态 |
|---|---|
| 合成操作数据（SDG，含资产级 real2sim）| ✅ RoboSmith 已落地 |
| 运动生成（collision-free IK/trajopt）| ✅ rocRobo 已落地 |
| 仿真 / 并行 env / 渲染 | 🟢 Genesis-ROCm 现成 |
| 数字孪生 / 遥操作 | 平台可做，优先级低 |
| 场景级 real2sim / sim2real / 世界模型数据 / RL | 🟡 待完善（见下） |

## Next plan

1. **C3 衔接（核心增量）**：rocRobo `serve` 出轨迹 → Genesis 执行 + 碰撞校验，封装统一平台 API。
2. **Genesis-ROCm 基线固化**：在本 repo 跑通 `gs.amdgpu` 加载机器人 + 场景 + 并行 env，与 RoboSmith 用法对齐。
3. **平台 API 收敛**：把仿真 rollout / plan→execute / render→obs 收敛成 API，供 RoboSmith 直接消费。
4. **待完善方向（中期）**：场景级 real2sim（3DGS 待 ROCm）、sim2real co-training、世界模型数据、RL 基线。
