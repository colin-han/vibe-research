# HumTone —— 哼唱转乐器演奏 App 技术调研报告

> 项目代号：**HumTone**（hum + tone）
> 调研日期：2026-06-24
> 调研方法：多源检索（19 个来源 / 84 条论点 / 25 条对抗式验证 / 17 条确认 / 8 条驳回）
> 平台目标：桌面端（macOS/Windows）+ 移动端（iOS/Android）
> 技术路线：经典 DSP+MIDI+合成管线 ＆ 端到端 AI 音色转换（两者并行调研）

---

## 目录

1. [结论速览](#1-结论速览)
2. [问题分解：哼唱到乐器演奏的完整链路](#2-问题分解哼唱到乐器演奏的完整链路)
3. [路线 1：经典 DSP + MIDI + 音色合成](#3-路线-1经典-dsp--midi--音色合成)
4. [路线 2：端到端 AI 音色转换](#4-路线-2端到端-ai-音色转换)
5. [平台与音频框架](#5-平台与音频框架)
6. [MVP 推荐技术栈与实现路径](#6-mvp-推荐技术栈与实现路径)
7. [许可证与商业化注意](#7-许可证与商业化注意)
8. [重要限制与开放问题](#8-重要限制与开放问题)
9. [引用来源](#9-引用来源)

---

## 1. 结论速览

这个 App **完全可行，并且不需要从零实现 DSP**。两条技术路线都有成熟的开源积木：

- **路线 1（DSP+MIDI+合成）风险最低**：CREPE / librosa pYIN 做音高检测、Basic Pitch 一站式音频转 MIDI、pretty_midi 组装并渲染、FluidSynth 用 SoundFont 合成真实乐器音色——**一个周末即可出可听 MVP**。
- **路线 2（端到端 AI 音色转换）更自然但更重**：DDSP Timbre Transfer 可用预训练模型做人声→乐器，RAVE 支持低延迟实时音色迁移；但需要 GPU、通常按乐器单独训练。

**推荐策略**：先用路线 1 跑通完整管线得到可用 MVP，再按需引入路线 2 获得更丰富的非 General-MIDI 音色。

---

## 2. 问题分解：哼唱到乐器演奏的完整链路

无论哪条路线，核心都是把「人声哼唱信号」映射为「目标乐器演奏信号」。可拆为四段：

```
┌─────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐
│ 哼唱音频输入 │ → │ 音高/音符提取 │ → │ MIDI 编排 │ → │ 目标乐器合成 │
│ (humming)   │   │ (pitch+notes)│   │ (MIDI)   │   │ (instrument) │
└─────────────┘   └──────────────┘   └──────────┘   └──────────────┘
   录音/低延迟        音高检测            音色选择        合成/采样/AI
```

| 阶段 | 子任务 | 关键技术 |
|---|---|---|
| 音频输入 | 录音、降噪、缓冲 | 平台音频框架（Core Audio / Oboe / Web Audio / JUCE） |
| 音高提取 | 基频(F0)估计、有声/无声判断 | autocorrelation / YIN / pYIN / CREPE |
| 音符切分 | onset detection、量化 | onset detection + 节拍量化 |
| MIDI 编排 | 音高→MIDI 音符号、力度、乐器映射 | MIDI 库（pretty_midi 等） |
| 乐器合成 | 用目标音色回放 | SoundFont 采样器 / 合成器 / AI 音色转换 |

---

## 3. 路线 1：经典 DSP + MIDI + 音色合成

这是**低风险、易原型**的主路线。每一段都有现成、经过验证的开源库。

### 3.1 音高检测（Pitch Detection）

| 方案 | 许可 | 置信度 | 关键点 |
|---|---|---|---|
| **CREPE** | MIT | 🟢 高 | 单音 CNN 音高追踪器，直接在时域波形上工作，`pip install crepe` 即可用（CLI 或 Python 模块）。训练数据含人声（如 MIR-1K），**对人声/哼唱效果好** |
| **librosa pYIN** (`librosa.pyin`) | ISC | 🟢 高 | YIN 的概率化两阶段改进。单次调用返回 `(F0, voiced_flag, voiced_prob)`；用 Viterbi 解码 + HMM 平滑，**对噪声单音人声输入比朴素 YIN 更鲁棒** |

> 这两个方案让非 DSP 专家**完全跳过从零实现自相关/YIN**。CREPE 精度更高但更重（神经网络）；pYIN 更轻量、纯算法。

### 3.2 音频 → MIDI（可选一站式）

| 方案 | 许可 | 置信度 | 关键点 |
|---|---|---|---|
| **Spotify Basic Pitch** | Apache-2.0 | 🟢 高 | 给一段音频直接吐出**带 pitch bend 的 MIDI 文件**。CLI：`basic-pitch <output> <input>`。正好替代「检测音高 → 切音符 → 生成 MIDI」整段，对单音（含哼唱）友好 |

> ⚠️ **被验证驳回的论点**：原本期望 Basic Pitch 自带全平台端侧推理 runtime（CoreML/TFLite/ONNX），对抗式投票**驳回**了该说法。**不要假设有开箱即用的移动端部署方案**——端侧部署需要你自己做模型格式转换与原生集成（社区有 TFLite/CoreML 转换报告，但本轮未被主源验证）。

### 3.3 MIDI 生成与编排

| 方案 | 许可 | 置信度 | 关键点 |
|---|---|---|---|
| **pretty_midi** | MIT | 🟢 高 | 纯 Python。把乐器名映射到 General MIDI 音色号（如 `instrument_name_to_program('Cello')` → 42），附加 Note 对象（音高/力度/起止时间），写出 `.mid` |

### 3.4 音色合成回放

| 方案 | 许可 | 置信度 | 关键点 |
|---|---|---|---|
| **pretty_midi `fluidsynth()`** | — | 🟢 高 | 用内置 SoundFont（TimGM6mb.sf2）把 MIDI 直接合成出**真实乐器音色**，无需独立 synth 宿主，全 Python 闭环 |
| **FluidSynth / FluidLite** | LGPL | 🟢 高 | 工业级 SoundFont 合成引擎；FluidLite 是其轻量版，适合嵌入 |
| **sfizz** | BSD | 🟢 高 | SFZ 格式采样器，活跃维护 |

> ⚠️ **细节坑**：pretty_midi 中 `synthesize()` 只生成 NumPy 正弦波；要用 `fluidsynth()` 方法才能得到真实乐器音色。

### 3.5 路线 1 最小流水线（纯 Python）

```python
# 伪代码：CREPE/pYIN → pretty_midi → fluidsynth
import crepe, pretty_midi, librosa

y, sr = librosa.load("humming.wav")
# 方案 A：手动音高检测
time, freq, conf, *_ = crepe.predict(y, sr, viterbi=True)
# 方案 B（更省事）：basic-pitch CLI 直接产出 humming.mid

pm = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Cello'))
# 把 freq 序列聚合成 Note 追加到 inst.notes ...
pm.instruments.append(inst)
audio = pm.fluidsynth()   # ← 真实乐器音色，不是 synthesize()
# 保存/播放 audio
```

---

## 4. 路线 2：端到端 AI 音色转换

更自然、能覆盖非 General-MIDI 的特定/拟真音色，但更重（GPU、训练、潜在音高漂移）。

| 方案 | 许可 | 置信度 | 关键点 |
|---|---|---|---|
| **DDSP**（Google Magenta） | Apache-2.0 | 🟢 高 | 路线 2 首选。**Timbre Transfer** demo 用预训练模型在声源间转换（如「把你的声音变成小提琴」），还能用自己的乐器数据训练自定义 autoencoder 并导出 `.zip` 模型。原理：提取音高/响度，用可微 DSP 以目标音色重合成。适合 Colab 原型 |
| **RAVE**（IRCAM） | MIT | 🟡 中 | 实时 VAE，支持通过 AdaIN 做音色/风格迁移；`--streaming` 缓存卷积模式 + causal 模式，**低延迟**适合交互 App |
| **DDSP-SVC** | MIT | 🟢 高 | 混合管线：ContentVec/HubertSoft 编码器 + RMVPE 音高提取 + DDSP 合成 + NSF-HiFiGAN 声码器，可实时 |
| **Amphion** | MIT | 🟢 高 | 支持 SVC/VC/SVS/TTA 多任务，含 diffusion 方案（DiffComoSVC）等可复用实现 |
| **so-vits-svc** | **AGPL-3.0** ⚠️ | 🟢 高 | SoftVC→VITS 保留音高语调；但**不含任何预训练模型、不含合成**，需自训；且**已官方归档停更**，活跃维护版是 `34j/so-vits-svc-fork` |

### 路线 2 已知局限（验证中暴露）

- **DDSP**：噪声输入音高追踪会出错；拨弦乐器瞬态较弱。
- **RAVE**：基础版本**不显式解耦音高与音色**，直接迁移可能出现音高漂移/伪影，干净的人声→乐器通常需要按乐器单独训练或用 RAVE2 / pitch-conditioned 变体。
- **被驳回**：①「公开有预训练 streaming RAVE 模型可直接下载」**被驳回**——预期要自己按乐器训练，没有开箱即用的哼唱→乐器模型；② DDSP-SVC 的「实时 GUI 接近离线音质」质量声明被弱化。

---

## 5. 平台与音频框架

> ⚠️ **诚实声明**：本轮验证的 17 条结论几乎全部位于算法/模型/MIDI 层（Python / CPU / 服务端）。**跨平台宿主框架本轮未形成验证过的结论**，下表为通用知识参考，建议后续单独验证。

| 平台 / 目标 | 推荐框架 | 说明 |
|---|---|---|
| **一套 C++ 代码覆盖全部四端** | **JUCE** | macOS/Windows/iOS/Android 全支持，音频/MIDI 插件与 App 的事实标准 |
| **iOS / macOS** | Core Audio / AudioUnit | Apple 官方，低延迟 |
| **Android** | **Oboe** | Google 官方低延迟音频库 |
| **Web（含移动浏览器）** | Web Audio API + AudioWorklet | 无需安装、跨平台 |
| **SoundFont / 采样合成** | FluidSynth / FluidLite / sfizz | 嵌入式合成引擎 |

**把 Python 算法库嵌入宿主** 的常见路径：把模型导出为 **ONNX / CoreML / TFLite**，在原生 runtime 中跑推理。注意 Basic Pitch 的多平台端侧 runtime 并非开箱即用（见 §3.2）。

---

## 6. MVP 推荐技术栈与实现路径

### 阶段 0：概念验证（周末）
- **纯 Python**：`librosa`（读音频）→ `crepe` 或 `librosa.pyin`（音高）→ `pretty_midi`（编排）→ `pm.fluidsynth()`（合成回放）。
- 或更省事：`basic-pitch` CLI 一键产 MIDI，再用 FluidSynth 回放。
- 目标：**听到「哼一段 → 大提琴/钢琴演奏出来」**。

### 阶段 1：本地桌面 App
- 用 **JUCE**（C++）或 **Python + GUI 框架**（如 PyQt/Tauri/Electron）封装阶段 0 的管线，加录音 UI 与乐器选择。

### 阶段 2：移动端
- 将音高检测模型导出为 **CoreML（iOS）/ TFLite（Android）**，在 **Oboe（Android）/ AudioUnit（iOS）** 上做低延迟实时处理。
- ⚠️ 实现实时（<50ms 延迟）的端侧音高检测是本项目的核心技术难点，需专项攻关。

### 阶段 3：丰富音色（按需）
- 引入 **DDSP Timbre Transfer** 或 **RAVE**，为目标乐器训练专属音色模型。

---

## 7. 许可证与商业化注意

| 方案 | 许可证 | 商业化影响 |
|---|---|---|
| CREPE | MIT | 友好 |
| librosa | ISC | 友好 |
| Basic Pitch | Apache-2.0 | 友好 |
| pretty_midi | MIT | 友好 |
| DDSP | Apache-2.0 | 友好 |
| RAVE | MIT | 友好 |
| DDSP-SVC | MIT | 友好 |
| Amphion | MIT | 友好 |
| **so-vits-svc** | **AGPL-3.0** | ⚠️ 网络 copyleft，**对商业部署影响很大**；且已归档 |
| FluidSynth / FluidLite | LGPL | 动态链接 OK，静态链接需注意 |

> 商业化前务必复核各依赖的许可证及其传递性。

---

## 8. 重要限制与开放问题

### 本轮调研的局限
1. **WebSearch 全程被限流（HTTP 429）**：许多论点仅来自直接抓取的官方 README，缺少更广泛佐证。**能力存在性判断可靠，质量/性能判断偏弱。**
2. **商业 SaaS API 未覆盖**：云端音频转 MIDI、人声/乐器转换 API（可完全避免自部署模型）本轮未检索到——不是不存在，是限流错过。
3. **跨平台宿主 + 移动端实时低延迟**方案本轮未深入。

### 待回答的开放问题
- 用哪套跨平台宿主（JUCE / React Native + Web Audio / Flutter）能共享代码覆盖四端，并如何嵌入所选音高/MIDI/synth 组件（Python 库 vs ONNX/CoreML runtime）？
- 如何在移动端实现 <50ms 延迟的实时音高检测？用哪个端侧 runtime 与模型？
- 是否存在开箱即用的 RAVE2 / Amphion Vevo2 哼唱→乐器预训练模型？这决定路线 2 是「周末原型」还是「数周数据采集」。
- 有哪些商业 SaaS API 可作为「完全不跑模型」的兜底方案？

---

## 9. 引用来源

### 路线 1（DSP + MIDI + 合成）
- CREPE — https://github.com/marl/crepe
- librosa pYIN — https://librosa.org/doc/latest/generated/librosa.pyin.html
- Basic Pitch — https://github.com/spotify/basic-pitch
- pretty_midi — https://craffel.github.io/pretty-midi/
- FluidSynth — https://github.com/FluidSynth/fluidsynth
- FluidLite — https://github.com/FluidSynth/fluidlite
- sfizz — https://github.com/sfztools/sfizz

### 路线 2（端到端 AI 音色转换）
- DDSP（Magenta）— https://github.com/magenta/ddsp
- RAVE（IRCAM）— https://github.com/acids-ircam/RAVE
- DDSP-SVC — https://github.com/yxlllc/DDSP-SVC
- Amphion — https://github.com/open-mmlab/Amphion
- so-vits-svc（已归档）— https://github.com/svc-develop-team/so-vits-svc

### 平台与框架
- JUCE — https://juce.com/ · https://github.com/juce-framework/JUCE
- Oboe（Android）— https://github.com/google/oboe
- AudioUnit（Apple）— https://developer.apple.com/documentation/audiounit

---

*报告由 deep-research 工作流生成（102 个子代理 / 多轮对抗式验证）。存在性结论高置信，性能/质量结论待后续验证补强。*
