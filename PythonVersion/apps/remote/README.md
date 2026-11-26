# EDOLAB Relay

EDOLAB Relay 是本地优先的统一实验计算平台。只有执行算法—问题的 **JSONL 科研实验**
可以选择内置“本机”设备或 SSH 服务器，并进入任务冻结、后台恢复、运行记录、结果校验
和删除流程。Benchmark、NBN 与结果绘图是 Relay 进程内的本机即时操作，不经过这条实验
调度链。浏览器、SQLite 控制数据库和实验最终结果都由 Relay 管理；SSH 服务器只接收
JSONL 实验所需的不可变代码快照，并在结果完整回传和校验通过后清理对应远端目录。

详细状态机、安全边界与传输协议见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 平台范围

当前 Relay 把以下入口放在同一套控制平面中：

- **运行设备**：内置本机设备和使用 OpenSSH 连接的服务器；设备只供 JSONL 算法实验
  选择。
- **任务配置**：浏览仓库 `tasks/*.jsonl` 任务库，或在新建运行时直接粘贴 JSONL。
- **实验运行**：每次运行独立记录输入、代码身份、设备身份、阶段进度、退出码、日志和
  完整结果。
- **问题 / NBN**：二维等高线、曲面、NBN 三视图和交互模型都由 Relay 本机即时计算，
  当场返回页面；不选择设备、不检查执行环境、不排队，也不创建运行记录或代码快照。
- **结果分析**：运行完成后只形成“候选结果”；用户从候选结果或外部标准结果中显式
  导入，才生成独立的“已导入结果”。只有已导入结果可以预览、筛选、导出、绘制趋势图
  或箱线图，并把兼容 runset 加入 SA（statistical analysis，统计分析）工作台。绘图在
  Relay 本机即时完成，不选择设备、不进入运行队列，也不创建实验 provenance。

## Python 一键启动（推荐）

在仓库根目录运行：

```bash
python startup_remote.py
```

启动器会完成以下检查后，在 `http://127.0.0.1:8766` 打开控制台：

1. 定位 conda `EDOLAB` 环境的 Python 3.11，并在需要时直接重新执行自身；
2. 运行仓库要求的 `python -m pip install -e .`；
3. 运行 `python -m cli list` 验证 registry 与依赖导入；
4. 确认已审阅的生产前端 `frontend/dist/index.html` 存在；
5. 启动唯一的 Relay controller 与本地 Web 服务。

启动器不会使用 `conda run`。如果 conda 不在常见位置，可显式指定：

```bash
EDOLAB_PYTHON=/absolute/path/to/envs/EDOLAB/bin/python python startup_remote.py
```

常用选项：

```bash
python startup_remote.py --port 8877
python startup_remote.py --no-open-browser
python startup_remote.py --data-root /absolute/path/to/relay-data
```

Relay 没有远程登录认证，因此 Python 启动器只允许绑定 loopback 地址。

## 运行设备：本机与 SSH

本节的“运行设备”只适用于 JSONL 算法实验。Python 和 Docker 启动方式都会注册一个受管
的内置本机设备。本机实验仍然遵循完整的科研来源协议：Relay 先固定当前 source snapshot
和输入 SHA-256，再把 snapshot 展开到本次运行专属的私有目录，最后从该目录启动可恢复
的后台 worker。它不会直接在持续变化的源码 checkout 中执行实验。Relay 重启后根据
持久化的运行记录、进程身份和工作目录继续接管；结果完成校验后才清理该次临时执行目录。

这里的“本机”准确地说是 **Relay 控制器所在的运行时**：

- 使用 `python startup_remote.py` 时，本机就是当前宿主机及其 `EDOLAB` 环境；
- 使用 Docker Compose 时，本机就是容器运行时，不等同于不受限制的宿主机。容器只能
  使用已挂载的目录和已暴露给容器的 CPU/GPU；只读挂载的宿主机 checkout 仍会先生成
  容器内该次运行的冻结私有副本。

SSH 设备必须使用 Relay 运行机器上的私钥文件。Relay 只保存私钥路径并调用本机
OpenSSH，不接收密码，不把私钥内容写入 SQLite、代码快照、结果或远端目录。对应公钥
必须事先加入服务器账户的 `authorized_keys`；服务器设置中的用户名、端口、私钥路径和
远端根目录共同构成需要环境检查和指纹固定的执行身份。

## 前端构建与 Git/package-data 策略

`frontend/src/` 是 React/TypeScript 源码，`frontend/dist/` 是 Python 启动器实际提供的
生产 UI。`dist/` 有意进入 Git，并通过 `pyproject.toml` 的 `apps.remote` package-data
进入 wheel/editable 安装；运行 Relay 不需要安装 Node.js。

前端开发统一使用 npm 和现有 `package-lock.json`，不混用 pnpm 或另外生成锁文件。

每次修改前端后必须刷新并提交对应 `dist/`：

```bash
cd apps/remote/frontend
npm ci
npm run build
cd ../../..
git status --short apps/remote/frontend
```

Vite 开发服务器把 `/api` 代理到 `127.0.0.1:8766`。生产源码只连接真实 Relay API；
API 断开时直接显示本地服务不可达及实际错误，不存在演示数据或静默 fallback。

## SSH 主机密钥信任

首次“环境检查”使用 OpenSSH `StrictHostKeyChecking=accept-new`（TOFU）：未见过的主机
密钥会自动写入 Relay 私有的 `known_hosts`，后续任何密钥变化都会被 OpenSSH 拒绝。
服务器卡片会显示完整 SHA-256 指纹；发起科研运行前，应通过服务器管理员、云控制台
或其他可信渠道核对该指纹。未成功完成环境检查、未得到指纹或 Python、Conda、远端
目录、磁盘等硬条件不合格时，后端会阻止任务准入。SSH 可达只表示密钥认证成功。
服务器负载单独显示为非阻断提醒，不会仅因负载偏高拒绝 JSONL 实验；GPU 仅在 task 明确请求
GPU 时成为硬条件。

服务器页将每台服务器显示为一条摘要记录，可展开查看逐项检查结论和 CPU、内存、
load、GPU 的历史曲线。轻量 SSH 采样由本地后台控制器执行并先写入 SQLite；浏览器的
总览、服务器摘要与展开详情都只读本地数据库，不会因打开页面或增加浏览器标签而新增
SSH 请求。默认采样周期为 3 秒、历史保留 72 小时，两者都可在设置页修改。关闭
“允许调度”只会禁止新运行选择该服务器，后台监控仍继续；在成功完成环境检查并固定
SSH 主机指纹前，资源监控显示为暂停。完整环境检查仍由用户显式触发，不会为了资源
监控反复执行 Conda、registry 或工作目录探测。

## 任务库、粘贴 JSONL 与冻结

“任务配置”页读取仓库中的 `tasks/*.jsonl` 任务库，可以查看配置并带入“新建运行”。
新建运行也可以直接粘贴 JSONL。粘贴内容不会写入或改动仓库的 `tasks/` 目录；服务端
会按 UTF-8 原文解析、检查预算与准入，并把原文、SHA-256 和解析后的任务配置冻结到
本次运行的本地证据中。后续本机执行、远端同步、恢复和结果审计始终复用这份冻结输入，
不能被页面中的后续编辑或同名 task 文件替换。

选择设备不会改变 JSONL 的科研语义。算法、问题、参数、problem/algorithm seed、run
数量和并行度仍来自同一份配置；设备选择及其检查快照作为额外 provenance 随运行记录
保存。正式运行前仍应先用固定 seed、小预算配置验证算法和问题。

## Benchmark 与 NBN：本机即时预览

“问题 / NBN”页使用 registry 中真实的问题参数规格。二维等高线和曲面要求
`Dimension == 2`；NBN 三视图与 NBN 模型可用于更高维问题。问题参数、环境序号、seed、
绘图分辨率和 NBN 样本数只构成本次浏览器请求。Relay 直接调用本机已有的 benchmark/NBN
函数并当场返回结果；页面没有设备选择，也不会触发 SSH、环境检查、代码快照、队列、
SQLite 运行记录或重启恢复。

同一个问题的等高线、曲面、NBN 三视图与交互 NBN 使用独立预览标签；绘制另一种视图
不会清除已有视图，同类重画只替换对应标签，切换问题才清空本页预览。需要文件承载的
PNG 写入 `results/remote_control/analyses/interactive/benchmarks/`。该目录是临时预览
缓存，不是科研结果归档；Relay 最多保留最近 128 个受管 UUID 目录，并在新建预览时清理
更早目录。曲面和交互 NBN 模型直接返回浏览器，不创建文件目录。

NBN 的计算量会随样本数增加；页面仍会执行输入与资源上限校验，但这不是服务器准入检查，
也不应把即时预览当作正式科研实验记录。需要可复现、可恢复、可审计的算法—问题计算时，
应把它写成 JSONL 实验运行。

## 运行模块与结果分析模块

Remote 工作台沿用 `GUIMode.m` 的导入驱动模型，并明确拆成两个生命周期互不绑定的模块：

```text
运行模块：提交任务 → 执行与校验 → 运行记录 + 候选结果
结果分析：候选结果/外部标准结果 → 用户显式导入 → 已导入结果
          已导入结果 → 预览/筛选/导出/绘图 → 选择 Runset → SA 工作台 → 分析历史
```

运行模块只负责执行状态、日志、设备、快照、校验和运行证据。一个 batch 运行完成且结果
校验通过后，会自动成为“候选结果”，但候选只是一条可导入来源：不会自动创建结果库记录，
也不能预览、绘图、导出或进入 SA。用户必须在结果分析的“候选结果”中明确选择导入，或者
从当前访问网页的电脑上传 EDOLAB 标准结果 ZIP。

结果分析包含四个独立视图：

1. **候选结果**：只列出运行模块产出的完整标准结果，支持搜索、选择和显式导入；
2. **已导入结果**：管理独立结果副本，并提供预览、筛选、绘图和 Runset 选择；切换 bundle 不清空选择，Runset 可跨 bundle、跨来源任务导出或加入 SA；
3. **SA 工作台**：只接收已导入结果中人工选择或按基准筛选出的兼容 Runset；
4. **分析历史**：保存每次统计分析及其冻结来源快照。

导入时 Relay 把 `summary.sqlite` 与 `trend.sqlite` 复制到独立受管目录并重新校验；删除运行
记录不影响已经导入的结果，删除结果也不删除运行历史，SA 历史则保存自己的冻结来源快照。

从候选结果导入与浏览器 ZIP 上传使用同一套 Runset 内容指纹检查。即使候选来自刚完成的
运行，也会逐个比较完整参数、运行样本、统计值和趋势；重复 Runset 直接跳过，部分重复时
只把非重复 Runset 裁剪为独立、可校验的结果副本，并按 Runset 数量报告导入与跳过结果。

已导入结果按“结果 → Runset”组织。绘图嵌入当前导入结果详情，趋势图读取 `trend.sqlite`，
箱线图读取 `summary.sqlite` 的逐次标量结果；默认只显示当前结果的 Runset，需要跨结果比较
时再明确添加其他导入来源。原始结果 bundle 始终只读。

超大趋势不会因为超过交互绘图点数预算而直接失败。Relay 会在内存中确定性降采样，保留
首尾点及各区间的主要峰值、谷值，并让 `stderr`、`count` 与选中的横坐标保持同一索引；
降采样只作用于本次图表，不会回写或截断 `trend.sqlite` 中的完整趋势。

“生成预览”把 PNG 直接渲染到内存并以 `Cache-Control: no-store` 返回浏览器，不创建文件、
下载链接或运行记录。只有“导出图形”才按勾选的 PNG/PDF/SVG 格式写入
`results/remote_control/analyses/interactive/plots/`；每次正式导出使用独立受管 UUID 目录，
文件受路径、格式、大小与 SHA-256 校验，并按最多 128 个目录滚动清理。

升级前通过旧版本提交的 Benchmark/NBN 或 plot 后台运行仍保留在历史运行记录中，Relay
继续支持读取、下载和按原规则删除这些记录；兼容读取接口不会让新页面再次创建这类运行。

## 已导入结果与 SA

SA 候选是结果库之上的独立列表。第一条候选建立问题、问题参数和运行次数基准；之后手工
加入的 Runset 必须兼容。GUIMode 式“一键筛选”只有列表至少已有一条时才能使用，它以第一
条为模板，从全部导入结果中补齐兼容 Runset。加入候选、选择本次分析项、执行分析和查看
历史是不同步骤。

旧实验结果可以通过“导入外部标准结果”加入同一目录。网页只接收从访问端电脑选择并上传的
EDOLAB 标准 ZIP，不要求浏览器所在电脑与 Relay 共享文件路径。ZIP 可以包含一条或多条结果；
Relay 先在私有 staging 中安全解包，核对归档布局与 manifest、逐文件大小和 SHA-256，再对
每组 `summary.sqlite` 与 `trend.sqlite` 执行标准 bundle 校验，再以 Runset 为单位比较完整问题与
算法参数、运行样本、统计值和趋势内容。重复 Runset 会跳过；同一 bundle 中的非重复 Runset
会被安全裁剪成仍可独立校验的标准结果副本，再原子提升到 `analyses/results/<result-id>/`。
因此部分重复不会阻止其他 Runset 导入，页面显示的导入、跳过和校验失败数量也以 Runset 为
单位。导入记录不会伪装成一条已完成运行。已完成且
校验通过的运行只会出现在候选结果中，也必须由用户显式执行“导入”才进入结果库；普通成功
运行不会自动出现在结果预览、绘图或 SA 中。删除原运行记录也不会改变已导入的受管结果副本。

运行记录、导入结果、当前 SA 候选和 SA 历史是四层独立状态：删除运行记录不删除已导入
结果；删除结果会删除其受管副本，并级联移除当前候选，但已完成 SA 历史保存的是冻结快照，
不会被改写。结果目录中的勾选只用于批量删除结果副本；Runset 选择属于整个结果分析工作区，
切换 bundle 时继续保留。导出前弹窗按 bundle 列出全部所选 Runset，确认后 Relay 同时裁剪
`summary.sqlite` 和 `trend.sqlite`，只保留所选 Runset 及其任务、运行、统计值和趋势，再生成
带校验 manifest 的标准 ZIP。跨 bundle 加入 SA 也会先整体检查兼容性与重复配置，再写入候选
列表，避免只加入部分 bundle。该 ZIP 可直接从
浏览器上传回导，系统仍逐个 Runset 检测、去重并汇报结果。历史 Benchmark/NBN 或 plot 运行
仍按原有终态记录删除规则处理；新的即时预览不属于运行记录，并由 128 目录上限自动轮换。

## JSONL 实验的快照与增量代码同步

以下机制只适用于 JSONL 算法实验；即时 Benchmark/NBN 和结果绘图不会创建或读取源码
快照。Relay 会复用远端已经校验的代码快照，并只上传新增或内容发生变化的文件。删除文件、
权限位变化和仅 Git provenance 变化不会上传源码；完全相同的 snapshot 命中缓存时，
源码上传量为零。每次运行仍会得到独立可写的 `source/`，任务 JSONL 只属于该次运行，
不会进入共享快照缓存。远端缓存使用只读目录树；命中时优先通过 reflink/COW 生成
不同 inode 的运行副本，不支持 reflink 的文件系统安全回退为普通复制。小差异不会再
触发完整快照的重复解压、重压或全量 archive 哈希；旧 archive 缓存只在首次复用时
校验并迁移一次。

运行抽屉把同步细分为“准备独立副本、计算代码差异、上传代码、校验远端副本”四步，
文件/字节进度与科研 run 进度分开显示。运行快照会排除仅供回归测试使用的
测试专用的参考数据，只保留运行所需的文件。

本机设备不需要上传步骤，但不会跳过 snapshot：它同样从内容固定的缓存生成独立
`source/` 副本，再在受管工作目录中运行。这样本机与 SSH 结果可以使用同一套 provenance
和恢复逻辑比较，而不会因为运行期间继续编辑源码导致实际执行版本漂移。

## Docker 启动

Docker 与 Python 启动器使用同一个 `results/remote_control/` 数据库和产物布局；控制器
锁会阻止二者同时写入。启动前确保 Python 版本已经停止。

```bash
mkdir -p results/remote_control
export EDOLAB_RELAY_SSH_KEY="$HOME/.ssh/edolab_relay_ed25519"
export EDOLAB_RELAY_UID="$(id -u)"
export EDOLAB_RELAY_GID="$(id -g)"
docker compose -f compose.remote.yaml up --build
```

然后访问 `http://127.0.0.1:8766`。停止服务：

```bash
docker compose -f compose.remote.yaml down
```

Docker 安全边界：

- host 端口固定发布到 `127.0.0.1`，不会监听局域网或公网接口；
- 整个仓库以只读方式挂载到 `/workspace`，只有
  `/workspace/results/remote_control` 是嵌套的可写挂载；
- SSH 私钥必须通过 `EDOLAB_RELAY_SSH_KEY` 指定绝对路径，并以同一个绝对路径只读挂载
  到容器；它不进入 build context，也绝不复制进镜像。这样 Python 与 Docker 可以共用
  `control.sqlite` 中同一条服务器配置，不需要改写私钥路径；
- 容器使用只读根文件系统、移除全部 Linux capabilities，并启用
  `no-new-privileges`；
- `EDOLAB_RELAY_UID/GID` 应与私钥所有者一致，私钥权限必须为 `0600` 或 `0400`。

Compose 使用镜像内安装好的控制器、依赖和生产前端；当前仓库只读挂载到 `/workspace`
作为科研代码与 task 的事实来源。每次远端 JSONL 实验都从该挂载重新构建不可变 snapshot，
因此镜像不会遮蔽本机最新代码差异，宿主机 checkout 也不会遮蔽镜像内已审阅的 UI。
选择 Docker 中的“本机”设备时，worker 在容器运行时内执行；若实验需要宿主机 GPU，
还必须按宿主机、Docker 运行时和 Compose 的要求显式把 GPU 暴露给容器，仅在宿主机
安装 GPU 驱动并不代表容器自动可用。

## 强制科研不变量

- 所有 SSH 连接只接受 Relay 运行机器上的私钥路径，不保存密码或 key 内容，也不把
  私钥上传到服务器。
- 每次 JSONL 实验运行在同步前固定 commit、dirty diff、snapshot SHA、task 原文与 task SHA；
  直接粘贴的 JSONL 不写入仓库 `tasks/`。
- 本机 JSONL 实验也只从每次运行的私有 snapshot 副本执行，不直接运行可变源码 checkout。
- JSONL 实验的所有声明结果逐文件回传并校验；错误运行同样保存退出码、错误代码和日志。
- checksum 校验始终开启，不能由 UI 或设置关闭。
- 只有完整回传和校验成功后才删除精确标记的远端运行目录；此规则不能关闭。
- 本地磁盘、协议或 checksum 错误不会迫使用户丢弃远端结果：终态记录会显示“恢复回传”，
  继续使用同一份冻结 snapshot、task、服务器身份和 marker-owned 工作区。
- 确定未进入 launch intent 的失败工作区会自动重试安全清理；存在启动不确定性时绝不
  猜测删除，而是先恢复幂等进程身份。
- 删除本地实验运行记录会同时删除该次已经回传的本地结果，且只允许终态记录执行；
  历史 Benchmark/plot 运行沿用同一删除保护。

## 本地状态

默认状态位于 `results/remote_control/`，不进入 Git。数据库、known-hosts、JSONL 实验
快照、日志、本机执行工作区、结果与远端清理状态必须作为一个整体保留。即时绘图缓存位于
`analyses/interactive/`，可由 128 目录轮换机制自动清理，不是恢复实验所需状态；完整目录说明见
[results/README.md](../../results/README.md)。
