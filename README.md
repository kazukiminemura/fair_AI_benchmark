# fairbench

ラップトップのCPU・GPU・NPUを、同一のONNXモデルと入力で比較するベンチマークです。
デバイスとランタイムは起動時に自動検出します。利用できないデバイスには理由を表示します。

PCの標準計測はCPU・GPU・NPUすべてでFP16とINT8を測定します。FP32・FP8・INT4は標準計測の対象外です。
FP16の結果はTFLOPS、INT8の結果はTOPSで表示します。単位はデバイスの種類ではなく測定精度で決まります。
記録されたモデル精度ごとに成功した結果の最高値を表示し、値がない精度は「未計測」とします。
概要にはFP16とINT8だけを表示します。過去のFP32結果は詳細表・履歴で確認できます。
FP32の結果や精度ヒントから別精度の性能値を推定しません。実際の実行精度はランタイムに依存します。
「精度別性能」または標準計測で、全デバイスに同じFP16・INT8の専用GEMM負荷を実行します。
FP16は半精度MatMul、INT8は量子化入出力のQLinearMatMulです。
入力変換と出力変換を含む推論全体を測定します。FP16のCPU実行にはランタイムによるエミュレーションを含む場合があります。
対応しない精度は「実行不可」と理由を表示します。NPUが検出されないPCではNPU性能は測定できません。
詳細表とCLIもFP16はTFLOPS、INT8はTOPSで表示します。FP32は正解照合用の内部参照として利用します。
モデル精度とは別に、演算ノードの実行精度を表示します。OpenVINOはランタイムの演算ノード情報、
ORT CPUは別セッションのプロファイルに含まれる演算ノードの入力型を記録します。
後者は命令・積算精度の検証ではありません。DirectMLなどで演算精度を取得できない場合は「未確認」と表示します。
GPUの入力転送・GPU常駐推論・出力転送は、I/O Bindingで各段階の完了を待ち、それぞれ測定します。
これは同期を含むホスト実時間であり、GPUイベントで測った純粋なカーネル時間ではありません。
GPU常駐推論にはCast・量子化・呼び出し処理も含みます。モデルの準備・GPUバッファの初回確保は分離測定の区間外です。
分離測定は全体性能の測定後に実行し、通常の推論出力と照合します。対応しない経路では取得不可の理由を記録します。
通常・簡易測定とも既定で3回繰り返します。簡易測定は各回1秒以上です。
成功した独立試行が1回以下の項目は再現性を判定できず、「合格」にはしません。

## セットアップと実行

[uv](https://docs.astral.sh/uv/getting-started/installation/)をインストールし、このディレクトリで実行します。
Pythonは`.python-version`に従ってuvが用意します。

```sh
uv sync
uv run fairbench devices
uv run fairbench serve
```

ダッシュボードから計測を開始できます。端末から実行する場合：

```sh
uv run fairbench verify
uv run fairbench run --quick
uv run fairbench run --full
```

計測セットは画面の「計測セット」または`--suite`で選べます。

```sh
uv run fairbench run --suite vision --quick
uv run fairbench run --suite compute --mode throughput --full
```

`standard`（全モデル）・`compute`（行列計算）・`vision`（画像処理）・
`nlp`（Transformer）・`llm`（prefill/decode）から選択できます。
INT8追加は従来どおりで、不要なら`--no-int8`を指定します。

NPUの処理量を測る場合は、画面で「NPU向け INT8・並列測定」を選択します。
CLIでは次を実行します（利用する機器は`--targets`で指定可能）。

```sh
uv run fairbench run --suite npu_peak
```

このセットは静的QDQ形式で入力・重みをINT8量子化したMLPを使用し、並列スループットを測ります。
固定した8組の合成入力で事前校正し、測定入力は別の固定シードで生成します。
NPUでは1/2/4/8/16/32並列を試し、その範囲で最も速かった結果を採用します。
`--requests`で並列数を固定できます。CPU/GPUは自動時に1/2/4/8並列を試します。
既存の動的INT8モデルは維持し、このセットには`dense_static_int8`を使います。
INT8モデルでも全演算がINT8実行になる保証はなく、実行精度はコンパイラと機器に依存します。
理論TOPSへの到達や、この探索範囲外を含む絶対的な最大性能を保証する測定ではありません。

各計測の前に、ホストRAMのコピー帯域を測り、JSONと「信頼性・条件」に保存します。
最大256 MiBの配列2本を使い、ページを確保した後の5回の中央値をGB/sで表示します。
読出しと書込みの合計バイト数を数えたNumPyコピーの実測値で、NPU専用の転送帯域や理論帯域ではありません。
バッファサイズ・各試行の値も記録します。キャッシュやOSの影響を含みます。
Windowsが返す各メモリのデータ幅と設定速度も記録しますが、モジュール数からチャネル数を推測しません。
画面ではセット選択後に個別調整も可能です。CLIでは`--workloads`がセットより優先されます。

バックエンドも画面から選択できます。CLIでは`--backend`を指定します。

```sh
uv run fairbench run --backend winml --suite vision --quick
```

`all`（既定）・`ort`（ONNX Runtime）・`openvino`・`winml`（Windows ML）を指定できます。
画面には利用可能なバックエンドだけを表示します。機器を個別指定した場合は、その条件とバックエンドの両方で絞り込みます。

最大スループットを測る場合：

```sh
uv run fairbench run --mode throughput --full
```

並列数1・2・4・8で探索し、各繰り返しの最高実測値を採用します。
`--requests 16`などで並列数を固定できます（上限32）。
ダッシュボードの「測定モード」からも選べます。
1候補ごとに指定の測定時間を使うため、自動探索では計測が約4倍になります。
これは探索範囲内の最大値で、ハードウェアの理論最大値を保証しません。
単発遅延モードとは別の条件として比較してください。

比較するPCはAC電源につなぎ、電源設定をそろえて実行してください。
結果は`results/`にJSONで保存され、ダッシュボードで比較できます。
`--quick`は動作確認、`--full`は本計測用です。

## Windows MLでGPU・NPUを計測

Windows 11 24H2（ビルド26100）以降では、Windows ML経由でも計測できます。

```sh
uv run fairbench setup-winml
uv run fairbench verify --targets winml
uv run fairbench run --targets winml --mode throughput --full
```

専用の`.venv-winml`を作成し、MicrosoftのWindows ML EPカタログから
対応する認証済みプロバイダーを取得します。初回はネット接続が必要です。
GPU・NPU・CPUを個別に検出し、ダッシュボードにも「Windows ML」として表示します。
GPU・NPUはONNX RuntimeによるCPUへのフォールバックを禁止し、非対応モデルは失敗理由を表示します。
この条件では、一部の演算だけCPUを必要とするモデルも計測不可になります。
対応ドライバーとメーカー別プロバイダーの対応状況に依存します。
詳細は[MicrosoftのEP対応表](https://learn.microsoft.com/windows/ai/new-windows-ml/supported-execution-providers)を参照してください。

## 自動セットアップの範囲

`uv sync`はOS・CPUアーキテクチャに応じて依存関係を選びます。
実際に搭載されているデバイスは、起動時にONNX RuntimeとOpenVINOへ問い合わせます。
ハードウェアごとのextras指定は不要です。

| 環境 | 標準で導入するランタイム |
| --- | --- |
| Windows x64 | ONNX Runtime DirectML（CPU・Intel/AMD/NVIDIA GPU）とOpenVINO（Intel CPU・GPU・NPU） |
| Linux x64 | ONNX Runtime CPUとOpenVINO（Intel CPU・GPU・NPU） |
| その他 | ONNX Runtime CPU。x64ではOpenVINOも導入 |

GPU・NPUの利用には対応ハードウェアとメーカー製ドライバーが必要です。
ドライバーやメーカー配布SDKは`uv sync`ではインストールしません。
ランタイムの提供状況によってはOS・Pythonバージョンに制約があります。
LinuxのGPUドライバー・権限設定も別途必要です。

WindowsのAMD NPU（Vitis AI）は次のコマンドでセットアップします。

```sh
uv run fairbench setup-npu
```

Windows MLの認証済み`VitisAIExecutionProvider`だけを取得し、AMD NPUを検出して
推論できることまで確認します。GPUしか見つからない場合や推論が失敗する場合は、
セットアップ未完了としてエラーを返します。Windows 11 24H2以降と対応するAMD NPUドライバーが必要です。
ドライバーは[AMD公式手順](https://ryzenai.docs.amd.com/projects/WinML/en/latest/installation.html)に従って導入してください。
EPカタログに対象が出ない場合は、Windows Update・ドライバー・ハードウェアの対応状況を確認してください。

`uv run fairbench setup-npu --check`で再インストールせず診断・推論確認できます。
従来のRyzen AI SDK導入経路は`--sdk`、取得済みのSDKは`--installer "C:\path\ryzen-ai-1.8.0.exe"`で指定できます。
SDK経路では同意・管理者権限確認とインストーラー画面の操作が必要です。
Conda環境を自動検出できない場合は、そのPythonを`FAIRBENCH_PYTHON`に指定してください。
AMD環境に通常のONNX RuntimeやDirectML版を追加するとAMD版が上書きされるため避けてください。

Qualcomm NPUも対応するQNN環境が別途必要です。複数のPythonはOSのパス区切り文字で指定できます。

NVIDIAでCUDAを使う場合は、専用環境を作成します（Windowsの標準GPU計測には不要）。

```sh
uv run fairbench setup-gpu --cuda
uv run fairbench devices
```

ONNX RuntimeのCPU・DirectML・CUDA・QNN版は同じモジュールを上書きするため、
同じ環境に混在させないでください。追加環境は`.venv-gpu`・`.venv-cuda`、
または`FAIRBENCH_PYTHON`から自動検出します。

## 主なコマンド

| コマンド（`uv run fairbench`に続ける） | 用途 |
| --- | --- |
| `devices` | デバイス・ランタイム・利用不可の理由 |
| `verify` | 各デバイスで推論できるか確認 |
| `serve` | ダッシュボード |
| `run --full` | 全計測 |
| `show results/<file>.json` | 保存した結果を表示 |
| `workloads` | モデル一覧 |
| `info` / `selftest` | 環境情報・動作診断 |

測定対象などの指定は`uv run fairbench run --help`を参照してください。
同一モデル・入力、ウォームアップ、CPU参照出力との照合、プロセス分離を使います。
測定条件と制限の詳細は[測定方法](docs/METHODOLOGY.md)を参照してください。

## 開発・配布

```sh
uv run python -m scripts.check_refactor
uv run ruff check fairbench scripts/check_refactor.py
```

Windows配布物は`pwsh -File scripts/build.ps1`で作成します。
`dist/`のZIPはPython不要のCPU・DirectML版、wheelはPython環境向けです。
ZIPにはOpenVINOを含みません。
