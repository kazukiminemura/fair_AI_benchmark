# fairbench

ラップトップのCPU・GPU・NPUを、同一のONNXモデルと入力で比較するベンチマークです。
デバイスとランタイムは起動時に自動検出します。利用できないデバイスには理由を表示します。

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
