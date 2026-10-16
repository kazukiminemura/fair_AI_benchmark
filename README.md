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

WindowsのAMD NPUは、追加のセットアップを実行します。

```sh
uv run fairbench setup-npu
uv run fairbench verify
```

`setup-npu`はMiniforge・CMake・Visual Studio C++ Build Toolsの不足分をwingetで導入し、
AMDのSDKダウンロードページを開きます。ライセンスに同意して通常の「ダウンロード」フォルダーへ保存すると、
AMD署名を検証してドライバーとSDKのインストーラーを起動し、完了後にランタイムを再検出します。
ライセンス・管理者権限の確認とインストーラーの画面操作は必要です。
再起動を求められた場合は再起動後に再実行してください。AMD機でのみ動作します。
AMD公式の[対応環境と前提条件](https://ryzenai.docs.amd.com/en/latest/inst.html)に従います。
ダウンロード済みのSDKは`uv run fairbench setup-npu --installer "C:\path\ryzen-ai-1.8.0.exe"`で指定できます。
導入成功後も、ベンチマークモデルの対応状況は`verify`で確認してください。

問題がある場合は`uv run fairbench setup-npu --check`で、インストールを行わずに
各Python環境・実行プロバイダー・読み込みエラーを再取得できます。

Conda環境の検出と計測は`conda run`で実行し、SDKの環境設定を有効化します。
Condaに登録済みの環境は自動探索します。検出されない場合は、その環境のPythonを
`FAIRBENCH_PYTHON`に指定してください。AMD環境に通常のONNX RuntimeやDirectML版を追加すると
AMD版が上書きされるため避けてください。
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
