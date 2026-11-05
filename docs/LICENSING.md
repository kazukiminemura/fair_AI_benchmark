# 公開・再配布の権利整理

確認日: 2026-11-05。対象はこの作業後のソース・新規ビルドです。
Git履歴の整理方針と、Windows x64配布物のネイティブライセンス確認も含みます。

## 公開するもの

| 対象 | 扱い |
| --- | --- |
| 自作コード・文書・UI・画面画像 | MIT。所有者はこのチャットで、公開・ライセンス許諾の権限があると確認済み。LICENSEの名義はGit履歴のKazuki Minemuraに統一。寄稿者の著作権は各人に残る。 |
| GLSL・SPIR-V | プロジェクトのMITソースと生成物。ソース、再生成方法、ハッシュを保持。コンパイラー自体は配布しない。 |
| 合成モデル・入力 | `models.py` によりローカル生成。外部の学習済み重み・データセットを同梱しない。 |
| Python依存パッケージ | それぞれのライセンスを維持。ソース配布・wheelは依存ライブラリー本体を内包しない。第三者ライセンスの索引は `THIRD_PARTY_NOTICES.md`。 |
| メーカーSDK・EP・ドライバー | fairbenchのMIT対象外。別途導入し、利用者が各配布元の条件に従う。キャッシュや仮想環境を公開しない。 |

MITは商用利用・改変・再配布を許可し、ライセンスと著作権表示の保持を要求します。
本文は [OSIのMITライセンス](https://opensource.org/license/mit) に基づきます。
メーカー名は対応技術の説明用であり、提携・推奨・商標使用の許諾を意味しません。

## AMD設定を同梱しない理由と移行

旧 `fairbench/vitisai_int8_gemm.json` は、Windows ML 2.4.89 / ORT 1.27.1、
Vitis AI EPメタデータ版0.1.0の出力を基に変更された設定でした。
その出力物そのものの再配布許諾を裏付けられないため、所有者の選択により削除しました。
[AMDの公開VAIPコード](https://github.com/amd/vaip) がMITであることだけでは、
別の配布物・版に由来する当該ファイルの許諾根拠にはしていません。

初期状態ではメーカー既定のコンパイラー設定を使います。旧版の自動最適化と速度が変わる
可能性があるため、旧版のNPU結果と比較する場合は設定をそろえて再測定してください。
旧ファイルを復元したり、権利未確認の内容をREADMEへ転載したりはしません。

自身で利用権限を確認した設定だけ、ローカルに用意してください。

```powershell
$env:FAIRBENCH_VITISAI_CONFIG = (Resolve-Path '.\compiler.local.json').Path
uv run fairbench run --targets VitisAIExecutionProvider --workloads precision_gemm_int8
```

この指定はX2のINT8精度別GEMMに限り、ORTとWindows MLの共通経路で適用します。
明示的な `config_file`・`target_opts`・`xclbin`、X2以外のターゲットを優先します。
存在しないファイルは測定前にエラーとします。環境変数未指定時は追加設定しません。
`*.local.json` はGitの対象外です。ローカル設定の内容にMITを自動適用しません。

## 配布物ごとの確認

1. **新しいソーススナップショット / wheel / sdist:** MIT本文・第三者ライセンス索引を同梱します。sdistにはこの文書と寄稿方針も含めます。SDK・キャッシュ・測定結果・削除済み設定が混入していないことをアーカイブ内で確認してください。
2. **履歴付きGitリポジトリ:** 旧AMD設定のパスを全ローカル参照（main・タグ・追跡参照・内部参照）から `git-filter-repo` で除外します。作業前の完全なbundleと未コミット差分はリポジトリ外の非公開バックアップに保存します。公開済みmain・v1.0・v1.1・v1.2への反映には履歴の強制更新が必要です。公開側への反映状況は `git ls-remote` と `.git/filter-repo/commit-map` で照合してください。他者のfork・clone、GitHubの到達不能オブジェクトやキャッシュまで消したことは意味しません。
3. **Windows実行形式ZIP:** `scripts/build.ps1` がビルド環境のLICENSE・NOTICE・COPYING・AUTHORSとPythonのLICENSE.txtを収集し、版とハッシュを `THIRD_PARTY_LICENSES/inventory.json` に保存します。ライセンス本文が見つからないパッケージはビルドを止めます。収集は許諾の自動判定ではありません。ローカル設定 `*.local.json` はソース配布・wheel・ZIPのコピー対象から除外します。
4. **Windows x64 ZIPのネイティブDLL:** 下記の版・構成について配布条件と出所を確認しました。`NATIVE_INVENTORY.json` が全DLL・PYD・EXEのSHA256・コンポーネント・版・出所を記録します。未確認の出所・版やMicrosoftシステムランタイムの混入はビルドを失敗させます。ZIPはMITソースと、個別のライセンスに従うランタイムを合わせた配布物です。ZIP内の全DLLをMITと呼ばないでください。
5. **古い配布物:** 作成済みZIP・wheel・ソースアーカイブは自動で修正されません。今回の変更から再ビルドしたものを使い、既存リリース添付物も個別に差し替えてください。

## 今後の更新

依存関係・OS・アーキテクチャ・Python版を変えたら、収集された実物のライセンスを再確認します。
特にNumPyの同梱BLAS、OpenVINOのネイティブ依存、メーカーEPには親パッケージ以外の条件があります。
PyInstallerの結合実行形式に対する例外は、
[公式ライセンス説明](https://pyinstaller.org/en/stable/license.html) と収集したCOPYING本文で確認できます。
この例外は、別の依存ライブラリーの条件を免除するものではありません。

外部素材の追加・AI支援を含む寄稿は `CONTRIBUTING.md` に従い、出所と許諾を記録してください。

## Windows x64ネイティブ配布の確認結果

| 対象 | 確認・公開時の対応 |
| --- | --- |
| DirectML 1.15.4 | 公式 `Microsoft.AI.DirectML` NuGetからx64 Windows用だけを採用。パッケージSHA256とDLLのSHA256を固定し、ホイールの由来不明な別バイナリーは置き換える。署名も有効であることを確認済み。EULA第1条(a)はWindows/Xbox上の自作MLアプリへの同梱配布を許可。第3条の制限、第三者条項を保持。EULA・LICENSE-CODE・ThirdPartyNoticesを同梱する。 |
| Microsoft VC++ / UCRT / API-set | 旧ビルドが他アプリのPATHから収集していたDLLを含め、**すべてZIPから除外**。VC++ DLLはNumPyに含まれるハッシュ付きのMSVCP DLLも除外。Windows 10以降のOSと、利用者がMicrosoft公式から導入するVC++ x64 Redistributableを前提にする。こちらでは再配布権限を仮定しない。 |
| NumPyのMSVCP別名 | 一部のwheelはMSVCPをハッシュ付き名で参照する。配布するのは必要なファイル名だけ。`start-fairbench.cmd` が `prepare-runtime.ps1` を実行し、利用者のSystem32に導入済みのMSVCPからローカルコピーを作る。第三者バイナリーの改変・再リンクは行わない。このローカルコピーは再配布ZIPへ含めない。 |
| CPython 3.12.10 | Pythonソースと標準拡張を改変せずアプリに組み込む。インタープリターのLICENSE.txt、PSFの著作権・変更概要を保持。OpenSSL 3.0.16（Apache-2.0）、libffi 3.4.4（MIT）、bzip2 1.0.8、liblzma/xz 5.2.5のライブラリー部分（public domain）、zlib 1.3.1（Zlib）、Expat 2.7.1（MIT）の条文・表示を補完する。 |
| NumPy 2.5.4 / OpenBLAS / LAPACK | wheelのライセンスディレクトリーをそのまま収集。OpenBLASのBSD、LAPACKのBSD、GCC runtimeのGPL-3.0-or-later WITH GCC-exception-3.1も保持する。GCC例外の対象であるランタイムの結合だけを理由にfairbench全体をGPLへ変更しない。 |
| ONNX 1.23.2 / ORT DirectML 1.24.4 / ml_dtypes 0.6.0 / protobuf 7.36.2 / cffi 2.1.1 | インストール済みwheelと同一バイトのネイティブコードだけを採用し、各LICENSE・NOTICEを保持。Eigen等の追加条項も収集する。 |
| PyInstaller 6.22.3 | 生成EXEのbootloader例外とruntime hookの条文を保持。PyInstaller自身を単独の製品として再配布しない。 |

DirectMLの固定値:

- NuGet URL: https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg
- NuGet SHA256: `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9`
- `bin/x64-win/DirectML.dll` SHA256: `9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1`

根拠: [公式DirectMLパッケージ](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4) 内の `LICENSE.txt` 第1条(a)・第3条、
[MicrosoftのVC++再配布条件](https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files)、
[公式VC++導入先](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)、
[CPython 3.12.10の外部依存定義](https://github.com/python/cpython/blob/v3.12.10/PCbuild/python.props)、
[Pythonのライセンス](https://docs.python.org/3.12/license.html)。
取得した原文は `packaging/licenses/` にあり、ZIPでは `THIRD_PARTY_LICENSES/native/` に入ります。

ZIPは全体を展開し、`start-fairbench.cmd` から起動してください。VC++ランタイムが未導入なら公式URLを表示して停止します。
Python単体のインストールは不要です。ソースから利用する場合は従来どおり `uv sync` を使います。

検証コマンド:

```powershell
uv run python -m scripts.check_licenses --artifacts dist --frozen dist/fairbench-0.1.0-win64.zip
```

`--artifacts` はwheelとsdistの両方があるディレクトリーを指定します。`uv build --out-dir dist` で両方を生成できます。
この監査の対象はWindows x64の上記固定版です。別OS・別Python版・依存更新は再確認が必要です。
