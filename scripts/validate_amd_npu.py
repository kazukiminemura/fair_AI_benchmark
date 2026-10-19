"""Run on the AMD PC: uv run python -m scripts.validate_amd_npu."""
from fairbench import report, system
from fairbench.cli import _print_progress
from fairbench.config import RunConfig
from fairbench.devices import discover
from fairbench.runner import run_suite


def main() -> int:
    fingerprint = system.collect()
    discovery = discover(fingerprint)
    targets = [target for target in discovery.available() if target.backend == 'winml'
               and target.vendor == 'amd' and target.kind == 'npu'
               and target.device == 'VitisAIExecutionProvider']
    if not targets:
        print('AMD Windows ML NPU unavailable. Run uv run fairbench setup-npu on the AMD PC.')
        return 2
    cfg = RunConfig(backend='winml', gemm_size=4096)
    cfg.apply_quick()
    result = run_suite(cfg, target_ids=[target.id for target in targets],
                       fingerprint=fingerprint, discovery=discovery, progress=_print_progress)
    path = report.save(result)
    print(f'Saved: {path.resolve()}')
    expected = {(target.id, precision) for target in targets for precision in ('fp16', 'int8')}
    passed = {(row['target_id'], row['precision']) for row in result['results']
              if row.get('ok') and row.get('correctness_verified')
              and row.get('session', {}).get('cpu_fallback_disabled') is True
              and row.get('session', {}).get('fallback_used') is False
              and row.get('gemm_shape') == [4096, 4096, 4096]}
    if passed != expected:
        print('FAILED: both 4096 GEMMs must pass correctness on the AMD NPU with CPU fallback disabled.')
        return 1
    print('PASSED: AMD NPU FP16 and INT8 4096 GEMM execution and correctness.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
