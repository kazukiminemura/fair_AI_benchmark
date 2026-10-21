"""Isolated AMD GPU hardware checks through the Windows ML EP Catalog.

Run with .venv-winml/Scripts/python.exe -m scripts.validate_winml_gpu.
"""

import argparse
import json
import sys
from pathlib import Path

from fairbench import windowsml
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.report import json_safe
from fairbench.worker_process import spawn_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('migraphx', 'dml'), default='migraphx')
    parser.add_argument('--gemm-size', type=int, choices=(512, 1024, 2048, 4096), default=4096)
    parser.add_argument('--mode', choices=('latency', 'throughput'), default='throughput')
    args = parser.parse_args()
    provider = windowsml.MIGRAPHX_PROVIDER if args.provider == 'migraphx' else 'DmlExecutionProvider'
    runtime = windowsml.probe(provider_name=provider)
    selected = [device for device in runtime['winml_devices']
                if device['provider'] == provider and device['kind'] == 'gpu' and device['vendor_id'] == 0x1002]
    if not selected:
        print(f'AMD GPU unavailable: {provider}. Run fairbench setup-winml --migraphx and check drivers.')
        return 1
    device = selected[0]
    target = Target(f'winml:{provider}:gpu:{device["key"]}', 'winml', provider, 'gpu', 'amd',
                    f'Windows ML {args.provider} GPU', detail=device['description'], python=sys.executable,
                    options={'winml_key': device['key'], 'windowsml_version': runtime['winml_version']})
    cfg = RunConfig(mode=args.mode, requests=1, gemm_size=args.gemm_size, warmup=3,
                    min_seconds=3, max_seconds=8, repeats=1, measure_power=False)
    cells = []
    path = Path(f'results/validation-winml-{args.provider}-{args.gemm_size}-{args.mode}.json')
    path.parent.mkdir(exist_ok=True)
    for precision in ('fp16', 'int8'):
        result = spawn_worker({'target': target.to_dict(), 'cfg': cfg.to_dict(),
                               'workload': f'precision_gemm_{precision}', 'precision': precision},
                              timeout=240, tag=f'validate-winml-{args.provider}-{args.gemm_size}-{precision}-{args.mode}')
        cells.append(result)
        path.write_text(json.dumps(json_safe({'schema': 'fairbench.validation/1', 'runtime': runtime,
                                             'config': cfg.to_dict(), 'cells': cells}), indent=2), encoding='utf-8')
        print(f'{precision}: ' + (f'{result["units_per_s"]:.3f} requests/s' if result.get('ok') else result.get('error', 'failed')),
              flush=True)
    print(f'Saved {path.resolve()}')
    return 0 if all(cell.get('ok') and cell.get('correctness', {}).get('valid') for cell in cells) else 1


if __name__ == '__main__':
    raise SystemExit(main())
