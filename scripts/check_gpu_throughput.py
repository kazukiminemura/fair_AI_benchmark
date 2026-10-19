"""Hardware check: python -m scripts.check_gpu_throughput (Intel OpenVINO GPU)."""
import json
from pathlib import Path

from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.report import json_safe
from fairbench.worker import run_cell


def main():
    target = Target('openvino:GPU', 'openvino', 'GPU', 'gpu', 'intel', 'OpenVINO GPU')
    results = []
    for precision in ('fp16', 'int8'):
        for requests in (1, 4):
            cfg = RunConfig(mode='throughput', requests=requests, gemm_size=4096, quick=True,
                            min_seconds=3, max_seconds=8, warmup=3, repeats=1, measure_power=False)
            result = run_cell({'target': target.to_dict(), 'cfg': cfg.to_dict(),
                               'workload': f'precision_gemm_{precision}', 'precision': precision})
            results.append(result)
            path = Path('results/validation-gpu-throughput-4096.json')
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(json_safe({'schema': 'fairbench.validation/1', 'cells': results}),
                                       indent=2), encoding='utf-8')
            assert result.get('ok'), result.get('error')
            assert result['correctness']['valid']
            assert result['measurement_scope'] == 'persistent_requests'
            print(precision, requests, round(result['flops_per_s'] / 1e12, 3), flush=True)
    print(f'Saved {path.resolve()}')


if __name__ == '__main__':
    main()
