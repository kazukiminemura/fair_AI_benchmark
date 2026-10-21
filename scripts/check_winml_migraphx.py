"""python -m scripts.check_winml_migraphx (no AMD hardware required)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fairbench import cli, devices, windowsml
from fairbench.backends.windowsml_backend import WindowsMLBackend
from fairbench.config import RunConfig
from fairbench.interp import Interpreter
from fairbench.peak import search
from fairbench.system import SystemFingerprint
from scripts.check_winml_tensorrt import main as check_gpu_backend


def main():
    check_gpu_backend(windowsml.MIGRAPHX_PROVIDER)
    args = cli.build_parser().parse_args(['setup-winml', '--migraphx'])
    assert args.migraphx and not args.tensorrt and args.provider is None
    # Preserve adapter identity and the isolated interpreter used for WinML.
    info = Interpreter(path='C:/winml/python.exe', available=True, onnxruntime='test',
                       winml_version='test', winml_devices=[{
                           'provider': windowsml.MIGRAPHX_PROVIDER, 'kind': 'gpu',
                           'key': 'amd-adapter', 'vendor_id': 0x1002, 'vendor': 'AMD',
                           'description': 'AMD Radeon test'}])
    with patch('fairbench.interp.probe_all', return_value=[info]):
        targets, _, _ = devices._ort_targets(SystemFingerprint())
    target = targets[0]
    assert target.vendor == 'amd' and target.kind == 'gpu' and target.backend == 'winml'
    assert target.label == 'Windows ML MIGraphX GPU'
    assert target.precisions == ['fp32', 'fp16', 'int8']
    assert target.python == info.path and target.options['winml_key'] == 'amd-adapter'
    backend = object.__new__(WindowsMLBackend)
    backend.ort = MagicMock()
    session = MagicMock()
    session.io_binding.return_value.get_outputs.return_value = [object()]
    compiled = SimpleNamespace(handle=session, precision='int8', output_names=['y'])
    run, read = backend.prepare_resident(compiled, {'x': np.ones((2, 2), np.float32)}, target)
    # Catalog-owned GPU allocators are not assumed to be CUDA/ROCm allocators.
    assert backend.io_device(target) == 'cpu'
    assert backend.ort.OrtValue.ortvalue_from_numpy.call_args.args[1:] == ('cpu', 0)
    session.io_binding.return_value.bind_output.assert_called_once_with('y', 'cpu', 0)
    run()
    read()
    assert session.run_with_iobinding.call_count == 2
    with patch('fairbench.peak.spawn_worker', return_value={
            'ok': True, 'flops_per_s': 1000, 'correctness': {'checked': True, 'valid': True}}) as spawn:
        peak = search(target, 'precision_gemm_int8', RunConfig(peak_max_size=1024, requests=1))
    assert peak['selected']['resident_io'] and peak['scope'] == 'persistent_requests'
    assert spawn.call_args.args[0]['cfg']['resident_io']
    print('WinML AMD GPU discovery, precision workloads and MIGraphX setup option passed.')


if __name__ == '__main__':
    main()
