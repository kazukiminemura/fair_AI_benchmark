"""python -m scripts.check_winml_tensorrt (no WinML/NVIDIA hardware required)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fairbench import cli, windowsml, worker
from fairbench.backends.base import BackendError, CompiledModel
from fairbench.backends.windowsml_backend import WindowsMLBackend
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.models import build_precision_gemm
from fairbench.report import format_compute_rate
from fairbench.runner import _aggregate


def main(provider=windowsml.TENSORRT_RTX_PROVIDER):
    target = Target('winml:rtx', 'winml', provider, 'gpu', 'nvidia', 'WinML RTX', options={'winml_key': 'selected'})
    selected = object()
    backend = object.__new__(WindowsMLBackend)
    backend.ort = MagicMock()
    backend.ort.get_ep_devices.return_value = [selected]
    configs = []
    session = MagicMock()
    session.get_providers.return_value = [provider, 'CPUExecutionProvider']
    assigned_provider = provider

    def options(*args):
        config = MagicMock(intra_op_num_threads=1, inter_op_num_threads=1)
        configs.append(config)
        return config

    def create(*args, **kwargs):
        config = kwargs['sess_options']
        assert kwargs['enable_fallback'] is False
        path = Path(config.profile_file_prefix + '.json')
        events = [{'cat': 'Node', 'name': 'fused_kernel_time', 'args': {
            'provider': assigned_provider, 'op_name': provider + '_subgraph'}}]
        def finish():
            path.write_text(json.dumps(events))
            return str(path)
        session.end_profiling.side_effect = finish
        return session

    backend._session_options = options
    backend.ort.InferenceSession.side_effect = create
    info = {'key': 'selected', 'provider': provider, 'kind': 'gpu'}
    model = build_precision_gemm('fp16', 512)
    with patch.object(windowsml, 'device_info', return_value=info):
        compiled = backend.compile(model, target, RunConfig(), 'fp16')
        assert compiled.handle is session
        assert compiled.meta['execution_assignment']['source'] == 'measured_session_profile'
        assert compiled.meta['execution_assignment']['profiling_stopped_before_timing']
        assert compiled.meta['gpu_precision']['tensor_precisions'] == ['float16']
        assert compiled.meta['gpu_precision']['arithmetic_precision'] == 'unverified'
        configs[-1].add_provider_for_devices.assert_called_once_with(
            [selected], {'enable_cuda_graph': '0'} if provider == windowsml.TENSORRT_RTX_PROVIDER else {})
        configs[-1].add_session_config_entry.assert_called_with('session.disable_cpu_ep_fallback', '1')
        backend._profile_events(model, target, RunConfig(), model.make_inputs(1337))
        assert configs[-1].enable_profiling is True
        assigned_provider = 'CPUExecutionProvider'
        try:
            backend.compile(model, target, RunConfig(), 'fp16')
        except BackendError as exc:
            assert 'assignment' in str(exc)
        else:
            raise AssertionError('CPU fallback passed the WinML GPU gate')

    # Measure the QDQ model without claiming native INT8 arithmetic, and require
    # correctness even with --no-check. Exercise latency and throughput returns.
    output = [np.ones(2, dtype=np.float32)]
    fake = SimpleNamespace(compile=lambda *args: CompiledModel(None, 0, 'int8', ['y'], {
        'gpu_precision': {'available': False, 'arithmetic_precision': 'unverified'}}), run=lambda *args: output,
        diagnose=lambda *args: {'available': False},
        prepare_resident=lambda *args: (lambda: output, lambda: output), io_device=lambda *args: 'cpu')
    with patch.object(worker, 'get_backend', return_value=fake), \
            patch.object(worker, 'reference_fingerprint', return_value=worker._fingerprint(output)):
        for mode in ('latency', 'throughput'):
            result = worker.run_cell({'target': target.to_dict(), 'workload': 'precision_gemm_int8', 'precision': 'int8',
                                      'cfg': RunConfig(gemm_size=512, check_correctness=False, mode=mode, requests=1,
                                                       quick=True, min_iterations=1, max_iterations=2, min_seconds=0,
                                                       max_seconds=1, warmup=0, measure_power=False).to_dict()})
            assert result['correctness']['valid'] and result['ok'], result.get('error')
            assert result['units_per_s'] > 0 and result['flops_per_s'] is None
            if provider == windowsml.MIGRAPHX_PROVIDER and mode == 'throughput':
                assert result['measurement_scope'] == 'persistent_requests'
            assert result['measurement_validity'] == 'model_only'
            assert result['session']['measurement_validity'] == 'model_only'
            row = _aggregate([result])[0]
            assert row['measurement_validity'] == 'model_only' and row['flops_per_s'] is None
            assert row['model_operations_per_s'] > 0
            assert 'TOPS (model estimate; native INT8 unverified)' in format_compute_rate(row)
            assert format_compute_rate({'session': {'measurement_validity': 'model_only'}, 'flops': 137438953472,
                                        'units_per_s': 61.87970122051206}).startswith('8.505 TOPS (model estimate;')

    # Provider names in Microsoft's docs differ in capitalization from catalog
    # names. Match case-insensitively and register the catalog's actual name.
    ep = SimpleNamespace(name=provider, certification=1, ready_state=2, library_path='vendor.dll', ensure_ready=MagicMock())
    catalog = MagicMock()
    catalog.__enter__.return_value.find_all_providers.return_value = [ep]
    module = SimpleNamespace(EpCatalog=lambda: catalog, EpCertification=SimpleNamespace(Certified=1),
                             EpReadyState=SimpleNamespace(NotPresent=2))
    ort = MagicMock()
    with patch.dict(sys.modules, windowsml=module, onnxruntime=ort), \
            patch.object(windowsml, '_registered', set()):
        assert windowsml.register(download=True, provider_name=provider.swapcase()) == []
        ort.register_execution_provider_library.assert_called_once_with(provider, 'vendor.dll')
    args = cli.build_parser().parse_args(['setup-winml', '--tensorrt'])
    assert args.tensorrt and args.provider is None
    print(f'WinML {provider} checks passed: V2 device selection, measured-session profiling, no CPU fallback, '
          'model-only timing without native INT8 TOPS, provider spelling and setup command.')


if __name__ == '__main__':
    main()
