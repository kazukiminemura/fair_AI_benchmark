"""ARM GPU/NPU discovery and CoreML CPU-fallback regression checks (no GPU needed)."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fairbench import devices, interp, models, system
from fairbench.backends.base import BackendError
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig


def main():
    assert system.classify_vendor('Apple A18 Pro') == 'apple'
    assert system.classify_vendor('Qualcomm Adreno GPU') == 'qualcomm'
    assert system.classify_kind('Snapdragon Adreno GPU') == 'gpu'
    assert system.classify_kind('Snapdragon Hexagon NPU') == 'npu'
    values = {'machdep.cpu.brand_string': 'Apple M4', 'hw.physicalcpu': '10',
              'hw.logicalcpu': '10', 'hw.memsize': str(16 * 1024**3)}
    def run(args):
        if args[0] == 'sysctl':
            return values[args[-1]]
        return '{"SPDisplaysDataType":[{"sppci_model":"Apple M4","spdisplays_metal":"Metal 3"}]}'
    with patch.object(system, 'IS_WINDOWS', False), patch.object(system.platform, 'system', return_value='Darwin'), \
            patch.object(system, '_run', side_effect=run):
        cpu = system._collect_cpu()
        assert (cpu.name, cpu.vendor, cpu.physical_cores, cpu.logical_cores) == ('Apple M4', 'apple', 10, 10)
        assert system._collect_memory().total_gib == 16
        adapters = system._collect_adapters()
        assert adapters[0].vendor == 'apple' and adapters[0].kind == 'gpu'
        with patch.object(system, '_run', return_value='invalid JSON'):
            assert system._collect_adapters() == []
    fp = system.SystemFingerprint(cpu=cpu, adapters=adapters)
    info = SimpleNamespace(primary=True, available=True, onnxruntime='test', winml_devices=[],
                           providers=['CPUExecutionProvider', 'CoreMLExecutionProvider'], winml_notes=[],
                           winml_version='', path='/python', label='primary', to_dict=lambda: {})
    with patch.object(interp, 'probe_all', return_value=[info]), \
            patch('fairbench.backends.vulkan_backend.probe_runtime', return_value={}):
        discovery = devices.discover(fp, include_openvino=False)
    target = discovery.get('ort:CoreMLExecutionProvider')
    assert discovery.capabilities['apple:cpu']['available']
    assert discovery.capabilities['apple:gpu']['available']
    assert target.precisions == ['fp32', 'fp16']
    npu = discovery.get('ort:CoreMLExecutionProvider:npu')
    assert npu.kind == 'npu' and npu.vendor == 'apple'
    assert discovery.capabilities['apple:npu']['targets'] == [npu.id]
    assert npu.options['provider_options']['MLComputeUnits'] == 'CPUAndNeuralEngine'
    assert 'provider_options' not in target.options
    # CoreML on an Intel Mac must not advertise an Apple Neural Engine.
    fp.cpu.vendor = 'intel'
    with patch.object(interp, 'probe_all', return_value=[info]):
        assert not any(t.kind == 'npu' for t in devices._ort_targets(fp)[0])
    fp.cpu.vendor = 'apple'
    info.primary = False
    with patch.object(interp, 'probe_all', return_value=[info]):
        external = devices._ort_targets(fp)[0]
        assert next(t for t in external if t.kind == 'npu').id == 'ort:CoreMLExecutionProvider:npu@primary'
        assert all(t.python == '/python' for t in external)
    info.primary = True
    fp.cpu.vendor = 'qualcomm'
    fp.adapters = [system.AdapterInfo('Snapdragon Adreno GPU', 'qualcomm', 'gpu')]
    info.providers = ['CPUExecutionProvider', 'DmlExecutionProvider']
    with patch.object(interp, 'probe_all', return_value=[info]), \
            patch('fairbench.backends.vulkan_backend.probe_runtime', return_value={}):
        discovery = devices.discover(fp, include_openvino=False)
    assert discovery.capabilities['qualcomm:gpu']['available']
    assert discovery.get('ort:DmlExecutionProvider').vendor == 'qualcomm'

    backend = OrtBackend.__new__(OrtBackend)
    backend.ort = Mock()
    session = backend.ort.InferenceSession.return_value
    session.get_providers.return_value = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    session.get_outputs.return_value = [SimpleNamespace(name='Y')]
    cfg = RunConfig()
    with patch('fairbench.backends.ort_backend.platform.system', return_value='Darwin'), \
            patch('fairbench.backends.ort_backend.platform.mac_ver', return_value=('13.0', (), 'arm64')):
        for device, scope, policy in ((target, 'coreml_cpu_and_gpu', 'CPUAndGPU'),
                                      (npu, 'coreml_cpu_and_neural_engine', 'CPUAndNeuralEngine')):
            for key, precision in [('smoke', 'fp32'), ('precision_gemm_fp16', 'fp16')]:
                options = Mock()
                backend._session_options = Mock(return_value=options)
                compiled = backend.compile(models.build_one(key, 512), device, cfg, precision)
                assert backend.ort.InferenceSession.call_args.kwargs['providers'] == ['CoreMLExecutionProvider']
                options.add_session_config_entry.assert_called_once_with('session.disable_cpu_ep_fallback', '1')
                assert compiled.meta['execution_scope'] == scope
                assert compiled.meta['provider_options']['MLComputeUnits'] == policy
                assert compiled.meta['accelerator_execution_verified'] is False
            session.get_providers.return_value = ['CPUExecutionProvider']
            try:
                backend.compile(models.build_one('smoke'), device, cfg, 'fp32')
            except BackendError:
                pass
            else:
                raise AssertionError('CoreML initialization fallback must be rejected')
            session.get_providers.return_value = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        with patch('fairbench.backends.ort_backend.platform.mac_ver', return_value=('12.0', (), 'arm64')):
            try:
                backend.compile(models.build_one('smoke'), npu, cfg, 'fp32')
            except BackendError as exc:
                assert 'macOS 13' in str(exc)
            else:
                raise AssertionError('Unsupported OS must not report CPU-only inference as NPU')
    print('ARM GPU/NPU discovery, Apple fingerprint, CoreML policies and fallback checks passed.')


if __name__ == '__main__':
    main()
