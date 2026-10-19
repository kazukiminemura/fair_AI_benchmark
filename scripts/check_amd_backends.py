"""Run: python -m scripts.check_amd_backends."""
from unittest.mock import patch

from fairbench import devices, interp
from fairbench.system import SystemFingerprint


def main():
    fingerprint = SystemFingerprint()
    fingerprint.cpu.vendor = 'amd'
    targets = [devices.Target('cpu', 'ort', 'CPUExecutionProvider', 'cpu', 'amd', 'CPU'),
               devices.Target('npu', 'winml', 'VitisAIExecutionProvider', 'npu', 'amd', 'NPU'),
               devices.Target('ov', 'winml', 'OpenVINOExecutionProvider', 'cpu', 'amd', 'OpenVINO')]
    with patch.object(devices, '_ort_targets', return_value=(targets, 'test', [])), \
            patch.object(devices, '_openvino_targets', return_value=([], 'test', [])) as native, \
            patch.object(interp, 'probe_all', return_value=[]):
        discovery = devices.discover(fingerprint)
        native.assert_not_called()
        assert [target.id for target in discovery.targets] == ['cpu', 'npu']
        assert 'openvino' not in discovery.runtimes
        fingerprint.cpu.vendor = 'intel'
        assert len(devices.discover(fingerprint).targets) == 3
        native.assert_called_once()
    print('AMD discovery excludes native/Windows ML OpenVINO; Intel paths preserved.')


if __name__ == '__main__':
    main()
