"""Check PC identity collection and report naming without benchmark hardware."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fairbench import report, system


def main():
    with patch.object(system, 'IS_WINDOWS', True), patch.object(system, '_powershell_json', return_value={
            'Manufacturer': ' Example PC Co. ', 'Model': 'Laptop 123 '}):
        assert system._collect_computer() == ('Example PC Co.', 'Laptop 123')
    for data in (None, {}, {'Manufacturer': 'To Be Filled By O.E.M.', 'Model': 'System Product Name'}):
        with patch.object(system, 'IS_WINDOWS', True), patch.object(system, '_powershell_json', return_value=data):
            assert system._collect_computer() == (None, None)
    saved = {'schema': 'fairbench.result/1', 'system': {
        'manufacturer': '日本PC / Maker', 'model': 'Laptop', 'cpu': {'name': 'CPU <test>: 123'}}}
    with TemporaryDirectory() as folder:
        path = report.save(saved, folder)
        assert '日本PC-Maker-CPU-test-123-' in path.name
        assert report.load(path)['system'] == saved['system']
        assert report.list_results(folder)[0]['manufacturer'] == '日本PC / Maker'
        explicit = Path(folder) / 'chosen.json'
        assert report.save(saved, explicit) == explicit
        legacy = report.save({'system': {}}, folder)
        assert 'unknown-maker-unknown-cpu' in legacy.name
    assert len(report.default_name(system={'manufacturer': 'A'*1000, 'cpu': {'name': 'B'*1000}})) < 200
    print('PC manufacturer/model collection, missing SMBIOS, safe filenames, saved identity and explicit paths passed.')


if __name__ == '__main__':
    main()
