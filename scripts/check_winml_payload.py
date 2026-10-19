"""Run: python -m scripts.check_winml_payload."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fairbench.windowsml import local_provider_library


def main():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / 'store-package'
        (source / 'nested').mkdir(parents=True)
        (source / 'provider.dll').write_bytes(b'provider')
        (source / 'dependency.dll').write_bytes(b'dependency')
        (source / 'nested' / 'config.json').write_text('{}')
        original = Path.cwd()
        try:
            os.chdir(root)
            with patch('fairbench.windowsml.os.add_dll_directory', create=True) as add:
                local = local_provider_library(str(source / 'provider.dll'))
                assert local.read_bytes() == b'provider'
                assert (local.parent / 'dependency.dll').read_bytes() == b'dependency'
                assert (local.parent / 'nested' / 'config.json').read_text() == '{}'
                assert local_provider_library(str(source / 'provider.dll')) == local
                (source / 'dependency.dll').write_bytes(b'new dependency')
                updated = local_provider_library(str(source / 'provider.dll'))
                assert updated != local
                assert (updated.parent / 'dependency.dll').read_bytes() == b'new dependency'
                assert add.called
        finally:
            os.chdir(original)
    print('Complete Windows ML provider payload copy and version invalidation passed.')


if __name__ == '__main__':
    main()
