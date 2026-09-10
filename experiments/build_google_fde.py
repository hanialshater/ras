"""One-time native build. No model or dataset dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(folder, jobs=2):
    source = Path(__file__).resolve().parents[1] / 'third_party/google_fde'
    expected = json.loads((source / 'upstream.json').read_text())
    for file, checksum in expected['sha256'].items():
        if sha(source / file) != checksum:
            raise ValueError(f'Upstream source changed: {file}')
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which('cmake')
    if not cmake:
        raise RuntimeError('Install cmake>=3.24,<4 and ninja first')
    subprocess.run([cmake, '-S', str(source), '-B', str(folder), '-G', 'Ninja',
                    '-DCMAKE_BUILD_TYPE=Release'], check=True)
    subprocess.run([cmake, '--build', str(folder), '--target', 'google_fde', '-j', str(jobs)], check=True)
    library = folder / 'libgoogle_fde.so'
    info = dict(upstream=expected, library_sha256=sha(library),
                adapter_sha256=sha(source / 'bridge.cc'), cmake_sha256=sha(source / 'CMakeLists.txt'))
    (folder / 'build_info.json').write_text(json.dumps(info, indent=2))
    print(f'OFFICIAL_FDE_READY {library}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    build(args.build_dir, args.jobs)
