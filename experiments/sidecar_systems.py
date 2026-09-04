"""Benchmark the portable semantic sidecar as a search-stage system component.

Unlike the microkernel benchmark, this harness includes calibrated composition,
query-plan ordering, top-k maintenance, and optional mathematically safe early
exit.  ANN traversal, exact filters, network RPC, and downstream ranking remain
outside this stage and must be measured by an integrating search system.

Typical use after ``experiments.export_native_finalists``::

    python -m experiments.sidecar_systems \
      --index results/native_finalists_first_seed/sidecar_index \
      --programs results/native_finalists_first_seed/sidecar_programs \
      --resident-items 500000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import pandas as pd

from ras.semantic_program import ProgramStore


def _cpu_model() -> str:
    try:
        text = Path('/proc/cpuinfo').read_text(errors='ignore')
        for line in text.splitlines():
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown'


def _parse_kv(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        if '=' not in line or line.startswith('top_results'):
            continue
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip()
        try:
            if any(ch in v for ch in '.eE'):
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def _build_rust(repo_root: Path) -> Path:
    crate = repo_root / 'rust' / 'semantic_engine'
    subprocess.run(['cargo', 'build', '--release', '--bin', 'sidecar'], cwd=crate, check=True)
    exe = crate / 'target' / 'release' / ('sidecar.exe' if os.name == 'nt' else 'sidecar')
    if not exe.exists():
        raise FileNotFoundError(exe)
    return exe


def run(args) -> Path:
    repo = Path(args.repo_root).resolve()
    index = Path(args.index).resolve()
    programs = Path(args.programs).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ProgramStore(programs).names()
    if not names:
        raise RuntimeError(f'no programs found under {programs}')
    exe = _build_rust(repo)

    candidate_counts = [int(x) for x in args.candidate_counts.split(',')]
    predicate_counts = [int(x) for x in args.predicate_counts.split(',')]
    rows = []
    for npred in predicate_counts:
        if npred > len(names):
            continue
        chosen = names[:npred]
        positive = chosen
        negative = []
        if args.mixed_signs and npred >= 2:
            positive = chosen[:-1]
            negative = [chosen[-1]]
        for n in candidate_counts:
            topk = max(1, int(round(n * args.keep_fraction)))
            for early in [False, True]:
                cmd = [
                    str(exe),
                    '--index', str(index),
                    '--programs', str(programs),
                    '--positive', ','.join(positive),
                    '--negative', ','.join(negative),
                    '--candidate-count', str(n),
                    '--topk', str(topk),
                    '--repeats', str(args.repeats),
                ]
                if args.resident_items:
                    cmd += ['--resident-items', str(args.resident_items)]
                if not early:
                    cmd.append('--no-early-exit')
                p = subprocess.run(cmd, check=True, text=True, capture_output=True)
                kv = _parse_kv(p.stdout)
                kv.update({
                    'requested_candidates': n,
                    'requested_predicates': npred,
                    'keep_fraction': args.keep_fraction,
                    'mixed_signs': bool(args.mixed_signs),
                    'early_exit_requested': bool(early),
                    'positive': ','.join(positive),
                    'negative': ','.join(negative),
                })
                rows.append(kv)
                print(
                    f"n={n:>6} p={npred} early={early} "
                    f"median={kv.get('median_ms', float('nan')):.3f}ms "
                    f"p95={kv.get('p95_ms', float('nan')):.3f}ms "
                    f"eval={kv.get('evaluated_predicate_fraction', float('nan')):.3f}"
                )

    df = pd.DataFrame(rows)
    csv = out_dir / 'sidecar_latency.csv'
    df.to_csv(csv, index=False)
    env = {
        'timestamp_unix': time.time(),
        'cpu_model': _cpu_model(),
        'machine': platform.machine(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
        'cargo': subprocess.check_output(['cargo', '--version'], text=True).strip(),
        'index': str(index),
        'programs': str(programs),
        'program_names': names,
        'resident_items': args.resident_items,
        'candidate_counts': candidate_counts,
        'predicate_counts': predicate_counts,
        'keep_fraction': args.keep_fraction,
        'repeats': args.repeats,
        'scope_note': 'Measures semantic sidecar scoring + calibrated composition + top-k. Excludes ANN, exact filters, RPC, and downstream ranking.',
    }
    (out_dir / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    print('wrote', csv)
    return csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default='.')
    p.add_argument('--index', default='results/native_finalists_first_seed/sidecar_index')
    p.add_argument('--programs', default='results/native_finalists_first_seed/sidecar_programs')
    p.add_argument('--out-dir', default='results/sidecar_systems')
    p.add_argument('--resident-items', type=int, default=500_000)
    p.add_argument('--candidate-counts', default='5000,20000,100000')
    p.add_argument('--predicate-counts', default='1,2,4,8')
    p.add_argument('--keep-fraction', type=float, default=0.2)
    p.add_argument('--repeats', type=int, default=11)
    p.add_argument('--mixed-signs', action='store_true')
    a = p.parse_args()
    if not (0 < a.keep_fraction <= 1):
        p.error('--keep-fraction must be in (0, 1]')
    run(a)


if __name__ == '__main__':
    main()
