#!/usr/bin/env python3
"""Smoke regression: run tiny train/test optimization to catch crashes/NaN quickly."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / 'auto_optimize.py'),
        '--data-dir',
        str(root.parent),
        '--out-dir',
        str(root / 'results' / 'smoke'),
        '--roi-mode',
        'opencv',
        '--groups',
        '001,002,003',
        '--max-samples-per-group',
        '2',
        '--holdout-list',
        '001/3-3,002/3-4',
        '--search-budget',
        '2',
        '--objective-profile',
        'hr_lf_balanced',
        '--metrics',
        'hr,lf',
    ]
    print('[SMOKE] running:', ' '.join(cmd))
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
