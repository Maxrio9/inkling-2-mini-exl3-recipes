#!/usr/bin/env python3
"""Print a TensorFold serve config for Inkling 2 Mini, sized by the engine's own planner.

This is a thin wrapper around `tensorfold-native plan MODEL --device-mib N --parallel P --json`.
The planner reads the model branch's checkpoint headers and runs the engine's allocation sizing and
startup admission checks without a GPU. This script maps a card class to the planner's inputs and
turns its answer into TOML.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Card classes: device memory as nvidia-smi reports it for the smallest common card of the class,
# the model branch, the default request count, and the engine reserve each recipe starts with.
# reserve_gib None keeps TensorFold's default (discrete: max(1 GiB, memory/32); unified: max(4 GiB, memory/10)).
CARDS = {
    '16': dict(device_mib=16303, unified=False, branch='3.3bpw', parallel=1, reserve_gib=0.5),
    '24': dict(device_mib=24564, unified=False, branch='4.0bpw', parallel=2, reserve_gib=None),
    '32': dict(device_mib=32607, unified=False, branch='4.0bpw', parallel=4, reserve_gib=None),
    # DGX Spark: MemTotal and the MemAvailable measured before launch on an otherwise idle Spark.
    'spark': dict(device_mib=124608, free_mib=119280, unified=True, branch='4.0bpw', parallel=8, reserve_gib=None),
}
DESKTOP_GIB = 1.0


def planner_args(card, desktop, parallel, drafts, kv, reserve_gib):
    c = CARDS[card]
    args = ['--device-mib', str(c['device_mib']), '--parallel', str(parallel), '--kv-cache', kv]
    if c['unified']:
        args.append('--unified')
    if 'free_mib' in c:
        # Spark: the measured free memory already excludes the system; the desktop allowance comes off it.
        free = c['free_mib'] - (int(DESKTOP_GIB * 1024) if desktop else 0)
        args += ['--free-mib', str(free)]
    elif desktop:
        args += ['--desktop-reserve-gib', str(DESKTOP_GIB)]
    if reserve_gib is not None:
        args += ['--reserve-gib', str(reserve_gib)]
    if not drafts:
        args.append('--no-drafts')
    return args


def find_binary(given):
    for candidate in [given, shutil.which('tensorfold-native'), ROOT / 'bin/tensorfold-native', Path.cwd() / 'bin/tensorfold-native']:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    sys.exit('tensorfold-native not found: pass --binary PATH (a build that has `plan --device-mib`)')


def toml(report, card, desktop, reserve_gib, branch, kv):
    context = report['rounded_down']
    if not context:
        sys.exit(f"no context fits: {report['result'].get('message') or 'the planner refused every window'}")
    lines = [f'# Model branch: {branch}. Pass its downloaded directory to serve.']
    if reserve_gib is not None:
        lines.append(f'# Start with TENSORFOLD_MEMORY_RESERVE_GIB={reserve_gib:g} set.')
    if desktop:
        lines.append(f'# Leaves {DESKTOP_GIB:g} GiB for the desktop session on this GPU.')
    lines.append('# Context per request includes input and output; sized by tensorfold-native plan.')
    lines += [f'context = {context}', f"parallel = {report['parallel']}", f'kv-cache = "{kv}"', 'prompt-cache-gib = 0']
    if not report['drafts']:
        lines.append('no-drafts = true')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--card', choices=sorted(CARDS), required=True)
    p.add_argument('--model', required=True, help='downloaded model branch directory (its headers are read)')
    p.add_argument('--binary', help='tensorfold-native with the planner')
    p.add_argument('--desktop', action='store_true', help=f'leave {DESKTOP_GIB:g} GiB for a desktop session on this GPU')
    p.add_argument('--parallel', type=int, help='simultaneous requests (default per card)')
    p.add_argument('--no-drafts', action='store_true', help='plain decoding without DSpark')
    p.add_argument('--kv', choices=['fp8', 'bf16'], default='fp8')
    p.add_argument('--reserve-gib', type=float, help='engine reserve (default per card)')
    p.add_argument('--json', action='store_true', help="print the planner's full report instead of TOML")
    a = p.parse_args(argv)
    c = CARDS[a.card]
    reserve = a.reserve_gib if a.reserve_gib is not None else c['reserve_gib']
    parallel = a.parallel or c['parallel']
    cmd = [find_binary(a.binary), 'plan', a.model, *planner_args(a.card, a.desktop, parallel, not a.no_drafts, a.kv, reserve), '--json']
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode not in (0, 1) or not run.stdout.strip().startswith('{'):
        sys.exit(f"planner failed ({run.returncode}): {run.stderr.strip() or run.stdout.strip()}")
    report = json.loads(run.stdout)
    if a.json:
        print(json.dumps(report, indent=2))
        return
    sys.stdout.write(toml(report, a.card, a.desktop, reserve, c['branch'], a.kv))


if __name__ == '__main__':
    main()
