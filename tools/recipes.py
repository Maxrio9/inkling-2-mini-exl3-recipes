#!/usr/bin/env python3
"""Print a TensorFold serve config for Inkling 2 Mini, sized by the engine's own planner.

This is a thin wrapper around `tensorfold-native plan MODEL --device-mib N --parallel P --json`.
The planner reads the model branch's checkpoint headers and runs the engine's allocation sizing and
startup admission checks without a GPU. This script maps a card class to the planner's inputs and
turns its answer into TOML.
"""
import argparse
import json
import math
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
    '16': dict(device_mib=16303, unified=False, branch='3.3bpw', parallel=1, reserve_gib=0.5, kv='fp8'),
    '24': dict(device_mib=24564, unified=False, branch='4.0bpw', parallel=2, reserve_gib=None, kv='fp8'),
    '32': dict(device_mib=32607, unified=False, branch='4.0bpw', parallel=4, reserve_gib=None, kv='fp8'),
    # DGX Spark: MemTotal and the MemAvailable measured before launch on an otherwise idle Spark.
    'spark': dict(device_mib=124608, free_mib=119280, unified=True, branch='4.0bpw', parallel=8, reserve_gib=None, kv='fp8'),
}
DESKTOP_GIB = 1.0


def planner_args(card, desktop, parallel, drafts, kv, reserve_gib, drafter=None, vision=False, prompt_cache_gib=None):
    c = CARDS[card]
    args = ['--device-mib', str(c['device_mib']), '--parallel', str(parallel), '--kv-cache', kv]
    if c['unified']:
        args += ['--unified', '--compute-capability', '121']
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
    elif drafter:
        args += ['--drafter', str(drafter)]
    if vision:
        args.append('--vision')
    if prompt_cache_gib is not None:
        args += ['--prompt-cache-gib', str(prompt_cache_gib)]
    return args


def find_binary(given):
    for candidate in [given, shutil.which('tensorfold-native'), ROOT / 'bin/tensorfold-native', Path.cwd() / 'bin/tensorfold-native']:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    sys.exit('tensorfold-native not found: pass --binary PATH (a build that has `plan --device-mib`)')


def toml(report, card, desktop, reserve_gib, branch, kv):
    recipe = report.get('recipe', {})
    context = recipe.get('context', report['rounded_down'])
    if not context:
        sys.exit(f"no context fits: {report['result'].get('message') or 'the planner refused every window'}")
    lines = [f'# Model branch: {branch}. Pass its downloaded directory to serve.']
    if reserve_gib is not None:
        lines.append(f'# Start with TENSORFOLD_MEMORY_RESERVE_GIB={reserve_gib:g} set.')
    if recipe.get('plain_head'):
        lines.append('# Start with TF_INKLING2_LAZY_HEAD=0 set.')
    if desktop:
        lines.append(f'# Leaves {DESKTOP_GIB:g} GiB for the desktop session on this GPU.')
    lines.append('# Context per request includes input and output; sized by tensorfold-native plan.')
    lines += [f'context = {context}', f"parallel = {report['parallel']}", f'kv-cache = "{kv}"']
    if recipe.get('prompt_cache_gib') is not None:
        lines.append(f"prompt-cache-gib = {recipe['prompt_cache_gib']}")
    if recipe.get('vision'):
        lines.append('vision = true')
    if not report['drafts']:
        lines.append('no-drafts = true')
    return '\n'.join(lines) + '\n'


def run_planner(command, context=None):
    if context is not None:
        command = [*command, '--context', str(context)]
    run = subprocess.run(command, capture_output=True, text=True)
    try:
        report = json.loads(run.stdout)
    except json.JSONDecodeError:
        sys.exit(f"planner failed ({run.returncode}): {run.stderr.strip() or run.stdout.strip()}")
    if run.returncode not in (0, 1) or 'result' not in report:
        sys.exit(report.get('error_message') or f"planner failed ({run.returncode})")
    result = report['result']
    if result['admitted']:
        if 'pool_tokens' not in result:
            sys.exit('The engine planner must report shared KV pool capacity. Use the paged Mini archive.')
        retention = result.get('prefix_retention')
        if not isinstance(retention, dict) or not all(
            type(retention.get(key)) is int and retention[key] >= 0 for key in ('pages', 'bytes', 'marks')
        ):
            sys.exit('Use the paged Mini planner that reports prefix-retention limits.')
    return report


def startup(args):
    requested, *command = args
    if requested:
        context = int(requested)
        result = run_planner(command, context)['result']
        if result['admitted']:
            print(result['context'], result['pool_tokens'])
            return
        if context == 0:
            sys.exit(result['message'] or 'The full model window does not fit')
    report = run_planner(command)
    if not report['maximum']:
        sys.exit(report['result']['message'] or 'No usable context fits')
    print(report['result']['context'], report['result']['pool_tokens'])


def nonnegative_gib(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be a finite nonnegative number')
    return number


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--startup']:
        startup(argv[1:])
        return

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--card', choices=sorted(CARDS), required=True)
    p.add_argument('--model', required=True, help='downloaded model branch directory (its headers are read)')
    p.add_argument('--binary', help='tensorfold-native with the planner')
    p.add_argument('--desktop', action='store_true', help=f'leave {DESKTOP_GIB:g} GiB for a desktop session on this GPU')
    p.add_argument('--parallel', type=int, help='simultaneous requests (default per card)')
    p.add_argument('--no-drafts', action='store_true', help='plan plain decoding')
    p.add_argument('--drafter', type=Path, help='DFlash2 bundle (default: dflash2 in model directory)')
    p.add_argument('--draft-min-context', type=int, help='use plain decoding if the drafter cannot reach this context')
    p.add_argument('--plain-head', action='store_true', help='budget the plain output head without a lazy-head proxy')
    p.add_argument('--vision', action=argparse.BooleanOptionalAction, help='include vision memory (24 GB and up, default off)')
    cache_policy = p.add_mutually_exclusive_group()
    cache_policy.add_argument('--prompt-cache', action=argparse.BooleanOptionalAction,
                              help='keep reusable pages in the shared pool (default on)')
    cache_policy.add_argument('--prompt-cache-gib', type=nonnegative_gib,
                              help='cap shared pool pages held for prefixes and their restart state in GiB (0 disables caching)')
    p.add_argument('--branch', choices=['3.3bpw', '4.0bpw'], help='model branch written into the config')
    p.add_argument('--kv', choices=['fp8', 'bf16'], help='target KV precision (default per card)')
    p.add_argument('--reserve-gib', type=float, help='engine reserve (default per card)')
    p.add_argument('--json', action='store_true', help="print the planner's full report instead of TOML")
    p.add_argument('--record', type=Path, help='write the capacity report and selected recipe to a file')
    a = p.parse_args(argv)
    c = CARDS[a.card]
    reserve = a.reserve_gib if a.reserve_gib is not None else c['reserve_gib']
    branch = a.branch or c['branch']
    kv = a.kv or c['kv']
    vision = bool(a.vision)
    if vision and a.card == '16':
        p.error('--vision requires a 24 GB or larger card class')
    if a.parallel is not None and a.parallel < 1:
        p.error('--parallel must be positive')
    cache_cap = 0 if a.prompt_cache is False else a.prompt_cache_gib
    if a.plain_head:
        os.environ['TF_INKLING2_LAZY_HEAD'] = '0'
    prefix = [find_binary(a.binary), 'plan', a.model]
    draft = not a.no_drafts

    parallel = a.parallel or c['parallel']

    def command(with_draft):
        args = planner_args(a.card, a.desktop, parallel, with_draft, kv, reserve, a.drafter, vision, cache_cap)
        return [*prefix, *args, '--json']

    cmd = command(draft)
    report = run_planner(cmd)
    drafted = report if draft else None
    if draft and (report.get('drafter') or '').lower() != 'dflash2':
        sys.exit('DFlash2 is required for drafting; pass --drafter DIR')
    if draft and a.draft_min_context and (report['rounded_down'] or 0) < a.draft_min_context:
        cmd = command(False)
        report = run_planner(cmd)

    context = report['rounded_down']
    if not context:
        sys.exit(report['result']['message'] or 'No usable context fits')
    result = report['result'] if context == report['maximum'] else run_planner(cmd, context)['result']
    if not result['admitted']:
        sys.exit(result['message'] or 'The selected context does not fit')
    report['recipe'] = dict(context=context, vision=vision, result=result)
    if cache_cap is not None:
        report['recipe']['prompt_cache_gib'] = cache_cap
    if a.plain_head:
        report['recipe']['plain_head'] = True
    if drafted is not None and drafted is not report:
        report['recipe']['drafted_maximum'] = drafted['maximum']
    if a.record:
        a.record.write_text(json.dumps({**report, 'model': branch}, indent=2) + '\n')
    if a.json:
        print(json.dumps(report, indent=2))
        return
    sys.stdout.write(toml(report, a.card, a.desktop, reserve, branch, kv))


if __name__ == '__main__':
    main()
