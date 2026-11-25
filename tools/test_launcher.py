"""Exercise the RTX 4090 launcher with local model, download, and engine fixtures."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_4090_replans_stock_recipe_with_installed_bf16_drafter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stubs = root / 'bin'
            stubs.mkdir()
            calls = root / 'calls.jsonl'
            downloads = root / 'downloads.log'

            def executable(path, source):
                path.write_text(source)
                path.chmod(0o755)

            executable(stubs / 'nvidia-smi', '''#!/usr/bin/env bash
case "$*" in
    *memory.total*) echo 'GPU-test, RTX 4090, 615.78.08, 24564, 24000, 8.9, Disabled' ;;
    *memory.free*) echo '24000' ;;
    *) echo 'CUDA Version: 13.0' ;;
esac
''')
            executable(stubs / 'hf', '''#!/usr/bin/env bash
set -eu
while [[ "$1" != --local-dir ]]; do shift; done
mkdir -p "$2"
echo '{}' > "$2/config.json"
(cd "$2" && sha256sum config.json > SHA256SUMS)
''')
            executable(stubs / 'curl', '''#!/usr/bin/env bash
set -eu
while [[ "$1" != https://* ]]; do shift; done
printf '%s\n' "$1" >> "$DOWNLOADS"
[[ "$2" == -o ]]
printf 'BF16 %s\n' "${1##*/}" > "$3"
''')
            # The pinned upstream manifest still passes the launcher's filename checks.
            # Replace only verification of the tiny stand-in for the pinned payload.
            executable(stubs / 'sha256sum', '''#!/usr/bin/env bash
if [[ ${PWD##*/} == draft && "$*" == '--check --quiet --strict SHA256SUMS' ]]; then
    [[ $(cat model.safetensors) == 'BF16 model.safetensors' ]] &&
        grep -q '^217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee  model.safetensors$' SHA256SUMS
else
    exec "$REAL_SHA256SUM" "$@"
fi
''')
            payload = root / 'payload'
            binary = payload / 'bin/tensorfold-native'
            binary.parent.mkdir(parents=True)
            executable(binary, '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
drafter = pathlib.Path(args[args.index('--drafter') + 1])
assert (drafter / 'model.safetensors').read_text() == 'BF16 model.safetensors\\n'
with open(os.environ['CALLS'], 'a') as f:
    print(json.dumps(args), file=f)
assert all(os.environ[name] == '0' for name in (
    'TF_INKLING2_DRAFT_FP8_HEAD', 'TF_INKLING2_DRAFT_FP8_LAYERS', 'TF_INKLING2_LAZY_HEAD'))
if args[0] == 'plan':
    assert args[args.index('--compute-capability') + 1] == '89'
    assert args[args.index('--free-mib') + 1] == '23665'
    maximum = 300000
    context = int(args[args.index('--context') + 1]) if '--context' in args else maximum
    admitted = context <= maximum
    result = dict(admitted=admitted, context=context, message='fixture memory limit')
    if admitted:
        result.update(pool_tokens=300288, prefix_retention={'pages': 1173, 'bytes': 5074059264, 'marks': 64})
    print(json.dumps(dict(maximum=maximum, result=result)))
    sys.exit(0 if admitted else 1)
assert args[0] == 'serve'
''')
            sha = hashlib.sha256(binary.read_bytes()).hexdigest()
            (payload / 'SHA256SUMS').write_text(f'{sha}  bin/tensorfold-native\n')
            archive = root / 'engine.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                tar.add(payload, arcname='tensorfold-1.3.4-linux-x86_64')

            env = dict(os.environ, CALLS=str(calls), DOWNLOADS=str(downloads),
                       PATH=f"{stubs}:{os.environ['PATH']}", XDG_CACHE_HOME=str(root / 'cache'),
                       INKLING_RECIPES_UPDATED='1', REAL_SHA256SUM=shutil.which('sha256sum'))
            env.pop('CUDA_VISIBLE_DEVICES', None)
            result = subprocess.run([ROOT / 'start.sh', '--no-desktop', '--engine-archive', archive],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            events = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual([event[0] for event in events], ['plan', 'plan', 'serve'])
            self.assertIn('--context', events[0])
            self.assertNotIn('--context', events[1])
            self.assertIn('300288-token KV pool with 300000 tokens per request', result.stdout)
            self.assertEqual(events[2][events[2].index('--context') + 1], '300000')
            self.assertEqual(len({event[event.index('--drafter') + 1] for event in events}), 1)
            urls = downloads.read_text().splitlines()
            self.assertEqual(len(urls), 4)
            self.assertTrue(all(re.fullmatch(
                r'https://huggingface\.co/incoai/Inkling-2-Mini-DFlash2/resolve/'
                r'4f9088ea66c21f2b99ed657d1941c306b03e84d2/(config\.json|model\.safetensors|README\.md|LICENSE)', url
            ) for url in urls))
