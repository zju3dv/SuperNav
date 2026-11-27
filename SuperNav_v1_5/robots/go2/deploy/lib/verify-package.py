#!/usr/bin/env python3
"""Offline validation of release inputs, without loading robot dependencies."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from supernav.prepare import verify_inputs


def main():
    manifest, _ = verify_inputs()
    scripts = list((ROOT / 'deploy').rglob('*.sh'))
    scripts += [ROOT / 'deploy/system/gopro/gopro-usb-reconcile']
    scripts += list((ROOT / 'supernav/overlay').rglob('*.sh'))
    scripts += list((ROOT / 'deploy/config').glob('*.example'))
    for script in scripts:
        subprocess.run(['bash', '-n', str(script)], check=True)
    for path in (ROOT / 'supernav').rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    rules = (ROOT / 'deploy/system/gopro/70-gopro-webcam.rules').read_text()
    if 'gopro-webcam-reconcile.service' in rules:
        raise SystemExit('legacy video autostart is not supported')
    print(f"PASS: public DimOS {manifest['upstream_commit']}; overlay/patch hashes; Python and Bash syntax")


if __name__ == '__main__':
    main()
