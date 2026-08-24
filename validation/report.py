"""Collect the validation artefacts into one report.

Pulls together whatever is present:

* a Dieharder transcript (``dieharder -a`` output),
* a portable-battery result from ``localtests.py``,
* an SP 800-90B assessment from ``sp800_90b.py``,
* the capture summary written by ``radiarandom raw``,

and prints a single verdict. The point is to keep the two kinds of evidence
visibly separate: Dieharder speaks to the *pipeline*, the SP 800-90B estimators
speak to the *physics*, and a report that blurs them is worse than no report.

Usage::

    python validation/report.py --results validation/results
    python validation/report.py --results validation/results --markdown > REPORT.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

DIEHARDER_ROW = re.compile(
    r'^\s*(?P<name>[\w.]+)\s*\|\s*(?P<ntup>\d+)\s*\|\s*(?P<tsamples>\d+)\s*\|'
    r'\s*(?P<psamples>\d+)\s*\|\s*(?P<p>[\d.eE+-]+)\s*\|\s*(?P<verdict>PASSED|WEAK|FAILED)'
)


def parse_dieharder(path: str) -> dict:
    rows = []
    generator = None
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.replace('\x00', '')
            if '|' in line and 'rands/second' not in line:
                match = DIEHARDER_ROW.match(line)
                if match:
                    rows.append({
                        'test': match.group('name'),
                        'ntup': int(match.group('ntup')),
                        'p_value': float(match.group('p')),
                        'verdict': match.group('verdict'),
                    })
                    continue
            if 'stdin_input_raw' in line or 'file_input_raw' in line:
                generator = line.split('|')[0].strip()
    counts = {'PASSED': 0, 'WEAK': 0, 'FAILED': 0}
    for row in rows:
        counts[row['verdict']] += 1
    return {
        'path': path,
        'generator': generator,
        'total': len(rows),
        'counts': counts,
        'failures': [row for row in rows if row['verdict'] == 'FAILED'],
        'weak': [row for row in rows if row['verdict'] == 'WEAK'],
        'rows': rows,
    }


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def newest(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def render(sections: dict, markdown: bool) -> str:
    out: list[str] = []
    h1 = '# ' if markdown else ''
    h2 = '## ' if markdown else ''
    rule = '' if markdown else '=' * 68

    out.append(f'{h1}radiarandom validation report')
    out.append('')

    dieharder = sections.get('dieharder')
    if dieharder:
        counts = dieharder['counts']
        out.append(f'{h2}1. Dieharder -- does the output stream look uniform?')
        out.append('')
        out.append(f'source: `{dieharder["path"]}`  generator: {dieharder["generator"]}')
        out.append('')
        out.append(f'  tests run : {dieharder["total"]}')
        out.append(f'  PASSED    : {counts["PASSED"]}')
        out.append(f'  WEAK      : {counts["WEAK"]}')
        out.append(f'  FAILED    : {counts["FAILED"]}')
        out.append('')
        if dieharder['failures']:
            out.append('  FAILURES (investigate before using this build):')
            for row in dieharder['failures']:
                out.append(f'    {row["test"]} ntup={row["ntup"]} p={row["p_value"]:.8f}')
        elif counts['WEAK']:
            out.append('  Weak results (expected noise across ~114 tests at alpha=0.05):')
            for row in dieharder['weak']:
                out.append(f'    {row["test"]} ntup={row["ntup"]} p={row["p_value"]:.8f}')
        out.append('')
        out.append('  NOTE: this validates the pipeline, not the physics. The stream')
        out.append('  under test passes through HMAC_DRBG, and a correct DRBG passes')
        out.append('  Dieharder regardless of seed quality. See section 2.')
        out.append('')
        if rule:
            out.append(rule)

    entropy = sections.get('sp800_90b')
    if entropy:
        out.append(f'{h2}2. SP 800-90B -- how much entropy does the physics supply?')
        out.append('')
        out.append(f'  samples           : {entropy["samples"]:,}')
        out.append(f'  alphabet          : {entropy["alphabet_size"]} '
                   f'({entropy["bit_width"]} bits/sample)')
        out.append(f'  min-entropy       : {entropy["min_entropy_per_sample"]:.4f} bits/sample')
        out.append(f'  binding estimator : {entropy["binding_estimator"]}')
        if not entropy.get('sufficient_samples'):
            out.append('  WARNING: fewer than 1,000,000 samples; bounds are wider than')
            out.append('           the standard intends. Collect more before relying on this.')
        out.append('')
        out.append('  per-estimator:')
        width = entropy['bit_width']
        for name, detail in entropy['estimators'].items():
            value = detail['h_min'] * (width if name.endswith('_per_bit') else 1)
            note = f'  [{detail["reason"]}]' if 'reason' in detail else ''
            out.append(f'    {name:<24} {value:8.4f}{note}')
        out.append('')
        if rule:
            out.append(rule)

    local = sections.get('localtests')
    if local:
        out.append(f'{h2}3. Portable battery -- pre-flight')
        out.append('')
        out.append(f'  bytes tested : {local["bytes"]:,}')
        out.append(f'  tests graded : {local["tests_run"]}')
        out.append(f'  verdict      : {"PASS" if local["passed"] else "FAIL"}')
        if local['failures']:
            out.append(f'  failures     : {", ".join(local["failures"])}')
        if local['weak']:
            out.append(f'  weak         : {", ".join(local["weak"])}')
        out.append('')
        if rule:
            out.append(rule)

    capture = sections.get('capture')
    if capture:
        out.append(f'{h2}4. Capture summary')
        out.append('')
        out.append(f'  elapsed        : {capture.get("elapsed_s", 0):.0f}s')
        out.append(f'  photons        : {capture.get("photons", 0):,}')
        out.append(f'  count rate     : {capture.get("count_rate", 0):.2f}/s')
        out.append(f'  physical bytes : {capture.get("physical_bytes", 0):,}')
        if 'assessment' in capture:
            out.append(f'  assessment     : {capture["assessment"]}')
        health = capture.get('health', {})
        if health:
            out.append(f'  health         : '
                       f'{"ok" if health.get("healthy") else health.get("failure_reason")}')
        out.append('')

    verdict_lines = []
    if dieharder and dieharder['counts']['FAILED']:
        verdict_lines.append('Dieharder reported failures -- do not ship this build.')
    if local and not local['passed']:
        verdict_lines.append('The portable battery reported failures.')
    if entropy and entropy['min_entropy_per_sample'] <= 0.0:
        verdict_lines.append('The entropy assessment is zero -- the source is predictable.')
    out.append(f'{h2}Verdict')
    out.append('')
    if verdict_lines:
        out.extend(f'  {line}' for line in verdict_lines)
    else:
        out.append('  No failures in any section that was run.')
    out.append('')
    return '\n'.join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', default='validation/results',
                        help='directory holding the artefacts')
    parser.add_argument('--dieharder', help='explicit path to a dieharder transcript')
    parser.add_argument('--sp800-90b', help='explicit path to sp800_90b.py --json output')
    parser.add_argument('--localtests', help='explicit path to localtests.py --json output')
    parser.add_argument('--capture', help='explicit path to a radiarandom raw summary')
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()

    sections: dict = {}

    dieharder_path = args.dieharder or newest(os.path.join(args.results, 'dieharder-*.txt'))
    if dieharder_path and os.path.exists(dieharder_path):
        sections['dieharder'] = parse_dieharder(dieharder_path)

    for key, explicit, pattern in (
        ('sp800_90b', args.sp800_90b, 'sp800_90b*.json'),
        ('localtests', args.localtests, 'localtests*.json'),
        ('capture', args.capture, '*summary.json'),
    ):
        path = explicit or newest(os.path.join(args.results, pattern))
        if path and os.path.exists(path):
            try:
                sections[key] = load_json(path)
            except (OSError, ValueError) as exc:
                print(f'could not read {path}: {exc}', file=sys.stderr)

    if not sections:
        print(f'no validation artefacts found under {args.results}', file=sys.stderr)
        print('run validation/run_dieharder.sh, localtests.py or sp800_90b.py first',
              file=sys.stderr)
        return 1

    print(render(sections, args.markdown))
    dieharder = sections.get('dieharder')
    if dieharder and dieharder['counts']['FAILED']:
        return 1
    if sections.get('localtests') and not sections['localtests']['passed']:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
