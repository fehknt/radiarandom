"""Command-line surface, driven against a simulated detector.

`_open_source` is the single seam between the CLI and the hardware, so
replacing it is enough to exercise every subcommand end to end.
"""

from __future__ import annotations

import json
import re

import pytest

from radiarandom import cli

from conftest import FakeSource


@pytest.fixture
def stub_device(monkeypatch):
    """Point the CLI at a simulated detector and hand the test the source."""
    created = []

    def fake_open(args):
        source = FakeSource(count_rate=3000.0)
        created.append(source)
        return source

    monkeypatch.setattr(cli, '_open_source', fake_open)
    return created


def run(argv, extra=('--startup-samples', '64', '--quiet')):
    return cli.main(list(argv) + list(extra))


# ------------------------------------------------------------ plumbing


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(['--help'])
    assert excinfo.value.code == 0


def test_version_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(['--version'])
    assert excinfo.value.code == 0


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(['definitely-not-a-command'])


# ---------------------------------------------------------------- info


def test_info_reports_the_device(stub_device, capsys):
    assert run(['info', '--rate-window', '2'], extra=()) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert 'FAKE-000000' in out
    assert 'entropy rate' in out
    assert 'OS entropy pool integration' in out


def test_info_json_is_machine_readable(stub_device, capsys):
    assert run(['info', '--rate-window', '2', '--json'], extra=()) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload['serial'] == 'FAKE-000000'
    assert payload['measured_count_rate_per_s'] > 0
    assert payload['projected_entropy_bits_per_s'] >= 0
    assert 'os_pool_contribution' in payload


def test_info_closes_the_device(stub_device):
    run(['info', '--rate-window', '2'], extra=())
    assert stub_device[0].closed


# ----------------------------------------------------------------- gen


@pytest.mark.parametrize('fmt', ['hex', 'base64', 'base64url', 'dec', 'bits', 'c'])
def test_gen_every_text_format(stub_device, capsys, fmt):
    assert run(['gen', '-n', '16', '--format', fmt]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_gen_hex_length_matches_request(stub_device, capsys):
    assert run(['gen', '-n', '24', '--format', 'hex']) == cli.EXIT_OK
    text = capsys.readouterr().out.strip()
    assert re.fullmatch(r'[0-9a-f]{48}', text), text


def test_gen_writes_a_binary_file(stub_device, tmp_path):
    path = tmp_path / 'out.bin'
    assert run(['gen', '-n', '4096', '--format', 'bin', '-o', str(path)]) == cli.EXIT_OK
    data = path.read_bytes()
    assert len(data) == 4096
    # Binary output must not be mangled by newline translation.
    assert len(set(data)) > 200


def test_gen_physical_mode_produces_the_exact_length(stub_device, capsys):
    assert run(['gen', '-n', '32', '--physical', '--format', 'hex']) == cli.EXIT_OK
    assert re.fullmatch(r'[0-9a-f]{64}', capsys.readouterr().out.strip())


def test_gen_two_runs_differ(stub_device, capsys):
    run(['gen', '-n', '32', '--format', 'hex'])
    first = capsys.readouterr().out
    run(['gen', '-n', '32', '--format', 'hex'])
    assert capsys.readouterr().out != first


def test_gen_rejects_a_no_op_request(stub_device, capsys):
    assert run(['gen', '-n', '0']) == cli.EXIT_ERROR


# -------------------------------------------------------- typed output


def test_int_respects_bounds(stub_device, capsys):
    assert run(['int', '--min', '1', '--max', '6', '-c', '50']) == cli.EXIT_OK
    values = [int(v) for v in capsys.readouterr().out.split()]
    assert len(values) == 50
    assert all(1 <= v <= 6 for v in values)


def test_int_rejects_inverted_range(stub_device, capsys):
    assert run(['int', '--min', '10', '--max', '1']) == cli.EXIT_ERROR


def test_float_in_unit_interval(stub_device, capsys):
    assert run(['float', '-c', '20', '--json']) == cli.EXIT_OK
    values = json.loads(capsys.readouterr().out)
    assert len(values) == 20
    assert all(0.0 <= v < 1.0 for v in values)


def test_uuid_is_version_4(stub_device, capsys):
    import uuid
    assert run(['uuid', '-c', '5']) == cli.EXIT_OK
    lines = capsys.readouterr().out.split()
    assert len(lines) == 5
    for line in lines:
        assert uuid.UUID(line).version == 4


def test_password_uses_the_named_alphabet(stub_device, capsys):
    from radiarandom import formats
    assert run(['password', '-l', '24', '-c', '3', '--alphabet', 'digits']) == cli.EXIT_OK
    lines = capsys.readouterr().out.split()
    assert len(lines) == 3
    for line in lines:
        assert len(line) == 24
        assert set(line) <= set(formats.ALPHABETS['digits'])


def test_password_rejects_an_empty_alphabet(stub_device, capsys):
    assert run(['password', '--alphabet', '']) == cli.EXIT_ERROR


# ----------------------------------------------------------- selftest


def test_selftest_passes_on_a_healthy_source(stub_device, capsys):
    assert run(['selftest']) == cli.EXIT_OK
    assert 'SELFTEST PASSED' in capsys.readouterr().out


def test_selftest_json_shape(stub_device, capsys):
    assert run(['selftest', '--json']) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload['ok'] is True
    assert payload['checks']['startup_test'] is True


# -------------------------------------------------------------- bench


def test_bench_reports_rates(stub_device, capsys):
    assert run(['bench', '--duration', '1', '--json'], extra=()) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload['photons'] > 0
    assert payload['modelled_bits_per_s'] >= 0


# ----------------------------------------------------------- raw / seed


def test_raw_capture_writes_all_artefacts(stub_device, tmp_path, capsys):
    prefix = tmp_path / 'cap'
    assert run(['raw', '--duration', '1', '--prefix', str(prefix)], extra=()) == cli.EXIT_OK
    assert (tmp_path / 'cap.channels.u16').exists()
    assert (tmp_path / 'cap.batches.jsonl').exists()
    assert (tmp_path / 'cap.summary.json').exists()
    summary = json.loads((tmp_path / 'cap.summary.json').read_text())
    assert summary['photons'] > 0


def test_raw_channels_file_is_parseable_by_the_estimators(stub_device, tmp_path):
    prefix = tmp_path / 'cap'
    run(['raw', '--duration', '1', '--prefix', str(prefix)], extra=())
    raw = (tmp_path / 'cap.channels.u16').read_bytes()
    assert len(raw) % 2 == 0
    channels = [int.from_bytes(raw[i:i + 2], 'little') for i in range(0, len(raw), 2)]
    assert channels and all(0 <= c < 1024 for c in channels)


def test_seed_file_written_with_requested_length(stub_device, tmp_path, capsys):
    path = tmp_path / 'seed.bin'
    assert run(['seed-file', str(path), '-n', '64']) == cli.EXIT_OK
    assert len(path.read_bytes()) == 64


# --------------------------------------------------------- platform


def test_feed_refuses_off_linux(stub_device, capsys, monkeypatch):
    monkeypatch.setattr(cli.sys, 'platform', 'win32')
    assert run(['feed']) == cli.EXIT_ERROR
    captured = capsys.readouterr()
    assert 'Linux-only' in captured.out + captured.err


def test_platform_capability_is_reported_for_this_host():
    capability = cli._platform_capability()
    assert 'platform' in capability and 'detail' in capability
    assert isinstance(capability['supported'], bool)
