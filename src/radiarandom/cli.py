"""Command line interface.

    radiarandom info                      what the detector is and what it can do
    radiarandom bench                     measure the real count and entropy rates
    radiarandom selftest                  run the health tests and report
    radiarandom gen -n 32 --format hex    random bytes, to stdout or a file
    radiarandom int --min 1 --max 6 -c 5  unbiased integers
    radiarandom password -l 20            passwords, with an entropy figure
    radiarandom uuid -c 4                 version 4 UUIDs
    radiarandom raw --duration 3600       capture the noise source for analysis
    radiarandom feed                      contribute to the Linux kernel pool
    radiarandom serve                     serve entropy over a pipe or socket
    radiarandom gui                       graphical front end for quick numbers

A note on latency: the SP 800-90B start-up test needs 1024 photons before any
output is permitted, which on indoor background is two to four minutes. Every
command that produces output prints its progress to stderr while it waits.
Pass ``--startup-samples`` to trade that off deliberately.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

from . import __version__, formats
from .conditioner import BLOCK_BITS, BLOCK_BYTES
from .device import DEFAULT_POLL_INTERVAL_S, DeviceNotFound, RadiaCodeSource, SourceError
from .entropy import (
    Assessment,
    DEFAULT_H_CHANNEL,
    DEFAULT_SAFETY_FACTOR,
    min_entropy_of_histogram,
    normalise,
    projected_bit_rate,
    shannon_entropy_of_histogram,
)
from .generator import Generator, GeneratorError
from .health import HealthFailure, STARTUP_SAMPLES

_log = logging.getLogger('radiarandom')

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_DEVICE = 2
EXIT_HEALTH = 3


# --------------------------------------------------------------------- setup


def _add_common(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('device and entropy assessment')
    group.add_argument('--serial', help='USB serial number, when several are attached')
    group.add_argument('--poll-interval', type=float, default=DEFAULT_POLL_INTERVAL_S,
                       metavar='SECONDS', help='how often to read the device')
    group.add_argument('--h-per-photon', type=float, default=DEFAULT_H_CHANNEL,
                       metavar='BITS',
                       help='per-photon channel min-entropy assumed when sizing '
                            'the health-test cutoffs. Does NOT set the entropy '
                            'budget, which comes from the Poisson model over the '
                            f'measured rate and spectrum (default {DEFAULT_H_CHANNEL})')
    group.add_argument('--safety', type=float, default=DEFAULT_SAFETY_FACTOR,
                       metavar='FACTOR',
                       help=f'safety factor on the assessment (default {DEFAULT_SAFETY_FACTOR})')
    group.add_argument('--no-live-estimate', action='store_true',
                       help='do not let the live MCV estimator lower the claim')
    group.add_argument('--startup-samples', type=int, default=STARTUP_SAMPLES,
                       metavar='N',
                       help='photons the SP 800-90B start-up test must pass '
                            f'(default {STARTUP_SAMPLES}; lowering it is a '
                            'deliberate weakening)')


def _assessment_for(args, spectrum, count_rate: float) -> Assessment:
    """Build an assessment from a measured spectrum and count rate."""
    return Assessment(
        channel_probs=normalise(spectrum),
        count_rate=count_rate,
        safety_factor=args.safety,
        h_channel=args.h_per_photon,
        origin='measured spectrum and rate',
    )


def _open_source(args) -> RadiaCodeSource:
    source = RadiaCodeSource(serial_number=getattr(args, 'serial', None),
                             poll_interval=getattr(args, 'poll_interval', DEFAULT_POLL_INTERVAL_S))
    source.open()
    return source


def _build_generator(args, source: RadiaCodeSource) -> Generator:
    return Generator(
        source,
        startup_samples=args.startup_samples,
        use_live_estimate=not args.no_live_estimate,
        reference_spectrum=source.reference_spectrum(),
        safety_factor=args.safety,
        h_channel=args.h_per_photon,
    )


def _startup(generator: Generator, quiet: bool = False) -> None:
    """Run the start-up test, reporting progress on stderr."""
    if generator.monitor.started:
        return
    needed = generator.monitor.startup_progress[1]
    if not quiet:
        print(f'start-up test: {needed} photons required '
              f'(a few minutes on background radiation)', file=sys.stderr, flush=True)

    state = {'last': -1}

    def progress(passed: int, total: int) -> None:
        if quiet or passed == state['last']:
            return
        state['last'] = passed
        rate = generator.count_rate
        eta = ''
        if rate and rate > 0 and passed < total:
            eta = f'  eta {int((total - passed) / rate)}s'
        bar_width = 28
        filled = int(bar_width * passed / max(1, total))
        bar = '#' * filled + '.' * (bar_width - filled)
        print(f'\r  [{bar}] {passed}/{total}{eta}   ', end='', file=sys.stderr, flush=True)

    generator.wait_for_startup(progress=progress)
    if not quiet:
        print('\r  start-up test passed' + ' ' * 40, file=sys.stderr, flush=True)


def _reader_for(generator: Generator, physical: bool):
    """Return a ``read(n) -> bytes`` bound to the requested output mode."""
    if physical:
        return lambda n: generator.physical_bytes(n)
    return lambda n: generator.read(n)


def _describe_mode(physical: bool) -> str:
    if physical:
        return ('physical mode: full-entropy output rate-limited by the detector '
                '(seconds per 32 bytes)')
    return ('DRBG mode: HMAC_DRBG(SHA-512) seeded and continuously reseeded from '
            'the detector')


# ---------------------------------------------------------------- subcommands


def cmd_info(args) -> int:
    source = _open_source(args)
    generator = None  # info never starts a pump thread
    try:
        spectrum = source.reference_spectrum()
        total = sum(spectrum)
        a0, a1, a2 = source.energy_calibration()
        h_measured = min_entropy_of_histogram(spectrum)
        h_shannon = shannon_entropy_of_histogram(spectrum)

        print(f'measuring count rate for {args.rate_window:.0f}s...', file=sys.stderr, flush=True)
        rate = source.measure_count_rate(args.rate_window)
        assessment = _assessment_for(args, spectrum, rate)
        projected = assessment.bits_per_second()

        info = {
            'serial': source.serial(),
            'firmware': source.firmware(),
            'channels': len(spectrum),
            'lifetime_counts': total,
            'energy_calibration': {'a0': a0, 'a1': a1, 'a2': a2},
            'reference_min_entropy_bits_per_photon': h_measured,
            'reference_shannon_entropy_bits_per_photon': h_shannon,
            'assessed': assessment.describe(),
            'measured_count_rate_per_s': rate,
            'projected_entropy_bits_per_s': projected,
            'projected_physical_bytes_per_s': projected / 8.0,
            'seconds_per_256_bit_seed': (BLOCK_BITS + 64) / projected if projected > 0 else None,
            'startup_test_seconds': args.startup_samples / rate if rate > 0 else None,
            'os_pool_contribution': _platform_capability(),
        }
        if args.json:
            print(json.dumps(info, indent=2))
            return EXIT_OK

        print(f'device            {info["serial"]}  firmware {info["firmware"]}')
        print(f'channels          {info["channels"]}  '
              f'(energy = {a0:.3f} + {a1:.4f}*ch + {a2:.2e}*ch^2 keV)')
        print(f'lifetime counts   {total:,}')
        print()
        print('entropy source: independent Poisson counts per pulse-height channel')
        print(f'  reference spectrum   H_min = {h_measured:.3f} bits/photon, '
              f'H_shannon = {h_shannon:.3f}')
        print(f'  measured count rate  {rate:.2f} photons/s')
        print(f'  entropy rate         {projected:.2f} bits/s '
              f'({projected / 8:.2f} bytes/s)')
        print(f'  model                {assessment.describe()}')
        if projected > 0:
            print(f'  256-bit seed         ~{(BLOCK_BITS + 64) / projected:.0f}s')
            print(f'  start-up test        ~{args.startup_samples / rate:.0f}s '
                  f'({args.startup_samples} photons)')
            for factor in (10, 100):
                hot = projected_bit_rate(rate * factor, assessment)
                print(f'  with a {factor:>3}x source   {hot:.1f} bits/s '
                      f'({hot / 8:.1f} bytes/s)')
        print()
        print('OS entropy pool integration:')
        for line in _platform_capability()['detail'].splitlines():
            print(f'  {line}')
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def _platform_capability() -> dict:
    if sys.platform.startswith('linux'):
        from .pools import linux as linux_pool
        avail = linux_pool.entropy_avail()
        return {
            'platform': 'linux',
            'supported': True,
            'detail': (
                'supported via RNDADDENTROPY on /dev/random.\n'
                'Run "radiarandom feed" as root (or with CAP_SYS_ADMIN) to credit\n'
                'the kernel pool; see packaging/linux/ for a systemd unit.\n'
                f'current entropy_avail: {avail if avail is not None else "unknown"} bits'
            ),
        }
    if sys.platform == 'win32':
        from .pools import windows as windows_pool
        return {
            'platform': 'windows',
            'supported': False,
            'detail': (
                'NOT supported by Windows.\n'
                + windows_pool.os_pool_explanation()
            ),
        }
    return {
        'platform': sys.platform,
        'supported': False,
        'detail': 'no kernel pool integration for this platform; use "radiarandom serve".',
    }


def cmd_bench(args) -> int:
    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        deadline = time.perf_counter() + args.duration
        print(f'benchmarking for {args.duration:.0f}s...', file=sys.stderr, flush=True)
        blocks = 0
        while time.perf_counter() < deadline:
            generator.pump_once()
            while generator.pool.ready():
                generator.pool.extract_block()
                blocks += 1
            time.sleep(source.poll_interval)
        elapsed = args.duration
        stats = generator.stats()
        result = {
            'duration_s': elapsed,
            'photons': stats['photons'],
            'count_rate_per_s': stats['count_rate'],
            'count_rate_lower_bound': stats['count_rate_lower_bound'],
            'assessment': stats['assessment'],
            'live_channel_min_entropy': stats['live_channel_min_entropy'],
            'modelled_bits_per_s': stats['entropy_rate_bits_per_s'],
            'physical_blocks': blocks,
            'physical_bytes_per_s': blocks * BLOCK_BYTES / elapsed,
            'realised_bits_per_s': blocks * (BLOCK_BITS + 64) / elapsed,
            'pool': stats['pool'],
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            rate = result['count_rate_per_s'] or 0.0
            print(f'photons              {result["photons"]}')
            print(f'count rate           {rate:.2f}/s')
            print(f'assessment           {result["assessment"]}')
            live = result['live_channel_min_entropy']
            print(f'live channel H_min   '
                  f'{f"{live:.3f} bits/photon" if live else "(needs more samples)"}')
            print(f'full-entropy blocks  {blocks} '
                  f'({result["physical_bytes_per_s"]:.3f} bytes/s)')
            print(f'modelled rate        {result["modelled_bits_per_s"]:.2f} bits/s')
            print(f'realised rate        {result["realised_bits_per_s"]:.2f} bits/s')
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_selftest(args) -> int:
    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        print('running SP 800-90B start-up test and health checks...',
              file=sys.stderr, flush=True)
        try:
            _startup(generator)
        except HealthFailure as exc:
            print(f'FAIL: {exc}', file=sys.stderr)
            return EXIT_HEALTH

        # Exercise both output paths.
        block = generator.physical_block()
        drbg_bytes = generator.read(4096)
        status = generator.stats()
        checks = {
            'startup_test': generator.monitor.started,
            'health_tests': not generator.monitor.failed,
            'physical_block_size': len(block) == BLOCK_BYTES,
            'physical_block_not_constant': len(set(block)) > 1,
            'drbg_output_size': len(drbg_bytes) == 4096,
            'drbg_output_not_constant': len(set(drbg_bytes)) > 16,
            'device_no_resets': source.resets_seen == 0,
        }
        ok = all(checks.values())
        if args.json:
            print(json.dumps({'ok': ok, 'checks': checks, 'status': status}, indent=2, default=str))
        else:
            for name, value in checks.items():
                print(f'  {"PASS" if value else "FAIL"}  {name}')
            print()
            print(f'  count rate     {status["count_rate"]:.2f}/s' if status['count_rate']
                  else '  count rate     (unknown)')
            print(f'  assessment     {status["assessment"]}')
            print(f'  proportion     max {status["health"]["proportion_cutoff"]} of any '
                  f'one channel per {status["health"]["proportion_window"]} photons')
            print(f'  repetition     max {status["health"]["repeat_cutoff"]} identical batches')
            print()
            print('SELFTEST PASSED' if ok else 'SELFTEST FAILED')
        return EXIT_OK if ok else EXIT_HEALTH
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_gen(args) -> int:
    if args.bytes <= 0 and not args.stream:
        print('nothing to do: pass -n/--bytes or --stream', file=sys.stderr)
        return EXIT_ERROR

    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        print(_describe_mode(args.physical), file=sys.stderr, flush=True)
        _startup(generator, quiet=args.quiet)
        if not args.physical:
            # The DRBG generates far faster than the device is polled, so
            # without a pump thread the entropy pool would never refill and
            # reseeds would stall. Physical mode pumps inline instead.
            generator.run_background()
        read = _reader_for(generator, args.physical)

        binary = args.format == 'bin'
        if args.output and args.output != '-':
            handle = open(args.output, 'wb' if binary else 'w',
                          **({} if binary else {'encoding': 'ascii'}))
            close = True
        else:
            handle = sys.stdout.buffer if binary else sys.stdout
            close = False

        written = 0
        started = time.perf_counter()
        try:
            if args.stream:
                _install_sigint(generator)
                chunk = args.chunk
                while not generator.stopped:
                    data = read(chunk)
                    _emit(handle, data, args.format, binary, args.group)
                    written += len(data)
                    _progress(args, written, started)
            else:
                remaining = args.bytes
                chunk = min(args.chunk, remaining)
                while remaining > 0:
                    take = min(chunk, remaining)
                    data = read(take)
                    _emit(handle, data, args.format, binary, args.group)
                    remaining -= take
                    written += take
                    _progress(args, written, started)
            if not binary and not args.no_newline:
                handle.write('\n')
        finally:
            handle.flush()
            if close:
                handle.close()
            if not args.quiet and args.output and args.output != '-':
                elapsed = time.perf_counter() - started
                print(f'\rwrote {written} bytes to {args.output} in {elapsed:.1f}s'
                      + ' ' * 20, file=sys.stderr, flush=True)
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def _emit(handle, data: bytes, fmt: str, binary: bool, group: int) -> None:
    if binary:
        handle.write(data)
    else:
        handle.write(formats.format_bytes(data, fmt, group))


def _progress(args, written: int, started: float) -> None:
    if args.quiet or not sys.stderr.isatty():
        return
    elapsed = max(1e-9, time.perf_counter() - started)
    print(f'\r  {written} bytes  ({written / elapsed:.1f} B/s)   ',
          end='', file=sys.stderr, flush=True)


def _install_sigint(generator: Generator) -> None:
    def handler(signum, frame):
        generator.stop()
    try:
        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not on the main thread
        pass


def cmd_int(args) -> int:
    if args.max < args.min:
        print('--max must be >= --min', file=sys.stderr)
        return EXIT_ERROR
    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        _startup(generator, quiet=args.quiet)
        read = _reader_for(generator, args.physical)
        values = [formats.random_int(read, args.min, args.max) for _ in range(args.count)]
        if args.json:
            print(json.dumps(values))
        else:
            print(args.separator.join(str(value) for value in values))
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_float(args) -> int:
    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        _startup(generator, quiet=args.quiet)
        read = _reader_for(generator, args.physical)
        values = [formats.random_float(read) for _ in range(args.count)]
        if args.json:
            print(json.dumps(values))
        else:
            print('\n'.join(repr(value) for value in values))
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_uuid(args) -> int:
    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        _startup(generator, quiet=args.quiet)
        read = _reader_for(generator, args.physical)
        for _ in range(args.count):
            print(formats.random_uuid4(read))
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_password(args) -> int:
    alphabet = args.alphabet
    if alphabet in formats.ALPHABETS:
        alphabet = formats.ALPHABETS[alphabet]
    if not alphabet:
        print('empty alphabet', file=sys.stderr)
        return EXIT_ERROR
    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        _startup(generator, quiet=args.quiet)
        read = _reader_for(generator, args.physical)
        bits = formats.password_entropy_bits(args.length, len(set(alphabet)))
        for _ in range(args.count):
            print(formats.random_password(read, args.length, alphabet))
        if not args.quiet:
            print(f'({args.length} chars from {len(set(alphabet))} symbols = '
                  f'{bits:.1f} bits each)', file=sys.stderr)
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_raw(args) -> int:
    """Capture the unprocessed noise source for offline analysis."""
    source = _open_source(args)
    # startup_samples=0: we want every photon on disk, including whatever the
    # detector produced before a start-up test would have passed. Analysis
    # happens offline, where nothing is at stake.
    generator = Generator(
        source,
        startup_samples=0,
        use_live_estimate=not args.no_live_estimate,
        reference_spectrum=source.reference_spectrum(),
        safety_factor=args.safety,
        h_channel=args.h_per_photon,
    )
    prefix = args.prefix
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or '.', exist_ok=True)
    _install_sigint(generator)

    start = time.perf_counter()
    deadline = start + args.duration if args.duration > 0 else None
    photons = 0
    blocks = 0
    # A capture runs for hours and may well be interrupted. Buffered writes
    # would lose whatever had not reached disk, which on a source producing a
    # few photons a second is a lot of wall-clock time to throw away.
    last_flush = start
    flush_interval = 10.0
    try:
        with open(prefix + '.channels.u16', 'ab') as flat, \
             open(prefix + '.physical.bin', 'ab') as physical, \
             open(prefix + '.batches.jsonl', 'a', encoding='utf-8') as jsonl:
            while not generator.stopped:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                batch = generator.pump_once()
                if batch.count:
                    photons += batch.count
                    flat.write(b''.join(c.to_bytes(2, 'little') for c in batch.channels))
                    record = {
                        's': batch.seq,
                        't': round(batch.host_monotonic, 6),
                        'w': round(batch.host_time, 6),
                        'd': batch.device_seconds,
                        'c': list(batch.channels),
                    }
                    jsonl.write(json.dumps(record, separators=(',', ':')))
                    jsonl.write('\n')
                banked = generator.reservoir_bytes
                if banked >= BLOCK_BYTES:
                    chunk = banked - (banked % BLOCK_BYTES)
                    physical.write(generator.physical_bytes(chunk))
                    blocks += chunk // BLOCK_BYTES
                now = time.perf_counter()
                if now - last_flush >= flush_interval:
                    jsonl.flush()
                    flat.flush()
                    physical.flush()
                    last_flush = now
                if not args.quiet:
                    print('\r  {:.0f}s  {} photons  {} physical bytes   '.format(
                        now - start, photons, blocks * BLOCK_BYTES),
                        end='', file=sys.stderr, flush=True)
                time.sleep(source.poll_interval)
    except HealthFailure as exc:
        print('\nhealth failure during capture: {}'.format(exc), file=sys.stderr)
    finally:
        if generator is not None:
            generator.stop()
        source.close()

    elapsed = time.perf_counter() - start
    stats = generator.stats()
    summary = {
        'elapsed_s': elapsed,
        'photons': photons,
        'count_rate': photons / elapsed if elapsed else 0,
        'physical_bytes': blocks * BLOCK_BYTES,
        'assessment': stats['assessment'],
        'bits_credited': stats['bits_credited'],
        'health': stats['health'],
    }
    with open(prefix + '.summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, default=str)
    print('\n' + json.dumps(summary, indent=2, default=str), file=sys.stderr)
    print('\nnext: python validation/sp800_90b.py {}.channels.u16'.format(prefix),
          file=sys.stderr)
    return EXIT_OK


def cmd_gui(args) -> int:
    """Launch the graphical front end. Tkinter is imported lazily so the rest
    of the CLI still works on a headless box with no Tk installed."""
    from . import gui
    return gui.run(serial=args.serial, startup_samples=args.startup_samples)


def cmd_feed(args) -> int:
    if not sys.platform.startswith('linux'):
        print('radiarandom feed is Linux-only.', file=sys.stderr)
        print(_platform_capability()['detail'], file=sys.stderr)
        return EXIT_ERROR

    from .pools import linux as linux_pool

    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        _startup(generator, quiet=args.quiet)
        _install_sigint(generator)

        def on_status(stats: dict) -> None:
            if args.json:
                print(json.dumps(stats, indent=2, default=str), flush=True)
            else:
                print(f'credited {stats["bits_credited"]} bits, '
                      f'entropy_avail={stats["entropy_avail"]}, '
                      f'uptime={stats["uptime_s"]:.0f}s', flush=True)

        linux_pool.feed(
            generator,
            watermark=args.watermark,
            interval=args.interval,
            max_bits_per_second=args.max_rate,
            device=args.device,
            on_status=on_status,
            status_interval=args.status_interval,
        )
        return EXIT_OK
    except HealthFailure as exc:
        print(f'health failure, stopping: {exc}', file=sys.stderr)
        return EXIT_HEALTH
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_serve(args) -> int:
    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        print(_describe_mode(args.physical), file=sys.stderr, flush=True)
        _startup(generator, quiet=args.quiet)
        generator.run_background()
        read = _reader_for(generator, args.physical)
        _install_sigint(generator)

        if args.transport == 'auto':
            transport = 'pipe' if sys.platform == 'win32' else 'fifo'
        else:
            transport = args.transport

        if transport == 'pipe':
            from .pools.windows import NamedPipeEntropyServer, DEFAULT_PIPE_NAME
            server = NamedPipeEntropyServer(read, pipe_name=args.pipe_name or DEFAULT_PIPE_NAME,
                                            chunk_size=args.chunk)
            server.serve_forever(on_ready=lambda name: print(f'listening on {name}',
                                                             file=sys.stderr, flush=True))
        elif transport == 'fifo':
            from .pools.service import serve_fifo
            serve_fifo(read, path=args.fifo_path, chunk_size=args.chunk,
                       on_ready=lambda path: print(f'listening on FIFO {path}',
                                                   file=sys.stderr, flush=True))
        elif transport == 'tcp':
            from .pools.service import TcpEntropyServer
            if args.host != '127.0.0.1' and not args.allow_remote:
                print('refusing to bind a non-loopback address without '
                      '--allow-remote: this streams key material in the clear',
                      file=sys.stderr)
                return EXIT_ERROR
            server = TcpEntropyServer(read, host=args.host, port=args.port,
                                      chunk_size=args.chunk)
            server.serve_forever(
                on_ready=lambda addr: print(f'listening on tcp://{addr[0]}:{addr[1]}',
                                            file=sys.stderr, flush=True))
        else:
            print(f'unknown transport {transport}', file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK
    except KeyboardInterrupt:
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


def cmd_seed_file(args) -> int:
    from .pools.service import write_seed_file

    source = _open_source(args)
    generator = None
    try:
        generator = _build_generator(args, source)
        _startup(generator, quiet=args.quiet)
        data = generator.physical_bytes(args.bytes)
        write_seed_file(data, args.path)
        print(f'wrote {len(data)} bytes of full-entropy seed to {args.path}',
              file=sys.stderr)
        return EXIT_OK
    finally:
        if generator is not None:
            generator.stop()
        source.close()


# ------------------------------------------------------------------ argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='radiarandom',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f'radiarandom {__version__}')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='-v for info, -vv for debug logging')
    subparsers = parser.add_subparsers(dest='command', required=True)

    def sub(name, func, help_text, **kwargs):
        p = subparsers.add_parser(name, help=help_text, description=help_text, **kwargs)
        p.set_defaults(func=func)
        _add_common(p)
        p.add_argument('--json', action='store_true', help='machine-readable output')
        p.add_argument('-q', '--quiet', action='store_true', help='suppress progress on stderr')
        return p

    p_info = sub('info', cmd_info, 'Report device identity, entropy assessment and rates.')
    p_info.add_argument('--rate-window', type=float, default=20.0, metavar='SECONDS',
                        help='how long to measure the count rate for')

    p_bench = sub('bench', cmd_bench, 'Measure the achievable entropy rate.')
    p_bench.add_argument('--duration', type=float, default=60.0, metavar='SECONDS')

    sub('selftest', cmd_selftest, 'Run the health tests and exercise both output modes.')

    p_gen = sub('gen', cmd_gen, 'Generate random bytes to stdout or a file.')
    p_gen.add_argument('-n', '--bytes', type=int, default=32, help='how many bytes')
    p_gen.add_argument('-o', '--output', help='output file ("-" for stdout)')
    p_gen.add_argument('-f', '--format', choices=formats.FORMATS, default='hex')
    p_gen.add_argument('--group', type=int, default=0, metavar='N',
                       help='for hex output, insert a space every N bytes')
    p_gen.add_argument('--physical', action='store_true',
                       help='full-entropy output at the detector rate instead of DRBG output')
    p_gen.add_argument('--stream', action='store_true',
                       help='generate until interrupted, ignoring --bytes')
    p_gen.add_argument('--chunk', type=int, default=65536, metavar='BYTES')
    p_gen.add_argument('--no-newline', action='store_true')

    p_int = sub('int', cmd_int, 'Generate uniform integers by rejection sampling.')
    p_int.add_argument('--min', type=int, default=0)
    p_int.add_argument('--max', type=int, default=99)
    p_int.add_argument('-c', '--count', type=int, default=1)
    p_int.add_argument('--separator', default='\n')
    p_int.add_argument('--physical', action='store_true')

    p_float = sub('float', cmd_float, 'Generate uniform doubles in [0, 1).')
    p_float.add_argument('-c', '--count', type=int, default=1)
    p_float.add_argument('--physical', action='store_true')

    p_uuid = sub('uuid', cmd_uuid, 'Generate version 4 UUIDs.')
    p_uuid.add_argument('-c', '--count', type=int, default=1)
    p_uuid.add_argument('--physical', action='store_true')

    p_pass = sub('password', cmd_password, 'Generate passwords from physical randomness.')
    p_pass.add_argument('-l', '--length', type=int, default=20)
    p_pass.add_argument('-c', '--count', type=int, default=1)
    p_pass.add_argument('--alphabet', default='unambiguous',
                        help='a named set (' + ', '.join(formats.ALPHABETS) + ') or literal characters')
    p_pass.add_argument('--physical', action='store_true')

    p_raw = sub('raw', cmd_raw, 'Capture the raw noise source for offline analysis.')
    p_raw.add_argument('--prefix', default='data/raw', help='output path prefix')
    p_raw.add_argument('--duration', type=float, default=0.0,
                       help='seconds to capture; 0 means until interrupted')

    p_feed = sub('feed', cmd_feed, 'Contribute entropy to the Linux kernel pool (Linux only).')
    p_feed.add_argument('--device', default='/dev/random')
    p_feed.add_argument('--watermark', type=int, default=None, metavar='BITS',
                        help='only contribute while entropy_avail is below this')
    p_feed.add_argument('--interval', type=float, default=1.0)
    p_feed.add_argument('--max-rate', type=float, default=None, metavar='BITS_PER_S')
    p_feed.add_argument('--status-interval', type=float, default=60.0)

    p_serve = sub('serve', cmd_serve, 'Serve entropy to other processes.')
    p_serve.add_argument('--transport', choices=('auto', 'pipe', 'fifo', 'tcp'), default='auto')
    p_serve.add_argument('--pipe-name', default=None, help=r'Windows pipe name, e.g. \\.\pipe\radiarandom')
    p_serve.add_argument('--fifo-path', default='/run/radiarandom/entropy')
    p_serve.add_argument('--host', default='127.0.0.1')
    p_serve.add_argument('--port', type=int, default=7373)
    p_serve.add_argument('--allow-remote', action='store_true',
                         help='permit binding a non-loopback address (not recommended)')
    p_serve.add_argument('--chunk', type=int, default=4096)
    p_serve.add_argument('--physical', action='store_true',
                         help='serve full-entropy output at the detector rate')

    p_gui = subparsers.add_parser(
        'gui', help='Launch the graphical front end.',
        description='Launch the graphical front end (needs Tkinter).')
    p_gui.set_defaults(func=cmd_gui)
    p_gui.add_argument('--serial', help='USB serial number, if several are attached')
    p_gui.add_argument('--startup-samples', type=int, default=STARTUP_SAMPLES,
                       metavar='N',
                       help='photons the SP 800-90B start-up test must pass '
                            f'(default {STARTUP_SAMPLES})')

    p_seed = sub('seed-file', cmd_seed_file, 'Write a full-entropy seed file (mode 0600).')
    p_seed.add_argument('path')
    p_seed.add_argument('-n', '--bytes', type=int, default=64)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        stream=sys.stderr)

    try:
        return args.func(args)
    except DeviceNotFound as exc:
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_NO_DEVICE
    except HealthFailure as exc:
        print(f'health failure: {exc}', file=sys.stderr)
        print('the generator stops rather than emitting output it cannot vouch for.',
              file=sys.stderr)
        return EXIT_HEALTH
    except (SourceError, GeneratorError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print('\ninterrupted', file=sys.stderr)
        return EXIT_OK


if __name__ == '__main__':
    raise SystemExit(main())
