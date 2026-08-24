"""A small cross-platform GUI for pulling numbers out of the detector.

Tkinter, because it ships with CPython on Windows and macOS (``python3-tk`` on
Debian and Fedora) and needs no build step -- the same reasoning that picked
HMAC_DRBG over a compiled cipher. A dice roller does not justify a 100 MB Qt
dependency.

The shape of this window is dictated by the hardware, not by taste:

* **The start-up test takes 60 s to 4 minutes.** SP 800-90B wants 1024 photons
  before anything may be emitted, and at 4.4-16 counts/s that is a long time to
  stare at nothing. The window therefore opens immediately, shows a progress
  bar, and only enables the controls when the generator is actually ready.
* **The detector is exclusive.** One process can hold the USB device, so if
  ``radiarandom serve`` or ``feed`` is already running this has to say so in
  words rather than dying in a traceback.
* **Physical mode is slow.** At 0.7-2.5 bytes/s a single draw can take
  seconds, so every draw is asynchronous and the button reports what it is
  waiting for.
* **:class:`RadiaCodeSource` is not thread-safe.** Exactly one thread owns the
  device; the GUI talks to it through queues and polls with ``after()``.

Run it with ``radiarandom gui`` or ``python -m radiarandom.gui``.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from . import formats
from .conditioner import BLOCK_BYTES, BLOCK_COST_BITS
from .device import DeviceNotFound, RadiaCodeSource, SourceError
from .generator import Generator
from .health import HealthFailure

#: (label, low, high, value labels or None)
PRESETS = (
    ('Coin flip', 0, 1, ('Tails', 'Heads')),
    ('D6', 1, 6, None),
    ('1 – 10', 1, 10, None),
    ('1 – 100', 1, 100, None),
)

#: Draws per second the rate limiter will allow, and where it starts.
MIN_RATE, MAX_RATE, DEFAULT_RATE = 0.2, 20.0, 2.0

#: How often the GUI drains the worker's queue, in milliseconds.
POLL_MS = 50

#: How often the worker publishes count rate, entropy rate and pool level.
STATUS_INTERVAL_S = 1.0


class _Worker(threading.Thread):
    """Owns the detector. Everything device-shaped happens on this thread.

    Commands arrive on :attr:`commands`; events go back on :attr:`events` as
    ``(kind, payload)`` tuples for the GUI to render.
    """

    def __init__(self, serial: Optional[str], startup_samples: int) -> None:
        super().__init__(daemon=True, name='radiarandom-gui-worker')
        self.commands: queue.Queue = queue.Queue()
        self.events: queue.Queue = queue.Queue()
        self._serial = serial
        self._startup_samples = startup_samples
        self._stop = threading.Event()
        self._source: Optional[RadiaCodeSource] = None
        self._generator: Optional[Generator] = None
        self._status_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- lifecycle

    def stop(self) -> None:
        self._stop.set()
        self.commands.put(('quit', None))

    def run(self) -> None:  # pragma: no cover - needs hardware
        try:
            self._open()
            self._serve()
        except DeviceNotFound as exc:
            self.events.put(('fatal', str(exc)))
        except SourceError as exc:
            self.events.put((
                'fatal',
                f'{exc}\n\nIf "radiarandom serve" or "feed" is running, stop it '
                f'first -- only one process can hold the detector.'))
        except HealthFailure as exc:
            self.events.put(('fatal', f'Health test failed: {exc}'))
        except Exception as exc:  # pragma: no cover
            self.events.put(('fatal', f'{type(exc).__name__}: {exc}'))
        finally:
            if self._generator is not None:
                self._generator.stop()
            if self._source is not None:
                self._source.close()

    def _open(self) -> None:
        self.events.put(('status', 'Opening detector...'))
        source = RadiaCodeSource(serial_number=self._serial)
        source.open()
        self._source = source
        self.events.put(('opened', {
            'serial': source.serial(),
            'firmware': source.firmware(),
        }))

        self.events.put(('status', 'Calibrating and running the start-up test...'))
        generator = Generator(
            source,
            startup_samples=self._startup_samples,
            reference_spectrum=source.reference_spectrum(),
        )
        self._generator = generator

        def progress(passed: int, needed: int) -> None:
            self.events.put(('startup', (passed, needed, generator.count_rate)))

        generator.wait_for_startup(progress=progress)
        # From here the pump thread keeps the pool topped up, so a physical
        # draw can block on this thread without starving the device.
        generator.run_background()
        self._start_status_thread()
        self.events.put(('ready', None))

    def _start_status_thread(self) -> None:
        """Publish metrics on their own thread, independent of draws.

        These used to be emitted from the command loop, which meant they froze
        for the entire duration of a draw. In DRBG mode that is imperceptible;
        in physical mode a draw blocks for ten to twenty seconds, so the count
        rate and -- worse -- the pool gauge stopped moving exactly while the
        user was waiting on them and wanted to know how much longer.
        """
        def loop() -> None:
            while not self._stop.wait(STATUS_INTERVAL_S):
                try:
                    self._emit_status()
                except Exception:  # pragma: no cover - shutdown races
                    return

        self._status_thread = threading.Thread(
            target=loop, daemon=True, name='radiarandom-gui-status')
        self._status_thread.start()

    # -------------------------------------------------------------- commands

    def _serve(self) -> None:
        """Execute draw commands. Metrics are published elsewhere, on purpose:
        a physical draw blocks this loop for seconds at a time."""
        while not self._stop.is_set():
            try:
                kind, payload = self.commands.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == 'quit':
                return
            if kind == 'draw':
                self._draw(payload)

    def _draw(self, request: dict) -> None:
        generator = self._generator
        assert generator is not None
        low, high, count, physical = (
            request['low'], request['high'], request['count'], request['physical'])

        read = (generator.physical_bytes if physical else generator.read)
        try:
            values = [formats.random_int(read, low, high) for _ in range(count)]
        except HealthFailure as exc:
            self.events.put(('fatal', f'Health test failed mid-draw: {exc}'))
            return
        self.events.put(('result', {'values': values, 'request': request}))

    def _emit_status(self) -> None:
        generator = self._generator
        if generator is None:
            return
        drbg = generator.stats()['drbg']
        self.events.put(('metrics', {
            'count_rate': generator.count_rate,
            'entropy_rate': generator.entropy_rate_bits_per_s,
            'healthy': not generator.monitor.failed,
            'failure': generator.monitor.failure_reason,
            # The reserve is the real measure of readiness: bounded, and every
            # byte in it is conditioned output that can be handed out at once.
            'reservoir_bytes': generator.reservoir_bytes,
            'reservoir_capacity': generator.reservoir_capacity,
            'reservoir_fraction': generator.reservoir_fraction,
            # The pool behind it tops out at 512 bits, so its own fill level is
            # only interesting while the reserve is empty.
            'pool_fill': generator.pool.fill_fraction,
            'cost_bits_per_byte': BLOCK_COST_BITS / BLOCK_BYTES,
            'drbg_seeded': drbg is not None,
            'seconds_since_reseed': (drbg or {}).get('seconds_since_reseed'),
            'reseeds': (drbg or {}).get('reseeds', 0),
        }))


class RandomApp:
    """The window. Knows nothing about USB; only talks to :class:`_Worker`."""

    def __init__(self, root, serial: Optional[str] = None,
                 startup_samples: int = 1024) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        root.title('radiarandom')
        root.minsize(430, 560)

        self._ready = False
        self._pending = False
        self._last_draw = 0.0
        self._labels: Optional[tuple] = None
        self._history: list = []
        self._identity = ''
        self._metrics: dict = {}
        # A draw asked for before the start-up test finished. Exactly one is
        # remembered: clicking four presets while waiting should produce the
        # last one, not a backlog of four.
        self._queued = False

        self.worker = _Worker(serial, startup_samples)

        self._build()
        self.worker.start()
        self.root.after(POLL_MS, self._pump_events)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ------------------------------------------------------------------ view

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        pad = {'padx': 10, 'pady': 6}

        # --- status strip
        top = ttk.Frame(self.root)
        top.pack(fill='x', **pad)
        self.status_var = tk.StringVar(value='Starting...')
        ttk.Label(top, textvariable=self.status_var, anchor='w').pack(fill='x')
        self.metrics_var = tk.StringVar(value='')
        ttk.Label(top, textvariable=self.metrics_var, anchor='w',
                  foreground='#666').pack(fill='x')
        self.ready_var = tk.StringVar(value='')
        ttk.Label(top, textvariable=self.ready_var, anchor='w',
                  foreground='#666').pack(fill='x')
        self.progress = ttk.Progressbar(top, mode='determinate', maximum=1000)
        self.progress.pack(fill='x', pady=(6, 0))

        # --- the number
        result = ttk.Frame(self.root, relief='groove', borderwidth=1)
        result.pack(fill='both', expand=True, **pad)
        self.result_var = tk.StringVar(value='–')
        self.result_label = ttk.Label(result, textvariable=self.result_var,
                                      anchor='center', font=('TkDefaultFont', 44))
        self.result_label.pack(fill='both', expand=True, pady=18)

        # --- presets
        presets = ttk.LabelFrame(self.root, text='Quick picks')
        presets.pack(fill='x', **pad)
        row = ttk.Frame(presets)
        row.pack(fill='x', padx=6, pady=6)
        for label, low, high, labels in PRESETS:
            ttk.Button(row, text=label, width=10,
                       command=lambda l=low, h=high, n=labels: self._apply_preset(l, h, n)
                       ).pack(side='left', expand=True, fill='x', padx=2)

        # --- range
        rng = ttk.LabelFrame(self.root, text='Range')
        rng.pack(fill='x', **pad)
        grid = ttk.Frame(rng)
        grid.pack(fill='x', padx=6, pady=6)
        self.min_var = tk.StringVar(value='1')
        self.max_var = tk.StringVar(value='100')
        self.count_var = tk.StringVar(value='1')
        for col, (text, var, width) in enumerate((
                ('Min', self.min_var, 9), ('Max', self.max_var, 9),
                ('How many', self.count_var, 5))):
            ttk.Label(grid, text=text).grid(row=0, column=col * 2, sticky='e', padx=(0, 4))
            ttk.Spinbox(grid, from_=-1_000_000_000, to=1_000_000_000, width=width,
                        textvariable=var).grid(row=0, column=col * 2 + 1,
                                               sticky='w', padx=(0, 12))

        # --- generate + rate limit
        actions = ttk.Frame(self.root)
        actions.pack(fill='x', **pad)
        # Deliberately not disabled while calibrating. A dead button gives no
        # feedback; this one queues the request and says so.
        self.go = ttk.Button(actions, text='Generate when ready',
                             command=self._on_generate)
        self.go.pack(fill='x', ipady=6)

        limit = ttk.Frame(actions)
        limit.pack(fill='x', pady=(8, 0))
        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(limit, text='Repeat automatically', variable=self.auto_var,
                        command=self._on_auto).pack(side='left')
        ttk.Label(limit, text='no faster than').pack(side='left', padx=(12, 4))
        self.rate_var = tk.DoubleVar(value=DEFAULT_RATE)
        ttk.Spinbox(limit, from_=MIN_RATE, to=MAX_RATE, increment=0.2, width=6,
                    textvariable=self.rate_var).pack(side='left')
        ttk.Label(limit, text='/sec').pack(side='left', padx=(4, 0))

        # --- entropy mode
        mode = ttk.LabelFrame(self.root, text='Source')
        mode.pack(fill='x', **pad)
        self.physical_var = tk.BooleanVar(value=False)
        ttk.Radiobutton(mode, text='Fast — DRBG seeded by the detector',
                        variable=self.physical_var, value=False).pack(anchor='w', padx=6)
        ttk.Radiobutton(mode, text='True entropy — detector rate only (slow)',
                        variable=self.physical_var, value=True).pack(anchor='w', padx=6,
                                                                     pady=(0, 6))

        # --- history
        hist = ttk.LabelFrame(self.root, text='History')
        hist.pack(fill='both', expand=True, **pad)
        self.history = tk.Listbox(hist, height=6, activestyle='none')
        self.history.pack(fill='both', expand=True, padx=6, pady=(6, 0))
        buttons = ttk.Frame(hist)
        buttons.pack(fill='x', padx=6, pady=6)
        self.copy_last_button = ttk.Button(buttons, text='Copy last',
                                           command=self._copy_last, state='disabled')
        self.copy_last_button.pack(side='left')
        self.copy_all_button = ttk.Button(buttons, text='Copy all',
                                          command=self._copy_all, state='disabled')
        self.copy_all_button.pack(side='left', padx=6)
        self.clear_button = ttk.Button(buttons, text='Clear',
                                       command=self._clear, state='disabled')
        self.clear_button.pack(side='right')

    # --------------------------------------------------------------- actions

    def _apply_preset(self, low: int, high: int, labels: Optional[tuple]) -> None:
        self.min_var.set(str(low))
        self.max_var.set(str(high))
        self._labels = labels
        self._refresh_readiness()
        # Always ask. Before start-up completes this queues instead of drawing,
        # which is better than a preset button that appears to do nothing.
        self._on_generate()

    def _read_range(self) -> Optional[tuple]:
        from tkinter import messagebox
        try:
            low, high = int(self.min_var.get()), int(self.max_var.get())
            count = int(self.count_var.get())
        except ValueError:
            messagebox.showerror('radiarandom', 'Min, max and count must be whole numbers.')
            return None
        if high < low:
            messagebox.showerror('radiarandom', 'Max must be at least Min.')
            return None
        if not 1 <= count <= 1000:
            messagebox.showerror('radiarandom', 'How many must be between 1 and 1000.')
            return None
        return low, high, count

    def _min_interval(self) -> float:
        try:
            rate = float(self.rate_var.get())
        except (ValueError, self.tk.TclError):
            rate = DEFAULT_RATE
        return 1.0 / max(MIN_RATE, min(MAX_RATE, rate))

    def _on_generate(self) -> None:
        """Draw now, or as soon as the generator and the rate limit allow."""
        if self._pending:
            return
        if not self._ready:
            # One request, not a backlog: clicking four presets while waiting
            # should give the last one rather than four draws. Naming it makes
            # that visible instead of leaving the user to infer it.
            self._queued = True
            self.go.config(
                text=f'Queued: {self._draw_description()} — runs after calibration')
            return
        parsed = self._read_range()
        if parsed is None:
            self.auto_var.set(False)
            return
        low, high, count = parsed

        wait = self._last_draw + self._min_interval() - time.perf_counter()
        if wait > 0:
            # Deliberately visible: a silently ignored click reads as a bug.
            self.go.config(text=f'Rate limited — {wait:.1f}s')
            self.root.after(int(wait * 1000) + 20, self._on_generate)
            return

        self._pending = True
        self._last_draw = time.perf_counter()
        self.go.config(state='disabled',
                       text='Waiting for photons...' if self.physical_var.get()
                       else 'Generating...')
        self.worker.commands.put(('draw', {
            'low': low, 'high': high, 'count': count,
            'physical': bool(self.physical_var.get()),
        }))

    def _on_auto(self) -> None:
        if self.auto_var.get():
            self._on_generate()

    def _render(self, values: list) -> None:
        if self._labels and len(values) == 1 and 0 <= values[0] < len(self._labels):
            shown = self._labels[values[0]]
        else:
            shown = ', '.join(str(v) for v in values)
        size = 44 if len(shown) <= 6 else (28 if len(shown) <= 18 else 16)
        self.result_label.config(font=('TkDefaultFont', size))
        self.result_var.set(shown)
        self._history.insert(0, shown)
        self.history.insert(0, shown)
        for button in (self.copy_last_button, self.copy_all_button, self.clear_button):
            button.config(state='normal')
        if self.history.size() > 200:
            self.history.delete(200, 'end')
            del self._history[200:]

    def _copy_last(self) -> None:
        if self._history:
            self._to_clipboard(self._history[0])

    def _copy_all(self) -> None:
        if self._history:
            self._to_clipboard('\n'.join(self._history))

    def _to_clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _clear(self) -> None:
        self._history.clear()
        self.history.delete(0, 'end')
        self.result_var.set('–')
        for button in (self.copy_last_button, self.copy_all_button, self.clear_button):
            button.config(state='disabled')

    # ---------------------------------------------------------------- events

    def _pump_events(self) -> None:
        from tkinter import messagebox
        try:
            while True:
                kind, payload = self.worker.events.get_nowait()
                if kind == 'status':
                    self.status_var.set(payload)
                elif kind == 'opened':
                    self._identity = (
                        f'{payload["serial"]} — firmware {payload["firmware"]}')
                    self.status_var.set(self._identity)
                elif kind == 'startup':
                    passed, needed, rate = payload
                    self.progress['value'] = 1000 * passed / max(1, needed)
                    eta = ''
                    if rate:
                        eta = f'  ~{max(0, int((needed - passed) / rate))}s left'
                    self.metrics_var.set(
                        f'Start-up test {passed}/{needed} photons{eta}')
                elif kind == 'ready':
                    self._ready = True
                    self.progress['value'] = 1000
                    self.go.config(state='normal', text='Generate')
                    if self._queued:
                        self._queued = False
                        self.root.after(10, self._on_generate)
                    # Put the device identity back; the start-up message has
                    # been sitting there since before the test began.
                    self.status_var.set(self._identity or 'Ready')
                    self.metrics_var.set('Ready')
                elif kind == 'metrics':
                    self._render_metrics(payload)
                elif kind == 'result':
                    self._pending = False
                    self.go.config(state='normal', text='Generate')
                    self._render(payload['values'])
                    if self.auto_var.get():
                        self.root.after(10, self._on_generate)
                elif kind == 'fatal':
                    self._ready = False
                    self._queued = False
                    self.auto_var.set(False)
                    self.go.config(state='disabled', text='Unavailable')
                    self.status_var.set('Stopped')
                    messagebox.showerror('radiarandom', payload)
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._pump_events)

    def _draw_description(self) -> str:
        """A short name for the draw the current settings would produce."""
        if self._labels and len(self._labels) == 2:
            base = 'coin flip'
        else:
            try:
                base = f'{int(self.min_var.get())}–{int(self.max_var.get())}'
            except (ValueError, self.tk.TclError):
                return 'a draw'
        try:
            count = int(self.count_var.get())
        except (ValueError, self.tk.TclError):
            count = 1
        return f'{count} × {base}' if count > 1 else base

    def _draw_bytes(self) -> int:
        """Bytes one draw of the current range needs, at minimum."""
        try:
            low, high = int(self.min_var.get()), int(self.max_var.get())
            count = int(self.count_var.get())
        except (ValueError, self.tk.TclError):
            return 1
        if high < low or count < 1:
            return 1
        return formats.bytes_needed_for(high - low + 1) * count

    def _render_metrics(self, m: dict) -> None:
        self._metrics = m
        if not m.get('healthy', True):
            self.metrics_var.set(f'HEALTH FAILURE — {m.get("failure")}')
            self.ready_var.set('')
            return

        rate = m.get('count_rate') or 0.0
        bits = m.get('entropy_rate') or 0.0
        banked = m.get('reservoir_bytes', 0)
        capacity = m.get('reservoir_capacity', 1)
        fraction = m.get('reservoir_fraction', 0.0)

        self.metrics_var.set(
            f'{rate:.1f} counts/s · {bits:.1f} bits/s · '
            f'reserve {fraction * 100:.0f}% ({banked}/{capacity} B)')
        self.ready_var.set(self._readiness_text(m, banked, bits))

        if self._ready:
            # The bar always shows the reserve against its real capacity. When
            # the reserve is empty it falls back to the pool's progress toward
            # the first block, so it still moves while you wait.
            value = fraction if banked else m.get('pool_fill', 0.0) / capacity * BLOCK_BYTES
            self.progress['value'] = 1000 * min(1.0, max(0.0, value))

    def _readiness_text(self, m: dict, banked: int, bits: float) -> str:
        """What the reserve means for the draw that is actually configured."""
        need = self._draw_bytes()
        cost = m.get('cost_bits_per_byte', BLOCK_COST_BITS / BLOCK_BYTES)

        if not self.physical_var.get():
            if not m.get('drbg_seeded'):
                # The DRBG is instantiated lazily, on the first draw. Saying it
                # "needs 64 B" while 256 B sit banked was simply wrong: it is
                # not waiting on entropy, it is waiting on a request.
                need = 2 * BLOCK_BYTES
                if banked >= need:
                    return (f'DRBG ready — seeds from the reserve on your first '
                            f'draw ({banked} B banked, needs {need} B)')
                wait = (need - banked) * cost / bits if bits > 0 else float('inf')
                wait_text = f'{wait:.0f}s' if wait < 3600 else 'a long time'
                return (f'DRBG needs {need} B to seed — {banked} B banked, '
                        f'about {wait_text}')
            since = m.get('seconds_since_reseed')
            when = f'{since:.0f}s ago' if since is not None else 'not yet'
            return (f'DRBG ready — unlimited draws · reseeded {when} · '
                    f'{m.get("reseeds", 0)} reseeds')

        draws = banked // max(1, need)
        if banked >= need:
            text = f'ready now — {draws} draw{"s" if draws != 1 else ""} banked ({need} B each)'
        else:
            wait = (need - banked) * cost / bits if bits > 0 else float('inf')
            wait_text = f'{wait:.0f}s' if wait < 3600 else 'a long time'
            text = f'next draw needs {need} B — about {wait_text}'

        if bits > 0 and need > 0:
            sustainable = bits / (need * cost)
            text += f' · sustains {sustainable:.2g}/s'
            try:
                wanted = float(self.rate_var.get())
            except (ValueError, self.tk.TclError):
                wanted = 0.0
            if self.auto_var.get() and wanted > sustainable * 1.05:
                text += f' (repeat set to {wanted:g}/s — will throttle)'
        return text

    def _refresh_readiness(self) -> None:
        """Recompute the readiness line after a range or mode change."""
        if self._queued and not self._ready:
            self.go.config(
                text=f'Queued: {self._draw_description()} — runs after calibration')
        if self._metrics:
            self._render_metrics(self._metrics)

    def _on_close(self) -> None:
        self.worker.stop()
        self.root.destroy()


def run(serial: Optional[str] = None, startup_samples: int = 1024) -> int:
    """Open the window and block until it is closed.

    Tkinter is imported here rather than at module scope so that importing
    :mod:`radiarandom.gui` -- which the CLI does when dispatching -- does not
    fail on a headless box with no Tk installed.
    """
    try:
        import tkinter as tk
    except ImportError:
        print('Tkinter is not available, so the GUI cannot start.\n'
              '  Debian/Ubuntu : sudo apt install python3-tk\n'
              '  Fedora        : sudo dnf install python3-tkinter\n'
              '  macOS/Windows : reinstall Python with the Tcl/Tk option enabled\n'
              'Everything else in radiarandom works without it.')
        return 1

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f'No display available for the GUI: {exc}')
        return 1

    RandomApp(root, serial=serial, startup_samples=startup_samples)
    root.mainloop()
    return 0


def main(argv: Optional[list] = None) -> int:
    """Entry point for ``radiarandom-gui`` and ``python -m radiarandom.gui``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog='radiarandom-gui', description='Graphical front end for radiarandom.')
    parser.add_argument('--serial', help='USB serial number, if several are attached')
    parser.add_argument('--startup-samples', type=int, default=1024,
                        help='photons the SP 800-90B start-up test must pass '
                             '(default 1024; lowering it is a deliberate weakening)')
    args = parser.parse_args(argv)
    return run(serial=args.serial, startup_samples=args.startup_samples)


if __name__ == '__main__':
    raise SystemExit(main())
