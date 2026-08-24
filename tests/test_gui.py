"""GUI behaviour, with the detector stubbed out.

Builds the real widget tree against a fake worker, so the layout, the input
validation and the rate limiter are all exercised without hardware and without
entering the Tk main loop. Skipped where there is no Tk or no display.
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

tk = pytest.importorskip('tkinter', reason='Tkinter not installed')

from radiarandom import gui  # noqa: E402


class FakeWorker:
    """Stands in for gui._Worker; never touches USB."""

    def __init__(self, *args, **kwargs) -> None:
        self.commands: queue.Queue = queue.Queue()
        self.events: queue.Queue = queue.Queue()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(scope='session')
def tk_root():
    """One Tk interpreter for the whole session.

    A second tk.Tk() in the same process intermittently fails with "tk wasn't
    installed properly", so tests get Toplevel windows off a single root.
    """
    try:
        root = tk.Tk()
    except tk.TclError as exc:                       # pragma: no cover
        pytest.skip(f'no display: {exc}')
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:                              # pragma: no cover
        pass


@pytest.fixture
def window(tk_root, monkeypatch):
    """A built RandomApp with no device behind it, before start-up completes."""
    monkeypatch.setattr(gui, '_Worker', FakeWorker)
    top = tk.Toplevel(tk_root)
    top.withdraw()
    application = gui.RandomApp(top)
    yield application
    try:
        top.destroy()
    except tk.TclError:                              # pragma: no cover
        pass


@pytest.fixture
def app(window):
    """As above, but pretending the start-up test has already passed."""
    window._ready = True
    window.go.config(state='normal')
    return window


def pump_until(window, predicate, timeout=3.0):
    """Spin the Tk event loop until predicate() or the timeout expires.

    after() callbacks only run once their delay has elapsed, so a bare
    update() is a race.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        window.root.update()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def drain(worker):
    out = []
    while True:
        try:
            out.append(worker.commands.get_nowait())
        except queue.Empty:
            return out


# ------------------------------------------------------------- construction


def test_window_builds_and_starts_its_worker(app):
    assert app.worker.started
    assert app.result_var.get() == '–'


def test_nothing_is_drawn_before_the_startup_test_passes(window):
    """No output may leave before start-up completes -- but say so, not nothing."""
    window._on_generate()
    assert drain(window.worker) == [], 'must not reach the device yet'
    assert window._queued is True
    assert 'Queued' in window.go['text']


def test_a_queued_request_fires_once_the_generator_is_ready(window):
    """A preset clicked during calibration should still produce a number.

    Previously the preset buttons were live but _apply_preset only drew when
    already ready, so clicking "D6" while calibrating did nothing at all and
    gave no hint why.
    """
    window._apply_preset(1, 6, None)
    assert window._queued is True
    assert drain(window.worker) == []

    window.worker.events.put(('ready', None))
    window._pump_events()
    assert window._queued is False
    assert pump_until(window, lambda: not window.worker.commands.empty())
    commands = drain(window.worker)
    assert len(commands) == 1
    kind, payload = commands[0]
    assert kind == 'draw'
    assert (payload['low'], payload['high']) == (1, 6)


def test_only_one_request_is_queued_however_many_presets_are_clicked(window):
    for label, low, high, labels in gui.PRESETS:
        window._apply_preset(low, high, labels)
    assert window._queued is True
    window.worker.events.put(('ready', None))
    window._pump_events()
    assert pump_until(window, lambda: not window.worker.commands.empty())
    assert len(drain(window.worker)) == 1, 'no backlog of impatient clicks'


def test_a_queued_request_is_dropped_on_failure(window, monkeypatch):
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *a, **k: None)
    window._on_generate()
    assert window._queued is True
    window.worker.events.put(('fatal', 'detector unplugged'))
    window._pump_events()
    assert window._queued is False


def test_history_buttons_are_disabled_until_there_is_history(window):
    """Buttons that cannot do anything should look like it."""
    for button in (window.copy_last_button, window.copy_all_button,
                   window.clear_button):
        assert str(button['state']) == 'disabled'


def test_history_buttons_enable_once_a_result_arrives_and_disable_on_clear(app):
    app._render([7])
    for button in (app.copy_last_button, app.copy_all_button, app.clear_button):
        assert str(button['state']) == 'normal'
    app._clear()
    for button in (app.copy_last_button, app.copy_all_button, app.clear_button):
        assert str(button['state']) == 'disabled'


def test_every_preset_is_applicable(app):
    for label, low, high, labels in gui.PRESETS:
        app._apply_preset(low, high, labels)
        assert int(app.min_var.get()) == low
        assert int(app.max_var.get()) == high


# --------------------------------------------------------------- validation


def test_range_is_parsed(app):
    app.min_var.set('3'); app.max_var.set('9'); app.count_var.set('4')
    assert app._read_range() == (3, 9, 4)


@pytest.mark.parametrize('lo,hi,n', [
    ('x', '10', '1'),        # not a number
    ('10', '1', '1'),        # max below min
    ('1', '10', '0'),        # count too small
    ('1', '10', '99999'),    # count too large
])
def test_bad_input_is_rejected_without_queueing_work(app, monkeypatch, lo, hi, n):
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror',
                        lambda *a, **k: errors.append(a))
    app.min_var.set(lo); app.max_var.set(hi); app.count_var.set(n)
    assert app._read_range() is None
    assert errors, 'the user should have been told'
    app._on_generate()
    assert drain(app.worker) == []


def test_bad_input_cancels_auto_repeat(app, monkeypatch):
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *a, **k: None)
    app.auto_var.set(True)
    app.max_var.set('-5')
    app._on_generate()
    assert app.auto_var.get() is False, 'a bad range must not loop forever'


# ------------------------------------------------------------- rate limiting


def test_min_interval_follows_the_rate_box(app):
    app.rate_var.set(4.0)
    assert app._min_interval() == pytest.approx(0.25)
    app.rate_var.set(0.5)
    assert app._min_interval() == pytest.approx(2.0)


def test_rate_is_clamped_to_sane_bounds(app):
    app.rate_var.set(10_000.0)
    assert app._min_interval() == pytest.approx(1.0 / gui.MAX_RATE)
    app.rate_var.set(0.0)
    assert app._min_interval() == pytest.approx(1.0 / gui.MIN_RATE)


def test_first_draw_is_immediate(app):
    app.min_var.set('1'); app.max_var.set('6'); app.count_var.set('1')
    app._on_generate()
    commands = drain(app.worker)
    assert len(commands) == 1
    kind, payload = commands[0]
    assert kind == 'draw'
    assert (payload['low'], payload['high'], payload['count']) == (1, 6, 1)


def test_second_draw_inside_the_window_is_deferred_not_dropped(app):
    app.rate_var.set(1.0)                 # one per second
    app.min_var.set('1'); app.max_var.set('6')
    app._on_generate()
    assert len(drain(app.worker)) == 1
    app._pending = False                  # pretend the first result came back
    app._on_generate()                    # immediately again
    assert drain(app.worker) == [], 'should be throttled, not sent'
    assert 'Rate limited' in app.go['text'], 'and the user should see why'


def test_draw_is_allowed_again_once_the_interval_has_passed(app):
    app.rate_var.set(20.0)                # 50 ms apart
    app.min_var.set('1'); app.max_var.set('6')
    app._on_generate()
    drain(app.worker)
    app._pending = False
    time.sleep(0.06)
    app._on_generate()
    assert len(drain(app.worker)) == 1


def test_a_draw_in_flight_is_not_duplicated(app):
    app.min_var.set('1'); app.max_var.set('6')
    app._on_generate()
    assert len(drain(app.worker)) == 1
    app._on_generate()                    # _pending is still True
    assert drain(app.worker) == []


def test_physical_mode_is_passed_through(app):
    app.physical_var.set(True)
    app._on_generate()
    (_, payload), = drain(app.worker)
    assert payload['physical'] is True


# ------------------------------------------------------------------ results


def test_result_is_rendered_and_recorded(app):
    app._render([42])
    assert app.result_var.get() == '42'
    assert app._history[0] == '42'
    assert app.history.size() == 1


def test_coin_flip_uses_word_labels(app):
    app._apply_preset(0, 1, ('Tails', 'Heads'))
    app._render([1])
    assert app.result_var.get() == 'Heads'
    app._render([0])
    assert app.result_var.get() == 'Tails'


def test_multiple_values_are_joined(app):
    app._render([1, 2, 3])
    assert app.result_var.get() == '1, 2, 3'


def test_history_is_bounded(app):
    for i in range(250):
        app._render([i])
    assert app.history.size() <= 200
    assert len(app._history) <= 200


def test_clear_empties_everything(app):
    app._render([7])
    app._clear()
    assert app.history.size() == 0
    assert app._history == []
    assert app.result_var.get() == '–'


def test_copy_puts_the_last_value_on_the_clipboard(app):
    app._render([123])
    app._copy_last()
    assert app.root.clipboard_get() == '123'


def test_copy_all_joins_the_history(app):
    app._render([1]); app._render([2])
    app._copy_all()
    assert app.root.clipboard_get().split('\n') == ['2', '1']


def test_copy_on_empty_history_does_not_raise(app):
    app._copy_last()
    app._copy_all()


# ------------------------------------------------------------------- events


def test_ready_event_enables_the_button(app):
    app._ready = False
    app.go.config(state='disabled')
    app.worker.events.put(('ready', None))
    app._pump_events()
    assert app._ready is True
    assert str(app.go['state']) == 'normal'


def test_startup_event_reports_progress(app):
    app.worker.events.put(('startup', (256, 1024, 16.0)))
    app._pump_events()
    assert '256/1024' in app.metrics_var.get()
    assert app.progress['value'] == pytest.approx(250)


def test_result_event_clears_pending_and_renders(app):
    app._pending = True
    app.worker.events.put(('result', {'values': [5], 'request': {}}))
    app._pump_events()
    assert app._pending is False
    assert app.result_var.get() == '5'


def test_fatal_event_disables_the_ui(app, monkeypatch):
    shown = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *a, **k: shown.append(a))
    app.auto_var.set(True)
    app.worker.events.put(('fatal', 'detector unplugged'))
    app._pump_events()
    assert app._ready is False
    assert app.auto_var.get() is False, 'auto-repeat must stop on failure'
    assert str(app.go['state']) == 'disabled'
    assert shown, 'the user must be told'


def test_health_failure_is_surfaced_in_the_status_line(app):
    app.worker.events.put(('metrics', {
        'count_rate': 16.0, 'entropy_rate': 20.0,
        'healthy': False, 'failure': 'proportion: channel 25 dominates',
    }))
    app._pump_events()
    assert 'HEALTH FAILURE' in app.metrics_var.get()


HEALTHY_METRICS = {
    'count_rate': 16.2, 'entropy_rate': 21.0, 'healthy': True, 'failure': None,
    'reservoir_bytes': 1024, 'reservoir_capacity': 4096,
    'reservoir_fraction': 0.25, 'pool_fill': 0.5, 'cost_bits_per_byte': 10.0,
    'drbg_seeded': True, 'seconds_since_reseed': 4.0, 'reseeds': 1,
}


def test_metrics_are_rendered_when_healthy(app):
    app.worker.events.put(('metrics', dict(HEALTHY_METRICS)))
    app._pump_events()
    text = app.metrics_var.get()
    assert '16.2 counts/s' in text and '21.0 bits/s' in text
    assert 'reserve 25%' in text and '1024/4096' in text


def test_readiness_line_counts_banked_draws_in_physical_mode(app):
    app.physical_var.set(True)
    app._apply_preset(1, 6, None)
    app.worker.events.put(('metrics', dict(HEALTHY_METRICS)))
    app._pump_events()
    ready = app.ready_var.get()
    assert 'ready now' in ready
    assert 'draws banked' in ready
    assert 'sustains' in ready


def test_readiness_line_estimates_a_wait_when_the_reserve_is_short(app):
    app.physical_var.set(True)
    app.min_var.set('1'); app.max_var.set('100'); app.count_var.set('500')
    metrics = dict(HEALTHY_METRICS, reservoir_bytes=0, reservoir_fraction=0.0)
    app.worker.events.put(('metrics', metrics))
    app._pump_events()
    assert 'needs 500 B' in app.ready_var.get()
    assert 's' in app.ready_var.get()


def test_readiness_line_describes_the_drbg_in_fast_mode(app):
    app.physical_var.set(False)
    app.worker.events.put(('metrics', dict(HEALTHY_METRICS)))
    app._pump_events()
    ready = app.ready_var.get()
    assert 'DRBG ready' in ready and 'unlimited' in ready


def test_readiness_warns_when_auto_repeat_outruns_the_detector(app):
    app.physical_var.set(True)
    app.auto_var.set(True)
    app.rate_var.set(20.0)
    app.min_var.set('1'); app.max_var.set('1000000000'); app.count_var.set('50')
    app.worker.events.put(('metrics', dict(HEALTHY_METRICS)))
    app._pump_events()
    assert 'will throttle' in app.ready_var.get()


def test_closing_stops_the_worker(app):
    app._on_close()
    assert app.worker.stopped


# ------------------------------------------------- worker status publishing


class StubPool:
    entropy_bits = 160.0
    fill_fraction = 0.5
    blocks_available = 0
    capacity_bits = 512.0


class StubMonitor:
    failed = False
    failure_reason = None


class StubGenerator:
    """Enough of a Generator for _emit_status, with a draw that blocks."""

    def __init__(self, block_for: float = 0.0) -> None:
        self.pool = StubPool()
        self.monitor = StubMonitor()
        self.count_rate = 16.0
        self.entropy_rate_bits_per_s = 20.0
        self.reservoir_bytes = 1024
        self.reservoir_capacity = 4096
        self.reservoir_fraction = 0.25
        self._block_for = block_for
        self.draws = 0

    def stats(self) -> dict:
        return {'drbg': {'seconds_since_reseed': 3.0, 'reseeds': 2}}

    def physical_bytes(self, n: int) -> bytes:
        self.draws += 1
        time.sleep(self._block_for)
        return b'\x00' * n

    def read(self, n: int) -> bytes:
        return self.physical_bytes(n)

    def stop(self) -> None:
        pass


def test_metrics_keep_arriving_while_a_draw_is_blocked(monkeypatch):
    """The reported bug: clicking Generate froze the entropy counter.

    Status used to be emitted from the same loop that executes draws, so a
    physical draw -- ten to twenty seconds on real hardware -- stopped the count
    rate and the pool gauge dead, precisely while the user was waiting on them.
    """
    monkeypatch.setattr(gui, 'STATUS_INTERVAL_S', 0.05)
    worker = gui._Worker(None, 0)
    worker._generator = StubGenerator(block_for=1.0)
    worker._start_status_thread()
    try:
        worker.commands.put(('draw', {'low': 1, 'high': 6, 'count': 1,
                                      'physical': True}))
        serving = threading.Thread(target=worker._serve, daemon=True)
        serving.start()

        # While the draw is blocked, metrics must still be published.
        time.sleep(0.6)
        metrics = [k for k, _ in list(worker.events.queue) if k == 'metrics']
        assert len(metrics) >= 3, f'status froze during the draw: {metrics}'

        serving.join(timeout=5)
    finally:
        worker.stop()


def test_status_thread_stops_with_the_worker(monkeypatch):
    monkeypatch.setattr(gui, 'STATUS_INTERVAL_S', 0.02)
    worker = gui._Worker(None, 0)
    worker._generator = StubGenerator()
    worker._start_status_thread()
    worker.stop()
    worker._status_thread.join(timeout=2)
    assert not worker._status_thread.is_alive()


def test_emit_status_is_a_no_op_before_the_generator_exists():
    worker = gui._Worker(None, 0)
    worker._emit_status()
    assert worker.events.empty()


def test_emit_status_reports_reserve_and_readiness():
    worker = gui._Worker(None, 0)
    worker._generator = StubGenerator()
    worker._emit_status()
    kind, payload = worker.events.get_nowait()
    assert kind == 'metrics'
    assert payload['healthy'] is True
    assert payload['reservoir_bytes'] == 1024
    assert payload['reservoir_capacity'] == 4096
    assert 0.0 <= payload['reservoir_fraction'] <= 1.0
    assert payload['cost_bits_per_byte'] > 0
    assert payload['drbg_seeded'] is True


def test_drbg_message_does_not_claim_to_be_short_of_entropy(app):
    """Reported from the running UI: 256 B banked, "needs 64 B", not seeded.

    The DRBG is instantiated lazily on the first draw, so an unseeded DRBG with
    a full reserve is waiting on a request, not on the detector. Saying it
    needed entropy it already had was just wrong.
    """
    app.physical_var.set(False)
    metrics = dict(HEALTHY_METRICS, drbg_seeded=False, reservoir_bytes=256)
    app.worker.events.put(('metrics', metrics))
    app._pump_events()
    ready = app.ready_var.get()
    assert 'seeds from the reserve on your first draw' in ready
    assert '256 B banked' in ready


def test_drbg_message_does_report_a_genuine_shortfall(app):
    app.physical_var.set(False)
    metrics = dict(HEALTHY_METRICS, drbg_seeded=False, reservoir_bytes=8)
    app.worker.events.put(('metrics', metrics))
    app._pump_events()
    ready = app.ready_var.get()
    assert 'needs 64 B to seed' in ready and '8 B banked' in ready
