"""GUI behaviour, with the detector stubbed out.

Builds the real widget tree against a fake worker, so the layout, the input
validation and the rate limiter are all exercised without hardware and without
entering the Tk main loop. Skipped where there is no Tk or no display.
"""

from __future__ import annotations

import queue
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


def test_generate_is_disabled_until_ready(window):
    """Nothing may be drawn before the SP 800-90B start-up test passes."""
    assert str(window.go['state']) == 'disabled'
    window._on_generate()
    assert drain(window.worker) == []


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
        'pool_fraction': 0.5,
    }))
    app._pump_events()
    assert 'HEALTH FAILURE' in app.metrics_var.get()


def test_metrics_are_rendered_when_healthy(app):
    app.worker.events.put(('metrics', {
        'count_rate': 16.2, 'entropy_rate': 21.0,
        'healthy': True, 'failure': None, 'pool_fraction': 1.0,
    }))
    app._pump_events()
    text = app.metrics_var.get()
    assert '16.2 counts/s' in text and '21.0 bits/s' in text


def test_closing_stops_the_worker(app):
    app._on_close()
    assert app.worker.stopped
