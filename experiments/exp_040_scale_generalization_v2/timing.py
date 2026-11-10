"""Per-seed start/window rule from the seed's own unprotected run (pure Python).

Input: monitor full-label standard ICL accuracy a(t) at every 100k sequences
on [0, horizon]. Smoothed s(t) is the centered three-point mean, defined only
when both neighbours exist. With p = max s:

* t_near_peak: earliest t with s(t), s(t+h), s(t+2h) >= p - tolerance;
* E_decline: earliest t after the (earliest) argmax with s(t) <= p - anchor;
* t0 = max(t_near_peak, E_decline - continuation); end = t0 + continuation;
* window = the 11 stages in [end - 1M, end].

The rule is first evaluated on [0, 20M]; only if E_decline is undefined there
it is evaluated once on [0, 30M]. The horizon that defines the lock is the
smallest one at which E_decline exists, so later points never change it.
Applicability additionally requires p >= min_peak and a monitor decline of at
least min_window_decline from s(t0) to the mean raw accuracy in the window;
the last condition needs the normal run to reach ``end`` (<= max_normal_end).
No protected-arm output is ever read here.
"""
H = 100000


def smooth(curve, horizon):
    stages = sorted(t for t in curve if t <= horizon)
    present = set(stages)
    return {t: (curve[t - H] + curve[t] + curve[t + H]) / 3.0
            for t in stages if (t - H) in present and (t + H) in present}


def lock_on_horizon(curve, horizon, cfg):
    """Return the lock fields computable on [0, horizon] (no window decline yet)."""
    s = smooth(curve, horizon)
    if not s:
        return dict(defined=False, reason='no_smoothed_points', horizon=horizon)
    peak = max(s.values())
    t_peak = min(t for t, v in s.items() if v == peak)
    tol = cfg['near_peak_tolerance']
    near = [t for t in sorted(s) if all(s.get(t + k * H, -1.0) >= peak - tol for k in range(3))]
    t_near = near[0] if near else None
    decline = [t for t in sorted(s) if t > t_peak and s[t] <= peak - cfg['decline_anchor']]
    e_decline = decline[0] if decline else None
    base = dict(horizon=horizon, smoothed_peak=peak, peak_stage=t_peak,
                near_peak_stage=t_near, decline_stage=e_decline)
    if t_near is None:
        return dict(base, defined=False, reason='no_near_peak_run')
    if e_decline is None:
        return dict(base, defined=False, reason='no_decline_within_horizon')
    start = max(t_near, e_decline - cfg['continuation_sequences'])
    start = int(round(start / H)) * H
    end = start + cfg['continuation_sequences']
    window = list(range(end - cfg['window_sequences'], end + 1, H))
    return dict(base, defined=True, reason='locked', start_stage=start, end_stage=end,
                window_stages=window, start_smoothed=s[start])


def decide(curve, cfg):
    """Lock on the first horizon where it is defined; report the needed run end."""
    first = lock_on_horizon(curve, cfg['first_horizon'], cfg)
    lock = first
    if not first['defined'] and first['reason'] == 'no_decline_within_horizon':
        lock = lock_on_horizon(curve, cfg['extended_horizon'], cfg)
        lock['first_horizon_result'] = first
    return lock


def applicability(curve, lock, cfg):
    """Final applicability once the run covers the window (or cannot)."""
    if not lock['defined']:
        return dict(applicable=False, reason=lock['reason'])
    if lock['smoothed_peak'] < cfg['min_peak']:
        return dict(applicable=False, reason='peak_below_min_peak')
    if lock['end_stage'] > cfg['max_normal_end']:
        return dict(applicable=False, reason='end_beyond_max_normal_end')
    missing = [t for t in lock['window_stages'] if t not in curve]
    if missing:
        return dict(applicable=None, reason='window_not_yet_trained', missing=missing)
    window_mean = sum(curve[t] for t in lock['window_stages']) / len(lock['window_stages'])
    drop = lock['start_smoothed'] - window_mean
    ok = drop >= cfg['min_window_decline']
    return dict(applicable=bool(ok), reason='window_decline_ok' if ok else 'window_decline_below_min',
                window_mean=window_mean, window_decline=drop)


def required_end(curve_end, cfg, lock):
    """Normal-run end needed next: extend for the horizon or for the window."""
    if not lock['defined']:
        if lock['reason'] == 'no_decline_within_horizon' and curve_end < cfg['extended_horizon']:
            return cfg['extended_horizon']
        return None
    return max(curve_end, lock['end_stage']) if lock['end_stage'] <= cfg['max_normal_end'] else None
