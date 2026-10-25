"""Small-sample paired t summaries; standard library, no experiment execution."""
import math
import statistics


def _beta_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1., a - 1.
    c, d = 1., 1. - qab * x / qap
    tiny = 1e-300
    d = 1. / max(abs(d), tiny) * (1 if d >= 0 else -1)
    h = d
    for m in range(1, 301):
        aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
        d, c = 1. + aa * d, 1. + aa / c
        d = tiny if abs(d) < tiny else d
        c = tiny if abs(c) < tiny else c
        d = 1. / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d, c = 1. + aa * d, 1. + aa / c
        d = tiny if abs(d) < tiny else d
        c = tiny if abs(c) < tiny else c
        d = 1. / d
        delta = d * c
        h *= delta
        if abs(delta - 1.) < 2e-14:
            return h
    raise ArithmeticError('beta fraction did not converge')


def _ibeta(a, b, x):
    if x <= 0:
        return 0.
    if x >= 1:
        return 1.
    scale = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.) / (a + b + 2.):
        return scale * _beta_fraction(a, b, x) / a
    return 1. - scale * _beta_fraction(b, a, 1. - x) / b


def t_cdf(t, df):
    tail = .5 * _ibeta(df / 2., .5, df / (df + t * t))
    return 1. - tail if t >= 0 else tail


def t_quantile(p, df):
    lo, hi = -1000000., 1000000.
    for _ in range(120):
        mid = (lo + hi) / 2.
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.


def summarize(values, expected_n, threshold=0.):
    complete = len(values) == expected_n and all(v is not None and math.isfinite(v) for v in values)
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    result = dict(n=len(finite), expected_n=expected_n, complete=complete, values=values,
                  mean=statistics.mean(finite) if finite else None, p_greater=1.)
    if not complete or len(finite) < 2:
        return result
    n, mean = len(finite), statistics.mean(finite)
    sd = statistics.stdev(finite)
    se = sd / math.sqrt(n)
    df = n - 1
    q2, q1 = t_quantile(.975, df), t_quantile(.95, df)
    p = 1. - t_cdf((mean - threshold) / se, df) if se > 0 else (0. if mean > threshold else 1.)
    result.update(sd=sd, se=se, df=df, lower95=mean - q2 * se, upper95=mean + q2 * se,
                  lower_one_sided95=mean - q1 * se, upper_one_sided95=mean + q1 * se,
                  threshold=threshold, p_greater=p, positive_seeds=sum(v > 0 for v in finite),
                  inference='paired training-seed Student t; conditional on fixed evaluation pool')
    return result


def holm(pvalues):
    result, previous = [1.] * len(pvalues), 0.
    for i, index in enumerate(sorted(range(len(pvalues)), key=lambda k: pvalues[k])):
        previous = max(previous, min(1., (len(pvalues) - i) * pvalues[index]))
        result[index] = previous
    return result
