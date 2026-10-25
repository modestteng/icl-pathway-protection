"""Equivalent execution forms; main task/Adam and projection are separate."""

def make_objective(r, mode):
    assert mode == 'cached32', 'Only the reported execution form is released.'
    keys = tuple((r.jax.random.split(r.jax.random.fold_in(r.KEY, i), 32) for i in range(2)))

    @r.eqx.filter_jit
    def average(l0, l1, g0, g1):
        return ((l0 + l1) * 0.5, r.jax.tree_util.tree_map(lambda a, b: (a + b) * 0.5, g0, g1))

    def cached32(model, px, py):
        (l0, g0) = r.task_objective(model, px[0], py[0], keys[0])
        (l1, g1) = r.task_objective(model, px[1], py[1], keys[1])
        return average(l0, l1, g0, g1)
    return cached32
