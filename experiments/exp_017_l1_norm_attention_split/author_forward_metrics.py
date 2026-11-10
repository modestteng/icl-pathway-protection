from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import main
import opto

def make_forward_fn_ic_only(options, default_fn=opto.default_model_fwd_fn):
    '''
    Makes an opto-ified forward_fn that returns just in-context metrics
    '''
    call_fn = opto.make_fn_from_opts(options, default_fn=default_fn)

    def forward_fn(model, x, y, key):
        keys = jax.random.split(key, x.shape[0])
        all_activations = jax.vmap(partial(call_fn, model=model))(x=x, y=y, key=keys)
        query_ce = main.ce(all_activations['out'][:, -1, :], y[:, -1])

        in_context_mask = jnp.sum(jax.nn.one_hot(y[:, :-1], all_activations['out'].shape[-1]), axis=1) > 0
        in_context_pred_y = in_context_mask*all_activations['out'][:, -1, :] - (1-in_context_mask)*1e20
        fsl_loss = main.ce(in_context_pred_y, y[:, -1])
        fsl_output_prob = jnp.einsum('bc,bc->b', 
                                    jax.nn.softmax(in_context_pred_y), 
                                    jax.nn.one_hot(y[:, -1], in_context_pred_y.shape[-1]))
        fsl_output_acc = (jnp.argmax(in_context_pred_y, axis=-1) == y[:,-1])

        return {'in_context_loss': fsl_loss,
                'in_context_prob': fsl_output_prob,
                'in_context_acc': fsl_output_acc,
                'loss': query_ce}

    return eqx.filter_jit(forward_fn)
