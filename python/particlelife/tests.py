import jax
import jax.numpy as jp

from particle_lenia import Params, simple_fields_f, fields_f

def test_simple_same():
    key = jax.random.PRNGKey(8)

    simple_params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    params = Params(
        mu_k=jp.array(4.0).reshape(1, 1, 1),
        sigma_k=jp.array(1.0).reshape(1, 1, 1),
        w_k=jp.array(0.022).reshape(1, 1),
        mu_g=jp.array(0.6).reshape(1, 1),
        sigma_g=jp.array(0.15).reshape(1, 1),
        c_rep=jp.array(1.0).reshape(1, 1)
    )
    num_points = 200
    num_dims = 2
    map_size = 20
    points = jax.random.uniform(key, [num_points, num_dims], minval=-0.5, maxval=0.5) * map_size
    species = jp.zeros(num_points, dtype=jp.int32)

    x = points[0]
    s = species[0]

    simple_fields = simple_fields_f(simple_params, points, x)
    fields = fields_f(params, points, species, x, s)

    assert jp.allclose(simple_fields.U, fields.U)
    assert jp.allclose(simple_fields.G, fields.G)
    assert jp.allclose(simple_fields.R, fields.R)
    assert jp.allclose(simple_fields.E, fields.E)

def test_train_task_funcs():
    pass

if __name__ == "__main__":
    # test_simple_same()
    test_train_task_funcs()