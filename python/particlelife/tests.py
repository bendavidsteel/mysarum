import datetime

import jax
import jax.numpy as jp

from particle_lenia import Params, simple_fields_f, fields_f, motion_f, direct_motion_f, direct_U_f, direct_grad_G_f, direct_grad_R_f, direct_grad_U_f

def test_simple_same():
    key = jax.random.PRNGKey(8)

    simple_params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    params = Params(
        mu_k=jp.array(4.0).reshape(1, 1, 1),
        sigma_k=jp.array(1.0).reshape(1, 1, 1),
        w_k=jp.array(0.022).reshape(1, 1),
        mu_g=jp.array(0.6).reshape(1, 1, 1),
        sigma_g=jp.array(0.15).reshape(1, 1, 1),
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

def test_direct_motion_f():
    # check direct U is same as vmap U
    key = jax.random.PRNGKey(8)

    simple_params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    params = Params(
        mu_k=jp.array(4.0).reshape(1, 1, 1),
        sigma_k=jp.array(1.0).reshape(1, 1, 1),
        w_k=jp.array(0.022).reshape(1, 1),
        mu_g=jp.array(0.6).reshape(1, 1, 1),
        sigma_g=jp.array(0.15).reshape(1, 1, 1),
        c_rep=jp.array(1.0).reshape(1, 1)
    )
    num_points = 200
    num_dims = 2
    map_size = 20
    points = jax.random.uniform(key, [num_points, num_dims], minval=-0.5, maxval=0.5) * map_size
    species = jp.zeros(num_points, dtype=jp.int32)

    U = jax.vmap(lambda x, s: fields_f(params, points, species, x, s).U)(points, species)
    direct_U = direct_U_f(params, points, species)
    assert jp.allclose(U, direct_U)

    grad_R_f = jax.grad(lambda p, s: fields_f(params, points, species, p, s).R)
    grad_R = jax.vmap(grad_R_f)(points, species)
    direct_grad_R = direct_grad_R_f(params, points, species)
    assert jp.allclose(grad_R, direct_grad_R)

    grad_U_f = jax.grad(lambda p, s: fields_f(params, points, species, p, s).U)
    grad_U = jax.vmap(grad_U_f)(points, species)
    direct_grad_U = direct_grad_U_f(params, points, species)
    assert jp.allclose(grad_U, direct_grad_U, 1e-7, 1e-7)

    grad_G_f = jax.grad(lambda p, s: fields_f(params, points, species, p, s).G)
    grad_G = jax.vmap(grad_G_f)(points, species)
    direct_grad_G = direct_grad_G_f(params, points, species)
    assert jp.allclose(grad_G, direct_grad_G, 1e-6, 1e-6)

    # check direct motion is same as vmap motion

    start_time = datetime.datetime.now()
    v = motion_f(params, points, species)
    print("motion_f time:", datetime.datetime.now() - start_time)
    start_time = datetime.datetime.now()
    direct_v = direct_motion_f(params, points, species)
    print("direct_motion_f time:", datetime.datetime.now() - start_time)
    assert jp.allclose(v, direct_v)

def test_train_task_funcs():
    pass

if __name__ == "__main__":
    # test_simple_same()
    # test_train_task_funcs()
    test_direct_motion_f()