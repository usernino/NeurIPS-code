# adapting code for Neural ODE from https://docs.kidger.site/diffrax/examples/neural_ode/

import time
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

import keras
from keras import layers


class Func(eqx.Module):
    """
    Standard Conv Net used for our vector fields/residual blocks
    """
    out_scale: jax.Array
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, *, key):
        k1, k2 = jax.random.split(key, 2)

        self.out_scale = jnp.array(1.0)

        self.conv1 = eqx.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
            key=k1,
        )

        self.conv2 = eqx.nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            padding=1,
            key=k2,
        )


    def __call__(self, t, y, args=None):
        y = y.reshape(1, 28, 28)    
        y = jax.nn.relu(self.conv1(y))
        y = self.conv2(y)
        y = y.reshape(784)              
        return self.out_scale * y

class ConvResNet(eqx.Module):
    '''
    ResNet made from Func (usually a convolutional NN) residual blocks
    '''
    blocks: list
    head: eqx.nn.Linear
    num_blocks : int
    
    def __init__(self, num_blocks, Func, *, key):
        keys = jax.random.split(key, num_blocks + 1)

        self.blocks = [
            Func(key=keys[i])
            for i in range(num_blocks)
        ]

        self.head = eqx.nn.Linear(784, 10, key=keys[-1])

        self.num_blocks = num_blocks
        
    def __call__(self, x):
        for i, block in enumerate(self.blocks):
            x = x + i/self.num_blocks*block(None, x)
        return self.head(x)

class ConvTiedResNet(eqx.Module):
    '''
    ResNet made from Func (usually a convolutional NN) residual blocks with tied weights
    i.e. the blocks repeat
    '''
    func: eqx.Module
    head: eqx.nn.Linear
    num_blocks : int
    
    def __init__(self, func, num_blocks, *, key):
        keys = jax.random.split(key, 2)

        self.func = func

        self.head = eqx.nn.Linear(784, 10, key=key)

        self.num_blocks = num_blocks
        
    def __call__(self, x):
        step_size = 1/self.num_blocks
        
        for i in range(self.num_blocks):
            t = i / self.num_blocks
            x = x + step_size*self.func(t, x)
        return x

class NeuralODE(eqx.Module):
    '''
    Neural ODE class using Diffrax.
    solver_choice: may choose ODE solver choice parameters -
    0 = Euler
    1 = Heun
    2 = Tsit5

    tolerance: specify adaptive solver tolerance/fixed step sizes:
    False = fixed step sizes
    tuple of length 2 = (rtol, atol)
    '''
    func: eqx.Module
    solver_num: int
    tolerance: tuple
    step_size: float
    
    def __init__(self, func, solver_choice=0, tolerance=False, step_size=0.1, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        if solver_choice not in [0,1,2]:
            raise Exception("Invalid solver choice: 0 = Euler, 1 = Heun, 2 = Tsit5")
        self.solver_num = solver_choice
        if tolerance and (len(tolerance) != 2 or type(tolerance) not in [list, tuple]):
            raise Exception("tolerance must be None (fixed steps) or tuple specifying (rtol, atol)")
        self.tolerance = tolerance
        self.step_size = step_size
                            
    def __call__(self, y0):
        if self.solver_num == 0:
            solv = diffrax.Euler()
        if self.solver_num == 1:
            solv = diffrax.Heun()
        if self.solver_num == 2:
            solv = diffrax.Tsit5()

        if self.tolerance:
            cntrl = diffrax.PIDController(rtol=self.tolerance[0], atol=self.tolerance[1])
        else:
            cntrl = diffrax.ConstantStepSize()
            
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solv,
            t0=0.0,
            t1=1.0,
            dt0=self.step_size,
            y0=y0,
            stepsize_controller=cntrl,
            saveat=diffrax.SaveAt(t1=True),
        )
        return solution.ys[0]

class NODEClassifier(eqx.Module):
    '''
    Uses a Neural ODE as a feature map into affine layer for classification
    solver_choice: may choose ODE solver choice parameters -
    0 = Euler
    1 = Heun
    2 = Tsit5

    tolerances: specify adaptive solver tolerance/fixed step sizes:
    False = fixed step sizes
    tuple of length 2 = (rtol, atol)
    
    '''
    ode: NeuralODE
    head: eqx.nn.Linear
    
    def __init__(self, Func, solver_choice=0, tolerance=False, step_size=0.1, *, key, **kwargs):
        super().__init__(**kwargs)
        k1, k2 = jr.split(key, 2)
        self.ode = NeuralODE(Func(key=key), solver_choice, tolerance, step_size, key=k1)
        self.head = eqx.nn.Linear(784, 10, key=k2)

    def __call__(self, x):
        h1 = self.ode(x)
        logits = self.head(h1)
        return logits

    def run_ode(self, x):
        output_ode = self.ode(x)
        return output_ode


def main_ODE(
    x_train,
    y_train,
    solver_choice=0,
    tolerance=False,
    step_size=0.1,
    adaptive=False,
    Func=Func,
    dataset_size=256,
    batch_size=32,
    lr=3e-3,
    steps=1000,
    seed=5678,
    print_every=100
):
    '''
    MNSIT training
    training for a Neural ODE
    '''
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    shuffle_key = jr.PRNGKey(0)
    perm = jr.permutation(shuffle_key, x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]
    
    #model = NeuralODE(data_size, width_size, depth, key=model_key)
    model = NODEClassifier(Func, solver_choice, tolerance, step_size, key=model_key)
    optim = optax.adam(lr)

    @eqx.filter_value_and_grad
    def grad_loss(model, x_batch, y_batch):
        logits = jax.vmap(model)(x_batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(x_batch, y_batch, model, opt_state):
        loss, grads = grad_loss(model, x_batch, y_batch)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
     
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    num_samples = x_train.shape[0]
    losses = []

    # Training loop
    for step in range(steps):
        start = (step * batch_size) % num_samples
        end = start + batch_size

        x_batch = x_train[start:end]
        y_batch = y_train[start:end]

        loss, model, opt_state = make_step(x_batch, y_batch, model, opt_state)
        losses.append(loss)
        if step % print_every == 0:
            print(f"Step {step}, Loss {loss}")
    return model, losses