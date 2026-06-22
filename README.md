# diffusion-model-from-scratch

A compact, readable implementation of a Denoising Diffusion Probabilistic Model
(DDPM) built up from the math rather than pulled from a library. Everything that
matters is here in plain PyTorch: the noise schedule, the closed form forward
process, a small UNet that predicts noise, the training objective, and ancestral
sampling. The code is written to be small enough to read in one sitting and to
run on a CPU.

## The idea

Diffusion turns generation into a denoising problem. You take a clean image and
add Gaussian noise to it over many steps until nothing is left but static. Then
you train a network to undo one step of that corruption. Once it can do that, you
start from pure static and walk backwards, and a sample falls out.

There are two processes to keep straight.

The forward process is fixed and has no learned parameters. It gradually adds
noise according to a variance schedule beta_1 through beta_T. A useful fact is
that you never have to simulate it step by step. Writing alpha_t = 1 - beta_t and
alpha_bar_t for the running product of alphas up to t, you can jump straight to
any timestep:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

That closed form is the heart of training. It says a noisy sample at step t is a
known blend of the original image and a single Gaussian noise draw, with the
mixing weights coming entirely from the schedule.

The reverse process is what the network learns. Given a noisy x_t and the
timestep t, the UNet predicts the noise eps that was added. From that prediction
you compute the mean of x_{t-1} and take a step. Repeat from t = T down to t = 0
and you have a sample.

## What is in here

    src/schedule.py   noise schedules (linear and cosine) and the precomputed
                      buffers, plus q_sample for the closed form forward draw
    src/unet.py       a small UNet with sinusoidal timestep embeddings injected
                      into each residual block
    src/ddpm.py       the training loss (noise prediction MSE) and ancestral
                      sampling (p_sample for one step, sample for the full loop)

## The training objective

Diffusion can be derived from a variational bound, but in practice Ho et al.
showed that a stripped down objective works better and is far simpler. Pick a
random timestep for each image, corrupt it with known noise using the closed
form, ask the network to predict that noise, and minimise the mean squared error
between the prediction and the true noise. That is the whole loss. You can see it
in `DDPM.loss`.

## Running the tests

```
pip install -r requirements.txt
pytest tests/ -q
```

The tests are property checks rather than golden value comparisons:

* The closed form forward sample has the right mean and variance. For a fixed
  x_0 and many random noise draws, the empirical per pixel mean approaches
  sqrt(alpha_bar_t) * x_0 and the empirical variance approaches 1 - alpha_bar_t,
  checked across several timesteps.
* The training loss drops when you overfit a single tiny batch. Holding the
  timesteps and target noise fixed, a few hundred Adam steps cut the MSE well
  below its starting value, which is the signal that the network is actually
  learning to predict noise.
* Sampling returns the right shape. Ancestral sampling from pure noise gives back
  a tensor matching the requested batch and image dimensions, with finite values.

There are also smaller checks: betas stay strictly inside (0, 1), the cumulative
alpha product is non increasing, the UNet output shape matches its input, and the
network genuinely conditions on the timestep (the same image at two timesteps
gives two different outputs).

All fifteen run on CPU in a few seconds with no downloads.

## A note on scope

This is a teaching and portfolio implementation. The UNet is intentionally tiny
and there is no attention, no exponential moving average of weights, and no
classifier guidance. The structure mirrors what a production diffusion model
does, so the path from here to something larger is mostly a matter of widening
the network and pointing it at real data.
