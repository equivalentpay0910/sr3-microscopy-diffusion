# Project Notes

## The idea

- Super-resolution microscopy gives sharp images, but it's slow and the light damages live cells.
- Fast widefield imaging is gentle and quick, but the images are blurry and noisy.
- If a model can turn fast blurry frames into sharp ones, you get speed and detail at the same time.
- The paper this is based on did it with a GAN. GANs can be hard to train and sometimes drift away from the input. Diffusion models train on one simple loss and stay closer to the input, so I'm testing that instead.

## What SR3 does

- Training: add noise to a sharp image, then teach a U-Net to predict that noise, with the blurry input as a guide.
- Generating: start from random noise and clean it up step by step.
- The model is told the noise level directly, so you can sample with fewer steps than you trained with. I train with 2,000 noise levels and sample with 50.

## Changes I made to the starting code

The code started from a Medium tutorial on SR3. I changed a few things to make it work on microscopy data:

| Change | Why |
| --- | --- |
| BioSR microscopy data in place of nature photos | That's the actual project |
| The model predicts the *difference* between the input and the target | Plain SR3 gave pure noise on dark microscopy images in my first tests. Predicting the difference fixed it |
| GroupNorm in place of BatchNorm | More stable with small batches |
| L1 loss in place of MSE | The SR3 paper found it works a little better |
| 50 sampling steps in place of 2,000 | Way faster, close to what the SR3 paper does |
| Learning rate 1e-4 with warmup, plus EMA weights | Standard for diffusion models, makes training smoother |
| Fixed a few bugs in the tutorial code | For example, `10e-4` is actually 0.001 even though the tutorial says 0.0001 |

## What I'm seeing

- The model learns pretty fast. By step 15,000 the filaments already look sharp.
- From 30,000 to 60,000 steps it mostly cleans up the background and thins out the lines.
- The background is still a little hazy in some images.
- In crowded areas it sometimes adds lines that aren't in the real image. This matters for science, since a fake filament could be misread as real.

## Next steps

- [ ] Get test-set numbers (PSNR, SSIM) at noise levels 1, 5 and 9
- [ ] Run the original tutorial code as-is as a baseline to compare against
- [ ] Compare against a simple non-diffusion model (a regular U-Net)
- [ ] Try the other BioSR structures (CCPs, ER, F-actin)
- [ ] Measure how many frames per second it can do, since the end goal is video rate
- [ ] Try faster sampling (fewer steps) and see how much quality drops
- [ ] Use uncertainty maps (run the model a few times on the same image) to show where it's guessing
