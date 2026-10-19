# How My Code Differs From the Article

I started from an SR3 tutorial article and kept the same method. Then I adapted it for microscopy images. The biggest change: my model only predicts the missing detail and adds it to the blurry image, where the article's version draws the whole sharp image from scratch. The original approach gave pure noise on our dark microscope images, so I had to change it. I also fixed a few bugs and made image generation about 40x faster.

## What stayed the same

Both versions follow the core SR3 idea:

- Add noise to sharp images and train a U-Net to predict that noise.
- Show the model the blurry image the whole time as a guide.
- To make a new image, start from random noise and clean it up step by step.

## What's different

| | Article's code | My code | Why it matters |
| --- | --- | --- | --- |
| **Data** | Nature photos (color, 32x32 → 128x128) | Microscope images from BioSR (grayscale, blurry widefield → sharp SIM, both 128x128) | Built for the actual project |
| **What the model draws** | The whole sharp image | Only the missing detail, added on top of the blurry image | Biggest change. The original version gave pure noise on dark microscope images |
| **U-Net building blocks** | Simple blocks with BatchNorm | Sturdier blocks with GroupNorm, closer to the SR3 paper | More stable training |
| **Loss** | MSE (squares the errors) | L1 (plain absolute errors) | The SR3 paper found L1 works a bit better |
| **Noise levels in training** | Fixed, one per step | Random values in between, like the SR3 paper | Lets the model generate with fewer steps later |
| **Steps to make one image** | 2,000 | 50 | About 40x faster, which matters for video-rate imaging |
| **Learning rate** | 0.001 | 0.0001 with a slow warm-up | Smoother training |
| **Smoothed weights (EMA)** | No | Yes | Cleaner output images |
| **Bugs** | A few | Fixed | See below |
| **Extra tools** | None | Test scores, Colab notebook, saves to Google Drive, resumes after a disconnect, works on any image size | Makes it usable for research |

## Bugs I fixed

- The noise setting `beta_start = 10e-4` is actually 0.001. The article says it should be 0.0001.
- The image loader skipped the last file in the folder.
- Output images were saved in the model's internal -1 to 1 range without converting back to normal 0 to 1 brightness.

## Quick glossary

- **U-Net:** the neural network inside the model. It shrinks the image to get the big picture, then builds it back up, with shortcuts that keep fine details.
- **Loss:** a number for how wrong the model is. Training tries to make it smaller.
- **BatchNorm / GroupNorm:** layers that keep the numbers inside the network stable. GroupNorm works better here because each image has a different noise level.
- **EMA:** a running average of the model's weights that gives smoother, cleaner results.
- **Widefield / SIM:** widefield is a fast but blurry type of microscope image. SIM is a slower, sharper one. The model learns to turn widefield into SIM.

## Next step

Run the article's original code on the same BioSR data as a baseline. Then I can show whether these changes actually help.
