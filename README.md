# wgan_option



This project implements a Generative Adversarial Network (GAN), specifically a Wasserstein GAN (WGAN), to model and generate option return distributions. This approach uses convolutional neural networks (CNNs) in both the generator and discriminator to leverage the spatial structure of option return data, structured as 
(R,T) matrices where R is the number of return bins and T is the number of time-to-maturity intervals.

## Project Structure
```graphql
wgan_option/gan/
│
├── data/                           # Directory for the dataset
│
├── models/                         # Directory to save trained models
│
├── outputs/                        # Directory to save output distributions and generation results
│
├── gan/                            # GAN components
│   ├── __init__.py                 # Makes GAN a Python module
│   ├── discriminator.py            # Discriminator model
│   ├── generator.py                # Generator model
│   └── gan_model.py                # Script with the GAN class that includes training logic
│
├── utils/
│   ├── __init__.py                 # Makes utils a Python module
│   └── dataloader.py               # DataLoader utilities, custom dataset class
│
├── config.py                       # Configuration file for hyperparameters
└── train.py                        # Main script to run the training process
```

### Key Components
Generator (CNN-based): Uses transposed convolutional layers to generate option return distributions from latent variables.
Discriminator (CNN-based): Utilizes convolutional layers to differentiate between real and generated distributions.
Wasserstein Loss: Improves training stability by using the Earth Mover's distance, avoiding issues like mode collapse and vanishing gradient

### Configuration
Hyperparameters and other configurations are managed in config.py. Here's a brief on key configurations:

R: Number of return bins.
T: Number of time-to-maturity intervals.
num_samples: Total number of option return samples.
z_dim: Dimension of the latent space for the generator.
lr: Learning rate for the RMSprop optimizers.
num_epochs: Total number of epochs for training.
batch_size: Batch size used in training.
clip_value: Clipping value for the discriminator weights to satisfy the Lipschitz constraint.
n_critic: Number of times the discriminator is updated per generator update

