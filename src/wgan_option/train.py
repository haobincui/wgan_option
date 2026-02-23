from wgan_option.config import default_config
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.utils.dataloader import create_random_dataloader


def main():
    dataloader = create_random_dataloader(640, 32, 32)
    config = default_config

    # Initialize WGAN model
    gan = WGAN_GP(config)

    # Train the model
    print("Training the model...")
    gan.train(dataloader)
    print("Training complete!")


if __name__ == "__main__":
    main()
