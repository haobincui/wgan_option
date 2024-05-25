from gan.config import Config, default_config
from gan.models.gan_model import WGAN_GP
from gan.utils.dataloader import create_random_dataloader


dataloader = create_random_dataloader(640, 32, 32)
config = default_config

# Initialize WGAN model

gan = WGAN_GP(config)

# Train the model

print("Training the model...")
gan.train(dataloader)
print("Training complete!")
