import torch
import torch.optim as optim
import numpy as np
from models.discriminator import Critic
from models.generator import Generator
from utils.dataloader import get_dataloader
from utils.visualization import save_images
import os
import config


def train():
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize models
    critic = Critic(config.img_shape)
    generator = Generator(config.z_dim, config.img_shape)

    if torch.cuda.is_available():
        critic.cuda()
        generator.cuda()

    # Optimizers
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=config.lr)
    generator_optimizer = optim.RMSprop(generator.parameters(), lr=config.lr)

    # Data loader
    dataloader = get_dataloader(config.batch_size)

    # Training Loop
    for epoch in range(config.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.cuda()

            # ---------------------
            #  Train Critic
            # ---------------------
            for _ in range(config.n_critic):
                # Sample noise as generator input
                z = torch.randn(config.batch_size, config.z_dim).cuda()

                # Generate a batch of images
                fake_imgs = generator(z).detach()
                # Adversarial loss
                loss_critic = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))

                critic_optimizer.zero_grad()
                loss_critic.backward()
                critic_optimizer.step()

                # Clip weights of critic
                for p in critic.parameters():
                    p.data.clamp_(-config.clip_value, config.clip_value)

            # -----------------
            #  Train Generator
            # -----------------
            z = torch.randn(config.batch_size, config.z_dim).cuda()
            gen_imgs = generator(z)
            loss_generator = -torch.mean(critic(gen_imgs))

            generator_optimizer.zero_grad()
            loss_generator.backward()
            generator_optimizer.step()

            # Print losses occasionally and print to tensorboard
            if i % 400 == 0:
                print(
                    f"[Epoch {epoch}/{config.n_epochs}] [Batch {i}/{len(dataloader)}] [Critic loss: {loss_critic.item()}] [Generator loss: {loss_generator.item()}]"
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % 400 == 0:
                save_images(gen_imgs.data[:9], epoch, n_row=3, output_dir=config.output_dir)

    # Save models at the end of training
    torch.save(generator.state_dict(), f'{config.output_dir}/generator.pth')
    torch.save(critic.state_dict(), f'{config.output_dir}/critic.pth')


if __name__ == '__main__':
    train()
