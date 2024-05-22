import torch
import torch.optim as optim
from .discriminator import Discriminator
from .generator import Generator


class WGAN:
    def __init__(self, z_dim, R, T, lr, device, clip_value=0.01):
        self.generator = Generator(z_dim, R, T).to(device)
        self.discriminator = Discriminator(R, T).to(device)
        self.optim_g = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optim_d = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.device = device
        self.z_dim = z_dim
        self.clip_value = clip_value

    def train_discriminator(self, real_samples):
        batch_size = real_samples.size(0)

        self.optim_d.zero_grad()

        # Real samples
        real_pred = self.discriminator(real_samples)
        real_loss = -torch.mean(real_pred)

        # Generate fake samples
        z = torch.randn((batch_size, self.z_dim), device=self.device)
        fake_samples = self.generator(z)
        fake_pred = self.discriminator(fake_samples)
        fake_loss = torch.mean(fake_pred)

        # Total loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optim_d.step()

        # Clip weights of discriminator
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

        return d_loss.item()

    def train_generator(self, batch_size):
        self.optim_g.zero_grad()

        z = torch.randn((batch_size, self.z_dim), device=self.device)
        fake_samples = self.generator(z)
        fake_pred = self.discriminator(fake_samples)
        g_loss = -torch.mean(fake_pred)

        g_loss.backward()
        self.optim_g.step()

        return g_loss.item()

    def train(self, dataloader, num_epochs, n_critic=5):
        for epoch in range(num_epochs):
            for i, real_samples in enumerate(dataloader):
                real_samples = real_samples.to(self.device)

                # Train Discriminator more frequently than Generator
                for _ in range(n_critic):
                    d_loss = self.train_discriminator(real_samples)

                # Train Generator
                g_loss = self.train_generator(real_samples.size(0))

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(dataloader)} \
                          Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f}")
