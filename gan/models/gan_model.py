# gan/gan_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from .discriminator import Discriminator
from .generator import Generator

#
# class GAN:
#     def __init__(self, z_dim, R, T, lr, device):
#         self.generator = Generator(z_dim, R, T).to(device)
#         self.discriminator = Discriminator(R, T).to(device)
#         self.optim_g = optim.Adam(self.generator.parameters(), lr=lr)
#         self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr)
#         self.criterion = nn.BCELoss()
#         self.device = device
#         self.z_dim = z_dim
#
#     def train_discriminator(self, real_samples):
#         batch_size = real_samples.size(0)
#         real_labels = torch.ones((batch_size, 1), device=self.device)
#         fake_labels = torch.zeros((batch_size, 1), device=self.device)
#
#         self.optim_d.zero_grad()
#
#         # Real samples
#         real_pred = self.discriminator(real_samples)
#         real_loss = self.criterion(real_pred, real_labels)
#
#         # Generate fake samples
#         z = torch.randn((batch_size, self.z_dim), device=self.device)
#         fake_samples = self.generator(z)
#         fake_pred = self.discriminator(fake_samples.detach())
#         fake_loss = self.criterion(fake_pred, fake_labels)
#
#         # Total loss
#         d_loss = (real_loss + fake_loss) / 2
#         d_loss.backward()
#         self.optim_d.step()
#
#         return d_loss.item()
#
#     def train_generator(self, batch_size):
#         self.optim_g.zero_grad()
#
#         # We want fake predictions to be seen as real
#         z = torch.randn((batch_size, self.z_dim), device=self.device)
#         fake_samples = self.generator(z)
#         fake_pred = self.discriminator(fake_samples)
#         g_loss = self.criterion(fake_pred, torch.ones((batch_size, 1), device=self.device))
#
#         g_loss.backward()
#         self.optim_g.step()
#
#         return g_loss.item()
#
#     def train(self, dataloader, num_epochs):
#         for epoch in range(num_epochs):
#             for i, real_samples in enumerate(dataloader):
#                 real_samples = real_samples.to(self.device)
#                 batch_size = real_samples.size(0)
#
#                 # Train Discriminator
#                 d_loss = self.train_discriminator(real_samples)
#
#                 # Train Generator
#                 g_loss = self.train_generator(batch_size)
#
#                 if (i + 1) % 100 == 0:
#                     print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(dataloader)} \
#                           Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f}")



class GAN:
    def __init__(self, z_dim, R, T, lr, device):
        self.generator = Generator(z_dim, R, T).to(device)
        self.discriminator = Discriminator(R, T).to(device)
        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.device = device
        self.z_dim = z_dim

    def train_discriminator(self, real_samples):
        batch_size = real_samples.size(0)
        real_labels = torch.ones((batch_size, 1), device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)

        self.optim_d.zero_grad()

        # Real samples
        real_pred = self.discriminator(real_samples)
        real_loss = self.criterion(real_pred, real_labels)

        # Generate fake samples
        z = torch.randn((batch_size, self.z_dim), device=self.device)
        fake_samples = self.generator(z)
        fake_pred = self.discriminator(fake_samples.detach())
        fake_loss = self.criterion(fake_pred, fake_labels)

        # Total loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optim_d.step()

        return d_loss.item()

    def train_generator(self, batch_size):
        self.optim_g.zero_grad()

        # We want fake predictions to be seen as real
        z = torch.randn((batch_size, self.z_dim), device=self.device)
        fake_samples = self.generator(z)
        fake_pred = self.discriminator(fake_samples)
        g_loss = self.criterion(fake_pred, torch.ones((batch_size, 1), device=self.device))

        g_loss.backward()
        self.optim_g.step()

        return g_loss.item()

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for i, real_samples in enumerate(dataloader):
                real_samples = real_samples.to(self.device)
                batch_size = real_samples.size(0)

                # Train Discriminator
                d_loss = self.train_discriminator(real_samples)

                # Train Generator
                g_loss = self.train_generator(batch_size)

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(dataloader)} \
                          Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f}")
