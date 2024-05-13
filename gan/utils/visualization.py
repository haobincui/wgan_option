import numpy as np
import matplotlib.pyplot as plt

def save_images(gen_imgs, epoch, n_row=3, output_dir='./output'):
    """Saves a grid of generated digits ranging from 0 to n_row**2"""
    fig, axs = plt.subplots(n_row, n_row)
    cnt = 0
    for i in range(n_row):
        for j in range(n_row):
            axs[i, j].imshow(gen_imgs[cnt, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()
