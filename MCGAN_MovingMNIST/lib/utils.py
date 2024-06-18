import os
import os.path as pt
import matplotlib.pyplot as plt
import torch 
import numpy as np 
from PIL import Image

def make_dir(algo,p,q,dg,root='experiment/',suffix=0):
    dir_path = pt.join(root,f'{algo}_p{p}_q{q}_dg{dg}/{suffix}')
    if not pt.exists(dir_path):
        # if the experiment directory does not exist we create the directory
        os.makedirs(dir_path)
    else:
        dir_path=make_dir(algo,p,q,dg,root,suffix=suffix+1)
    return dir_path

def sample_indices(dataset_size, batch_size,device):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False)).to(device).long()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices


def save_loss(loss_history,dir_path):
        fig, axes = plt.subplots(len(loss_history), 1, figsize=(10, 8))

        for i, key in enumerate(loss_history.keys()):
            axes[i].plot(loss_history[key],label=key)
            axes[i].grid()
            axes[i].legend()
            axes[i].set_title(key)
        axes[-1].set_ylim(0,0.5)
        plt.savefig(pt.join(dir_path,'loss_history.png'),dpi=100)
        plt.close()


def array2gif(array: np.array, file_path = "data/test_1.gif", frames=20):
    image_list = []
    for i in range(frames):
        im = Image.fromarray(array[i].astype('uint8'),'L').convert('L').convert('P')
        image_list.append(im)
    image_list[0].save( file_path, save_all=True, append_images=image_list[1:], optimize=False, duration=40, loop=0)


def save_model(model, dir_path, file_name):
    if not pt.exists(dir_path):
        # if the experiment directory does not exist we create the directory
        os.makedirs(dir_path)
    torch.save(model.state_dict(), pt.join(dir_path,file_name))
    return None


def save_samples(x_real,x_fake, base_path, n=10,t=20,epoch=0):
    if not pt.exists(pt.join(base_path,f"{epoch}")):
        # if the experiment directory does not exist we create the directory
        os.makedirs(pt.join(base_path,f"{epoch}"))
    for i,(xr,xf) in enumerate(zip(x_real.cpu().numpy()*255,x_fake.cpu().numpy()*255)):
        array2gif(xr[:,0],file_path=pt.join(base_path,f"{epoch}/x_real_{i}_ep{epoch}.gif"),frames=t)
        array2gif(xf[:,0],file_path=pt.join(base_path,f"{epoch}/x_fake_{i}_ep{epoch}.gif"),frames=t)
    return None


def find_best_gpus(num_gpu_needs=1):
    import subprocess as sp
    gpu_ids = []
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [(int(x.split()[0]), i) for i, x in enumerate(memory_free_info) if i not in gpu_ids]
    print('memories left ', memory_free_values)
    memory_free_values = sorted(memory_free_values)[::-1]
    gpu_ids = [k for m, k in memory_free_values[:num_gpu_needs]]
    print("GPU:", gpu_ids)
    return gpu_ids
