import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from lib.utils import make_dir,save_loss, sample_indices, save_samples, save_model, find_best_gpus

import os 
from os import path as pt

from torchvision import datasets, models, transforms
from tqdm import tqdm
from collections import defaultdict
import pickle 

torch.manual_seed(0)
np.random.seed(0)

def run(config):
    # get parameters 
    algo = config.algo
    dataset = config.dataset
    root_dir = config.root_dir
    latent_dim = config.latent_dim
    frame_size = config.frame_size 
    hidden_dim = config.hidden_dim 
    p,q = config.p,config.q 
    num_frames = p+q

    #find the best gpu based on their memeory left 
    device = find_best_gpus(1)[0] if config.use_cuda and torch.cuda.is_available() else 'cpu' 
    
    #set the train size and val size
    train_size, val_size = 9000, 1000

    ## Training loop
    num_epochs = config.num_epochs#10000
    batch_size = config.batch_size# 16
    d_per_g = config.d_per_g#1

    ## make dir
    dir_path = make_dir(algo,p,q,d_per_g,root=root_dir)

    #load dataset
    movingmnist = np.load('data/mnist_test_seq.npy')[-num_frames:,:train_size + val_size] #(T,B,H,W) to [0,1]
    #downsample
    downsample = 64//frame_size
    movingmnist = movingmnist[:,:,::downsample,::downsample]/255

    #define the data transformation 
    datameans = (0.5,) 
    datastds = (0.5,) 

    data_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(256),
        #transforms.ToTensor(),
        transforms.Normalize(datameans, datastds)
        ])
    
    inv_transforms = transforms.Compose([
        transforms.Normalize( [ m/s for m,s in zip(datameans,datastds)], [1/s for s in datastds])
        ])

    #to torch tensor
    data_raw = torch.from_numpy(np.swapaxes(movingmnist, 0,1)[:,:,None]).to(device).float()# (B,T,1,H,W) scale [0,1]
    print(data_raw.max(),data_raw.min())
    channels = data_raw.shape[2]

    #data transform
    data_processed = torch.stack([data_transforms(data) for data in data_raw]) 
    train_data, valid_data = data_processed [:train_size], data_processed [train_size:]

    print(data_processed .max(),data_processed .min())
    print('datasize:',data_processed.shape, 'trainsize',data_processed.shape,'testsize',data_processed.shape)

    # Build Model 
    from lib.networks import Generator_LSTM,Discriminator_LSTM

    G_config = dict(num_channels=channels,
                num_kernels=hidden_dim,
                kernel_size=(3, 3),
                padding=(1, 1),
                activation="relu", 
                frame_size=(frame_size, frame_size), num_layers=config.G_num_layers, 
                num_frames=q,
                device= device)


    D_config = dict(num_channels=channels,
                num_kernels=hidden_dim,
                kernel_size=(3, 3),
                padding=(1, 1),
                activation="relu", 
                frame_size=(frame_size, frame_size), 
                num_layers=config.D_num_layers,
                device= device)

    print('Building models')
    print(f'G_config:\n{G_config}')
    print(f'D_config:\n{D_config}')

    generator = Generator_LSTM(**G_config).to(device)

    discriminator = Discriminator_LSTM(**D_config).to(device)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()# nn.BCELoss()#

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #build loss history 
    loss_history = defaultdict(list)

    for epoch in tqdm(range(num_epochs+1)):
        # Train Discriminator
        optimizer_D.zero_grad()
        # get real samples 
        indx = sample_indices(train_size, batch_size,device)
        # Real videos
        real_videos = train_data[indx] # Load a batch of real videos
        real_labels = torch.ones((batch_size, 1), requires_grad=False).to(device)
        real_noise = torch.randn((batch_size, q, 1,frame_size,frame_size), requires_grad=False).to(device)*0.0005

        # Generate fake videos
        z = torch.randn(batch_size, latent_dim).to(device)
        real_past, real_future =  real_videos[:,:p], real_videos[:,p:]
        fake_future = generator(real_past,z)
        fake_videos = torch.cat([real_past,fake_future],axis=1)
        fake_labels = torch.zeros((batch_size, 1), requires_grad=False).to(device)

        # Discriminator loss on real and fake videos
        d_real = discriminator(real_videos+real_noise)
        d_fake = discriminator(fake_videos.detach())


        # BCE LOSS 
        real_loss = adversarial_loss(d_real, real_labels)
        fake_loss = adversarial_loss(d_fake, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward(retain_graph=True )
        optimizer_D.step()

        

        if epoch % d_per_g == 0:
            # Train Generator
            optimizer_G.zero_grad()
            # Generate fake videos
            real_videos = train_data[indx] # Load a batch of real videos
            real_past, real_future =  real_videos[:,:p], real_videos[:,p:]
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_future = generator(real_past,z)
            fake_videos = torch.cat([real_past,fake_future],axis=1)
        
            d_fake = discriminator(fake_videos)
        
            # Generator loss
            if algo!='MCGAN':
                real_labels = torch.ones((batch_size, 1), requires_grad=False).to(device)
                g_loss = adversarial_loss(d_fake, real_labels)
            else:
                d_real = discriminator(real_videos)
                g_loss = torch.nn.functional.mse_loss(d_fake, d_real, reduction='mean')
            g_loss.backward(retain_graph=True )
            optimizer_G.step()

            #save loss
            loss_history['d_loss'].append(d_loss.item())
            loss_history['d_real'].append(d_real.mean().item())
            loss_history['d_fake'].append(d_fake.mean().item())
            loss_history['g_loss'].append(g_loss.item())

        # for eachj 500 epoch save samples and compute test metrics
        if epoch % 500 == 0:
            if epoch == 0:
                print('X_real shape and X_fake shape', real_videos.shape, fake_videos.shape)
            print(f"Epoch {epoch} / {num_epochs} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
        
            #compute test metrics MSE:
            with torch.no_grad():
                real_videos = valid_data[:]
                real_past, real_future =  real_videos[:,:p], real_videos[:,p:]
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_future = generator(real_past,z)
                fake_videos = torch.cat([real_past,fake_future],axis=1)
                #compute mse 
                mse = torch.nn.functional.mse_loss(fake_future , real_future, reduction='mean').detach().cpu().item()
                loss_history['mse_valid'].append(mse)
            #save samples 
            save_samples(real_videos[:20,:].detach(),fake_videos[:20,:].detach(), pt.join(dir_path,'saved_gif'), t=num_frames,epoch=epoch)
            save_samples(real_videos[:20,-q:].detach(),fake_videos[:20,-q:].detach(), pt.join(dir_path,'saved_gif_lastq'), t=q,epoch=epoch)
            #sava history
            with open(pt.join(dir_path,'loss_history.pkl'), 'wb') as f:
                pickle.dump(loss_history, f)

            #plot loss summary
            save_loss(loss_history,dir_path)
            #save models 
            save_model(generator,pt.join(dir_path,'saved_models'),f'G_state_dict_{epoch}.pt')
            save_model(discriminator,pt.join(dir_path,'saved_models'),f'D_state_dict_{epoch}.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser() 

    # Meta parameters
    parser.add_argument('-root_dir', default='./experiment', type=str)
    parser.add_argument('-use_cuda',  action='store_true')
    parser.add_argument('-dataset', default='movingmnist', nargs="+")
    
    #algo keyword supports only 'GAN' and 'MCGAN'
    parser.add_argument('-algo', default = 'GAN', nargs="+")#
    parser.add_argument('-latent_dim', default = 128, type=int)
    parser.add_argument('-num_epochs', default = 10000, type=int)
    parser.add_argument('-batch_size', default = 16, type=int)
    parser.add_argument('-d_per_g', default = 1, type=int)

    #model parameter 
    parser.add_argument('-hidden_dim', default = 64, type=int)
    parser.add_argument('-frame_size', default = 32, type=int)
    parser.add_argument('-p', default = 10, type=int)
    parser.add_argument('-q', default = 1, type=int)
    parser.add_argument('-G_num_layers', default = 2, type=int)
    parser.add_argument('-D_num_layers', default = 2, type=int)


    args = parser.parse_args()
    run(args)
