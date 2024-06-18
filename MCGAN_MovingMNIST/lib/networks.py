from  lib.convlstm import ConvLSTM, Seq2Seq
import torch.nn as nn
import torch

class Generator_LSTM(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers, num_frames, device ):

        super(Generator_LSTM, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size, device = device)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):
            out_channels  = num_kernels # channels  if l== num_layers else num_kernels 
            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=out_channels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size, device = device)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=out_channels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)
        self.frame_size = frame_size[0]
        self.num_frames  = num_frames 
        self.l1 = nn.Sequential(nn.Linear(128, 128 * self.num_frames * self.frame_size**2))

    def forward(self, x_past, z):
       # h = self.l1(z).view(z.shape[0],128,self.num_frames, self.frame_size, self.frame_size)
        # Forward propagation through all the layers

        output = self.sequential(x_past.transpose(1,2))#(h)
        # Return only the last output frame
        output = self.conv(output[:,:,-1]).unsqueeze(2)
        #outputs = []
        #for i in range(self.num_frames):
        #    outputs.append(self.conv(output[:,:,i]))
        #outputs = torch.stack(outputs,dim=2)
        #print(output.shape)
        return torch.tanh(output).transpose(1,2)



from math import prod
class Discriminator_LSTM(nn.Module):
    def __init__(self,num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers, device, sigmoid=False):
        super(Discriminator_LSTM,self).__init__()
        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size, device=device)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):
            out_channels  = num_kernels # channels  if l== num_layers else num_kernels 
            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=out_channels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size, device=device)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=out_channels)
                ) 

        
        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)
        self.fn  = nn.Linear(prod(frame_size),1)
        self.sigmoid = sigmoid 
        
        
    def forward(self,x_t):
        # Forward propagation through all the layers
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)
        x_t = x_t.transpose(2,1) # from (B,T,C,H,W) to (B,C,T,H,W)
        h_t = self.sequential(x_t)
        # Return only the last output frame
        h = self.conv(h_t[:,:,-1])
        #final prediction
        output  = self.fn(nn.ReLU()(h.flatten(1)))

        if not self.sigmoid:
            return output
        else:
            return nn.Sigmoid()(output)