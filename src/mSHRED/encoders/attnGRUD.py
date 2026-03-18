import torch                            # type: ignore
import torch.nn as nn                   # type: ignore
import torch.utils.data as utils        # type: ignore
import torch.nn.functional as F         # type: ignore

from torch.autograd import Variable         # type: ignore
from torch.nn.parameter import Parameter    # type: ignore

from tqdm import tqdm                   # type: ignore
import math                             # type: ignore    
import numpy as np                      # type: ignore
import pandas as pd                     # type: ignore
import time                             # type: ignore 


#filter squre matrix is the matrix that will be used to filter the weights of the linear layer
#it should have the same shape as the weight matrix of the linear layer and contain 
#-1sfor the weights that should be kept 
#-0s for the weights that should be masked out
class FilterLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        filter_square_matrix: torch.Tensor,
        bias: bool = True
    ):
        #initialize the linear layer with the given in_features and out_features
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        #check cuda availability and move the filter_square_matrix to the same device as the model parameters
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        #initialize the weight and bias parameters of the linear layer
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        #reset the parameters of the linear layer using the reset_parameters method
        self.reset_parameters()

    def reset_parameters(self):
        #He initialization for the weights of the linear layer
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        #apply the filter_square_matrix to the weights of the linear layer
        masked_weight = self.weight * self.filter_square_matrix

        #perform the linear transformation using the masked weights and the bias
        return F.linear(input, masked_weight, self.bias)

#the GRUD model is a recurrent neural network that is designed to handle missing data in time series.
#it uses a gated recurrent unit (GRU) architecture and incorporates a masking mechanism to handle missing values in the input data.

# Attention idea
# ----------------------------------------------------------------
# Attention mechanism (feature-wise, inspired by eq. 7-10).
# At each time step t the previous (decayed) hidden state h_{t-1}
# and the imputed input x_t are used to produce a score e_k for
# each feature k:
#   e_k  = v^T  tanh( W1 * h_{t-1}  +  W2 * x_t^k )
#   alpha = softmax(e)              
#   z_t  = alpha * x_t             
# ----------------------------------------------------------------

class AttnGRUD(nn.Module):
    def __init__(
        self,
        input_size: int,
        cell_size: int,
        hidden_size: int,
        X_mean: torch.Tensor,
        output_last: bool = False
    ):
        super().__init__()

        #store the needed parameters 
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        #check cuda availability and move the X_mean to the same device as the model parameters
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zero = Variable(torch.zeros(input_size).cuda())
            self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zero = Variable(torch.zeros(input_size))
            self.X_mean = Variable(torch.Tensor(X_mean))

        #initialize zl rl hl linear layers for the GRU cell
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)

        #initialize gamma_x and gamma_h linear layers for the decay mechanism
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size)

        #initialize the output linear layer
        self.output_last = output_last

        # ATTENTION parameters
        self.attn_size = hidden_size

        # W1: projects h  (hidden_size -> attn_size)
        self.W_attn_h = nn.Linear(hidden_size, self.attn_size, bias=False)
        # W2: scalar weight applied to each individual feature value
        self.W_attn_x = nn.Parameter(torch.Tensor(self.attn_size))
        # v:  scoring vector  (attn_size -> scalar per feature)
        self.v_attn    = nn.Parameter(torch.Tensor(self.attn_size))

        # initialise attention parameters with uniform distribution
        nn.init.uniform_(self.W_attn_x, -1. / math.sqrt(self.attn_size),
                                          1. / math.sqrt(self.attn_size))
        nn.init.uniform_(self.v_attn,   -1. / math.sqrt(self.attn_size),
                                          1. / math.sqrt(self.attn_size))


    def attention(self, x, h):
        
        # h_proj : (batch, attn_size)  ->  unsqueeze to (batch, 1, attn_size)
        h_proj = self.W_attn_h(h).unsqueeze(1)          # (B, 1, A)

        # x_proj : each feature value (scalar) scaled by W_attn_x
        # x.unsqueeze(-1) : (B, D, 1)  *  W_attn_x (A,) -> (B, D, A)
        x_proj = x.unsqueeze(-1) * self.W_attn_x        # (B, D, A)

        # energy  : (B, D, A)
        energy = torch.tanh(h_proj + x_proj)

        # scores  : (B, D)  — dot with v
        scores = (energy * self.v_attn).sum(dim=-1)      # (B, D)

        # alpha   : (B, D)  — normalised attention weights
        alpha = F.softmax(scores, dim=-1)                # (B, D)

        # z_t = alpha * x  (element-wise feature rescaling)
        return alpha * x

    def step(
        self,
        x: torch.Tensor,
        x_last_obsv: torch.Tensor,
        x_mean: torch.Tensor,
        h: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor
    ):
        #sizes
        batch_size = x.shape[0]
        dim_size = x.shape[1]

        #deltas decay
        delta_x = torch.exp(-torch.clamp(self.gamma_x_l(delta), min=0))
        delta_h = torch.exp(-torch.clamp(self.gamma_h_l(delta), min=0))

        # xnew computation (imputation of missing values)
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h

        # apply attention: re-weight the imputed input using the previous
        # (decayed) hidden state as context (eqs. 8-10)
        x = self.attention(x, h)

        #combined + z and r computation
        combined = torch.cat((x, h, mask), dim=1)
        z = torch.sigmoid(self.zl(combined))
        r = torch.sigmoid(self.rl(combined))

        #combined_r + h_tilda computation
        combined_r = torch.cat((x, r * h, mask), dim=1)
        h_tilda = torch.tanh(self.hl(combined_r))

        #final h computation
        h = (1 - z) * h + z * h_tilda
        return h
    
    def initHidden(self, batch_size):

        #initialize the hidden state of the GRU cell with zeros
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            return Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(batch_size, self.hidden_size))
        
    def forward(self, input):
        
        #sizes
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)

        #extract the different components from the input tensor
        Hidden_State = self.initHidden(batch_size)
        X =             torch.squeeze(input[:,0,:,:])
        X_last_obsv =   torch.squeeze(input[:,1,:,:])
        Mask =          torch.squeeze(input[:,2,:,:])
        Delta =         torch.squeeze(input[:,3,:,:])

        #iterate over the time steps and apply the GRU cell to each time step
        outputs = None
        for i in range(step_size):

            #call the step function to compute the hidden state for the current time step
            Hidden_State = self.step(
                torch.squeeze(X[:,i:i+1,:]),
                torch.squeeze(X_last_obsv[:,i:i+1,:]),
                self.X_mean[:, 0, i, :],
                Hidden_State,
                torch.squeeze(Mask[:,i:i+1,:]),
                torch.squeeze(Delta[:,i:i+1,:])
            )

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), dim=1)

        #if i want to output only the last hidden state, I return outputs[:,-1,:]
        #otherwise I return all the hidden states

        if self.output_last:
            outputs = outputs[:,-1,:]
            assert outputs.dim() == 2, "Expected output shape (batch_size, hidden_size), got {}".format(outputs.shape)
            return outputs
        else:
            return outputs
