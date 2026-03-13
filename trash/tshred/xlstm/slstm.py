import torch                    #type: ignore
import torch.nn as nn           #type: ignore


#slstm cell
class sLSTMCell(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        
        super().__init__()

        #store the input and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size

        #Wheights and biases for the input and hidden states
        #4 for the 4 gates:
        # z = candidate cell state
        # i = input gate
        # f = forget gate
        # o = output gate
        self.W = nn.Linear(input_size, 4*hidden_size)
        self.R = nn.Linear(hidden_size, 4*hidden_size, bias = False)    #no double counting of bias

    def forward(self, x, states):

        #unpack the states
        # h_prev = previous hidden state
        # c_prev = previous cell state
        # n_prev = previous n state 
        # m_prev = numerical stability state
        h_prev, c_prev, n_prev, m_prev = states

        #compute the gates
        gates = self.W(x) + self.R(h_prev)
        z_raw, i_raw, f_raw, o_raw = gates.chunk(4,1)

        #numerical stability state
        #instedad of computing the exponentials of i_raw and f_raw directly
        #we subtract the maximum of the two from both to prevent overflow
        mt = torch.max(i_raw, f_raw+m_prev)
        i_prime = torch.exp(i_raw - mt)
        f_prime = torch.exp(f_raw + m_prev- mt)

        #compute the activations
        zt = torch.tanh(z_raw)
        ot = torch.sigmoid(o_raw)

        #states update
        ct = f_prime * c_prev + i_prime * zt
        nt = f_prime * n_prev + i_prime
        ht = ot * ct / (nt+1e-8)                #numerical stability

        #Return the new hidden state and the new states
        return ht, (ht, ct, nt, mt)

#slstm layer
class sLSTMLayer(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        
        super().__init__()

        #the sLSTM cell
        self.cell = sLSTMCell(input_size, hidden_size)

    def forward(self, x):

        #extract the batch size and sequence length from the input
        batch_size, seq_len, _ = x.size()

        #initialize the states to zeros
        h0 = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        c0 = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        n0 = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        m0 = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)

        # initialize a list to store the outputs at each time step
        outputs = []
        states = (h0, c0, n0, m0)   

        #update the states for each time step
        for t in range(seq_len):
            xt = x[:,t,:]
            ht, states = self.cell(xt, states)
            outputs.append(ht.unsqueeze(1))

        #concatenate the outputs along the sequence dimension and return
        return torch.cat(outputs, dim=1)
        
#sLSTM model
class sLSTM(nn.Module): 

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        
        super().__init__()

        #store the input size, hidden size, and number of layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #create the sLSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(sLSTMLayer(layer_input_size, hidden_size))

    def forward(self, x):

        #pass the input through each layer
        for layer in self.layers:
            x = layer(x)

        return x