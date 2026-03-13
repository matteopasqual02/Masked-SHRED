import torch                 #type: ignore
import torch.nn as nn        #type: ignore  

# mLSTM cell
class mLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        head_size: int,
    ):
        super().__init__()
        # Store head_size to use for scaling in the forward pass
        self.head_size = head_size 

        # Query, key, value projections for matrix memory 
        self.Q = nn.Linear(input_size, head_size)
        self.K = nn.Linear(input_size, head_size)
        self.V = nn.Linear(input_size, head_size)

        # Gating mechanism: input (i), forget (f), and output (o) 
        self.I = nn.Linear(input_size, 1)
        self.F = nn.Linear(input_size, 1)
        self.O = nn.Linear(input_size, head_size)

    def forward(self, x, states):

        # Unpack the states
        # c_prev: matrix memory, n_prev: normalizer, m_prev: stabilizer 
        c_prev, n_prev, m_prev = states

        # Compute Q, K, V projections 
        qt = self.Q(x)
        kt = self.K(x) / (self.head_size ** 0.5)    # Scale keys for stable gradients
        vt = self.V(x)

        # Exponential gating mechanism pre-activations 
        it_raw = self.I(x)
        ft_raw = self.F(x)
        ot_raw = self.O(x)

        # Numerical stability state (m_t) 
        mt = torch.max(ft_raw + m_prev, it_raw)
        
        # Stabilized gates using the log-sum-exp trick
        i_prime = torch.exp(it_raw - mt)
        f_prime = torch.exp(ft_raw + m_prev - mt)
        o_prime = torch.sigmoid(ot_raw)             # Output gate can use sigmoid directly as it doesn't involve the same exponential scaling issues

        # Matrix memory update (Covariance rule) 
        ct = f_prime.unsqueeze(-1) * c_prev + i_prime.unsqueeze(-1) * torch.bmm(vt.unsqueeze(2), kt.unsqueeze(1))
        
        # Normalization state update
        nt = f_prime * n_prev + i_prime * kt

        # Memory recovery (Retrieval) 
        # h_tilde = (C_t @ q_t) / max(|n_t^T @ q_t|, 1)

        nq = torch.bmm(nt.unsqueeze(1), qt.unsqueeze(2)).squeeze(2)
        htilde = torch.bmm(ct, qt.unsqueeze(2)).squeeze(2) / torch.max(torch.abs(nq), torch.ones_like(nq))

        # Final hidden state output 
        ht = o_prime * htilde

        # Return the new hidden state and updated states for the next time step
        return ht, (ct, nt, mt)
    
# mLSTM layer
class mLSTMLayer(nn.Module):

    def __init__(
        self,
        input_size: int,
        head_size: int,
    ):
        
        super().__init__()

        # Initialize the mLSTM cell
        self.cell = mLSTMCell(input_size, head_size)

    def forward(self, x):

        #extract the batch size and sequence length from the input
        batch_size, seq_len, _ = x.size()

        #initialize the states to zeros
        c0 = torch.zeros(batch_size, self.cell.head_size, self.cell.head_size, device=x.device)  # matrix memory
        n0 = torch.zeros(batch_size, self.cell.head_size, device=x.device)                       # normalizer vector
        m0 = torch.zeros(batch_size, 1, device=x.device)                                         # stabilizer scalar

        # initialize a list to store the outputs at each time step
        outputs = []
        states = (c0, n0, m0)   

        #update the states for each time step
        for t in range(seq_len):
            xt = x[:,t,:]
            ht, states = self.cell(xt, states)
            outputs.append(ht.unsqueeze(1))

        #concatenate the outputs along the sequence dimension and return
        return torch.cat(outputs, dim=1)

# mLSTM model
class mLSTM(nn.Module): 

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

        #create the mLSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(mLSTMLayer(layer_input_size, hidden_size))

    def forward(self, x):

        #pass the input through each layer
        for layer in self.layers:
            x = layer(x)

        return x
        
        