import torch                    # type: ignore
import numpy as np              # type: ignore  
import matplotlib.pyplot as plt # type: ignore

def mae(datatrue, datapred):
    # Mean Absolute Error
    return (datatrue - datapred).abs().mean()

def mse(datatrue, datapred):
    # Mean Squared Error
    return (datatrue - datapred).pow(2).sum(axis=-1).mean()

def mre(datatrue, datapred):
    # Mean Relative Error
    return ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() 
        / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()

# Convert a probability to a percentage string with two decimal places
num2p = lambda prob: ("%.2f" % (100*prob)) + "%"

# Dimensionity reduction function using Singular Value Decomposition (SVD)
def dim_reduction(Xtrain, Xvalid, Xtest, modes):

    # reshape to vectors
    ntrain, ntimes, nstate = Xtrain.shape
    nvalid = Xvalid.shape[0]
    ntest = Xtest.shape[0]

    Xtrain = Xtrain.reshape(-1, nstate)
    Xvalid = Xvalid.reshape(-1, nstate)
    Xtest = Xtest.reshape(-1, nstate)

    # Perform Singular Value Decomposition on the training data
    U_POD, S_POD, V_POD = np.linalg.svd(Xtrain, full_matrices = False)
    POD_modes = modes

    # Project the training, validation, and test data onto the first 'modes' left singular vectors
    Xtrain_proj = Xtrain @ V_POD[:POD_modes, :].T
    Xtrain_proj = Xtrain_proj.reshape(ntrain, ntimes, POD_modes)
    Xtrain = Xtrain.reshape(ntrain, ntimes, nstate)

    Xvalid_proj = Xvalid @ V_POD[:POD_modes, :].T
    Xvalid_proj = Xvalid_proj.reshape(nvalid, ntimes, POD_modes)
    Xvalid = Xvalid.reshape(nvalid, ntimes, nstate)

    Xtest_proj = Xtest @ V_POD[:POD_modes, :].T
    Xtest_proj = Xtest_proj.reshape(ntest, ntimes, POD_modes)
    Xtest = Xtest.reshape(ntest, ntimes, nstate)

    # Print the residual energy and plot the decay of the singular values
    print(f"Residual energy is equal to  {np.sum(S_POD[POD_modes:]**2) / np.sum(S_POD**2)}\n")
    plt.plot(S_POD[:50], color = 'magenta', linewidth = 2)
    plt.axvline(POD_modes, color = 'teal', linestyle = '--', linewidth = 2)
    plt.title("Singular Values Decay")
    plt.show()

    return Xtrain, Xvalid, Xtest, Xtrain_proj, Xvalid_proj, Xtest_proj, V_POD
