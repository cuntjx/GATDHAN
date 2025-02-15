# GATDHAN  
Dual hierarchical attention network combined with residual mechanism for predicting circRNA–miRNA associations  

# Data and code  
## CMI-9589  
9589_circRNA_sequence.csv：This file stores the name and the sequence of circRNAs.  

9589_miRNA_sequence.csv: This file stores the name and the sequence of miRNAs.  

A_9589.csv: This file stores the association information between circRNAs and miRNAs. 

merged_9589_circRNA_3.pkl: This file stores the features of circRNA which can be download from URL: https://github.com/ntitatip/A-Knowledge-Graph-Enhanced-Pre-trained-Large-Language-Model-for-Predicting-MicroRNA-circRNA-Interac.  

merged_9589_micRNA_3.pkl: This file stores the features of miRNA which can be download from URL: https://github.com/ntitatip/A-Knowledge-Graph-Enhanced-Pre-trained-Large-Language-Model-for-Predicting-MicroRNA-circRNA-Interac. 

## CMI-9905  
circSequence_9905.csv： This file stores the name and the sequence of circRNAs.  

miSequence_9905.csv: This file stores the name and the sequence of miRNAs.  

A.csv: This file stores the association information between circRNAs and drugs. 

merged_9905_circRNA_3.pkl: This file stores the features of circRNA which can be download from URL: https://github.com/ntitatip/A-Knowledge-Graph-Enhanced-Pre-trained-Large-Language-Model-for-Predicting-MicroRNA-circRNA-Interac.  

merged_9905_micRNA_3.pkl: This file stores the features of miRNA which can be download from URL: https://github.com/ntitatip/A-Knowledge-Graph-Enhanced-Pre-trained-Large-Language-Model-for-Predicting-MicroRNA-circRNA-Interac}. 

GATDHAN_CMI_9905.ipynb: GATDHAN is run on Jupyter Notebook, and all the code for CMI-9905 dataset is contained within this file.   

GATDHAN_CMI_9589.ipynb: GATDHAN is run on Jupyter Notebook, and all the code for CMI-9589 dataset is contained within this file.   

# Environment  
GATDHAN is implemented to work under Python 3.9 and Jupyter Notebook  
torch 2.5.*  
numpy 1.22.*   
pandas 1.5.*  
torch_geometric 2.5.*  
 
# Run steps  
GATDHAN_CMI_9905.ipynb and GATDHAN_CMI_9589.ipynb can be upload into Jupyter Notebook and select ‘run all’ to train the model and obtain prediction scores for circRNA-miRNA associations.  
## Function descriptions for in files GATDHAN_CMI_9905.ipynb and GATDHAN_CMI_9589.ipynb   
Integration of feature  
def get_syn_sim(A, seq_sim, str_sim, mode)  
    PARAMETERS:  
    A(numpy matrix): The adjacency matrix of circRNAs between miRNAs.  
    seq_sim(numpy matrix): The similarity matrix of circRNAs.  
    str_sim(numpy matrix): The similarity matrix of miRNAs.  
    mode: 0 = GIP kernel sim.  
    return the new similarity of circRNAs and miRNAs.  
    
Get the index and type of the edges  
def get_edge_index(matrix, new_A, threshold)   
    PARAMETERS:  
    matrix(numpy matrix): The similarity matrix of circRNAs or miRNAs.  
    new_A(numpy matrix): The adjacency matrix of circRNAs between miRNAs.  
    threshold (int): Threshold for distinguishing between head and tail node. Here, we do not distinguish between the head and tail nodes. 
    
Dual attention layer  
class DGATConv (self, in_hid, out_hid, num_edge_types, negative_slope=0.2, dual=True, heads=1, mask=None, global_weight=True)  
    PARAMETERS:  
    in_hid (int): Dimension of input features.  
    out_hid (int): Dimension of output features.  
    num_edge_types(int): Number of edge types.  
    negative_slope(float, optional): Controls the angle of the negative slope (which is used for negative input values). (default: 0.2)  
    dual(bool, optional): Controls whether dual attention mechanisms are used. (default: True)  
    heads(int, optional): Number of multi-head-attentions. (default: 1)  
    mask(int, optional): Feature extraction by specific types of edges. (default: None)  
    global_weight (bool, optional): Controls the use of global weights. (default: True)  

Hierarchical attention layer  
class HetGATConv (self, in_hid, out_hid, negative_slope=0.2, norm=True, dual=True, global_weight=True)   
    PARAMETERS:    
    in_hid (int): Dimension of input features.  
    out_hid (int): Dimension of output features.  
    norm(bool, optional): Parameter reservation. (default: True)  
    dual(bool, optional): Controls whether dual attention mechanisms are used. (default: True)  
    global_weight (bool, optional): Controls the use of global weights. (default: True)  

GAT layer
class GATConv(MessagePassing):
PARAMETERS:
in_channels (int): Dimension of input features
out_channels (int): Dimension of output features
heads (int): Number of heads of multi-attention mechanisms

Our model  
class GATDHAN(self,n_hid_layers: int, hid_features: list, hid_features2: list, n_heads: list, n_dis: int, n_pi: int,
                 dropout: float = 0.6, n_hid_layers2: int = 1, dual: bool = False)  
    PARAMETERS:
    n_hid_layers (int): The number of GAT layer.
    hid_features (list): Dimension of circRNA features.  
    hid_features2 (list): Dimension of miRNA features.  
    n_heads (int): Number of heads of multi-attention mechanisms  
    n_dis (int): Number of circRNAs.  
    n_pi (int): Number of miRNAs.     
    n_hid_layers2 (int): Number of Dual attention layers. (default: 1)  
    dropout(float, optional): Dropout probability of the normalized attention coefficients which exposes each node to a 
    stochastically sampled neighborhood during raining. (default: 0.6)  
    dual (bool, optional): If “True” then the dual mechanism is used in the DHAN layer. (default: False) 
    



