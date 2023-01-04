# ProteinMPNN Dissection
01-04-2023

### The `featurize` function:

found in `model_utils.py` converts a batch into all of the desired matrices.

The Main Data Matrices:
* `X` (the raw data matrix) is a 4D tensor storing the coordinate-frames for each backbone position for each protein complex in the batch. Some of the coords in raw data are stored in the batch dictionaries as `np.nan`, likely because they're gaps in the PDB structure these are converted to all zeros in featurize and are recorded in the mask matrix.
	* Dim 1: Number of proteins in batch
	* Dim 2: Number of AA of largest protein
	* Dim 3: Tracked atom bins (N, C$\alpha$, C, O)
		* C$\beta$ is computed during forward pass.
	* Dim 4: XYZ coords of atoms

* `S` (the sequence matrix) is essentially the labels matrix, these don't appear to be One-Hot Encoded during the forward pass:
	* Dim 1: Number of proteins in batch
	* Dim 1: Number of AA of largest protein (contains label 0->19) corresponding to elements of alphabet. Seems like zeros are used both as a character and also as padding... Needs slicing by lengths matrix.
 
* `mask` records which elements of the data matrix X are NaN or hidden during training/testing.  NaNs are introduced when padding the smaller proteins in the batch up to the length of the longest in the batch. They are also introduced in the pre-processing of the data as there are NaNs stored in the raw coordinate arrays. These are set to zeros in the X that is returned, and their positions are used to create the mask matrix. It seems like not all training example protein complexes have a visible chain, I'm not sure why as this seems wasteful for training.
	* Dim 1: Number of proteins in batch
	* Dim 2: Number of AA of largest protein in batch 
 
* `chain_M` records which elements of the data matrix X are hidden during training/testing as well but rather than masking for the lengths of the protein like `mask` this keeps track of which chains are visible. Chains that are visible are marked with 0 and hidden chains are marked with 1
	* Dim 1: Number of proteins in batch
	* Dim 2: Number of AA of largest protein in batch 

* `residue_idx` constructs a unique index for each residue in each chain of a given protein complex. The code that creates this is 
	```python
	for i,b in enumerate(batch):
		residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
	```
	which is executed for each chain in the loop where c is the 1-index of the chain in the protein complex example in the batch which is incremented for each chain. The index is unique regardless of whether the chain is visible or hidden and each chain is offset by 100 for some reason. For example for three residues chain A and chain B the indices would be 0,1,2 and 103,104,105. This matrix is initialized to -100 at every index corresponding to an empty index for the chains that are not the longest one in the batch it is then filled in as the chains are looped over with the above formula.
	* Dim 1: Number of proteins in batch
	* Dim 2: Number of AA of largest protein in batch 

### The Forward Pass:

##### The `gather_nodes(nodes, neighbor_idx)` Function
* This relies on the `torch.gather` function which given a matrix and an matrix to index it with, returns the requested indices along a specified dimension. 

Relies on the `gather_nodes` function which is responsible for collecting the neighbor information for each node. `cat_neighbors_nodes(nodes, neighbors, idx)` does the same thing by calling this function and then concatenating the neighbors to the nodes after the nodes are gathered according to the index.

1) Constructs E (features), E_idx (node indices) from the input data (using `self.features`):
 * `E` is the matrix of edge features which are the RBF encoded distances of all of the atoms per coordinate frame node, also computes virtual C$\beta$ coordinates for these distances.
	 * Dim 1: Number of proteins in batch
	 * Dim 2: Number of AA of largest protein in batch
		 * This index corresponds to either the source source node and the distance encodings to its K nearest neighbors.
	 * Dim 3: K nearest neighbors
	 * Dim 4: Hidden Dimensionality of edges
		 * RBF distances and positional encodings$^ \star$ of relative distance in sequence and whether from same or different chain (see `PositionalEncodings` in `model_utils.py`)
* `E_idx` is the matrix holding the edge connectivity which is used for gathering the neighbor nodes of each index. Because of the padding a lot of the smaller proteins in the batch will be padded with some random index corresponding to the residue closest to (0,0,0).
	* Dim 1: Number of proteins in batch
	* Dim 2: Number of AA of largest protein in batch
	* Dim 3: K nearest neighbors
		* The indices correspond to the nearest nodes. The first index is the node itself, the remaining nodes are the nearest neighbors, by doing this, they have self-loops in their graphs. Because of the batching, these are filled with some random residue corresponding to the closest to (0,0,0). Multiply by mask to zero out the non-existent indices.

> *$^\star$Regarding the Utility of Positional Encoding Information:*
	"For relative positional encoding we used AlphaFold-like (10) discrete (one-hot encoded) tokens -32, -31,..., 31, 32 within the protein chains and additional token 33 if residues are in different chains. Ablating positional encodings showed almost the same performance" 
		- pMPNN Supplement

Presumably, the information about whether the residues are within the same chain or not would provide additional information to the model though.

2) Nodes are initialized to zeros in the specified dimensionality.
3) Edge features are mapped by a linear layer into their specified dimensionality.
4) Create a `mask_attend` matrix which is just the mask repeated along the KNN dimension.
5) Compute forward pass through ==Encoder Layer==.
	* See git repo for line-by-line analysis of how these layers work inside training Jupyter notebook.
6) Map sequence into node embedding space and concatenate it to the edge features.
	* Multiplies the index-encoding of the sequence (not one-hot encoding for some reason) into node-embedding dimensionality
7) Construct `h_EXV_encoder`:
	* This is a matrix with zeros-like the sequence embedding concatenated to the encoder node and edge embeddings.
	* This matrix will be used to make predictions when sequence information is unavailable.
9) Construct `order_mask_backward`:
	* This is a decoding matrix corresponding to which amino acid indices of the protein complex are visible for the current step of the auto-regressive decoder. It is an NxN matrix where N is the length of the longest protein and is constructed using the biased random shuffle of the indices of the residues where the bias ensures that chains that are visible in the mask (both chain mask and mask) are decoded first and masked indices are decoded second.
10) Update `mask_attend` for decoding:
	* Corresponds to which indices of E_idx are visible for auto-regressive decoding. A binary mask of the KNN of indices of E_idx corresponding to their value in each row of `order_mask_backward`
11) Create `mask_bw` (and `mask_fw` which is the exact inverse of `mask_bw`):
	* The forward mask will be used to select the masked elements from `h_EXV_encoder` and construct `h_EXV_encoder_fw`. This is created and ==remains fixed== while passing through the decoder layers.
	* The backward mask will be used to select unmasked elements (those for which sequence context is available) from `h_ESV` which is created with updated node embeddings on each iteration through the decoder layer.
12) Apply ==Decoder Layers==:
	* For each decoder layer use the same `h_EXV_encoder_fw = mask_fw * h_EXV_encoder` (I think this makes the layers learn the effect of sequence context rather than to continue message passing) 
		* Each iteration recomputes `h_ESV` which is the sequence embedding concatenated to the source node embedding from the encoder concatenated to the edge information. 
	* `h_ES` is fixed going into the decoding layers, and updated node embeddings are concatenated on each iteration to form `h_ESV`
	* For input to decoder blocks:
		* `h_V` from the encoder or previous decoder block.
		* `h_ESV = (h_ESV * mask) + h_EXV_encoder_fw` which has ( h_V || E || h_S or $\emptyset$ )
			* This is odd to me because the decoder block will be have the encoder node features passed in twice.
	* Decoder layers actually work in ==3x the hidden dimensionality== ($3 \times 128$) because of the extra node features concatenated to the edge features unlike in the encoder layer. 

#### Forward Training Pass Thoroughly Commented

All modified code for stepping through a forward pass is found here:
https://github.com/benf549/ProteinMPNN/blob/main/Analyze%20ProteinMPNN%20training.ipynb

```python
from model_utils import ProteinFeatures, PositionWiseFeedForward, gather_nodes, cat_neighbors_nodes
import torch.nn as nn
import torch.nn.functional as F

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        
        # Compute node update.
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        ## Duplicates the node embeddings up to K (nearest neighbors) for concatenation with edge embeddings.
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        ## Compute message passing through linear layers in self.num_hidden embedding space.
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        
        ## mask_V and mask_attend will not be None during training at least.
        ## Zeros out the updates to the indices being ignored, before the update is applied to the node embeddings.
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
            
        ## Compute the message by summing over the neighbor indices and dividing by a scale factor (should probably be the degree)
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        
        ## mask_V and mask_attend will not be None during training at least.
        ## Zeros out the nodes being ignored, after the update is applied to the node embeddings (removes any layer biases added to the zeroed nodes).
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Compute edge update.
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        ## Looks like encoder block does not perform a 'dense layer' update for edge embeddings.
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # The decoding layer works exactly like the encoding layer, but it does not perform an edge update. See line-by-line explanation from above.

        # This block computes the node update.
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1) 
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale 
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
            
        # Return updated node embeddings.
        return h_V

class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        
        # Prepare node and edge embeddings
        # Constructs edge features from X.
        # Computes virtual Cb atom position and encodes into RBF, adds additional edge features as needed.
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        
        # Node features are intialized to zeros in the same hidden dimensionality as edges (128 by default)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        # Linear transformation over the edges to move them into the hidden dimensionality
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        # Duplicate the mask out to K (nearest neighbors) so that it can be used to mask which nearest neighbors will be used.
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        # Perform the forward pass of the encoder layers with checkpointing.
        # Computes a node and edge update which is used as input to the next layer and builds up the node representation.
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        # Concatenates linear transformation of sequence (not one-hot encoded for some reason) to edge embeddings in place of node embeddings. 
        # Moves sequence from [B, N] -> [B, N, (node embedding) 128]
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        
        # (Justas comment) Build encoder embeddings [idk what this means]
        ## Concatenates zeros to edges in same shape as nodes would be concatenated to edges.
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        ## Concatenates node embeddings from encoder to zeros concatenated to edges.
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        ## select only visible chains and only the residues that actually exist (not the padded residues added during batching) 
        ## element-wise multiplication of boolean vectors performs logical AND.
        ## Update chain_M to include missing regions (Justas comment, this implies to me that mask also tracks gaps in structure/sequence)
        chain_M = chain_M*mask 
        
        # Apply auto-regressive masking
        ## Creates a biased random shuffle of the indices in range [0,N), 0-indices in chain_M will be at the beginning of the shuffle while 1-indices will be at the end.
        ## (Justas Comment) [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) 
        
        ## Creates a permutation matrix by constructing one-hot encodings for all of the indices in the biased random decoding order.
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        
        ## Creates a random matrix for each protein that encodes the decoding order as a mask. Each row in the mask for each protein complex
        ##   corresponds to which indices are visible for each step of the decoding process. Follows the 
        ##   For example:
        ###    [0, 1, 0, 0, 0]
        ###    [0, 0, 0, 0, 0]
        ###    [1, 1, 0, 0, 0]
        ###    [1, 1, 0, 1, 0]
        ###    [1, 1, 0, 1, 1] for a 5 amino acid protein complex.
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        
        # print(chain_M.shape)
        # print()
        # print(chain_M[0, :])
        # print(E_idx*chain_M.unsqueeze(-1).expand(-1, -1, 5)[0, :])
        
        ## This creates a mask for which neighbors are visible for each index in the random decoding order.
        ###  since each row in the order_mask_backward corresponds to a different index in the protein complex during auto-regressive decoding
        ###  each index has its K nearest neighbors stored in the E_idx matrix. 
        ###  This call of the gather function creates a mask to find visible neighbors for each connection in E_idx.
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        print(mask_attend.shape)
              
        ## Convert the mask from [B, N] -> [B, N, 1, 1] moves data to deepest index in a compatible dimensionality with mask_attend.
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        
        ## Construct a mask and an inverse mask. Masks[B, N, KNN k, 1]
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        # Add sequence where appropriate.
        ## Using the forward mask select the indices of the zeros concat to node embeddings concat to edges for fw matrix.
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            ## Construct a matrix using the node embeddings updated on each iteration merged concat to sequence embedding concat to edges.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            ## Use the inverse mask to select only the visible indices of the seq embedding concat to node embedding concat to the edges 
            ##   add forward mask so we have values for every index.
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            ## Compute node update with masked indices.
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        # Linear map of logits to output dimensionality, softmax to prepare logits for NLL loss and return.
        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
```
