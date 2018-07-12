Binary_matrix.m - The conventional feature representation of amino acid composition uses 20 binary bits to represent an amino acid. To deal with the problem of sliding windows spanning out of N-terminal or C-terminal, one additional bit is appended to indicate this situation. Then a vector of size (20+1) bits is used for representing a sample. For example, the amino acid A is represented by '100000000000000000000' and R is represented by '010000000000000000000'.

AAindex_Ubi31_21.mat -  A set of 31 informative physicochemical properties proposed by Tung and Ho (2008).

AAindex.m - A feature extraction method using the 31 informative physicochemical properties.

CKspace.m - The CKSAAP encoding means the composition of k-spaced residue pairs in the protein sequence. For example, there are 441 residue pairs (i.e., AA, AC, ..., XX). Therefore, the feature vector can be defined as the frequency of the 441 residue pairs in the segment.

PseAAC1.m -  Chou's pseudo amino acid composition is a set of discrete serial correlation factors combined with traditional 20 amino acids component. In the study, we select 20 correlation factors and the weight of these factors is 0.05, then a 40-dimension vector is acquired.

text_cnn.py - A CNN module constructed to deal with the protein segments.

CNN_train.py -  A program for training parameters of the convolution neural network.
