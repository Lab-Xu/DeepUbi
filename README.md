# DeepUbi
A Deep Learning Framework for Prediction of Ubiquitination Sites in Proteins
## Requirements
* Python>=3.6
* Matlab2016a
* Tensorflow =1.6.0
## Introduction of four encoding method
### One-hot encoding
The conventional feature representation of amino acid composition used 20 binary bits to represent an amino acid. To deal with the problem of sliding windows spanning out of N-terminal or C-terminal, one additional bit is appended to indicate this situation.
### Informative Physicochemical Properties (IPCP)
In all PTM sites prediction, physicochemical properties are essential to extract the instinct information for a fragment or protein.
The value of main effect difference (MED) was used to estimate the individual effects of physicochemical properties and the property with the largest value of MED is the most effective in predicting ubiquitylation sites.
### Composition of K-space amino acid pairs(CKSAAP)
The CKSAAP encoding scheme reflects the information of amino acid pairs in small range within the peptides.
### Pseudo Amino Acid Composition (PseAAC)
Pseudo amino acid composition is a set of discrete serial correlation factors combined with traditional 20 amino acids component.
## Algorithm flow
 The flow as shown below:
 <img src="https://github.com/Sunmile/DeepUbi/blob/master/picture/Fig.1.png"> 
## DeepLearing Framework
We constructed a convolutional neural network (CNN) as below:
<img src="https://github.com/Sunmile/DeepUbi/blob/master/picture/Fig.2.png"> 
## Results
* First of all, on the One-Hot encoding data set, we used the 4-fold, 6-fold, 8-fold and 10-fold cross validations. The ROC curves is shown below:

  <img src="https://github.com/Sunmile/DeepUbi/blob/master/picture/Fig.3.png"> 
* The ROC curves of the second model which used different feature representations with 10-fold cross validations is also shown below:

  <img src="https://github.com/Sunmile/DeepUbi/blob/master/picture/Fig.4.png"> 
