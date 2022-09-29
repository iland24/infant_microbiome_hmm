# Background and purpose:
This project is a binary classifier model, whose purpose is to predict preborn infants (prematurely born babies) as growth normal (GN) or growth faltering (GF) using the infants' gut microbiome community data. Hidden Markov Model(HMM) is used to train on the microbiome data of GN and GF infant data, producing two HMM outputs. The two outputs will be used to calculate the probability of observations (P(O)) of given test data, labeling GN or GF based on higher P(O) value produced by the two models.  

HMM is suitable for sequential or time series data. In this project, Infant microbiome data was collected at multiple timepoints, specifically at certain post menstral age (PMA) of each infant. Because the microbiome abundance were collected from infant fecal matter, samples were collected at varying timepoints for each infant. Among all PMA timepoint samples, 10 timepoints were selected; however, many infants have missing timepoint samples. HMM allows learning a model even when there are missing values by taking into account all possible assignments of the hidden state considering their probability, making HMM a suitable model for handling clinical data with missing values. 

The hidden state for each timepoint sample is assigned using MicrobeDMM - Software for fitting Dirichlet multinomial mixtures to microbial communities. DMM clustering is a probabilistic method for community detection in microbial samples.

This project is designed so that the code runs in Unix-like systems. The writer used WSL (Windows subsystem for Linux). Scripts were coded using Python language (Python 3.7.2).

* User can run infant_microbiome_hmm.py to simply run HMM without dividing data into two sets: GN and GF.

* User can run infant_microbiome_two_hmm.py if the user has label data that classifies each infant either as GN or GF. This script will run DMM to assign states to all samples in the combined data, then divides the data into GN and GF set to run HMM separately on the two data sets. HMM outputs will be saved in different folders, saving GN HMM output first, then GF HMM output next.

* DMM outputs will be saved in dmm_outputs/dmm_out_#
* HMM outputs will be saved in hmm_outputs/hmm_out_#
    * If multiple rounds of code were ran, output of each round will be saved in a different directory with different indexing 
    
* sample data is given at "infant_gut_microbiome_hmm/data/otu_table_sample.csv"
* sample label is given at "infant_gut_microbiome_hmm/class_labels.tsv"

## Input data:
* Input data is Operational taxonomic unit (OTU) table csv file.
* OTU table in this project contains abundance of bacteria taxa (data type = float)
* The first column of OTU should be the "Taxa" column that contains name of bacterial species (data type = string)

&nbsp;

# Steps to use the code:
## Clone code:
> git clone https://github.com/iland24/infant_microbiome_hmm

> cd ./infant_gut_microbiome_hmm/

&nbsp; 

## Run initiate.py
* Downloads DMM code zip file
* Extracts DMM zip file
* Saves parameters.txt at current working directory (which shoulod be infant_gut_microbiome_hmm). parameters.txt file must be read either by infant_microbiome_hmm.py or infant_microbiome_two_hmm.py to run DMM and HMM

&emsp;**sample command:**
> python3 initiate.py

* After running initiate_hmm.py, user can type in the parameters that will be used in scripts that run DMM and HMM
    * **User must type in path to input data in the parameters.txt file for running the next script**
* Default parameter values are set for provided sample data

&nbsp;
    
## Run either single HMM output (a) or two HMM output (b)
## &emsp;(a) infant_microbiome_hmm.py (single HMM output script)
* Preprocesses data 
    1. Select rows (taxa) with highest variance (User can define fraction of the taxa with highest variance)
    1. Select 10 timepoint samples of each infant based on start PMA timepoint
    1. Drop infants with more than 5 missing timepoint samples
* Runs DMM code to assign states to each sample
* Combines state outputted by DMM code with infant gut microbiome data
* Runs HMM 

* **Input**:
    * parameters.txt
    
* **Output**: 
    * dmm_outputs/dmm_out_1
    * hmm_outputs/hmm_out_1 
        * HMM output from given dataset
        
&emsp;**sample command:**
> python3 infant_microbiome_hmm.py -p './parameters.txt'  

&nbsp;

## &emsp;(b) infant_microbiome_two_hmm.py (two HMM output script)
* Preprocesses data 
    * Select rows (taxa) with highest variance (User can define fraction of the taxa with highest variance)
    * Select 10 timepoint samples of each infant based on start PMA timepoint
    * Drop infants with more than 5 missing timepoint samples
* Runs DMM code to assign states to each sample
* Combines state outputted by DMM code with infant gut microbiome data
* Separates data into GN and GF sets based on given label data
* Runs HMM on GN data set
* Runs HMM on GF data set

* **Input**:
    * parameters.txt
    * class_labels.tsv
        * must be a tsv file, not csv

* **Output**: 
    * dmm_outputs/dmm_out_1
    * hmm_outputs/hmm_out_1 
        * HMM output from GN dataset
    * hmm_outputs/hmm_out_2
        * HMM output from GF dataset
        
&emsp;sample command:
> python3 infant_microbiome_hmm.py -p './parameters.txt' -l './class_labels.tsv'

&nbsp;

## output_p_of_o.py:
* Preprocesses test OTU_table.csv file same way input data was preprocessed in infant_microbiome_hmm.py / infant_microbiome_two_hmm.py. 
    * This script does not select rows based on variance. Specific rows (taxa) must be preselected by the user before feeding the test data to this script.
* Calculates P(O) for all the infants in the test data
* Saves ID and correlating P(O) value as a csv file
* **Input**:
    * file path to data
    * directory path to hmm_out
    * path to save output
    * number of states used to run DMM and HMM
    * PMA start date used to run DMM and HMM

* **Output**: 
    * csv file with infant id as columns and P(O) (probability of observing data) of each infant as the first row.
        * **output_p_of_o.py should be ran using GN and GF HMM outputs**
        * **Each infant id and P(O) should be compared to make prediction if the infant will be GN or GF**

&emsp;**sample command:**
> python3 output_p_of_o.py -f 'data/otu_table_sample.csv' -m './hmm_outputs/hmm_out_1' -o './prediction.csv' -s '6' -p '196'

&nbsp;

## Further Notes
* Additional notes are provided in the beginning of the python scripts.

* Caveats to using DMM code:
  * If problem occur in running DMM code, Error message will be printed out in the terminal: "Error occured while running DMM"
  * This is possibly due to missing library package for compiling DMM C code. If such is suspected, try installing package at the therminal by typing:
  * > sudo apt-get install libgsl0ldbl
    * visit https://askubuntu.com/questions/490465/install-gnu-scientific-library-gsl-on-ubuntu-14-04-via-terminal for more information.
  * When the dimension of the input data is too big, error occurs in the DMM code. So, my code is designed to check if the column number goes over 1800. DMM code doesn't run with sample data (shape=(444 x 2997)) included in infant_microbiome_hmm code. It helped to rename the column names to integers, shortening each column name. 
  * My code initially checks if the column number of the input data goes over 1800. If it does, it trims the columns based on the starting PMA date first before running DMM. If the column number doens't go over 1800, it runs DMM first before trimmming down columns. 
* Likelihood of the HMM is calculated using Viterbi path, not the full joint probability.
* Infant label file must be a tsv file not a csv file. 