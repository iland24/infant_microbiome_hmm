```
# Code designed to run in linux
python 3.7
Ubuntu 22.04.3 LTS
```

### Background / Purpose:
This project is a Hidden Markov Model (HMM) binary classifier that predicts preborn infants (prematurely born babies) as growth normal (GN) or growth faltering (GF) using the infants' gut microbiome community data collected over time. HMM is used to train on the microbiome data of GN and GF infant data separately, producing two HMM outputs. The two outputs are be used to calculate the probability of observations (P(O)) of test samples, classifiying the samples as GN or GF based on higher P(O) value produced by the two models.

HMM is suitable for modeling sequential or time series data, and, in this project, Infant microbiome data was collected at multiple timepoints, labled using post menstral age (PMA) of the infants. During the microbiome abundance sample collection step, the samples were collected at varying timepoints for each infant, so sampling of 10 PMA timepoints was aligned with all samples as much as possible. Each PMA timepoint is 7 days apart. Due to this reason, some samples had missing timepoint data. However, HMM allows learning a model even when there are missing values by taking into account all possible assignments of the hidden state, considering their probability, making HMM a suitable model for handling clinical data with missing values. 

The hidden state for each timepoint sample is assigned using MicrobeDMM - Software for fitting Dirichlet multinomial mixtures (DMM) to microbial communities. DMM clustering is a probabilistic method for community detection in microbial samples.

* User can run infant_microbiome_hmm.py to simply run HMM without dividing data into two sets based on given labels.

* User can run infant_microbiome_two_hmm.py if the user has label data that classifies each infant either as GN or GF. This script runs DMM to assign states to all samples in the combined data, then divides the data into GN and GF set to run HMM separately on the two data sets. HMM outputs are saved in separate folders, saving GN HMM output first, then GF HMM output next.

* DMM outputs will be saved in dmm_outputs/dmm_out
* HMM outputs will be saved in hmm_outputs/hmm_out
    * If multiple rounds of code were ran, output of each round will be saved in a different directory with different indexing 
    
* Sample microbiome abundance data is given at "infant_microbiome_hmm/data/otu_table_sample.csv"
* Sample label is given at "infant_microbiome_hmm/data/class_labels.tsv"
* No test data is provided.

### DMM Dependencies
```
# install gcc
sudo apt update
sudo apt install build-essential

# install & unzip GSL
wget https://ftp.kaist.ac.kr/gnu/gsl/gsl-1.15.tar.gz
cd gsl-1.15

# build library
make
sudo make install

# Add below to .bashrc file in home directory
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export CFLAGS="-I/usr/local/include"
export LDFLAGS="-L/usr/local/lib"
```

#### Output File System
```
├── MicrobeDMMv1.0.tar.gz
├── MicrobeDMMv1.0
├── data
│   ├── class_labels.tsv
│   └── otu_table_sample.csv
├── dmm_img_outputs
│   └── dmm_clusters.png
├── dmm_outputs
│   └── dmm_out
│       ├── stub.err
│       ├── stub.fit
│       ├── stub.mixture
│       ├── stub.z
│       └── stubk.z
├── hmm_outputs
│   └── hmm_out
│       ├── emission_matrix_state_1.txt
│       ├── emission_matrix_state_2.txt
│       ├── emission_matrix_state_3.txt
│       ├── emission_matrix_state_4.txt
│       ├── emission_matrix_state_5.txt
│       ├── emission_matrix_state_6.txt
│       ├── emission_matrix_state.npy
│       ├── hmm_log_likelihood.png
│       ├── initial_matrix.txt
│       ├── initial_matrix.npy
│       ├── top_x_variance_taxa_per_state.txt
│       ├── transition_matrix.txt
│       └── transition_matrix.npy
├── helper.py
├── infant_microbiome_hmm.py
├── infant_microbiome_two_hmm.py
├── initiate_hmm.py
├── output_p_of_o.py
├── parameters.txt
└── readme.md
```

### Input data:
* Input data is Operational taxonomic unit (OTU) table csv file.
* OTU table in this project contains abundance of bacteria taxa (data type = float)
* The first column of OTU should be the "Taxa" column that contains name of bacterial species (data type = string)

&nbsp;

### Steps to use the code:
##### **Clone code**:
> git clone https://github.com/iland24/infant_microbiome_hmm
> cd ./infant_microbiome_hmm/

&nbsp; 

### Run initiate_hmm.py
* Downloads DMM code zip file
* Extracts DMM zip file
* Saves parameters.txt at current working directory (which shoulod be infant_microbiome_hmm). parameters.txt file must be read either by infant_microbiome_hmm.py or infant_microbiome_two_hmm.py to run DMM and HMM

&emsp;**sample command:**
> python3 initiate_hmm.py

* After running initiate_hmm.py, user can type in the parameters that will be used in scripts that run DMM and HMM
    * **User must type in path to input data in the parameters.txt file for running the next script**
* Default parameter values are set for provided sample data

&nbsp;
    
### Run single HMM (a) or two HMM (b)
Run single HMM script to train HMM using microbiome abundance timeseries train data in the infant_microbiome_hmm/data directory.

Run two HMM script if there is label file in the infant_microbiome_hmm/data directory along with the training data. Two HMM script will outputs a pair of HMM outputs. Each of the output can be used to calculate the P(O) of test data using the next script.

* The format of the training data and the label file must match the given sample to run these scripts.

#### (a) infant_microbiome_hmm.py (single HMM output script)
-
  Preprocesses data 
      1. Select rows (taxa) with highest variance (User can define fraction of the taxa with highest variance)
      2. Select 10 timepoint samples of each infant based on start PMA timepoint
      3. Drop infants with more than 5 missing timepoint samples
  * Runs DMM code to assign states to each sample
  * Combines state outputted by DMM code with infant microbiome abundance timeseries train data
  * Runs HMM 

* **Input**:
    * parameters.txt
    
* **Output**: 
    * dmm_outputs/dmm_out
    * hmm_outputs/hmm_out
        * HMM output from given dataset
        
&emsp;**sample command:**
> python3 infant_microbiome_hmm.py -p ./parameters.txt 

&nbsp;

#### (b) infant_microbiome_two_hmm.py (two HMM output script)
-
  * Preprocesses data 
      1. Select rows (taxa) with highest variance (User can define fraction of the taxa with highest variance)
      2. Select 10 timepoint samples of each infant based on start PMA timepoint
      3. Drop infants with more than 5 missing timepoint samples
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
    * dmm_outputs/dmm_out
    * hmm_outputs/hmm_out_1
        * HMM output from GN dataset
    * hmm_outputs/hmm_out_2
        * HMM output from GF dataset
        
&emsp;**sample command**:
> python3 infant_microbiome_two_hmm.py -p ./parameters.txt -l ./data/class_labels.tsv

&nbsp;

### output_p_of_o.py:
* Preprocesses test OTU_table.csv file same way input data was preprocessed in infant_microbiome_hmm.py / infant_microbiome_two_hmm.py. 
    * This script does not select rows based on variance. Specific rows (taxa) must be preselected by the user before feeding the test data to this script.
* Calculates P(O) for all the infants in the test data
* Saves ID and correlating P(O) value as a csv file
* **Input**:
    * file path to microbiome abundance timeseries test data
    * directory path to hmm_out
    * path to save output
    * number of states used to run DMM and HMM
    * PMA start date used to run DMM and HMM

* **Output**: 
    * csv file with infant id as columns and P(O) (probability of observing data) of each infant as the first row.
        * **output_p_of_o.py should be ran using GN and GF HMM outputs**
        * **Each infant id and P(O) should be compared to make prediction if the infant will be GN or GF**

&emsp;**sample command:**
> python3 output_p_of_o.py -f data/otu_table_test_data.csv -m ./hmm_outputs/hmm_out_1 -o ./prediction.csv -s 6 -p 196

&nbsp;

### Notes
* Additional notes are provided in the beginning of the python scripts.
* Caveats to using DMM code:
  * If problem occur in running DMM code, Error message will be printed out in the terminal: "Error occured while running DMM"
  * This is possibly due to missing library package for compiling DMM C code. If such is suspected, try installing package at the therminal by typing:
  * > sudo apt-get install libgsl0ldbl
    * visit https://askubuntu.com/questions/490465/install-gnu-scientific-library-gsl-on-ubuntu-14-04-via-terminal for more information.
  * When the dimension of the input data is too big, error occurs in the DMM code. So, my code is designed to check if the column number goes over 1800. DMM code doesn't run with sample data (shape=(444 x 2997)) included in infant_microbiome_hmm code. It helped to rename the column names to integers, shortening each column name. 
  * My code initially checks if the column number of the input data goes over 1800. If it does, it trims the columns based on the starting PMA date first before running DMM. If the column number doens't go over 1800, it runs DMM first before trimmming down columns. 
* The preset parameters in the parameters.txt file genereated from initiate_hmm.py file are values determined based on the writer's code runs.
* Likelihood of the HMM is calculated using Viterbi path, not the full joint probability.
* Infant label file must be a tsv file not a csv file. 