import helper_functions_hmm as hf
import argparse
'''
This script outputs a csv file with infant id as columns 
and P(O) (probability of observing data) of each infant as the first row.
The script preprocesses the input test OTU table the same way as infant_microbiome_hmm_main.py
before calculating P(O). Preprocessing includes 
selecting samples based on PMA and dropping infants with less than 5 samples.

*Row selection based on variance is not conducted for preprocessing.
To calculate P(O) using this script, user must feed this script with test OTU table 
with specific set of taxa used to run DMM and HMM.   

This script takes in 4 command line parameters:
1. -f
    path to test OTU table data 
2. -o
    path that user wants to output the prediction dataframe as csv file 
3. -s
    number of states used to run DMM and HMM
4. -p
    PMA (post mentral age) start date.
    This should be the same as how it was set to run HMM.

example command line)
python3 output_p_of_o.py -f 'data/otu_table_sample.csv' -m './hmm_outputs/hmm_out_1' -o './prediction.csv' -s '6' -p '196' -t '0.0338'

'''
################# script argument #################
parser = argparse.ArgumentParser()
parser.add_argument('-f', required=True, help='filepath path to user test data (OTU table)')
parser.add_argument('-m', required=True, help='directory path to the hmm_out directory user wants to use to calculate P(O)')
parser.add_argument('-o', required=True, help='filepath to output prediction df')
parser.add_argument('-s', required=True, help='number of states (clusters) used for DMM and HMM')
parser.add_argument('-p', required=True, help='pma start date')

args = parser.parse_args()

csv_data_filepath = args.f
path_to_hmm_out = args.m
prediction_output_path = args.o
n_states = args.s
pma_start_date = args.p
####################################################

path = hf.make_unique_file_or_dir_names(prediction_output_path)

# convert numercial arguments(string) to int/float
n_states = int(n_states)
pma_start_date = int(pma_start_date)

data = hf.convert_otu_to_data_for_prediction(csv_data_filepath, pma_start_date)
prediction_df = hf.output_p_of_o_df(path_to_hmm_out, data, n_states)
prediction_df.T.to_csv(path, index=False, header=False)