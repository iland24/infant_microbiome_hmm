import helper_functions_hmm as hf
import argparse
'''
This script can be run the same way as infant_microbiome_hmm_main.py except that 
it takes in one additional commandline argument (-l) which is a path to .tsv file
that contains true label to each infant.
    -l
    - label tsv file:
        column1: Astarte ID (infant ID)
        column2: gf (1 = GF, 0 = GN)

This script outputs two hmm outputs; one from Growth Normal (GN) data 
and the other from Growth Faltering (GF) data.

Two outputs will be saved under hmm_outputs directory.
'''
def main():
    ################# script argument #################
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='filepath to parameters.txt outputted by initiate_hmm.py')
    parser.add_argument('-l', required=True, help='filepath to class_labels.tsv')
    
    args = parser.parse_args()
    parameters_filepath = args.p
    label_filepath = args.l
    
    ###### read in parameters from parameter.txt #######
    csv_data_filepath, \
    fraction_of_taxa_selection,\
    pma_start_date, \
    n_states, \
    likelihood_threshold, \
    n_taxa_result_to_be_displayed_in_terminal = hf.read_parameters(parameters_filepath)

    ####################################################

    print()
    print("Data File:", csv_data_filepath)
    print()

    otu_df = hf.read_data_as_df(csv_data_filepath)
    otu_df = hf.convert_data_to_int(otu_df)
    otu_df, taxa_names = hf.choose_top_x_percent_of_taxa_with_highest_variance(otu_df,frac = fraction_of_taxa_selection)

    cluster_assignments_df, col_trimmed_train_otu_df = hf.check_column_numbers_and_run_DMM(otu_df,\
                                                                                            csv_data_filepath,\
                                                                                            n_states,\
                                                                                            pma_start_date)

    data = hf.combine_cluster_and_otu_table(cluster_assignments_df, col_trimmed_train_otu_df)
    
    # Divide data based on given label
    gn_df, gf_df = hf.divide_based_on_true_labels(otu_df, taxa_names, label_filepath)
    gn_data, gf_data = hf.convert_gn_gf_df_to_data_and_add_states_from_dmm(gn_df, gf_df, data)  

    # run hmm for gn and gf separately 
    hf.run_hmm(gn_data, n_states, taxa_names, likelihood_threshold, n_taxa_result_to_be_displayed_in_terminal)
    hf.run_hmm(gf_data, n_states, taxa_names, likelihood_threshold, n_taxa_result_to_be_displayed_in_terminal)
    
if __name__=="__main__":
    main()
