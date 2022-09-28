import helper_functions_hmm as hf
import argparse

def main():
    ################# script argument #################
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='filepath to parameters.txt outputted by initiate_hmm.py')
    args = parser.parse_args()
    parameters_filepath = args.p
    
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
    
    hf.run_hmm(data, n_states, taxa_names, likelihood_threshold, n_taxa_result_to_be_displayed_in_terminal)

if __name__=="__main__":
    main()
