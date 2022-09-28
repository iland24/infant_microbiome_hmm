import subprocess
import os.path
import random
import copy
import csv
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# Abbreviations / Terms used in infant microbiome hmm project
# DF (df): DataFrame, SR (sr): Series

# OTU 
#   -Operational taxonomic unit
#   -OTU table shows abundance of bacteria taxa
#   -OTU DF's first column should be "Taxa" (data type = obj / non-numeric)

# infant 
#   -infant is identified through ID number alone or 
#    ID and site where fecal samples were gathered
#   -multiple samples were collected from an infant 
#    and 10 samples are chosen for 10 timepoint data

# PMA
#   -post mentral age of infant
#   -PMA measures gestational age and post-natal age of infants
#   -PMA is used to choose the 10 timepoints

# sample
#   -a set of microbiome taxa abundance is included in a sample

# data
#   -word 'data' in this project is a list of lists (list of infants)
#   -Each infant in initial data outputted from combine_cluster_and_otu_table() contains the following:
#    [0] = abundance data (df), [1] = states at 10 timepoints (df)
#   -Missing sample is represented as a column of Nan values in each df. 

#   -Each infant in data after e_step():
#    [0] = abundance data (df), [1] = states at 10 timepoints (df), 
#    [2] = alpha matrix (np.array), [3] = beta matrix (np.array), 
#    [4] = s_t(i) matrix (np.array), [5] = s_t(i,j) matrix (np.array)

#   -Each infant in data after m_step() and find_viterbi_path()
#    [0] = abundance data, [1] = states at 10 timepoints, 
#    [2] =alpha matrix (np.array), [3] = beta matrix (np.array), 
#    [4] = s_t(i) matrix (np.array), [5] = s_t(i,j) matrix (np.array), 
#    [6] = viterbi timepoint path of states (list)

######### Function for reading / writing data #########

def read_data_as_df(file_name):
    '''
    Returns OTU DF and a Series 
    of bacteria taxa name.
    
    Input: OTU csv file
    
    Renames column of taxa name to "Taxa"
    
    '''
    otu_df = pd.read_csv(file_name, engine='python')
    if 'ID' in list(otu_df):
        otu_df.rename(columns={"ID":"Taxa"},inplace=True)

    return otu_df

def read_parameters(path):
    '''
    Returns parameters written in 
    parameter.txt file.

    Input: path to parameter.txt file
    outputted by initiate_hmm.py
    '''
    parameter_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line)==0:
                continue
            x = re.search('(?<=\=).*',line[0]).group()
            dtype = re.match('\([a-zA-Z]+\)',line[0]).group()
            dtype = dtype[1:-1]
            if dtype == 'int':
                parameter_list.append(int(x.strip()))
            elif dtype == 'float':
                 parameter_list.append(float(x.strip()))
            elif dtype == 'string':
                 parameter_list.append(x.strip())
    return parameter_list

def make_unique_file_or_dir_names(path):
    '''
    Returns new file/directory name.

    Using the path parameter, checks if the 
    given file/directory exists in the path,
    and creates a new name for the file/directory.

    *designed so that user puts in same file 
    name multiple times and this function creates
    filename with increasing index number at the end.
    ex)
        path = dir/file_name.csv
        run this funciton three times:
        dir/file_name_1.csv
        dir/file_name_2.csv
        dir/file_name_3.csv
        
    '''
    filename, extension = os.path.splitext(path)
    
    first_path = filename+'_1'+extension
    if os.path.exists(first_path) == False:
        return first_path
    else:
        counter = 2
        while os.path.exists(first_path):
            first_path = filename + "_" + str(counter) + extension
            counter += 1
        return first_path

def df_to_csv_for_DMM(otu_df, filepath):
    '''
    Saves the DF as csv file @
    "MicrobeDMMv1.0/Data/"+filename
    Returns the filepath to the outputted csv file.
    If csv file existed before at "MicrobeDMMv1.0/Data," 
    filepath with new file name will be returned. 

    Input: 
    1. OTU DF
    2. file path

    If otu_table.csv already exists, 
    this function saves the file in
    a different name.
    '''
    filename = re.search('[a-zA-Z_]+.csv',filepath).group()
    path = make_unique_file_or_dir_names("MicrobeDMMv1.0/Data/"+filename)

    column_names = list(otu_df)
    taxa = otu_df["Taxa"]
    otu_df.drop(columns=["Taxa"],inplace=True)
    col_names = [i for i in range(len(otu_df.columns))]

    otu_df.columns = col_names
    otu_df.insert(0,"Taxa",taxa)
    otu_df.to_csv(path, index=False)
    otu_df.columns = column_names
    print("File saved as:"+ path)
    print()
    return path

######### Functions for cleaning and priming data for DMM and HMM #########

def convert_data_to_int(otu_df):
    '''
    Returns OTU DF with taxa abundance 
    values as int type.
    
    Input: OTU DF
    
    Sorts sample ID (column names) lexicographically.
    Multiplies data by a 1e6 & adds 0.01 
    to all data to rid of zeros.
    
    '''
    if "Taxa" in list(otu_df):
        taxa = otu_df["Taxa"]
        numeric_df = otu_df.drop(columns=['Taxa'])
    else:
        numeric_df = list(otu_df)
    
    numeric_df.sort_index(axis=1,inplace=True)
    numeric_df = numeric_df.multiply(1e6).round().astype(int)
    numeric_df = numeric_df.add(1)
    numeric_df.insert(0,"Taxa",taxa)
    
    return numeric_df

def calc_variance(otu_df):
    '''
    Takes bacteria taxa x samples abundance DF as a parameter.
    Calculate variance of each row.
    returns a list of tuples (idx, variances).
    '''
    if "Taxa" in list(otu_df):
        numeric_df = otu_df.drop(columns="Taxa")
    else:
        numeric_df = otu_df
        
    n = len(numeric_df.columns)
    mean = np.array([])
    for row in numeric_df.index:
        mean = np.append(mean, np.mean(numeric_df.loc[row,:]))
    numeric_df["mean"] = mean
    
    var = []
    for idx,row in numeric_df.iterrows():
        deviations_sum = ((numeric_df.loc[idx,:] - mean[idx])**2).sum()
        var.append((idx,(deviations_sum / n)))
    var = sorted(var,key=lambda x:x[1])
    
    return var

def choose_top_x_percent_of_taxa_with_highest_variance(otu_df, frac=0.0338):
    '''
    Returns otu_df with selected taxa (row).
    Taxa is sorted from high to low variation 
    in data and taxa with only high variance 
    can be selected using frac parameter.
    
    Input: 
    1.otu_df
    2. fraction of taxa to be dropped
    
    '''
    print('Number of taxa (rows) before selection:', len(otu_df))
    
    variance = calc_variance(otu_df)
    
    new_taxa_len = int(len(otu_df)*frac)
    len_of_taxa_to_be_dropped = len(otu_df) - new_taxa_len
    
    indices_to_be_dropped=[]
    for var in variance[:len_of_taxa_to_be_dropped]:
        indices_to_be_dropped.append(var[0])
    
    otu_df.drop(index=indices_to_be_dropped,inplace=True)
    
    taxa_name = otu_df["Taxa"]
    
    print('Number of taxa (rows) after selection:', len(otu_df))
    print()
    
    return otu_df, taxa_name

def drop_infants_with_less_than_5_samples(otu_df):
    '''
    Returns OTU DF after dropping 
    infants with less than 5 samples.
    
    Input: OTU DF
    '''
    
    column_names = list(otu_df)[1:]

    # extract tuple of (id + site, column name) as list of sets
    infants = []
    for col_name in column_names:
        #search for id+site (1001_A)
        infants.append((re.search("[0-9]+[_\-.,/\s][A-Za-z]",col_name).group(), col_name)) 
    infants = sorted(infants, key = lambda x: x[0] )

    #drop ID with less than 5 samples
    cnt = 0
    tmp_id = []
    for i,infant in enumerate(infants):    
        cnt = cnt+1
        tmp_id.append(infant[1])
        curr_id = infant[0]
        if i != 0:
            prev_id = infants[i-1][0]
            if prev_id != curr_id:
                last_element = tmp_id[len(tmp_id)-1]
                if cnt <= 4: 
                    otu_df.drop(columns = tmp_id[:len(tmp_id)-1], inplace = True)
                tmp_id.clear()
                tmp_id.append(last_element)
                cnt = 0
    
    return otu_df



######### Functions for data (list of infants) selection #########

def convert_otu_df_to_data(otu_df, include_nan=True):
    '''
    Returns list of lists (= list of infants). 
    Each infant contrains one DF: abundance OTU DF

    If include_nan is True, columns containing nan values will be included.

    Input: OTU DF (int)
    
    Column name example: 1000_A_1 (ID_Site_timepoint)
    SampleNum: positive integer
    Site: single letter
    '''
    if "Taxa" in list(otu_df):
        samples = list(otu_df.drop(columns=['Taxa']))
    else:
        samples = list(otu_df)

    infants = []
    
    # find unique ID (sampleID_Site_)
    for sample in samples:
        infants.append(re.search("[0-9]+[_\-.,/\s][A-Za-z]_",sample).group())
    
    unique_infants = sorted(list(set(infants)))
    
    # create empty dfs:
    if include_nan == False:
        tmp_data = []
        for _ in unique_infants:
            tmp_data.append([pd.DataFrame()])  
    else:
        tmp_data = []
        for infant in unique_infants:
            tmp_data.append([ pd.DataFrame(columns = [infant+str(x+1) for x in range(10)]) ]) 

    # Store abundance data.
    index = 0
    for i,sample in enumerate(samples):
        identifier = re.search("[0-9]+[_\-.,/\s][A-Za-z]_",sample).group()
        if identifier == unique_infants[index]:
            tmp_data[index][0][sample] = otu_df[sample] #add columns
        elif i==len(samples)-1:
            continue
        else: 
            tmp_data[index+1][0][sample] = otu_df[sample] #add columns
            index+=1
    return tmp_data

def convert_list_of_infants_to_otu_df(data, taxa_names):
    '''
    When converting list of infants to otu_df,
    excludes nan value.
    '''
    container = []
    for infant in data:
        container.append(infant[0])
    otu_df = pd.concat([pd.Series(taxa_names,name="Taxa")])
    container.insert(0,otu_df)
    otu_df = pd.concat(container, axis=1)
    
    return otu_df

######### DMM functions #########

def run_dmm(csv_file_path, out, n_clusters = 6):
    '''
    Runs DMM in MicrobeDMMv1.0 directory.
    DMM c code saves output as stub.z 
    in dmm_out folder.
    
    Input: 
    1. OTU Table csv file path
    2. n_clusters (or n_states or hidden_label): 
    number of clusters. DMM  will 
    assign a cluster label to each sample.
    
    *n_clusters (n_states) value begins from 1, not 0.
    *DMM code returns label of states that begins from 1.
    '''
    if os.path.exists('MicrobeDMMv1.0/DirichletMixtureGHPFit') == False:
        subprocess.run(['make'], shell=True, cwd="MicrobeDMMv1.0")

    process = subprocess.Popen(['MicrobeDMMv1.0/DirichletMixtureGHPFit', 
                    '-in', csv_file_path, 
                    '-out', out, 
                    '-k', str(n_clusters),
                    '-l', str(random.randint(1,7000)) ])
    if process.wait() == 0:
        print("Successfully ran DMM.")
        print()
    else:
        print("Error occured while running DMM")
        print()

def read_dmm_output(dmm_output_path, otu_df):
    '''
    Returns cluster DF with 2 columns: sample & cluster label.
    Retrieves DMM output from stub.z file 
    @ MicrobeDMMv1.0/Data/otu_table.csv
    
    Inputs: 
    1. file path to stub.z 
    (DMM cluster assignment output)
    2. OTU DF
    
    '''
    dmm_out = []
    with open(dmm_output_path,"r") as f:
        reader = csv.reader(f)
        _ = next(reader) #header

        for line in reader:
            tmp=[]
            for i, value in enumerate(line):
                if i!=0:
                    tmp.append(float(value))
                else:
                    tmp.append(value)
            dmm_out.append(tmp)
    
    #select the cluster label with the highest probability
    cluster_assignment=[]

    for row in dmm_out:
        max = -1.0
        idx = -1
        for j,v2 in enumerate(row):
            if j!=0 and v2 > max:
                max = v2
                idx = j
        cluster_assignment.append((row[0],idx))
        
    #output as pd.dataframe
    cluster_df=pd.DataFrame(cluster_assignment)
    cluster_df.columns=["sample","cluster"]

    #rename 'sample' column values as the sample (column) names from OTU DF
    cluster_df['sample'] = list(otu_df)[1:]

    return cluster_df

def check_if_all_states_are_included_in_dmm_output(cluster_assignments_df, n_clusters = 6):
    '''
    Returns 0 if the number if number of n_cluster 
    designated in run_dmm() by the user matches the 
    output of DMM code.
    
    If the numbers match, returns 1.
    
    Input: 
    1. Output from read_dmm_output
    2. Number of states
    
    '''
    rerun_dmm = False
    
    cluster_sr = cluster_assignments_df['cluster'].value_counts()
    if len(cluster_sr) != n_clusters:
        rerun_dmm = True
    if rerun_dmm == True:
        print('Number of DMM output states did not match the user input.')
        print('Rerunning DMM')
        print()
        return 0 # sth went wrong
    else:
        print('Number of DMM output states matches the user input.')
        print()
        return 1 # correct

def dmm(otu_df, csv_file_path, dmm_output_name, path_to_dmm_output_filename, n_clusters = 6,):
    '''
    Returns cluster assignments of all 
    the samples (columns of otu_df) as df.
    Column 1: sample names
    Column 2: cluster assignments

    *This fuction is made to checks if the output df 
    contains the same number of unique clusters
    users designated. Often, DMM output doesn't
    contain all clusters(states).
    '''

    final_ok = False
    while final_ok == False: 
        run_dmm(csv_file_path=csv_file_path, out= dmm_output_name, n_clusters=n_clusters)
        cluster_assignments_df = read_dmm_output(path_to_dmm_output_filename, otu_df)

        ################# outputs pie chart of dmm output ##################
        cluster_assignments_df["cluster"].value_counts().plot(kind="pie",figsize=(6,6))
        if os.path.exists('dmm_img_outputs') == False:
            os.mkdir('dmm_img_outputs')

        path = 'dmm_img_outputs/dmm_clusters.png'
        new_path = make_unique_file_or_dir_names(path)
        plt.savefig(new_path)
        #####################################################################

        final_ok = check_if_all_states_are_included_in_dmm_output(cluster_assignments_df, n_clusters=n_clusters)

    return cluster_assignments_df

def check_column_numbers_for_DMM(otu_df):
    '''
    Returns 1 if column numbers of otu_df 
    is smaller or equal than 1800.
    Else, returns 0.
    '''
    if len(list(otu_df)) <= 1800:
        return 1 #True
    else:
        return 0 #False

def transpose_cluster_assignment_df(cluster_assignments_df):
    columns_sr = list(cluster_assignments_df['sample'])
    states_sr = cluster_assignments_df['cluster']
    transposed_cluster_df = pd.DataFrame(columns=columns_sr,index=[0])
    transposed_cluster_df.iloc[0]=states_sr
    return transposed_cluster_df

def transpose_cluster_df_back(c_df):
    sample_names = list(c_df)
    states_sr = list(c_df.iloc[0])
    transposed_cluster_df = pd.DataFrame({'sample':sample_names,'cluster':states_sr})
    return transposed_cluster_df

def check_column_numbers_and_run_DMM(otu_df, csv_data_filepath, n_states, pma_start_date):
    '''
    dmm_output_as: ...dmm_out/stub
        *used as a paramter for DMM code for dmm output name
    dmm_output_path: ...dmm_out/stub.z 
        *used to read the output to get cluster assignment
    '''
    # make dmm_outputs directory. 
    # Everythime dmm is run, dmm_out_# directories will be saved here
    if os.path.exists('dmm_outputs') == False:
        os.mkdir('dmm_outputs')

    path_to_dmm_out_dir = make_unique_file_or_dir_names('dmm_outputs/dmm_out')
    os.mkdir(path_to_dmm_out_dir)

    #if column # <= 1800 run dmm here
    if check_column_numbers_for_DMM(otu_df) == True:
        print('otu_table columns were not trimmed to run DMM')
        print()
        # gf_or_gn = re.search('/[a-zA-Z]+', data_filepath).group()

        #to run DMM, must save modified otu table df in 'MicrobeDMMv1.0/Data'
        path_to_dmm_otu_csv = df_to_csv_for_DMM(otu_df, csv_data_filepath) 

        #Run DMM before trimming
        cluster_assignments_df = dmm(otu_df = otu_df,\
                                csv_file_path = path_to_dmm_otu_csv,\
                                dmm_output_name = path_to_dmm_out_dir+'/stub',\
                                path_to_dmm_output_filename = path_to_dmm_out_dir+'/stub.z',\
                                n_clusters = n_states)
        
        #trim columns of cluster df (because dmm was ran with all samples)
        transposed_cluster_df =transpose_cluster_assignment_df(cluster_assignments_df)
        trimmed_cluster_df = drop_infants_with_less_than_5_samples(transposed_cluster_df)
        timepoints = select_10_pma(pma_start_date=pma_start_date) 
        selected_columns = sample_based_on_timepoints(trimmed_cluster_df, timepoints)
        trimmed_cluster_df = drop_samples_not_part_of_timepoints(selected_columns, trimmed_cluster_df)
        cluster_assignments_df = transpose_cluster_df_back(trimmed_cluster_df)

        # trim columns (samples)
        otu_df = drop_infants_with_less_than_5_samples(otu_df)
        timepoints = select_10_pma(pma_start_date=pma_start_date) 
        selected_columns = sample_based_on_timepoints(otu_df, timepoints)
        otu_df = drop_samples_not_part_of_timepoints(selected_columns, otu_df)
        
        return cluster_assignments_df, otu_df
    else:
        print('There were too many columns in the otu_table. Columns will be trimmed to run DMM')
        print()
        # trim columns (samples) 1
        otu_df = drop_infants_with_less_than_5_samples(otu_df)
        timepoints = select_10_pma(pma_start_date=pma_start_date) 
        selected_columns = sample_based_on_timepoints(otu_df, timepoints)
        # trim columns (samples) 2
        otu_df = drop_samples_not_part_of_timepoints(selected_columns, otu_df)

        #to run DMM, must save modified otu table df in 'MicrobeDMMv1.0/Data'
        path_to_dmm_otu_csv = df_to_csv_for_DMM(otu_df, csv_data_filepath)

        # no need to trim columns because DMM was ran after trimming
        cluster_assignments_df = dmm(otu_df = otu_df,\
                                    csv_file_path = path_to_dmm_otu_csv,\
                                    dmm_output_name = path_to_dmm_out_dir+'/stub',\
                                    path_to_dmm_output_filename = path_to_dmm_out_dir+'/stub.z',\
                                    n_clusters = n_states)
                                    
        return cluster_assignments_df, otu_df

######### Functions handling data (list of infants (infant = list of abundance df and state label df) #########
def select_10_pma(pma_start_date=196):
    '''
    Returns a list of 10 PMA dates that are 7 days 
    apart from the start date.
    
    Takes in start PMA date (int) as a parameter.
    
    *Estimate range of PMA dates for all infants: 0 - 273.
    
    Default input is 196 because most data were 
    able to be selected starting from 196 PMA.

    Default output range: 196 - 260
    '''
    end = pma_start_date + 63
    return list(range(pma_start_date,end+1,7))


def compare_pma_and_select_data(pma_and_id, timepoints):
    '''
    Returns list of samples that are selected
    and those that are not selected. 
    All samples belong to a single infant.
    
    Input: 
    1. [PMA, sample ID] of all samples that belongs to one infant in a list
    2. list of 10 PMA time points (output from select_10_pma())
    
    This function selects samples that are less than
    3 days away from each time point. 
    
    Selected samples contain list of 3 elements: 
    [PMA, sample ID, ID_Site_timepoint]
    
    Samples that weren't selected contain 2 elements.
    '''
    idx = 0 
    for j,pma in enumerate(pma_and_id):
        if timepoints[idx]-pma[0] < -3: 
            while idx < 10 and timepoints[idx]-pma[0] < -3:
                idx+=1
            if idx == 10:
                break
        if timepoints[idx]-pma[0] > 3:
            continue  
        elif 3 >= abs(timepoints[idx]-pma[0]):
            if j > 0:
                prev_d = timepoints[idx]-pma_and_id[j-1][0]
                curr_d = timepoints[idx]-pma[0]
                if 3 >= abs(prev_d) >= abs(curr_d):
                    pma_and_id[j-1].pop()
                if abs(curr_d) > abs(prev_d):
                    continue
                pma.append(re.search("[0-9]+[_\-.,/\s][A-Za-z]_",pma[1]).group() + str(idx+1))
            else:
                pma.append(re.search("[0-9]+[_\-.,/\s][A-Za-z]_",pma[1]).group() + str(idx+1))
                
    tmp = [i for i in pma_and_id]
    pma_and_id.clear()
    
    return tmp

def sample_based_on_timepoints(df, timepoints):
    '''
    Returns a list of infants. Samples with 
    PMA date that falls within 3 days of 
    selected timepoints (output from select_10_pma()) 
    have len of 3. Samples that are not selected 
    have len of 2.
    
    Input: 
    1.OTU df
    2.list of 10 time points
    
    Selected samples have lenth of 3:
    [227, '1002_B_227', '1002_B_227_5'] 
    (pma, id_site_pma, id_site_pma_timepoint_label)
    
    Samples that's not selected have lenth of 2:
    [227, '1002_B_227']
    '''
    pma_and_id = []
    selected_columns = []
    
    if "Taxa" in list(df):
        columns = list(df.drop(columns=['Taxa']))
    else:
        columns = list(df)
    
    for i,col in enumerate(columns):
        curr_id = re.search("[0-9]+[_\-.,/\s][A-Za-z]",col).group()
        pma_and_id.append([int(re.search("[0-9]+$",col).group()),col])
        
        if i != 0 & i !=len(columns)-1:
            prev_id = re.search("[0-9]+[_\-.,/\s][A-Za-z]",columns[i-1]).group()
            if prev_id != curr_id:
                last_element = pma_and_id[len(pma_and_id)-1]
                pma_and_id.pop()
                out = compare_pma_and_select_data(pma_and_id, timepoints)
                selected_columns += out
                pma_and_id.append(last_element)
        if i==len(columns)-1:
            out = compare_pma_and_select_data(pma_and_id, timepoints)
            selected_columns += out
            
    return selected_columns

def drop_samples_not_part_of_timepoints(selected_columns, df):
    '''
    Returns OTU DF
    
    Input: 
    1. list of selected samples (columns)
    2. OTU DF 
    
    Drops columns(samples) that are selected.
    Replace PMA in all column names with timepoints (1-10).
    '''
    sth_went_wrong = False
    for col in selected_columns:
        if len(col) == 2:
            df.drop(columns=col[1],inplace=True)
        elif len(col) == 3:
            df.rename(columns={col[1]:col[2]},inplace=True)
        else:
            sth_went_wrong == True
    if sth_went_wrong == True:
        print("Wrong_format: please check if all column names are formatted as so: \
            \"id(integer)\" + \"site(A-Z)\" + \"Post-mentral age or PMA(integer)\", \
            each string joined by underscore or space.")
    return df

def drop_infants_with_less_than_5_after_selecting_infants(data):
    '''
    Returns data (list of infants) with
    infants with more than four samples.
    
    Input: data (output from combine_cluster_and_abund_data())
    
    This function is applied after samples
    for each timepoint has been selected and DMM.
    '''
    new_data = []
    for infant in data:
        if len(list(infant[0].dropna(axis=1)))>=5:
            new_data.append(infant)
    return new_data

def combine_cluster_and_otu_table(df_cluster, df_abund):
    '''
    Returns list of infants (data). 
    Each infant contrains 2 DFs:
    1st DF: abundance OTU DF
    2nd DF: infant's samples 10 timepoints-cluster labels
    *Samples with missing data have NaN values for both DFs. 
    
    Inputs:
    1. DF output from read_dmm_output() 
    2. OTU DF (int)
    
    Column name example: 1000_A_1 (ID_Site_timepoint)
    SampleNum: positive integer
    Site: single letter
    Timepoints: 1-10
    '''
    samples = df_cluster["sample"]
    samples = samples.sort_values()
    infants = []
    
    # find unique ID (sampleID_Site_)
    for sample in samples:
        infants.append(re.search("[0-9]+[_\-.,/\s][A-Za-z]_",sample).group())
    
    unique_infants = sorted(list(set(infants)))
    
    # create empty dfs:
    data = []
    for infant in unique_infants:
        data.append([pd.DataFrame(columns = [infant+str(x+1) for x in range(10)]),
                     pd.DataFrame(columns = [infant+str(x+1) for x in range(10)])])  
    
    # Store abundance and labels in data.
    index = 0
    for i,sample in enumerate(samples):
        identifier = re.search("[0-9]+[_\-.,/\s][A-Za-z]_",sample).group()
        cluster = pd.Series(data = [int(df_cluster.loc[df_cluster["sample"]==sample,"cluster"])])
        if identifier == unique_infants[index]:
            data[index][0][sample] = df_abund[sample]
            data[index][1][sample]= cluster
        elif i==len(samples)-1:
            continue
        else: 
            data[index+1][0][sample] = df_abund[sample]
            data[index+1][1][sample]= cluster
            index+=1
    data = drop_infants_with_less_than_5_after_selecting_infants(data)

    return data

######### Funcitons related to infant_microbiome_hmm_with_labels.py  #########

def divide_otu_df_based_on_true_labels(otu_df, labels_df, taxa_names):
    '''
    not in helperfunctionshmm.py file
    divides otu_df into gf and gn otu table absed on labels

    *labels_df['Astarte ID'] does not contain all ids in otu_df columns 
    (some infant samples not present in labels_df)
    '''
    no_taxa_otu_df = otu_df.drop(columns=['Taxa'])
    col_names = list(no_taxa_otu_df) # number_site_PMA
    id_list = []
    for an_id in col_names:
        id_list.append(re.search('[0-9]+',an_id).group())
    # *id_list length == column length of no_taxa_otu_df

    #loop thorugh labels_df and make 2 dfs based on labels. 
    # Columns = ['Astarte ID', 'gf']
    # ex) id = 1002, label= 0 or 1 
    id_and_label = {} 
    set_id_list = set(id_list) # unique infant id in string
    for _, row in labels_df.iterrows():
        if str(row.iloc[0]) in set_id_list:
            id_and_label[str(row.iloc[0])] = row.iloc[1] 
            
    gf_list_of_sr=[]
    gn_list_of_sr=[]

    keys = list(id_and_label.keys()) # unique ids
    for i, an_id in enumerate(id_list): #same length as otu_df columns
        if an_id in keys:
            if id_and_label[an_id] == 0:
                gn_list_of_sr.append(no_taxa_otu_df.iloc[:,i].T)
            if id_and_label[an_id] == 1:
                gf_list_of_sr.append(no_taxa_otu_df.iloc[:,i].T)

    gn_df = pd.concat(gn_list_of_sr,axis=1)
    gf_df = pd.concat(gf_list_of_sr,axis=1)

    gn_df.insert(0,'Taxa',taxa_names)
    gf_df.insert(0,'Taxa',taxa_names)
    return gn_df, gf_df

def add_states_to_train_data(data, gn_gf_data):
    '''
    identify infants in gf_data or gn_data in data,
    then adds infant timepoint states (data[infant][1]) to gf_data or gn_data in data.
    '''
    list_of_all_ids = []
    for infant in data:
        an_id = re.search('[0-9]+',list(infant[0])[0]).group()
        list_of_all_ids.append(an_id)
        
    for infant in gn_gf_data:
        an_id = re.search('[0-9]+',list(infant[0])[0]).group()
        if an_id in set(list_of_all_ids):
            idx = list_of_all_ids.index(an_id)
            infant.append(data[idx][1])
    
    #find infant with state data
    new_data = []
    for infant in gn_gf_data:
        if len(infant)==2:
            new_data.append(infant)
            
    return new_data

def divide_based_on_true_labels(otu_df, taxa_names, label_filepath):
    '''
    reads in label.tsv
    divides original data based on label
    '''
    labels_df = pd.read_csv(label_filepath, delimiter='\t')
    gn_df, gf_df = divide_otu_df_based_on_true_labels(otu_df, labels_df, taxa_names)
    return gn_df, gf_df
    
def convert_gn_gf_df_to_data_and_add_states_from_dmm(gn_df, gf_df, data):
        gn_data = convert_otu_df_to_data(gn_df,include_nan=True)
        gf_data = convert_otu_df_to_data(gf_df,include_nan=True)
        gn_data = add_states_to_train_data(data, gn_data)
        gf_data = add_states_to_train_data(data, gf_data)
        return gn_data, gf_data

######### Funcitons related to Initial probabilties  #########

def logdiffexp(la, lb):
    '''
    Returns log of difference 
    of exponent values of la 
    and lb. Smaller value is 
    subtracted from bigger value, 
    so return value is always positive.
    
    Inputs: two values in 
    natural log space.
    
    *When using this function, an 
    exception cases where la and lb values are 
    the same is required! Those exceptions are
    not handled in this function.
    '''
    if lb > la:
        tmp = la
        la = lb
        lb = tmp
        
    if (lb - la < -0.693):
        return la + np.log1p(-np.exp(lb - la, dtype=np.float128), dtype=np.float128)
    else:
        return la + np.log(-np.expm1(lb - la, dtype=np.float128), dtype=np.float128)
    
def generate_initial_matrix(data, n_states = 6):
    '''
    Returns initial probabilities of each first_state as pd.Series.
    
    Input: 
    1.data (output from combine_cluster_and_abund_data())
    2.n_states (number of states or n_clusters) (natural number). This must match 
    the number of states put in for DMM.
    
    States in data must be labeled with natural numbers. No floats.
    '''
    list_of_first_pos = np.array([], dtype=np.float128)
    for infant in data:
        no_nan_ser = infant[1].iloc[0,:].dropna()
        list_of_first_pos = np.append(list_of_first_pos, no_nan_ser[0])
    
    tot_cnt = len(list_of_first_pos)
    ser = pd.Series(list_of_first_pos).value_counts()
    init = ser.sort_index() 
    
    if len(init) == n_states:        
        x = (np.log(init / tot_cnt, dtype=np.float128))
        return x.to_numpy()
    else:
        states = [i+1 for i in range(n_states)]
        excluded_states = []
        for s1 in states:
            if int(s1) not in init.index:
                excluded_states.append(int(s1))
        init = pd.concat([init,pd.Series([0]*len(excluded_states),index=excluded_states, dtype="float128")])
        init += 1
        init = init / (tot_cnt+n_states)
        init.sort_index(inplace=True)
        init = np.log(init, dtype=np.float128)
        return init.to_numpy()

######### Funcitons related to transition probabilties  #########

def generate_transition_matrix(data, n_states=6):
    '''
    Returns n_states by n_states transition matrix
    
    Input: 
    1. data (output from combine_cluster_and_abund_data())
    2.n_states (number of states or n_clusters). This 
    must match the number of states put in for DMM.
    '''        
    tot_cnt = np.zeros((n_states,), dtype=np.float128)
    m = np.zeros((n_states,n_states), dtype=np.float128)
    
    for sample in data:
        states_sr = sample[1].iloc[0,:]
        no_nan_sr = list(states_sr.dropna())
        
        for j,state in enumerate(no_nan_sr):
            #total count of each state upto last element
            if j != len(no_nan_sr)-1:
                tot_cnt[state-1]+=1 # 1st position of moving window of 2 (doesn't count last element)
            if j != 0:
                m[no_nan_sr[j-1]-1][state-1]+=1 # 2nd position of moving window of 2 (doesn't count first element)
    if 0 in m:
        #add 1 for smoothing
        m += 1 
        tot_cnt += n_states
        for i in range(n_states):
            m[i] = m[i] / tot_cnt[i]
    else:
        for i in range(n_states):
            m[i] = m[i] / tot_cnt[i]
    m = np.log(m, dtype=np.float128)
    return m

######### Funcitons related to Emission probabilties  #########

def divide_observations_based_on_states(data, n_states=6):
    '''
    Returns a DF with columns of 
    means and variances of each taxa
    of each hidden state.
    
    Input: 
    1. data (output from combine_cluster_and_abund_data())
    2. n_states (number of states or n_clusters). 
    This must match the number of states put in for DMM.
    '''
    # Make 6 empty DFs
    obs_based_on_states = []
    for i in range(n_states):
        obs_based_on_states.append(pd.DataFrame())
    
    # Get abundance data based on the hidden states
    for infant in data: #240 samples
        for i, col_name in enumerate(list(infant[1].columns)):
            smp_state = infant[1].loc[0,col_name]
            if np.isnan(smp_state): #handle nan values
                continue
            obs_based_on_states[smp_state-1] = pd.concat([obs_based_on_states[smp_state-1],infant[0][col_name]],axis=1)

    return obs_based_on_states

def calc_mean_var_of_each_state(obs_based_on_states, n_states = 6): 
    '''
    Returns DF of mu and var of all states.
    shape=(12,n_taxa)
    
    Input:
    1. output from divide_observations_based_on_states():
    list of abundance DFs divded based on 
    hidden state.
    2. n_states (number of states or n_clusters). This 
    must match the number of states put in for DMM.
    Cacluates mean and variance of each state.
    Mean and variance in df_e_m are not in log space.
    '''
    
    # make empty DF for means and variances of each taxon
    col_names = []
    for i in range(n_states):
        col_names.append("mu"+str(i+1))
        col_names.append("var"+str(i+1))
    df_e_m = pd.DataFrame(columns = col_names)
    df_e_m.fillna(0,inplace=True)
    
    #check if all abund values are the same
    for i,df in enumerate(obs_based_on_states):
        np_version = df.to_numpy()
        row_val_all_same = False
        for row in np_version:
            if (row[0]==row).all():
                row_val_all_same = True
        if row_val_all_same == True:
            df['tmp']= [1.1]*len(df)
    
    # calculate means and variances and store them in df_e_m
    for i,df in enumerate(obs_based_on_states):
        sample_cnt = len(list(df))
        sum_sr = df.sum(axis = 1)
        mu_sr = sum_sr.divide(sample_cnt)
        df_e_m["mu"+str(i+1)] = mu_sr
        
        var = []
        for j in range(len(df)): #loop through rows of df
            deviations_sum = ((df.iloc[j,:] - mu_sr.iloc[j])**2).sum() 
            var.append(deviations_sum / sample_cnt)
        df_e_m["var"+str(i+1)] = var
    return df_e_m

def e_m_to_np(df_e_m):
    '''
    Returns np version of emission matrix 
    (output of calc_mean_var_of_each_state())
    Shape = (n_states,2,n_taxa)
    
    Input:output of calc_mean_var_of_each_state()
    
    *Output means and variances are in natural log space.
    '''
    n_states = int(len(list(df_e_m))/2)
    n_taxa = len(df_e_m)
    np_e_m = np.zeros((n_states,2,n_taxa), dtype=np.float128)
    
    for col in list(df_e_m):
        state = int(re.search('[0-9]+',col).group())-1
        if col[:2]=="mu":
            np_e_m[state,0] = df_e_m[col].to_numpy()
        else:
            np_e_m[state,1] = df_e_m[col].to_numpy()
    return np.log(np_e_m, dtype=np.float128)

def normal_dist_PDF(abund , mean , var):
    '''
    Returns probability calculated using 
    normal distribution pdf of the abundance 
    value in log space (natural log). 
    
    Input:
    1. abund (abundance value) must 
    not be in log space. 
    2.mean must be in log space. 
    3.var (variance) must be in log space. 
    '''
    sd = np.multiply(0.5,var,dtype=np.float128) 
    a = - (sd + 0.5*np.log(2 * np.pi))
    if np.round(np.log(abund, dtype=np.float128),decimals=4) != np.round(mean,decimals=4):
        b = - np.exp((np.log(0.5, dtype=np.float128) + (2*(logdiffexp(np.log(abund,dtype=np.float128),mean)-sd))), dtype=np.float128)
        return a+b
    else:
        return a
    
def calc_emis_prob(e_m, sample, state):
    '''
    Returns emission probability
    of a sample in log space.
    Input:
    1. emission matrix (e_m)
    2. sample (shape=(1 x 444))
    3. state the sample is in.
    State value begins from 0 for this function, not 1.
    '''
    n_taxa = len(sample)
    prob = np.array([], dtype=np.float128)
    for i in range(n_taxa):
        mean = e_m[state,0,i]
        var = e_m[state,1,i]
        prob = np.append(prob, normal_dist_PDF(sample.iloc[i],mean,var))
    ans = np.sum(prob, dtype=np.float128)
    return ans

######### Funcitons for  calculating alpha / beta (e step) #########

def calc_alpha(infant, i_m, t_m, e_m, n_states=6):
    '''
    Returns calculated alpha variables.
    
    Input: infant, which is a list of DFs 
    ([0] = abundance data, [1] = states)
    
    Output matrix is in natural log space.
    '''
    abund = infant[0]
    timepoint = len(list(infant[0]))

    alpha = np.zeros((n_states,timepoint), dtype=np.float128)
    
    for tp in range(timepoint):
        sample = abund.iloc[:,tp]
        for i in range(n_states):
            if tp == 0:
                if np.isnan(sample.iloc[0]):
                    alpha[i,tp] = i_m[i]
                else:
                    alpha[i,tp] =  calc_emis_prob(e_m, sample, i) + i_m[i]
            else:
                a_i_j = []
                a_i = 0.0
                for j in range(n_states): 
                    if np.isnan(sample.iloc[0]):
                        a_i_j.append(t_m[j,i] + alpha[j,tp-1])
                    else:
                        a_i_j.append(calc_emis_prob(e_m,sample,i) + t_m[j,i] + alpha[j,tp-1])
                a_i = logsumexp(a_i_j)
                alpha[i,tp] = a_i
    return alpha

def calc_beta(infant, i_m, t_m, e_m, n_states=6):
    '''
    Returns calculated beta variables.
    
    Input: infant, which is a list of DFs 
    ([0] = abundance data, [1] = states)
    
    Output matrix is in natural log space.
    '''
    abund = infant[0]
    timepoint = len(list(infant[1]))
    beta = np.zeros((n_states,timepoint), dtype=np.float128)
    beta[:,-1] = 0 #inialize last timepoint as 0 (because calculating beta in log space)
    
    for tp in range(timepoint-2,-1,-1): # timepoint loop => loop through 8~0 (not 9~1)
        sample = abund.iloc[:,tp+1]
        for i in range(n_states): # i loop (transition mtx idx)
            b_i_j = []
            for j in range(n_states): # j loop (beta / transition mtx idx)
                if np.isnan(sample.iloc[0]):
                    b_i_j.append(t_m[i,j] + beta[j,tp+1])
                else:
                    b_i_j.append(t_m[i,j] + beta[j,tp+1] + calc_emis_prob(e_m,sample,j))
            b_i = logsumexp(b_i_j)
            beta[i,tp] = b_i
    return beta


def calc_alphas_and_betas(data, i_m, t_m, e_m, n_states=6):
    '''
    Returns data with updated alpha & beta matricies for all infants.
    
    Input:
    1. data (output from combine_cluster_and_abund_data())
    2. i_m (np.array initial matrix from generate_initial_matrix())
    3. t_m (np.array transition matrix from generate_transition_matrix())
    4. t_m (np.array emission matrix from e_m_to_np())
    5. n_states (number of states or n_clusters). 
    This must match the number of states put in for DMM.
    
    This function either calculates or updates alpha and beta 
    matrices at index 2 and 3 for all infants in data. 
    '''
    for infant in data:
        if len(infant) == 2: #1st e step of EM
            infant.append(calc_alpha(infant,i_m, t_m, e_m))
            infant.append(calc_beta(infant,i_m, t_m, e_m))
        elif len(infant) == 3:
            infant[2] = calc_alpha(infant,i_m, t_m, e_m)
            infant.append(calc_beta(infant,i_m, t_m, e_m))
        else: # updating on steps after the first
            infant[2] = calc_alpha(infant,i_m, t_m, e_m)
            infant[3] = calc_beta(infant,i_m, t_m, e_m)
    return data

######### Funcitons for calculating st_i / st_i_j for e step #########

def calc_st_i(infant):
    '''
    Returns calculated probability matrix of state i 
    at timepoint t given the observation: s_t(i)
    Shape = (n_states x timepoints)
    
    Input: infant, which is a list of DFs:
    [0] = abundance data, [1] = states at 10 timepoints, [2] =alpha matrix, [3] = beta matrix

    Output matrix is in log space.
    '''
    # Calculate S_t(i) of an infant
    n_states = len(infant[2]) 
    n_timepoints = len(infant[2][0])
    alpha = infant[2]
    beta = infant[3]
    
    st_i = np.zeros((n_states,n_timepoints), dtype=np.float128)
    
    for tp in range(n_timepoints):
        sum_of_prod_of_ab_all_states = np.array([],dtype=np.float128) #설마....... # denom
        sum_of_prod_of_ab_all_states = logsumexp(alpha[:,tp] + beta[:,tp]) # sum of alpha*beta (all states) of each timepoint (denom)
        for i in range(n_states):
            st_i[i,tp] = alpha[i,tp] + beta[i,tp]
        st_i[:,tp] = st_i[:,tp] - sum_of_prod_of_ab_all_states
    return st_i

def calc_st_i_j(infant, t_m, e_m):
    '''
    Returns calculated transition matrices 
    of each timepoint given the observation: s_t(i,j)
    Shape = (n_timepoints-1, n_states, n_states)
    
    Input: infant, which is a list of DFs:
    [0] = abundance data, [1] = states at 10 timepoints, [2] =alpha matrix, [3] = beta matrix

    Output matrices are in log space.
    '''
    n_states = len(infant[2])
    n_timepoints = len(infant[2][0])
    abund_data = infant[0]
    alpha = infant[2]
    beta = infant[3]
    
    st_i_j_list=[]
    for tp in range(n_timepoints-1): # loop through 0-8 (not 0-9)
        st_i_j = np.zeros((n_states,n_states), dtype=np.float128) # n_states by n_states
        
        sum_of_prod_of_ab_all_states = np.array([],dtype=np.float128)
        sum_of_prod_of_ab_all_states = logsumexp(alpha[:,tp] + beta[:,tp]) # sum of alpha*beta (all states) of each timepoint (denom)
        for i in range(n_states):
            for j in range(n_states):
                if np.isnan(abund_data.iloc[0,tp+1]):
                    st_i_j[i,j] = alpha[i,tp] + beta[j,tp+1] + t_m[i,j]
                else:
                    st_i_j[i,j] = alpha[i,tp] + beta[j,tp+1] + t_m[i,j] + calc_emis_prob(e_m, abund_data.iloc[:,tp+1], j)
            st_i_j[i,:] = st_i_j[i,:] - sum_of_prod_of_ab_all_states
        
        st_i_j_list.append(st_i_j)
    return st_i_j_list

def calc_sts(data, t_m, e_m):
    '''
    Returns data with updated st_i & st_ij matricies for all infants.
    
    Input:
    1. data (output from combine_cluster_and_abund_data())
    2. t_m (np.array transition matrix from generate_transition_matrix())
    3. t_m (np.array emission matrix from e_m_to_np())
    4. n_states (number of states or n_clusters). This must 
    match the number of states put in for DMM.
    
    This function either calculates or updates st_i & st_ij
    matrices at index 4 and 5 for all infants in data.
    '''
    for infant in data:
        if len(infant) == 4:
            infant.append(calc_st_i(infant))
            infant.append(calc_st_i_j(infant, t_m, e_m))
        elif len(infant) == 5:
            infant[4] = calc_st_i(infant)
            infant.append(calc_st_i_j(infant, t_m, e_m))
        else:
            infant[4] = calc_st_i(infant)
            infant[5] = calc_st_i_j(infant, t_m, e_m)
    return data

######### Funcitons for  calculating i_m, t_m, e_m for m step #########

def m_step_initial_matrix(data):
    '''
    Returns initial matrix.
    Shape = (n_states,)
    
    Input: updated data with alpha, beta, sti and stij.
    
    Generates initial probability matrix
    using all s_t(i) matricies in data.
    
    Output matrix is in log space.
    '''
    n_states = len(data[0][2]) #n_rows of alpha matrix
    i_m = np.array([],dtype=np.float128)
    all_states_tot_sum = np.float128(0) # 의미가 있나?
    
    for x, infant in enumerate(data):
        sti = infant[4]
        tp_1_sum = logsumexp(sti[:,0]) #sum of 1st column
        if x == 0:
            i_m = sti[:,0]
            all_states_tot_sum = tp_1_sum
        else:
            for i in range(n_states):
                i_m[i] = logsumexp([i_m[i],sti[i,0]])
            all_states_tot_sum = logsumexp([all_states_tot_sum,tp_1_sum])
    
    i_m = i_m - all_states_tot_sum
    return i_m
            
def m_step_transition_matrix(data):
    '''
    Returns transition matrix.
    Shape = (n_states, n_states)
    
    Input: updated data with alpha, beta, sti and stij.
    
    Generates transition probabilities
    using all s_t(i,j) matricies in data.
    
    Output matrix is in log space
    '''
    n_states = len(data[0][2]) #n_rows of alpha matrix == number of states
    t_m = np.array([],dtype=np.float128)
    all_states_tot_sum = np.zeros((n_states), dtype=np.float128) # tot sum of probabilistic count of each state
    
    for x, infant in enumerate(data):
        for tp, stij in enumerate(infant[5]):
            for i in range(n_states):
                row_sum = logsumexp(stij[i,:])
                if x == 0 and tp == 0:
                    all_states_tot_sum[i] = row_sum
                    if i == n_states-1: # x==0 , tp ==0 일때 stij 한번만 넣어주기위해 마지막에 state에 넣음.
                        t_m = stij
                else:
                    all_states_tot_sum[i] = logsumexp([all_states_tot_sum[i],row_sum])
                    for j in range(n_states):
                        t_m[i,j] = logsumexp([t_m[i,j],stij[i,j]])
    for s in range(n_states):
        t_m[s] = t_m[s] - all_states_tot_sum[s]
    return t_m

def m_step_emission_matrix(data):
    '''
    Returns emission matrix
    Shape=(n_states,2,n_taxa)
    At axis=1, index 0 is mu and 1 is variance.
    
    Input: updated data with alpha, beta, sti and stij.
    
    Mu and variances are calculated using 
    abundance and s_t(i) matricies in updated data. 
    '''
    n_states = len(data[0][2]) #n_rows of alpha matrix == number of states
    timepoint = len(data[0][2][0])
    n_taxa = len(data[0][0])
    n_infants = len(data)
    
    #compile all data in stack_of_states
    stack_of_states = np.array([], dtype=np.float128) # shape =(n_state, n_infant, n_taxa)
    
    denom_sum_of_weights = np.zeros((6), dtype=np.float128)
    
    for state in range(n_states):
        stack_of_samples = np.array([], dtype=np.float128) #exists for each state
        
        for idx, infant in enumerate(data):
            abund = infant[0]
            sti = infant[4]
            for tp in range(timepoint):
                if np.isnan(abund.iloc[0,tp]) == False: # if abund value not nan
                    if denom_sum_of_weights[state] == 0:
                        denom_sum_of_weights[state] = sti[state,tp]
                    else:
                        denom_sum_of_weights[state] = logsumexp([denom_sum_of_weights[state], sti[state,tp]])
                    if idx == 0:
                        stack_of_samples = np.expand_dims( (np.log(abund.iloc[:,tp], dtype=np.float128) + sti[state,tp]), 0) #becomes 2D (stack_of_samples)
                    else:
                        stack_of_samples = np.append(stack_of_samples, np.expand_dims( (np.log(abund.iloc[:,tp], dtype=np.float128) + sti[state,tp]), 0), axis=0)
        
        if state == 0: #becomes 3D (stack_of_states)
            stack_of_states = np.expand_dims(stack_of_samples,0)
        else:
            stack_of_states = np.append(stack_of_states, np.expand_dims(stack_of_samples,0),axis=0) 
            
    #compute mu and sigma of each state and store them in e_m
    e_m = np.zeros((n_states, 2, n_taxa), dtype=np.float128) # (6, 2, 444)
    
    
    for i in range(n_states):
        for j in range(n_taxa):
            weighted_taxa_data_at_state_i = stack_of_states[i,:,j]
            #calc mu 
            mu = logsumexp(weighted_taxa_data_at_state_i) - denom_sum_of_weights[i]
            e_m[i,0,j] = mu
            
            #calc var
            cnt_when_abund_same_as_mu = 0
            deviations = np.array([], dtype=np.float128)
            for abund_val in weighted_taxa_data_at_state_i:
                if np.round(abund_val,decimals=4) != np.round(mu,decimals=4):
                    deviations = np.append(deviations, 2*logdiffexp(abund_val, mu))
                else:
                    cnt_when_abund_same_as_mu+=1
            e_m[i,1,j] = logsumexp(deviations) - np.log((len(deviations)+cnt_when_abund_same_as_mu), dtype=np.float128)
            
    return e_m

######### hmm em algorithm #########
def e_step(data,i_m,t_m,e_m,states=6):
    '''
    Inference step
    
    Returns updated data:
    
    Input:
    1. data
    2. i_m (np.array initial matrix)
    3. t_m (np.array transition matrix)
    4. t_m (np.array emission matrix)
    5. n_states (This must match the number of states put in for DMM.)
    
    After e step, all infants contains:
    [0] = abundance data, [1] = states at 10 timepoints, [2] =alpha matrix, [3] = beta matrix
    '''
    updated_data = calc_alphas_and_betas(data,i_m, t_m, e_m, n_states=states)
    updated_data = calc_sts(updated_data, t_m, e_m)
    return updated_data

def m_step(data):
    '''
    MLE step
    
    Returns initial, transition and emission matrix.
    
    Input:data updated in e step.
    
    '''
    i_m = m_step_initial_matrix(data)
    t_m = m_step_transition_matrix(data)
    e_m = m_step_emission_matrix(data)
    return i_m, t_m, e_m

def find_viterbi_path(data, i_m, t_m, e_m):
    '''
    Return updated data appended with highest probability path of states for all infants.
    use this func after e and m step 
    to find the most probable path.
    
    Viterbi path of all infants calculated
    using i_m, t_m, e_m.
    
    Returned v_path_of_states state starts from index 0 not 1.
    '''
    n_states = len(i_m)
    n_timepoints = len(list(data[0][0]))
    
    v_m = np.zeros((n_states,n_timepoints),dtype=np.float128)
    
    for infant in data:
        
        v_path_of_states = np.array([],dtype=np.int8) #np.int8 => -128 ~ 127 
        abund = infant[0]
        for tp in range(n_timepoints):
            sample = abund.iloc[:,tp]
            prev_tp_max = -1e100 #initialize prev_tp_max with a negative every timepoint
            prev_tp_max_idx = -1
            for i in range(n_states):
                if tp==0:
                    if np.isnan(sample.iloc[0]):
                        v_m[i,tp] = i_m[i]
                    else:
                        v_m[i,tp] = calc_emis_prob(e_m,sample,i) + i_m[i]
                else:
                    if i == 0:
                        for j in range(n_states):
                            if v_m[j,tp-1] > prev_tp_max:
                                prev_tp_max = v_m[j,tp-1]
                                prev_tp_max_idx = j
                        v_path_of_states = np.append(v_path_of_states, int(prev_tp_max_idx))
                    if np.isnan(sample.iloc[0]):
                        v_m[i,tp] = t_m[prev_tp_max_idx,i] + prev_tp_max
                    else:
                        v_m[i,tp] = calc_emis_prob(e_m,sample,i) + t_m[prev_tp_max_idx,i] + prev_tp_max
            if tp == n_timepoints-1:
                last_tp_max = -1e100
                last_tp_max_idx = -1
                for j in range(n_states):
                    if v_m[j,tp] > last_tp_max:
                        last_tp_max = v_m[j,tp]
                        last_tp_max_idx = j
                v_path_of_states = np.append(v_path_of_states, int(last_tp_max_idx))
        
        if len(infant) == 6:
            infant.append(v_path_of_states)
        elif len(infant) == 7:
            infant[6] = v_path_of_states
        elif len(infant) > 7:
            print('Infant length bigger than 7')
        elif len(infant) <6:
            print('Infant length smaller than 6. Infant is missing one or more of the following: alpha, beta, s_t(i) or s_t(i,j)')
    return data

def calc_loglikelihood_viterbi(data,i_m,t_m,e_m):
    '''
    Returns loglikelihood of hmm model,
    calculated using viterbi path.
    
    Input: data updated in e step
    '''
    n_states = len(i_m)
    n_timepoints = len(list(data[0][0]))
    likelihood = np.array([],dtype=np.float128)
    
    for infant in data:
        abund = infant[0]
        v_path = infant[6]
        for tp in range(n_timepoints):
            state = v_path[tp]
            sample = abund.iloc[:,tp]
            if tp == 0:
                if np.isnan(sample.iloc[0]):
                    likelihood = np.append(likelihood, i_m[state])
                else:
                    likelihood = np.append(likelihood, i_m[state] + calc_emis_prob(e_m,sample,state))
            else:
                prev_state = v_path[tp-1]
                
                if np.isnan(sample.iloc[0]):
                    likelihood = np.append(likelihood, t_m[prev_state, state])
                else:
                    likelihood = np.append(likelihood, t_m[prev_state, state] + calc_emis_prob(e_m,sample,state))
                    
    return np.sum(likelihood, dtype=np.float128)

def calc_percent_diff(likelihoods):
    '''
    Returns percent difference of 
    last and second to last element 
    of likelihoods list.
    
    Input: list of likelihoods
    
    *len(likelihoods) must be > 2
    '''
    last_idx = len(likelihoods)-1
    bf = likelihoods[last_idx-1]
    af = likelihoods[last_idx]
    
    return abs((af-bf)/(af+bf)/2)*100

def em_algo(data, init_i_m, init_t_m, init_e_m, threshold):
    '''
    Returns caclulated likelihoods (list), 
    final initial matrix, final transition matrix 
    and final emission matrix.

    Input:
    1. data (output from combine_cluster_and_otu_table())
    2. init_i_m (output from generate_initial_matrix())
    3. init_t_m (output from generate_transition_matrix())
    4. init_e_m (output from e_m_to_np())
    5. threshold (threshold parameter is compared with the 
    percent difference of the current and previous log likelihood.)

    *This function prints percent difference of each step.
    *If the iteration goes over 20, it loops out and converges.
    '''
    print('Threshold: ',threshold)
    print()
    likelihoods = np.array([],dtype=np.float128)
    i_m, t_m, e_m = np.array([], dtype=np.float128), np.array([], dtype=np.float128), np.array([], dtype=np.float128)
    percent_diff = 123456789 #initialize with big number
    i = 0
    
    while True:
        if  percent_diff < threshold:
            break
        elif i == 20:
            break
        if i == 0:
            data = e_step(data, init_i_m, init_t_m, init_e_m)
            i_m, t_m, e_m = m_step(data)
            data = find_viterbi_path(data, i_m, t_m, e_m)
            likelihoods = np.append(likelihoods, calc_loglikelihood_viterbi(data,i_m, t_m, e_m))
        else: 
            data = e_step(data, i_m, t_m, e_m)
            i_m, t_m, e_m = m_step(data)
            data = find_viterbi_path(data, i_m, t_m, e_m)
            likelihoods = np.append(likelihoods, calc_loglikelihood_viterbi(data,i_m, t_m, e_m))
            
            percent_diff = calc_percent_diff(likelihoods)
        i+=1
        
        if i == 1:
            print('============================ Iteration: ',i)
            print('Current Likelihood: ', likelihoods[len(likelihoods)-1])
            print()
        else:
            print('============================ Iteration: ',i)
            print('Current Likelihood: ', likelihoods[len(likelihoods)-1])
            print('Likelihood percent difference: ', percent_diff)
            print()
            if likelihoods[len(likelihoods)-2] > likelihoods[len(likelihoods)-1] and percent_diff > 0.05:
                print("likelihood dropped more than 0.05. Exclude the current likelihood that dropped.")
                return likelihoods[:len(likelihoods)-2], i_m, t_m, e_m

    return likelihoods, i_m, t_m, e_m

def output_likelihood_graph(score, filepath):
    '''
    Saves a graph of likelihood 
    score in current working directory.

    Input: score (list of likelihood values)
    '''
    plt.figure(dpi=150)
    plt.rc("font", size=6.5)
    plt.plot(range(0,len(score)), score, color = "black", marker = "o")
    plt.title("Log Likelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Log Likelihood")
    plt.savefig(filepath+"/hmm_log_likelihood.png")
    plt.close(None)
    print('HMM log likelihood graph saved as: hmm_log_likelihood.png')

def output_i_t_e_matrix(taxa_name,i_m,t_m,e_m, path):
    '''
    Prints out initial, transition and
    emission matrices. Outputs them in 
    path (parameter) directory as well.
    '''
    print('initial matrix:')
    print()
    print(np.exp(i_m))
    print()
    print('===================================')
    print('transition matrix:')
    print()
    print(np.exp(t_m))
    print()
    print('===================================')
    print('emission matrix:')
    print()
    pd.set_option('display.max_rows', None)
    for i, state in enumerate(np.exp(e_m)):
        print('state: ',str(i+1))
        print(pd.DataFrame(state,columns=taxa_name,index=['mean','variance']).T)
        print()
    pd.reset_option('display.max_columns')

    np.savetxt(path+'/initial_matrix.txt', np.exp(i_m), delimiter=',')
    np.savetxt(path+'/transition_matrix.txt', np.exp(t_m), delimiter=',')
    
    for i, state in enumerate(np.exp(e_m)):
        tmp = pd.DataFrame(state,columns=taxa_name,index=['mean','variance']).T
        np.savetxt(path+'/emission_matrix_state_'+str(i+1)+'.txt', tmp.values, delimiter=',')
        

def output_top_x_variance_taxa_per_state(taxa_names, e_m, n, filepath):
    '''
    Prints taxa with top n (parameter) variance 
    per state. Outputs it in filepath (parameter) directory
    as txt file as well.
    
    Input: 
    1. OTU df
    2. emission matrix
    3. n (number of taxa to be outputed)
    
    Example output: 
    (Bifidobacterium_breve, 10.0)
    '''
    top_x_variance_taxa_per_state = []
    for i in range(len(e_m)):
        tmp = []
        for j in range(len(taxa_names)):
            tmp.append((taxa_names.iloc[j], e_m[i,1,j])) #bind taxa name and variance as tuple
        top_x_variance_taxa_per_state.append(tmp)

    for i in range(len(top_x_variance_taxa_per_state)):
        top_x_variance_taxa_per_state[i] = sorted(top_x_variance_taxa_per_state[i],key=lambda x:x[1], reverse=True)

    for i,s in enumerate(top_x_variance_taxa_per_state):
        print('state '+str(i+1))
        for j,taxon in enumerate(s):
            if j < n:
                print(taxon)
        print()

    with open(filepath + '/top_x_variance_taxa_per_state.txt','w') as f:
        for i,s in enumerate(top_x_variance_taxa_per_state):
            f.write('state '+str(i+1))
            for j,taxon in enumerate(s):
                if j < n:
                    f.write(str(taxon))
            f.write('\n')

def hmm_output(taxa_names, i_m, t_m, e_m, n_taxa, score):
    '''
    Outputs initial matrix, transition matrix, 
    emission matrix and top n_taxa with the 
    highest variance as txt file. Also outputs change 
    in likelihood over iterations.
    '''
    if os.path.exists("hmm_outputs") == False:
        os.mkdir('hmm_outputs')

    new_path = make_unique_file_or_dir_names('hmm_outputs/hmm_out')
    os.mkdir(new_path)
    
    output_i_t_e_matrix(taxa_names, i_m, t_m, e_m, new_path)
    output_top_x_variance_taxa_per_state(taxa_names, e_m, n_taxa, new_path)
    output_likelihood_graph(score, new_path)
    print('Output saved in:'+new_path)


def run_hmm(data, n_states, taxa_names, likelihood_threshold, n_taxa_for_display):
    # EM initialization: calculate first initial, transition, emission matrix
    init_i_m= generate_initial_matrix(data,n_states)
    init_t_m = generate_transition_matrix(data, n_states)
    obs_based_on_states = divide_observations_based_on_states(data, n_states)
    df_e_m = calc_mean_var_of_each_state(obs_based_on_states, n_states)
    init_e_m = e_m_to_np(df_e_m)
    print()
    print('Initial, transition and emission matrix generated.')
    
    print()
    print('Moving on to EM algorithm...')
    likelihoods, i_m, t_m, e_m = em_algo(data, init_i_m, init_t_m, init_e_m, threshold = likelihood_threshold)
    
    print()
    print('EM algorithm outputs: ')
    hmm_output(taxa_names, \
                        i_m, \
                        t_m, \
                        e_m, \
                        n_taxa_for_display, \
                        likelihoods)

##### Functions for outputting predictions (classification of GN or GF) #####

##### read in initial, transsition & emission matrix from hmm out #####

def read_in_i_m_from_hmm_output(path_to_hmm_out):
    '''
    Returns initial matrix read 
    from hmm_out directory. Output
    is in log space.
    
    *hmm_out directory might be named
    differently depending on the number
    of times the code was run.
    ex) 
    If HMM was run twice, there could be:
    hmm_out, hmm_out_2
    '''
    i_m_df = pd.read_csv(path_to_hmm_out+'/initial_matrix.txt',header=None)
    return np.log(i_m_df).to_numpy()

def read_in_t_m_from_hmm_output(path_to_hmm_out):
    '''
    Returns transition matrix read 
    from hmm_out directory. Output
    is in log space.
    
    *hmm_out directory might be named
    differently depending on the number
    of times the hmm model was run.
    ex) 
    If HMM was run twice, there could be:
    hmm_out, hmm_out_(1) 
    '''
    t_m_df = pd.read_csv(path_to_hmm_out+'/transition_matrix.txt',header=None)
    return np.log(t_m_df).to_numpy()

def read_in_e_m_from_hmm_output(path_to_hmm_out, n_states):
    '''
    Returns the emission matrix 
    as np.array with values in log space.
    
    Input:
    1.Path_to_hmm_out parameter is 
    the path to hmm_out directory.
    Do not put the full path to
    emssion_matrix.txt file.
    
    2. Number of states of the HMM model
    
    *hmm_out directory might be named
    differently depending on the number
    of times the hmm model was run.
    ex) 
    If HMM was run twice, there could be:
    hmm_out, hmm_out_(1) 
    '''
    n_taxa = len(pd.read_csv(path_to_hmm_out+'/emission_matrix_state_1.txt',header=None)) #get taxa length from file
    e_m = np.zeros((n_states,2,n_taxa))
    for i in range(n_states):
        em_state = pd.read_csv(path_to_hmm_out+'/emission_matrix_state_'+str(i+1)+'.txt',header=None)
        em_state.columns=['mu','var']
        e_m[i,0] = np.log(em_state['mu'])
        e_m[i,1] = np.log(em_state['var'])
    return e_m

def output_im_tm_em_from_hmmout_dir(path, n_states =6):
    '''
    path = path to hmm_out dir
    '''
    i_m = read_in_i_m_from_hmm_output(path)
    t_m = read_in_t_m_from_hmm_output(path)
    e_m = read_in_e_m_from_hmm_output(path, n_states)
    return i_m, t_m, e_m

def convert_otu_to_data_for_prediction(csv_data_filepath, pma_start_date=196):
    '''
    difference:
    data only contains infants with abundance data.
    Abundance data contains nan values for missing samples in 10 timepoints.
    '''
    test_otu_df= read_data_as_df(csv_data_filepath)
    test_otu_df = convert_data_to_int(test_otu_df) # columns of test_otu_df sorted
    test_otu_df, taxa_names = choose_top_x_percent_of_taxa_with_highest_variance(test_otu_df, frac=1)

    test_otu_df = drop_infants_with_less_than_5_samples(test_otu_df)
    timepoints = select_10_pma(pma_start_date=pma_start_date) 
    selected_columns = sample_based_on_timepoints(test_otu_df, timepoints)
    # trim columns (samples) 2
    test_otu_df = drop_samples_not_part_of_timepoints(selected_columns, test_otu_df)
    data = convert_otu_df_to_data(test_otu_df,include_nan=True)
    data = drop_infants_with_less_than_5_after_selecting_infants(data)

    return data

def calc_alphas_for_classification(infant, i_m, t_m, e_m, n_states):
    '''
    Returns data with alpha matrix 
    appended to each infant in data.
    Input:
    1. data (list of infants)
    2. initial matrix (np.array())
    3. transition matrix (np.array())
    4. emisison matrix (np.array())
    '''
    if len(infant)==1:
        infant.append(calc_alpha(infant,i_m,t_m,e_m,n_states))
    else:
        print('Length of each infant is bigger than 1. Each infant should only contain abundance data.')
    return infant

def calc_p_of_o(infant):
    '''
    Returns caclulated P(O)s of all infants
    in data in a list.
    Input: data (list of infants)
    '''
    n_tps = len(list(infant[0]))
    
    p_of_o = logsumexp(infant[1][:,n_tps-1])
    return p_of_o

def output_p_of_o_for_infant(test_infant, np_i_m, np_t_m, np_e_m, n_states): #path_to_test_otu_table
    '''
    Returns df with infant id and p(O).
    P(O) is in natural log space as
    the values are very small.
    
    Input:
    im,tm,em in log space
    '''
    #preprocess test otu_df data
    copy_infant = copy.deepcopy(test_infant)
    
    copy_infant = calc_alphas_for_classification(copy_infant, np_i_m, np_t_m, np_e_m, n_states)
    p_of_o = calc_p_of_o(copy_infant)
    return p_of_o

def output_p_of_o_df(path_to_hmm_out, test_data, n_states):
    '''
    takes test data
    Output df 

    column1: infant ID
    Column2: predicted label (0 or 1)
    0 = GN
    1 = GF

    prediction = P(O) = probability of observed data
    '''
    prediction_df = pd.DataFrame({'id':[],'prediction':[]})
    i_m, t_m, e_m = output_im_tm_em_from_hmmout_dir(path_to_hmm_out)
    
    for infant in test_data:
        infant_id = re.search('[0-9]+',list(infant[0])[0]).group()
        p_of_o = output_p_of_o_for_infant(infant, i_m, t_m, e_m, n_states)
        prediction_df = pd.concat([prediction_df, pd.Series([infant_id, p_of_o], index=['id','prediction']).to_frame().T])
                
    return prediction_df