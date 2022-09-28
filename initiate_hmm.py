import helper_functions_hmm as hf
import subprocess
import os.path

'''
This script does three things:
1. Download DMM code zip file
2. Extracts DMM zip file
3. Makes a initialized parameter.txt file
    * every time this script is run, parameter text file with a different name will be created.
    * user must type in the location of data in parameter.txt file before running infant_microbiome_hmm_main.py 
'''
######### script functions #########
def download_dmm_code():
    '''
    MicrobeDMMv1.0.tar.gz file required to run this project.
    MicrobeDMMv1.0.tar.gz located at downloads tab at:
    https://code.google.com/archive/p/microbedmm/
    '''
    if os.path.exists('MicrobeDMMv1.0.tar.gz'):
        print("MicrobeDMMv1.0.tar.gz file exists in current directory. File not downloaded.")
    else:
        process = subprocess.Popen(['curl',
                    '-o','MicrobeDMMv1.0.tar.gz',
                    'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/microbedmm/MicrobeDMMv1.0.tar.gz'])
        if process.wait() == 0:
            print("Successfully Downloaded DMM code tar.gz file.")
            print()
        else:
            print("Error occured while downloading DMM code.")
            print("You can manually download MicrobeDMMv1.0.tar.gz located at: https://code.google.com/archive/p/microbedmm/")

def extract_dmm_zip():
    '''
    Extracts MicrobeDMMv1.0.tar.gz file located
    at current working directory.
    '''
    if os.path.exists('MicrobeDMMv1.0'):
        print('DMM directory (MicrobeDMMv1.0) already exists. Zip file not extracted.')
    else:
        if os.path.exists('MicrobeDMMv1.0.tar.gz'):
            unzip_f = subprocess.run(['tar','-xzf' 'MicrobeDMMv1.0.tar.gz'])
            if unzip_f.returncode == 0:
                print("Successfully extracted MicrobeDMMv1.0.tar.gz at current working directory.")
            else:
                print("Unsuccessful file extraction.")
        else:
            print("MicrobeDMMv1.0.tar.gz file is required in the current working directory. The code located at downloads tab at: https://code.google.com/archive/p/microbedmm/")

def write_parameter_layout_txt_file():
    '''
    Writes parameter.txt file that
    lays out all the parameter names 
    and default parameter values.

    Each parameter is searched based on '=' sign.
    * '=' sign required
    * order of the parameters must be maintained.
    '''
    parameters = ['(string) data_filepath = data/otu_table_sample.csv\n',
                '(float) fraction_of_taxa_selection = 0.0338\n'
                '(int) pma_start_date (range: 0 - 273) = 196\n', 
                '(int) n_states = 6\n', 
                '(float) likelihood_threshold = 0.017\n',
                '(int) n_taxa_result_to_be_displayed_in_output = 5']
                

    new_path = hf.make_unique_file_or_dir_names("./parameters.txt")
    with open(new_path, 'w') as f:
        f.writelines(parameters)


############ Run Functions ############
download_dmm_code()
extract_dmm_zip()
write_parameter_layout_txt_file()