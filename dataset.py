import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import rpy2.robjects as ro

# Import necessary R libraries
ro.r('library(DirichletMultinomial)')
ro.r('library(reshape2)')

class InfantMicrobiomeDataset():
    def __init__(self, data_df, label_df):
        """
        data_df (pd.DataFrame): taxa x timepoint microbiome "frequency" data (*not normalized)
            row index: 
                'Taxa'
            column naming format: 
                "<sample id(int)>_<hospital site code(str)>_<post menstral age(int)>"

        label_df (pd.DataFrame): sample id to binary label data
            row index: 'Astarte ID'
                Astarte ID is sample id
            column name: gf
                1 = growth falter
                0 = growth normal

        InfantMicrobiomeDataset initializes & holds 2 versions of microbiome data: 
            1.matrix (taxa by all samples' timepoints mtx)
                => for runnning Dirichelet Multinomial Mixture (DMM)
            2.dictionary (keys:sample ids, values: infant data&label)
                => for train/test split
        """
        self.data_df = data_df
        self.label_df = label_df

        self.data_dict = self.create_data_dict(self.data_df) 
        # dict[str,SimpleNamespace]

    def sliceout_sample_from_df(self, df, sample_id):
        """
        return sliced sample df from OTU frequency df
        
        sample_id (str): <sample_id>
        
        OTU df column name format: 
            str(<sample id(int)>_<hospital site code(str)>_<post menstral age(int))
        """
        # Select columns where column name contains sample_id
        slice_df = df.loc[:, df.columns.str.contains(sample_id)]
        return slice_df

    def create_data_dict(self, df):
        """
        return dictionary where
            keys: sample ids
            values: SimpleNamespace(data,label,mask])
        """
        # Output datset dict container
        id_to_tup={}

        # Create a set for str sample ids
        sample_id_set=set()
        for column in list(df):
            # Get sample id
            try:
                idx = column.find('_')
                id_str = column[:idx]
            except:
                print(f"Wrong column name format:{column}. \
                    Correct format: <sample id(int)>_<hospital site code(str)>_<post menstral age(int)>")
            else:
                sample_id_set.add(id_str)
        
        # Insert data to id_to_tup dictionary
        for id_str in sample_id_set:
            # Create infant data for each id
            if (id_to_tup.get(id_str)) is None:
                
                # 1.Retrieve data from data_df & convert to np.array
                sample_df = self.sliceout_sample_from_df(df, id_str)
                
                # 2.Retrieve label
                try:
                    label = self.label_df.loc[int(id_str),'gf']
                except:
                    print(f"Label missing for sample id: <{id_str}> \n=> not added to dictionary dataset")
                    cols_to_remove = df.columns[df.columns.str.contains(id_str)]
                    df.drop(columns=cols_to_remove, inplace=True)
                    print(f"sample id: <{id_str}> removed from microbiome data",end='\n\n')
                
                # 3.Create Infant SimpleNamespace obj
                else:
                    missing_data_mask = ~sample_df.columns.str.contains('NA')
                    id_to_tup[id_str] = SimpleNamespace(data=sample_df.to_numpy().T, # (tp x taxa)
                                                        label=label,
                                                        mask=missing_data_mask # seq length bool mask: False if tp data is missing, else True
                                                        )
        return id_to_tup
    
    def train_test_split(self, sample_ids=[], test_size=None, random_state=32):
        """
        stores df and dict versions of train and test split as instance attributes
        
        attrib names:
        train_df, train_dict, test_df, test_dict
        
        params:
            sample_ids (list): 
                if given list of sample ids, use those samples as test set
            test_size (int or float): 
                test ratio if float, test count if int
        """
        if sample_ids:
            # Create test set
            try:
                test_dict = {sample_id: self.data_dict[sample_id] for sample_id in sample_ids}
            except KeyError as e:
                print(f'{e}: one or more input sample_ids not found in microbiome dictionary keys')
                print('Check input sample_ids and try again')
                return
            
            test_df = pd.DataFrame(index=self.data_df.index)
            for sample_id in sample_ids:
                sample_df = self.data_df.loc[:,self.data_df.columns.str.contains(sample_id)]
                test_df = pd.concat([test_df,sample_df],axis=1)
            
            # Create training set
            train_dict = {k: v for k, v in self.data_dict.items() if k not in sample_ids}
            
            cols_to_remove=[]
            for sample_id in sample_ids:
                cols_to_remove += list(self.data_df.columns[self.data_df.columns.str.contains(sample_id)])            
            train_df = self.data_df.drop(columns=cols_to_remove)

            # store train_df, train_dict, test_df, test_dict
            self.train_df = train_df
            self.train_dict = train_dict
            self.test_df = test_df
            self.test_dict = test_dict
        
        elif test_size:
            if isinstance(test_size, float):
                assert test_size>0 and test_size<1, "Float input range: (0, 1)"
                n_test = int(np.ceil(test_size * len(self.data_dict)))
            elif isinstance(test_size, int):
                assert test_size>0 and test_size<len(self.data_dict), "Int input range: (0, len(self.data_dict))"
                n_test = test_size
            else:
                assert isinstance(test_size, float) or isinstance(test_size, int),\
                    "Input type for 'test_size' parameter neither int nor float"

            sample_ids = list(self.data_dict.keys())
            labels = [self.data_dict[sample_id].label for sample_id in sample_ids]

            train_sample_ids, test_sample_ids = \
                train_test_split(sample_ids,
                                 test_size = n_test,
                                 stratify = labels, # *stratify based on labels
                                 random_state = random_state)

            train_dict = {sample_id: self.data_dict[sample_id] for sample_id in train_sample_ids}
            test_dict  = {sample_id: self.data_dict[sample_id] for sample_id in test_sample_ids}
            
            test_df = pd.DataFrame(index=self.data_df.index)
            for sample_id in test_sample_ids:
                sample_df = self.data_df.loc[:,self.data_df.columns.str.contains(sample_id)]
                test_df = pd.concat([test_df,sample_df],axis=1)

            cols_to_remove=[]
            for sample_id in test_sample_ids:
                cols_to_remove += list(self.data_df.columns[self.data_df.columns.str.contains(sample_id)])            
            train_df = self.data_df.drop(columns=cols_to_remove)

            # Store train_df, train_dict, test_df, test_dict
            self.train_df = train_df
            self.train_dict = train_dict
            self.test_df = test_df
            self.test_dict = test_dict 
        else:
            raise Exception(f"{self.__class__.__name__}.train_test_split:",
                            "Please provide a value for 'test_size' parameter")
    
    def get_single_class_df(self, is_label_1, remove_missing_tps=False, from_train=False):
        """
        return class=1 or class=0 dataframe from train dataset
        
        is_label_1 (bool): 
            True to retrieve label 1 sample, False to retrive label 0 sample

        remove_missing_tps (bool):
            remove columns with missing data if True
            * missing timepoints (columns) contain "NA" in column names

        *When runing for DMM => data should not contrain 0 sum rows/columns;
            -set remove_missing_tps=True to remove missing timepoints (columns)
        """
        label = 1 if is_label_1 else 0

        # Get single class
        single_class_df = pd.DataFrame(index=self.data_df.index)
        
        # Choose from what dataset to return singel class df (train or mb df)
        if from_train:
            data_dict = self.train_dict
            data_df =self.train_df
        else:
            data_dict = self.data_dict
            data_df =self.data_df

        for sample_id, data in data_dict.items():
            if data.label == label:
                sample_df = data_df.loc[:, data_df.columns.str.contains(sample_id)]
                single_class_df = pd.concat([single_class_df,sample_df],axis=1)
        
        if remove_missing_tps:
            # remove columns with missing data
            cols_to_remove = single_class_df.columns[single_class_df.columns.str.contains('NA')]
            single_class_df = single_class_df.drop(columns=cols_to_remove)

        return single_class_df
    
    @staticmethod
    def add_pseudo_cnt(df, pseudo_cnt):
        return df + pseudo_cnt
    
    @staticmethod
    def get_idx_w_less_than_n_datapoints(df, unobserved_val=0, n=3):
        """
        unobserved_val: 
            0 entries in sparce matrix data
        """
        idx_to_drop=[]
        for r_idx, row_sr in df.iterrows():
            if len(row_sr[row_sr.ne(unobserved_val)]) < n:
                idx_to_drop.append(r_idx)
        return idx_to_drop
    
    @staticmethod
    def run_dmm(out_dir, data_path, n_comp_range_pair, save_fit_plot=False, random_state=12):
        """
        saves state assigment csv file after running DMM using R code;
        *can save laplace, aic, bic plot if save_fit_plot=True
        *chooses best fit based on laplace, not aic/bic
        
        saves to output_dir:
            1.state assignment csv file
            2.DMM fitness image (laplace/aic/bic)

        out_dir (str):
            directory where outputs will be saved 
                ex) out_dir: "data/<sample_id>/"
        in_mb_filename (str):
            microbiome csv file name
        n_comp_range_pair (tup): 
            number of components will be searched to find best DMM fit;
            e.g. (3,6) will search best fit for 3,4,5,6 components
        random_state (int): 
            random seed for reproducibility
        save_fit_plot (bool):
            saves fit plot that shows fitness (y axis) over the 
            number of components (x axis)
        """
        # Set random seed
        ro.r(f'set.seed({random_state})')
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        filename = os.path.basename(data_path)

        # ex) 'data/1033/gf.csv'
        state_assignment_out_path = out_dir+os.path.splitext(filename)[0]+'_state_assignments.csv' 
        # ex) 'data/1033/state_assignment.csv'

        # load data to R
        ro.r(f'data_path = "{data_path}"')
        ro.r(f'state_ass_path = "{state_assignment_out_path}"')
        
        # read data as matrix (taxa x samples/timepoints)
        ro.r('mb_data <- as.matrix(t(read.csv(data_path, row.names=1, check.names=FALSE)))')

        # Fit Dirichlet-Multinomial model
        n_comp_min, n_comp_max = n_comp_range_pair
        print()
        ro.r(f'fit <- lapply({n_comp_min}:{n_comp_max}, dmn, count = mb_data, verbose=TRUE)')

        # Calculate Laplace, AIC, and BIC
        ro.r('lplc <- sapply(fit, DirichletMultinomial::laplace)')

        if save_fit_plot:
            ro.r('aic  <- sapply(fit, DirichletMultinomial::AIC)')
            ro.r('bic  <- sapply(fit, DirichletMultinomial::BIC)')

            # Save fit plot to 
            ro.r('all_y_values <- c(lplc, aic, bic)')
            ro.r(f'png("{out_dir+os.path.splitext(filename)[0]}_fit_plot.png", width = 800, height = 600)')
            ro.r(f'plot(seq({n_comp_min}, {n_comp_max}), lplc, type="b", lty = 2, col="red", \
                 ylim = range(all_y_values), xlim = c({n_comp_min}, {n_comp_max}), \
                    xlab="Number of Dirichlet Components", ylab="Model Fit")')
            ro.r(f'lines(seq({n_comp_min}, {n_comp_max}), aic, type="b", lty = 2, col="blue")')
            ro.r(f'lines(seq({n_comp_min}, {n_comp_max}), bic, type="b", lty = 2, col="green")')
            ro.r('legend("topright",legend=c("lplc","aic","bic"),fill=c("red","blue","green"),box.lty=0,cex=1.5)')
            ro.r('dev.off()')

        # Get the best fit model; minimized Laplace
        ro.r('best <- fit[[which.min(unlist(lplc))]]')

        # Generate state assignments
        ro.r('ass <- apply(mixture(best), 1, which.max)')

        # Write state assignments to CSV
        ro.r(f'write.table(ass, state_ass_path, col.names = FALSE, sep = ",")')
        
    @staticmethod
    def normalize_n_clr(df, norm_col=True):
        """
        return 
        1. normalized & center log ratio (clr) data
            -convert data to probability by samples(tp)
            -apply clr transformation
        2.  mask for the DataFrame where each element 
        is True if it's non-zero in the original sparce data else False

        if norm_col = True
            normalize and apply clr columns
        else
            normalize and apply clr rows
        """
        
        if norm_col:
            # convert data to proability by column
            col_sums = df.sum(axis=0)
            df_prob = df.div(col_sums, axis='columns')

            # clr transformation
            df_log = np.log(df_prob)
            df_clr = df_log.sub(df_log.mean(axis=0), axis='columns')
            return df_clr
        
        else:
            # convert data to proability by column
            row_sums = df.sum(axis=1)
            df_prob = df.div(row_sums, axis='index')

            # clr transformation
            df_log = np.log(df_prob)
            df_clr = df_log.sub(df_log.mean(axis=1),axis='index')
            return df_clr
