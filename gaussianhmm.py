# Gaussian Emission HMM
import pandas as pd
import numpy as np
from types import SimpleNamespace

from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from mylogger import logger


class InfantMBGaussianHMM:
    def __init__(self, n_states=3, n_iter=10, threshold=0.01, name="infant_hmm", random_state=34):
        self.random_state = random_state
        self.name = name

        self.n_states = n_states
        self.n_feat = None

        self.data_df = None
        self.data_dict = None
        
        self.eps = 1e-10
        self.initial = None # (np arr)
        self.transition = None # (np arr)
        self.emission = None # (list of SimpleNamespace(mean=np arr,cov=np arr))

        self.inference_var = SimpleNamespace(alpha={}, beta={}, gamma={}, xi={})
        # alpha(forward)/beta(backward)/gamma(posterior)/xi(posterior_transition)
        # all => (dict[str, np arr])

        self.curr_iter = 0
        self.n_iter = n_iter
        
        self.loglikelihood = None # current log likelihood
        self.loglikelihood_history = [] # log likelihood history
        self.threshold = threshold
    
    def __str__(self):
            return f"<Instance '{self.name}'>"
    
    def initialize_hmm_params_from_state_assignments(self, data_df, sa_path):
        self.data_df = data_df
        self.n_feat = data_df.shape[0]
        
        # read in state assignments
        self.sa_df = pd.read_csv(sa_path, header=None, index_col=0).T
        assert self.sa_df.T.value_counts().ne(0).all(), \
            "Some states have 0 counts; try running DMM with less number of components."
        
        self.create_state_assignments_dict()
        self.calc_initial_prob_from_state_assignments()
        self.calc_transition_prob_from_state_assignments()
        self.calc_emission_mean_n_var_from_state_assignments()

    def create_state_assignments_dict(self):
        """
        return state assignment dict from state assigment df (taxa x sample):
            dict[str, np arr[int]]
            keys: infant ids
            values: state sequence
        """
        # Output datset dict container
        id_to_ls={}

        # Create a set for str infant ids
        sample_id_set=set()
        for column in list(self.sa_df):
            # Get infant id
            try:
                idx = column.find('_')
                id_str = column[:idx]
            except:
                print(f"Wrong column name format:{column}. \
                    Correct format: <infant id(int)>_<hospital site code(str)>_<post menstral age(int)>")
            else:
                sample_id_set.add(id_str)
        
        for id_str in sample_id_set:
            # Create infant data for each id
            if (id_to_ls.get(id_str)) is None:
                sample_sa_df = self.sa_df.loc[:, self.sa_df.columns.str.contains(id_str)]
                # store state seq as np arr
                id_to_ls[id_str] = sample_sa_df.to_numpy()[0]
        
        self.sa_dict = id_to_ls

    def calc_initial_prob_from_state_assignments(self):
        i = np.zeros(self.n_states)
        
        for state_seq in self.sa_dict.values():
            first_state = state_seq[0]-1 
            # subtract 1 to all states because DMM outputs states is 1-indexing not 0
            i[first_state]+=1
        
        i = i/len(self.sa_dict)
        assert np.allclose(np.sum(i), 1.0) ,"Error; initial vector sum!=1."
        i[i==0] = self.eps
        self.initial = np.log(i)
    
    def calc_transition_prob_from_state_assignments(self):
        t = np.zeros((self.n_states, self.n_states))
        
        for state_seq in self.sa_dict.values():
            # subtract 1 to all states because DMM outputs states is 1-indexing not 0
            state_seq = state_seq-1
            
            for idx, curr_state in enumerate(state_seq):
                # total count of each state to normalize freq of transition
                # except last timepoint where no transition occurs
                if idx!=len(state_seq)-1:
                    # count transitions
                    next_tp_state = state_seq[idx+1]
                    t[curr_state][next_tp_state]+=1
            
        # normalize freq of each state with total cnts
        t = t/np.sum(t, axis=1)[:, None]
        assert np.allclose(np.sum(t, axis=1), 1.0) ,"Error; transition matrix sum(row(s))!=1."
        t[t==0] = self.eps
        self.transition = np.log(t)
        
    def calc_emission_mean_n_var_from_state_assignments(self):
        e = [SimpleNamespace(mean=None, cov=None) for _ in range(self.n_states)]
        
        no_missing_tp_col_names = self.data_df.columns[~self.data_df.columns.str.contains('NA')]
        assert (no_missing_tp_col_names==self.sa_df.columns).all(), \
                f"<label>.csv and <label>_state_assignments.csv columns do not match;"\
                "Please check column naming format in InfantMicrobiomeDataset constructor in dataset.py file."

        # store data by state in list
        data_per_state = [pd.DataFrame(index=self.data_df.index) for _ in range(self.n_states)]

        for col_name in self.sa_df.columns:
            # subtract 1 to state because DMM output is 1-indexed not 0
            sample_state = self.sa_df[col_name].iat[0]-1
            data_per_state[sample_state] = pd.concat([data_per_state[sample_state], self.data_df[col_name]], axis=1)
            
        # calc mean and var of taxas by state and store to e
        for state, state_df in enumerate(data_per_state):
            e[state].mean = state_df.mean(axis=1).to_numpy()
            e[state].cov = np.diag(np.var(state_df.to_numpy(), axis=1, ddof=1))

        self.emission = e

    def calc_alpha_n_beta(self, data_dict:dict):
        """
        calculate alpha & beta (dict[str, np.arr]) of all 
        samples in data_dict using instance initial/transition/emisison
        
        data_dict: 
            sample_id to data dict
            dict[str, SimpleNamespace(data, label, mask)]

        *alpha & beta arr => log scale
        """
        alpha_dict = {}
        beta_dict = {}
        
        for sample_id, sample in data_dict.items():
            data = sample.data # (tp x n_feat)
            mask = sample.mask # (tp) False if missing tp data, else True
            n_tp = data.shape[0]

            # initialize alpha arr
            alpha_arr = np.full((self.n_states, n_tp), -np.inf)
            alpha_arr[:,0] = self.initial + np.array([self.gauss[st].logpdf(data[0]) for st in range(self.n_states)])
            
            # calc alpha
            for tp in range(1, n_tp):
                if mask[tp]:
                    tp_emission = np.array([self.gauss[st].logpdf(data[tp]) for st in range(self.n_states)])
                else:
                    tp_emission = np.zeros((self.n_states))
                
                alpha_arr[:,tp] = logsumexp(self.transition + alpha_arr[:,tp-1].reshape(-1,1), axis=0) + tp_emission            
            
            alpha_dict[sample_id] = alpha_arr
            
            # initialize beta arr
            beta_arr = np.full((self.n_states, n_tp), -np.inf)
            beta_arr[:,n_tp-1] = 0 # log(1)=0
            
            # calc beta
            for tp in range(n_tp-2, -1, -1):
                if mask[tp+1]:
                    nexttp_emission = np.array([self.gauss[st].logpdf(data[tp+1]) for st in range(self.n_states)])  
                else:
                    nexttp_emission = np.zeros((self.n_states))
                    
                beta_arr[:,tp] = logsumexp(self.transition + beta_arr[:,tp+1] + nexttp_emission, axis=1)
            
            beta_dict[sample_id]=beta_arr

        return alpha_dict, beta_dict

    def calc_gamma_n_xi(self, data_dict):
        """
        update gamma & xi using alpha & beta

        gamma: posterior state probabiliy
        xi: posterior transition probability

        data_dict: 
            sample_id to data dict
            dict[str, SimpleNamespace(data, label, mask)]
        """
        alpha_dict = self.inference_var.alpha
        beta_dict = self.inference_var.beta
         
        gamma = self.inference_var.gamma
        xi = self.inference_var.xi
        
        for sample_id, sample in data_dict.items():
            data = sample.data
            mask = sample.mask # (tp) False if missing tp data, else True

            alpha = alpha_dict[sample_id] # (n_state, tp)
            beta = beta_dict[sample_id] 
            alpha_beta_sum = alpha+beta
            
            # calc posterior state prob
            gamma_arr = alpha_beta_sum.copy()
            gamma_arr -= logsumexp(gamma_arr, axis=0)
            gamma[sample_id] = gamma_arr
            
            # calc posterior transition prob
            n_tp = data.shape[0]
            xi_arr = np.full((n_tp-1, self.n_states, self.n_states), -np.inf)

            for tp in range(n_tp-1):
                if mask[tp+1]:
                    nexttp_emission = np.array([self.gauss[st].logpdf(data[tp+1]) for st in range(self.n_states)])
                else:
                    nexttp_emission = np.zeros((self.n_states))
                 
                tp_tr_mtx = alpha[:,tp].reshape(-1,1) + self.transition + nexttp_emission + beta[:,tp+1]
                tp_tr_mtx -= logsumexp(alpha_beta_sum[:,tp])

                xi_arr[tp,:,:] = tp_tr_mtx
                
            xi[sample_id] = xi_arr

    def e_step(self, data_dict):
        """
        perform e step, updating alpha, beta, gamma and xi
        using current HMM params in inference_var attribute
        
        *alpha, beta, gamma and xi are stored in 
        inference_var attribute (SimpleNamespace(alpha,beta,gamma,xi))

        data_dict: 
            sample_id to data dict
            dict[str, SimpleNamespace(data, label, mask)]
        """
        assert self.data_dict is not None, \
            "data_dict attrib missing; use InfantMicrobiomeDataset "\
            "instance method create_infant_data_dict(data_df) to create data_dict"
        
        # create scipy's multivariate normal instances per state (updated each iteration of EM algo)
        self.gauss = [multivariate_normal(mean=self.emission[state].mean, cov=self.emission[state].cov) 
                      for state in range(self.n_states)]
            
        alpha_dict, beta_dict = self.calc_alpha_n_beta(data_dict)
        self.inference_var.alpha = alpha_dict
        self.inference_var.beta = beta_dict
        
        self.calc_gamma_n_xi(data_dict)

    def m_step_with_assigned_state(self, data_dict):
        """
        calculate initial/transition/emission 
        using state assigned data.
        => divide data based on states & calculate HMM params
        """
        sa_dict = self.get_state_path(data_dict, is_viterbi=False)
        df = self.data_df
        i=np.zeros((self.n_states))
        t=np.zeros((self.n_states,self.n_states))
        e=[SimpleNamespace(mean=None,cov=None) for _ in range(self.n_states)]
        data_per_state = [pd.DataFrame(index=df.index) for _ in range(self.n_states)]

        for sample_id, sa_arr in sa_dict.items():
            sa_arr-=1
            #initial
            i[sa_arr[0]]+=1
            
            #transition
            for idx, curr_state in enumerate(sa_arr):
                # total count of each state to normalize freq of transition
                # except last timepoint where no transition occurs
                if idx!=len(sa_arr)-1:
                    # count transitions
                    next_tp_state = sa_arr[idx+1]
                    t[curr_state][next_tp_state]+=1
            
            # emission
            sample_df = df.loc[:,df.columns[df.columns.str.contains(sample_id)]]
            assert len(sa_arr)==len(list(sample_df))

            for state, col in zip(sa_arr, sample_df.columns):
                data_per_state[state] = pd.concat([data_per_state[state], sample_df[col]],axis=1)
        
        # self.tmp_ls = [None for _ in range(self.n_states)]
        for state, st_df in enumerate(data_per_state):
            # logger.info(f"Data shape in state<{state+1}>: {st_df.shape}")
            # self.tmp_ls[state] = st_df
            
            e[state].mean = st_df.mean(axis=1).to_numpy()
            e[state].cov = np.diag(st_df.var(axis=1))
        
        i = i/np.sum(i)
        assert np.allclose(np.sum(i), 1.0) ,"Error; initial vector sum!=1."
        i[i==0] = self.eps
        self.initial = np.log(i)

        # normalize freq of each state with total cnts
        t = t/np.sum(t, axis=1)[:, None]
        assert np.allclose(np.sum(t, axis=1), 1.0) ,"Error; transition matrix sum(row(s))!=1."
        t[t==0] = self.eps
        self.transition = np.log(t)

        self.emission = e

    def check_updated_emission(self):
        """
        checks if any covariance matrix is all zero
        """
        for state in range(self.n_states):
            cov = self.emission[state].cov
            if np.all(cov == 0):
                raise ValueError(f"Current Iter: <{self.curr_iter}> State:<{state+1}>'s covariance matrix is all zeros!")

    def m_step(self, data_dict):
        """
        perform m step by either 
            1.using gamma as weight
            2.using data assigned with states using gamma
        and update initial/transition/emission attributes.
        """
        self.m_step_with_assigned_state(data_dict)
            
        self.check_updated_emission()
    
    def calc_internal_loglikelihood(self):
        """
        Loop alpha attribute & calculate P(O)
        to estimate model loglikelihood
        """
        self.loglikelihood = 0
        for alpha in self.inference_var.alpha.values():
            self.loglikelihood += logsumexp(alpha[:, -1])

    def record_likelihood(self):
        """
        1.check if log likelihood is increasing/dropping
        2.record log likelihood & current iteration number
        """
        tolerance = np.finfo(float).eps**(0.5)
        if self.loglikelihood_history:
            if self.loglikelihood-self.loglikelihood_history[-1] < -tolerance:
                logger.info(f"Curr iter:{self.curr_iter} log likelihood dropped; model not converging.\n"\
                            f"Current log likelihood: <{self.loglikelihood}>.\n"\
                            f"Current should be bigger than last: <{self.loglikelihood_history[-1]}>.")

        self.loglikelihood_history.append(self.loglikelihood)
        self.curr_iter += 1

    def check_convegence(self):
        """
        return True if finished iterations OR LL difference is smaller than threshold 
        else return False
        """
        if self.curr_iter == self.n_iter:
            logger.info("Max iteration reached. Stopping EM algorithm iterations.\n"\
                        f"Current log likelihood: <{self.loglikelihood}>\n")
            return True
        elif (len(self.loglikelihood_history) > 1) and \
            (self.loglikelihood_history[-1] - self.loglikelihood_history[-2] < self.threshold):
            logger.info("Stopping EM algorithm iterations.\n")
            return True
        else:
            return False 
    
    def fit(self, data_dict):
        """
        run em algorithm iteration loops

        data_dict: data HMM model will be fit on
            dict[str(sample id), SimpleNamespace(data, label, mask)]
        """
        logger.info(f"Fitting <{self.name}>")
        for i in range(self.n_iter):
            logger.info(f"EM algorithm's {i}_th iteration")
            
            self.e_step(data_dict)
            
            self.calc_internal_loglikelihood()
            self.record_likelihood()            

            # check likelkihood using calcuated alpha before updating HMM parameters
            if self.check_convegence():
                break
            self.m_step(data_dict)
            
            # check if any state transition probs sum to 0
            if (zero_st_idx := self.transition.sum(axis=1)==0).any():
                logger.info(f"Zero transition prob sum state(s) encountered: {np.where(zero_st_idx)[0]+1}")
    
    def viterbi_decoding(self, data_dict):
        
        def backtrack(vit_arr):
            n_tp = vit_arr.shape[1]
            state_path_arr = np.zeros(n_tp)
            for tp in range(n_tp-1,-1,-1):
                state_path_arr[tp] = np.argmax(vit_arr[:,tp])+1
            return state_path_arr
        
        viterbi_path_dict = {}
        
        for sample_id, sample in data_dict.items():
            data = sample.data # (tp x n_feat)
            mask = sample.mask # (tp) False if missing tp data, else True
            n_tp = data.shape[0]

            # initialize vit_arr
            vit_arr = np.full((self.n_states, n_tp), -np.inf)
            vit_arr[:,0] = self.initial + np.array([self.gauss[st].logpdf(data[0]) for st in range(self.n_states)])
            
            for tp in range(1, n_tp):
                for curr_st in range(self.n_states):
                    if mask[tp]:
                        tp_st_emission = self.gauss[curr_st].logpdf(data[tp])
                    else:
                        tp_st_emission = 0

                    tmp_arr = np.zeros(self.n_states)
                    
                    for prev_st in range(self.n_states):
                        tmp_arr[prev_st] = tp_st_emission + \
                                            self.transition[prev_st,curr_st] + \
                                            vit_arr[prev_st,tp-1].reshape(-1,1)
                        
                    vit_arr[curr_st,tp] = tmp_arr.max()

            viterbi_path_dict[sample_id] = backtrack(vit_arr)

        return viterbi_path_dict

    def get_state_path_from_gamma(self, data_dict):
        """
        return sample_id to state_path dict (dict[str, np arr])
        *calculates alpha, beta & gamma using current initial/transition/emission

        data_dict: 
        """
        alpha_dict, beta_dict = self.calc_alpha_n_beta(data_dict)
        
        id_to_state_dict = {}
        
        for sample_id in data_dict.keys():
            alpha = alpha_dict[sample_id] # (n_state, tp)
            beta = beta_dict[sample_id] 
            
            # calc posterior state prob
            gamma_arr = alpha+beta
            gamma_arr -= logsumexp(gamma_arr, axis=0)

            # most probable state at each timepoint
            id_to_state_dict[sample_id] = (np.argmax(gamma_arr, axis=0) + 1)

        return id_to_state_dict

    def get_state_path(self, data_dict, is_viterbi=False):
        """
        return most probable state path given HMM parameters
        stored in sample_id to state_path dict (dict[str, np arr])
        """
        if is_viterbi:
            sa_dict = self.viterbi_decoding(data_dict)
        else:
            sa_dict = self.get_state_path_from_gamma(data_dict)
        return sa_dict

    def calc_loglikelihood(self, data_dict):
        """
        calculate alpha & calculate P(O) to estimate 
        loglikelihood of the model given the data (data_dict)
        """
        # calc_alpha using HMM params
        alpha_dict, _ = self.calc_alpha_n_beta(data_dict)
        
        sample_loglikelihood ={}

        for sample_id, alpha in alpha_dict.items():
            sample_loglikelihood[sample_id] = logsumexp(alpha[:, -1])
        return sample_loglikelihood