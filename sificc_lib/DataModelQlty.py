import numpy as np
from sificc_lib import utils

class DataModelQlty():
    '''Data model for the features and targets to train SiFi-CC Quality 
    Neural Network. The training data should be generated seperately 
    from a trained SiFi-CC Neural Network.
    
    Features R_n*(9*clusters_limit) format: {
        cluster entries, 
        cluster energy, 
        cluster energy uncertainty, 
        cluster position (x,y,z), 
        cluster position uncertainty (x,y,z) 
    } * clusters_limit
        
    Targets R_n*11 format: {
        event type (is ideal Compton or not),
        e energy,
        p energy,
        e position (x,y,z),
        p position (x,y,z),
        e cluster index,
        p cluster index,
    }
    
    Reco R_n*9 format: {
        event type (is ideal Compton or not),
        e energy,
        p energy,
        e position (x,y,z),
        p position (x,y,z),
    }
    
    Quality R_n*4 format: {
        e energy quality,
        p energy quality,
        e position quality,
        p position quality,
    }
    '''
    def __init__(self, file_name, *, batch_size = 64, validation_percent = .05, test_percent = .1, 
                 weight_compton = 1, weight_non_compton = 1):
        self.__validation_percent = validation_percent
        self.__test_percent = test_percent
        self.batch_size = batch_size
        self.weight_compton = weight_compton
        self.weight_non_compton = weight_non_compton
        
        self.cluster_size = 9
        self.append_dim = True
        
        self.__std_factor = 15
        self.__balanced_training = False
        
        # loading training matrices
        with open(file_name, 'rb') as f_train:
            npz = np.load(f_train)
            self._features = npz['features']
            self._targets = npz['targets']
            self._reco = npz['reco']
            self._seq = npz['sequence']
            self._qlty = npz['quality']
            
        # assert number of columns is correct
        assert self._features.shape[1] % self.cluster_size == 0
        
        # define clusters limit
        self.clusters_limit = self._features.shape[1] // self.cluster_size
        
        #normalize features, targets, and reco
        self._features = (self._features - self.__mean_features) / self.__std_features
        self._targets = (self._targets - self.__mean_targets) / self.__std_targets
        self._reco = (self._reco - self.__mean_targets[:-2]) / self.__std_targets[:-2]
        
        # compute the starting position of the validation and test sets
        self.validation_start_pos = int(self.length * (1-self.validation_percent-self.test_percent))
        self.test_start_pos = int(self.length * (1-self.test_percent))
        
    def _denormalize_features(self, data):
        if data.shape[-1] == self._features.shape[-1]:
            return (data * self.__std_features) + self.__mean_features
        raise Exception('data has invalid shape of {}'.format(data.shape))
    
    def _denormalize_targets(self, data):
        if data.shape[-1] == self._targets.shape[-1]:
            return (data * self.__std_targets) + self.__mean_targets
        elif data.shape[-1] == self._reco.shape[-1]:
            return (data * self.__std_targets[:-2]) + self.__mean_targets[:-2]
        else:
            raise Exception('data has invalid shape of {}'.format(data.shape))
    
    def normalize_targets(self, data):
        if data.shape[-1] == self._targets.shape[-1]:
            return (data - self.__mean_targets) / self.__std_targets
        elif data.shape[-1] == self._reco.shape[-1]:
            return (data - self.__mean_targets[:-2]) / self.__std_targets[:-2]
        else:
            raise Exception('data has invalid shape of {}'.format(data.shape))
    
    def get_targets_dic(self, start=None, end=None):
        start = start if start is not None else 0
        end = end if end is not None else self.length
        
        return {
            'type': self._target_type[start:end],
            'e_cluster': self._target_e_cluster[start:end],
            'p_cluster': self._target_p_cluster[start:end],
            'pos_x': self._target_pos_x[start:end],
            'pos_y': self._target_pos_y[start:end],
            'pos_z': self._target_pos_z[start:end],
            'energy': self._target_energy[start:end],
            'quality': self._target_qlty[start:end]
        }
    
    def get_features(self, start=None, end=None):
        start = start if start is not None else 0
        end = end if end is not None else self.length
        
        if self.append_dim:
            return self._features[start:end].reshape((-1, self._features.shape[1], 1))
        else:
            return self._features[start:end]
        
    def shuffle(self, only_train=True):
        # if balancing the data is activated, select another random sample from the 
        # background events 
        if self.__balanced_training:
            non_comptons = np.random.choice(self.__background_pool, self.__n_comptons)
            index = np.concatenate([non_comptons, self.__base_index], axis=0)
            self._features = self.__features_all[index]
            self._targets = self.__targets_all[index]
            self._reco = self.__reco_all[index]
            self._seq = self.__seq_all[index]
            self._qlty = self.__qlty_all[index]
            
        limit = self.validation_start_pos if only_train else self.length
        sequence = np.arange(self.length)
        sequence[:limit] = np.random.permutation(limit)
        
        self._features = self._features[sequence]
        self._targets = self._targets[sequence]
        self._reco = self._reco[sequence]
        self._seq = self._seq[sequence]
        self._qlty = self._qlty[sequence]
        
    @property
    def steps_per_epoch(self):
        return int(np.ceil(self.validation_start_pos/self.batch_size))
    
    @property
    def balance_training(self):
        '''Balance the samples in the training set in order to make the number of Compton
        samples equal to the number of background samples. Default value is False.'''
        return self.__balanced_training
    
    @balance_training.setter
    def balance_training(self, value):
        # when balancing is activated
        if (not self.__balanced_training) and (value==True):
            # copy the original datasets
            self.__features_all = self._features.copy()
            self.__targets_all = self._targets.copy()
            self.__reco_all = self._reco.copy()
            self.__seq_all = self._seq.copy()
            self.__qlty_all = self._qlty.copy()
            
            # compute the list of background events to choose a sample from and
            # the list of base features (all the comptons + validation set + test set)
            train_type = self.train_y['type'].ravel().astype(bool)
            self.__n_comptons = train_type.sum()
            comptons = np.where(train_type)[0]
            self.__background_pool = np.where(~train_type)[0]
            valid_test = np.arange(self.validation_start_pos, self.length)
            self.__base_index = np.concatenate([comptons, valid_test], axis=0)
            
            # select a sample from the background and add it to the base features to
            # compose a balanced training set
            non_comptons = np.random.choice(self.__background_pool, self.__n_comptons)
            index = np.concatenate([non_comptons, self.__base_index], axis=0)
            self._features = self.__features_all[index]
            self._targets = self.__targets_all[index]
            self._reco = self.__reco_all[index]
            self._seq = self.__seq_all[index]
            self._qlty = self.__qlty_all[index]
            
            # fix the position of the validation and test starting positions
            diff = self.__targets_all.shape[0] - self._targets.shape[0]
            self.validation_start_pos = self.validation_start_pos - diff
            self.test_start_pos = self.test_start_pos - diff
            
            # shuffle the training part
            self.shuffle(only_train=True)
            
        # when balancing is deactivated
        elif self.__balanced_training and (value==False):
            # compute the difference in size
            diff = self.__targets_all.shape[0] - self._targets.shape[0]
            
            # restore the original values
            self._features = self.__features_all
            self._targets = self.__targets_all
            self._reco = self.__reco_all
            self._seq = self.__seq_all
            self._qlty = self.__qlty_all
            
            # fix the position of the validation and test starting positions
            self.validation_start_pos = self.validation_start_pos - diff
            self.test_start_pos = self.test_start_pos - diff
            
        self.__balanced_training = value
        
    
    def generate_batch(self, augment=False):
        while True:
            self.shuffle(only_train=True)

            for step in range(self.steps_per_epoch):
                start = step * self.batch_size
                end = (step+1) * self.batch_size
                # end should not enter the validation range
                end = end if end <= self.validation_start_pos else self.validation_start_pos
                
                features_batch = self.get_features(start, end)
                targets_batch = self.get_targets_dic(start, end)
                
                if augment:
                    sequence, expanded_sequence = self.__get_augmentation_sequence()
                    features_batch = features_batch[:,expanded_sequence]
                    targets_batch['e_cluster'][:,1] = np.where(np.equal(targets_batch['e_cluster'][:,[1]], sequence))[1]
                    targets_batch['p_cluster'][:,1] = np.where(np.equal(targets_batch['p_cluster'][:,[1]], sequence))[1]
                
                yield (
                    features_batch, 
                    targets_batch, 
                    targets_batch['type'] * self.weight_compton + \
                        (1-targets_batch['type']) * self.weight_non_compton
                )
        
    def __get_augmentation_sequence(self):
        num_clusters = self.clusters_limit
        sequence = np.random.permutation(num_clusters)
        expanded_sequence = np.repeat(sequence * self.cluster_size, self.cluster_size) + \
                            np.tile(np.arange(self.cluster_size), num_clusters)
        return sequence, expanded_sequence
    
    def shuffle_training_clusters(self):
        # e_pos = 9
        # p_pos = 10
        for i in range(self.length):
            sequence, expanded_sequence = self.__get_augmentation_sequence()
            self._features[i] = self._features[i, expanded_sequence]
            self._targets[i,9] = np.where(np.equal(self._targets[i,9], sequence))[0]
            self._targets[i,10] = np.where(np.equal(self._targets[i,10], sequence))[0]
    
    ################# Properties #################
    @property
    def length(self):
        return self._targets.shape[0]
    
    @property
    def validation_percent(self):
        return self.__validation_percent
    
    @property 
    def test_percent(self):
        return self.__test_percent
    
    @property
    def train_x(self):
        return self.get_features(None, self.validation_start_pos)
    
    @property
    def train_y(self):
        return self.get_targets_dic(None, self.validation_start_pos)
    
    @property
    def train_row_y(self):
        return self._targets[:self.validation_start_pos]
    
    @property
    def validation_x(self):
        return self.get_features(self.validation_start_pos, self.test_start_pos)
    
    @property
    def validation_y(self):
        return self.get_targets_dic(self.validation_start_pos, self.test_start_pos)
    
    @property
    def validation_row_y(self):
        return self._targets[self.validation_start_pos: self.test_start_pos]
    
    @property
    def test_x(self):
        return self.get_features(self.test_start_pos, None)
    
    @property
    def test_y(self):
        return self.get_targets_dic(self.test_start_pos, None)
    
    @property
    def test_row_y(self):
        return self._targets[self.test_start_pos:]
    
    @property
    def reco_valid(self):
        return self._reco[self.validation_start_pos: self.test_start_pos]
    
    @property
    def reco_test(self):
        return self._reco[self.test_start_pos:]
    
    @property
    def __mean_features(self):
        # define normalization factors
        mean_entries = [1.7874760910930447]
        mean_energies = [1.3219832176828306]
        mean_energies_unc = [0.03352665535144364]
        mean_positions = [3.08466733e+02, 8.30834656e-02, -8.41913642e-01]
        mean_positions_unc = [1.05791671, 12.8333989, 0.94994155]
        
        # declare the mean of a single cluster and repeat it throughout the clusters
        mean = np.concatenate((
            mean_entries, 
            mean_energies, 
            mean_energies_unc, 
            mean_positions, 
            mean_positions_unc
        ))
        mean = np.tile(mean, self.clusters_limit)
        return mean
    
    @property
    def __std_features(self):
        # define normalization factors
        std_entries = [1.6479899119958636]
        std_energies = [1.8812291744163367]
        std_energies_unc = [0.025137531990537407]
        std_positions = [97.44675577, 30.56710605, 27.5600849]
        std_positions_unc = [1.01437355, 6.11019272, 0.76225179]
        
        std = np.concatenate((
            std_entries, 
            std_energies, 
            std_energies_unc, 
            std_positions, 
            std_positions_unc
        ))
        std = np.tile(std, self.clusters_limit)
        return std
    
    @property
    def __mean_targets(self):
        mean_e_energy = [1.207963305458394]
        mean_p_energy = [2.081498278344268]
        mean_e_position = [2.02256879e+02, 1.00478623e-02, -3.36698613e+00]
        mean_p_position = [3.93714750e+02, 1.02343097e-01, 1.31962800e+00]
        
        mean = np.concatenate((
            [0],
            mean_e_energy, 
            mean_p_energy,
            mean_e_position,
            mean_p_position,
            [0,0]
        ))
        return mean
    
    @property
    def __std_targets(self):
        std_e_energy = [1.7854439595674854] / np.array(self.__std_factor)
        std_p_energy = [1.675908762593649] / np.array(self.__std_factor)
        std_e_position = [20.45301063, 27.74893174, 27.19126733] / np.array(self.__std_factor)
        std_p_position = [23.59772062, 28.41093766, 28.10100634] / np.array(self.__std_factor)

        std = np.concatenate((
            [1],
            std_e_energy, 
            std_p_energy,
            std_e_position,
            std_p_position,
            [1,1]
        ))
        return std
    
    
    @property
    def _target_type(self):
        # [t]
        return self._targets[:,[0]]
    
    @property
    def _target_e_cluster(self):
        # [t, e_clus]
        return self._targets[:,[0,9]]
    
    @property
    def _target_p_cluster(self):
        # [t, p_clus]
        return self._targets[:,[0,10]]
    
    @property
    def _target_pos_x(self):
        # [t, e_clus, e_pos_x, p_clus, p_pos_x]
        return self._targets[:,[0,9,3,10,6]]
    
    @property
    def _target_pos_y(self):
        # [t, e_clus, e_pos_y, p_clus, p_pos_y]
        return self._targets[:,[0,9,4,10,7]]
    
    @property
    def _target_pos_z(self):
        # [t, e_clus, e_pos_z, p_clus, p_pos_z]
        return self._targets[:,[0,9,5,10,8]]
    
    @property
    def _target_energy(self):
        # [t, e_enrg, p_enrg]
        return self._targets[:,[0,1,2]]
    
    @property
    def _target_qlty(self):
        # [t, e_energy_qlty, p_energy_qlty, e_pos_qlty, p_pos_qlty]
        return np.concatenate([self._targets[:,[0]], self._qlty], axis=1)
    