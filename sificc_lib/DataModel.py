import numpy as np
from sificc_lib import utils

class DataModel():
    '''Data model of the features and targets for the simulated data.
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
    '''
    def __init__(self, file_name, *, batch_size = 64, validation_percent = .05, test_percent = .1, 
                 weight_compton = 1, weight_non_compton = .75):
        self.validation_percent = validation_percent
        self.test_percent = test_percent
        self.batch_size = batch_size
        self.weight_compton = weight_compton
        self.weight_non_compton = weight_non_compton
        
        self.cluster_size = 9
        self.append_dim = True
        
        self.__std_factor = 10
        
        # loading training matrices
        with open(file_name, 'rb') as f_train:
            npz = np.load(f_train)
            self._features = npz['features']
            self._targets = npz['targets']
            self._reco = npz['reco']
            self._seq = npz['sequence']
            
        # assert number of columns is correct
        assert self._features.shape[1] % self.cluster_size == 0
        
        # define clusters limit
        self.clusters_limit = self._features.shape[1] // self.cluster_size
        
        # define number of events
        self.length = self._targets.shape[0]
        
        #normalize features, targets, and reco
        self._features = (self._features - self.__mean_features) / self.__std_features
        self._targets = (self._targets - self.__mean_targets) / self.__std_targets
        self._reco = (self._reco - self.__mean_targets[:-2]) / self.__std_targets[:-2]
        
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
            'energy': self._target_energy[start:end]
        }
    
    def get_features(self, start=None, end=None):
        start = start if start is not None else 0
        end = end if end is not None else self.length
        
        if self.append_dim:
            return self._features[start:end].reshape((-1, self._features.shape[1], 1))
        else:
            return self._features[start:end]
        
    def shuffle(self, only_train=True):
        limit = self.validation_start_pos if only_train else self.length
        sequence = np.arange(self.length)
        sequence[:limit] = np.random.permutation(limit)
        
        self._features = self._features[sequence]
        self._targets = self._targets[sequence]
        self._reco = self._reco[sequence]
        self._seq = self._seq[sequence]
        
    @property
    def steps_per_epoch(self):
        return int(np.ceil(self.validation_start_pos/self.batch_size))
    
    def generate_batch(self, shuffle=True, augment=False):
        while True:
            if shuffle:
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
    def validation_start_pos(self):
        return int(self.length * (1-self.validation_percent-self.test_percent))
    
    @property
    def test_start_pos(self):
        return int(self.length * (1-self.test_percent))
    
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
        mean_entries = [2.040231921404413]
        mean_energies = [1.463238041955734]
        mean_energies_unc = [0.056992982647403614]
        mean_positions = [3.05663006e+02, 2.58387064e-01, -9.36406347e-01]
        mean_positions_unc = [1.18703742, 13.13392672, 0.99326574]
        
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
        std_entries = [2.0368607586297127]
        std_energies = [2.1517674544081133]
        std_energies_unc = [0.03662857474288101]
        std_positions = [96.10447476, 24.62908853, 27.47497502]
        std_positions_unc = [1.24972692, 10.70676995, 0.83927992]
        
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
        mean_e_energy = [1.1569136787161725]
        mean_p_energy = [1.9273711829259783]
        mean_e_position = [209.63565735, -0.23477532, -5.38639807]
        mean_p_position = [3.85999635e+02, 1.30259990e-01, 2.13816374e+00]
        
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
        std_e_energy = [1.78606941263188] / np.array(self.__std_factor)
        std_p_energy = [1.6663689936376904] / np.array(self.__std_factor)
        std_e_position = [41.08060207, 20.77702422, 27.19018651] / np.array(self.__std_factor)
        std_p_position = [43.94193657, 27.44766386, 28.21021386] / np.array(self.__std_factor)

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
    
    ################# Static methods #################
    @staticmethod
    def generate_training_data(simulation, output_name):
        '''Build and store the generated features and targets from a ROOT simulation'''
        features = []
        targets = []
        l_valid_pos = []
        l_events_seq = []
        
        for idx, event in enumerate(simulation.iterate_events()):
            if event.is_distributed_clusters:
                features.append(event.get_features())
                targets.append(event.get_targets())
                l_valid_pos.append(True)
                l_events_seq.append(idx)
            else:
                l_valid_pos.append(False)
                
        features = np.array(features, dtype='float64')
        targets = np.array(targets, dtype='float64')
        
        # extract the reco data for the valid events
        reco = np.concatenate((
            np.zeros((sum(l_valid_pos),1)), # event type
            simulation.tree['RecoEnergy_e']['value'].array()[l_valid_pos].reshape((-1,1)),
            simulation.tree['RecoEnergy_p']['value'].array()[l_valid_pos].reshape((-1,1)),
            utils.l_vec_as_np(simulation.tree['RecoPosition_e']['position'].array()[l_valid_pos]),
            utils.l_vec_as_np(simulation.tree['RecoPosition_p']['position'].array()[l_valid_pos]),
        ), axis=1)
        # reco type is true when e energy is not 0
        reco[:,0] = reco[:,1] != 0
        
        # save features, targets, reco as numpy tensors
        with open(output_name, 'wb') as f_train:
            np.savez_compressed(f_train, 
                                features=features, 
                                targets=targets, 
                                reco=reco,
                                sequence = l_events_seq
                               )
        