# Author: Awal Awal
# Date: Jul 2020
# Email: awal.nova@gmail.com

import math
import numpy as np
from sificc_lib import utils
from uproot_methods.classes.TVector3 import TVector3

class Event:
    '''Represents a single event in a ROOT file
    '''
    
    # list of leaves that are required from a ROOT file to properly instantiate an Event object
    l_leaves = ['Energy_Primary', 'RealEnergy_e', 'RealEnergy_p', 'RealPosition_source', 'SimulatedEventType',
                'RealDirection_source', 'RealComptonPosition', 'RealDirection_scatter', 'RealPosition_e', 
                'RealInteractions_e', 'RealPosition_p', 'RealInteractions_p', 'Identified', 'PurCrossed',
                'RecoClusterPositions.position', 'RecoClusterPositions.uncertainty', 'RecoClusterEnergies', 
                'RecoClusterEnergies.value', 'RecoClusterEnergies.uncertainty', 'RecoClusterEntries', 
               ]
    
    def __init__(self, real_primary_energy, real_e_energy, real_p_energy, real_e_positions, 
                 real_e_interactions, real_p_positions, real_p_interactions, real_src_pos, real_src_dir, 
                 real_compton_pos, real_scatter_dir, identification_code, crossed, clusters_count, 
                 clusters_position, clusters_position_unc, clusters_energy, clusters_energy_unc, 
                 clusters_entries, event_type,
                 scatterer, absorber, clusters_limit
                ):
        # define the main values of a simulated event
        self.event_type = event_type
        self.real_primary_energy = real_primary_energy
        self.real_e_energy = real_e_energy
        self.real_p_energy = real_p_energy
        self.real_e_position_all = real_e_positions
        self.real_e_interaction_all = real_e_interactions
        self.real_p_position_all = real_p_positions
        self.real_p_interaction_all = real_p_interactions
        self.real_src_pos = real_src_pos
        self.real_src_dir = real_src_dir
        self.real_compton_pos = real_compton_pos
        self.real_scatter_dir = real_scatter_dir
        self.identification_code = identification_code
        self.crossed = crossed
        self.clusters_count = clusters_count
        self.clusters_position = clusters_position
        self.clusters_position_unc = clusters_position_unc
        self.clusters_energy = clusters_energy
        self.clusters_energy_unc = clusters_energy_unc
        self.clusters_entries = clusters_entries
        self.clusters_limit = clusters_limit
        
        # check if the event is a valid event by considering the clusters associated with it
        # the event is considered valid if there are at least one cluster within each module of the SiFiCC
        if self.clusters_count >= 2 \
                and scatterer.is_any_point_inside_x(self.clusters_position) \
                and absorber.is_any_point_inside_x(self.clusters_position):
            self.is_distributed_clusters = True
        else:
            self.is_distributed_clusters = False
        
        # check if the event is a Compton event
        self.is_compton = True if self.real_e_energy != 0 else False
        
        # check if the event is a complete Compton event
        # complete Compton event= Compton event + both e and p go through a second interation in which
        # 0 < p interaction < 10
        # 10 <= e interaction < 20
        # Note: first interaction of p is the compton event
        if self.is_compton \
                and len(self.real_p_position_all) >= 2 \
                and len(self.real_e_position_all) >= 1 \
                and ((self.real_p_interaction_all[1:] > 0) & (self.real_p_interaction_all[1:] < 10)).any() \
                and ((self.real_e_interaction_all[0] >= 10) & (self.real_e_interaction_all[0] < 20)):
            self.is_complete_compton = True 
        else:
            self.is_complete_compton = False
        
        # initialize e & p first interaction position
        if self.is_complete_compton:
            for idx in range(1,len(self.real_p_interaction_all)):
                if 0 < self.real_p_interaction_all[idx] < 10:
                    self.real_p_position = self.real_p_position_all[idx]
                    break
            for idx in range(0,len(self.real_e_interaction_all)):
                if 10 <= self.real_e_interaction_all[idx] < 20:
                    self.real_e_position = self.real_e_position_all[idx]
                    break
        else:
            self.real_p_position = TVector3(0,0,0)
            self.real_e_position = TVector3(0,0,0)
            
        # check if the event is a complete distributed Compton event
        # complete distributed Compton event= complete Compton event + each e and p go through a secondary 
        # interaction in a different module of the SiFiCC
        if self.is_complete_compton \
                and scatterer.is_any_point_inside_x(self.real_p_position_all) \
                and absorber.is_any_point_inside_x(self.real_p_position_all):
            self.is_complete_distributed_compton = True
        else:
            self.is_complete_distributed_compton = False
            
        # check if the event is an ideal Compton event and what type is it (EP or PE)
        # ideal Compton event = complete distributed Compton event where the next interaction of both 
        # e and p is in the different modules of SiFiCC
        if self.is_complete_compton \
                and scatterer.is_point_inside_x(self.real_e_position) \
                and absorber.is_point_inside_x(self.real_p_position) \
                and self.event_type == 2:
            self.is_ideal_compton = True
            self.is_ep = True
            self.is_pe = False
        elif self.is_complete_compton \
                and scatterer.is_point_inside_x(self.real_p_position) \
                and absorber.is_point_inside_x(self.real_e_position) \
                and self.event_type == 2:
            self.is_ideal_compton = True
            self.is_ep = False
            self.is_pe = True
        else:
            self.is_ideal_compton = False
            self.is_ep = False
            self.is_pe = False
        
    def _aggregate_max_clusters(self):
        '''Aggregate the top clusters in term of cluster energy within the `cluster_limit` (inplace sorting)'''
        
        # redeclare clusters position to np array to be able to switch values within
        self.clusters_position = np.array(self.clusters_position)
        self.clusters_position_unc = np.array(self.clusters_position_unc)
        
        # updates to the clusters only if their number is bigger than the limit
        if self.clusters_count > self.clusters_limit:
            # minimum energy of top clusters should be higher than the rest
            while np.min(self.clusters_energy[:self.clusters_limit]) < \
                    np.max(self.clusters_energy[self.clusters_limit:]):
                # if it is not the case, get the position of the minimum and maximum energies of both ends
                # and replace the clusters
                min_pos = np.argmin(self.clusters_energy[:self.clusters_limit])
                max_pos = np.argmax(self.clusters_energy[self.clusters_limit:]) + self.clusters_limit
                
                self.clusters_energy[min_pos], self.clusters_energy[max_pos] = \
                    self.clusters_energy[max_pos], self.clusters_energy[min_pos]
                self.clusters_energy_unc[min_pos], self.clusters_energy_unc[max_pos] = \
                    self.clusters_energy_unc[max_pos], self.clusters_energy_unc[min_pos]
                self.clusters_position[min_pos], self.clusters_position[max_pos] = \
                    self.clusters_position[max_pos], self.clusters_position[min_pos]
                self.clusters_position_unc[min_pos], self.clusters_position_unc[max_pos] = \
                    self.clusters_position_unc[max_pos], self.clusters_position_unc[min_pos]
                self.clusters_entries[min_pos], self.clusters_entries[max_pos] = \
                    self.clusters_entries[max_pos], self.clusters_entries[min_pos]
                
    def _sort_clusters(self):
        '''Sort event clusters in reverse based on their energy (inplace sorting)'''
        energy_sort = np.flip(np.argsort(self.clusters_energy))
        self.clusters_energy = self.clusters_energy[energy_sort]
        self.clusters_energy_unc = self.clusters_energy_unc[energy_sort]
        self.clusters_position = self.clusters_position[energy_sort]
        self.clusters_position_unc = self.clusters_position_unc[energy_sort]
        self.clusters_entries = self.clusters_entries[energy_sort]
        
    def _align_clusters(self):
        '''Align the cluster so that their count matches with `clusters_limit` either by trimming or padding'''
        # if cluster count is more than the limit then trim
        if self.clusters_count >= self.clusters_limit:
            self.clusters_energy = self.clusters_energy[:self.clusters_limit]
            self.clusters_energy_unc = self.clusters_energy_unc[:self.clusters_limit]
            self.clusters_position = self.clusters_position[:self.clusters_limit]
            self.clusters_position_unc = self.clusters_position_unc[:self.clusters_limit]
            self.clusters_entries = self.clusters_entries[:self.clusters_limit]
            
        # otherwise, pad the clusters with the difference to reach `clusters_limit`
        else:
            padding = self.clusters_limit - self.clusters_count
            
            self.clusters_energy = np.concatenate((self.clusters_energy, [0] * padding))
            self.clusters_energy_unc = np.concatenate((self.clusters_energy_unc, [0] * padding))
            self.clusters_position = np.concatenate((self.clusters_position, [TVector3(0,0,0)] * padding))
            self.clusters_position_unc = np.concatenate((self.clusters_position_unc, [TVector3(0,0,0)] * padding))
            self.clusters_entries = np.concatenate((self.clusters_entries, [0] * padding))
        # clusters count now is identical to clusters limit
        self.clusters_count = self.clusters_limit
        
    @property
    def e_clusters_count(self):
        '''Calculates and returns the count of clusters matching with e position'''
        return self._count_e_clusters()
    
    @property
    def p_clusters_count(self):
        '''Calculates and returns the count of clusters matching with p position'''
        return self._count_p_clusters()
    
    @property
    def is_clusters_matching(self):
        '''Checks if only one cluster is assigned to each of e & p'''
        return True if self.e_clusters_count == 1 and self.p_clusters_count == 1 else False
    
    @property
    def is_clusters_overlap(self):
        '''Checks if their are overlap of clusters assigned to e or p'''
        return True if self.e_clusters_count > 1 or self.p_clusters_count > 1 else False
    
    def _count_e_clusters(self):
        '''Returns the count of clusters matching e position within their uncertainities'''
        return self._count_matching_clusters(self.real_e_position)
    
    def _count_p_clusters(self):
        '''Returns the count of clusters matching p position within their uncertainities'''
        return self._count_matching_clusters(self.real_p_position)
        
    def _count_matching_clusters(self, point):
        '''Returns the number of matching clusters to `point` within their uncertainities'''
        count = 0
        
        for cluster, cluster_unc in zip(self.clusters_position, self.clusters_position_unc):
            if utils.is_point_inside_cluster(point, cluster, cluster_unc):
                count += 1
        return count
    
    def _arg_matching_cluster(self, point):
        '''Gets a point and returns the index of the first cluster matching with this point 
        within its uncertainities. Returns -1 when no cluster matches with `point`'''
        
        for idx, (cluster, cluster_unc) in enumerate(zip(self.clusters_position, self.clusters_position_unc)):
            if utils.is_point_inside_cluster(point, cluster, cluster_unc):
                return idx
        else:
            return -1
            
    def _arg_closest_cluster(self, point):
        '''Gets a point a return the index of the closest cluster. Returns -1 when there are
        no clusters'''
        if self.clusters_count < 1:
            return -1
        
        closest_idx = 0
        min_euclidean = utils.euclidean_distance(point, self.clusters_position[0])
        
        for idx, cluster in enumerate(self.clusters_position):
            distance = utils.euclidean_distance(point, cluster)
            if distance < min_euclidean:
                min_euclidean = distance
                closest_idx = idx
                
        return closest_idx
    
    def get_features(self, sort=True):
        '''Generate and return the training features of the event. Features are of the form:
        [ cluster-1, cluster-2, ..., cluster-n ]
        
        Where the format of each cluster is:
        [
            cluster entries, 
            cluster energy, 
            cluster energy uncertainty, 
            cluster position (x,y,z), 
            cluster position uncertainty (x,y,z) 
        ]
        
        Output feature dimention is 1x(9*clusters_limit)'''
        
        # sort the clusters and align the number of clusters to match `clusters_limit`
        if sort:
            self._sort_clusters()
        else:
            self._aggregate_max_clusters()
        self._align_clusters()
        
        # build the list of clusters features
        # the abs is taken for position uncertainty because the y dimention is negative
        l_clusters = []
        for i in range(self.clusters_limit):
            cluster = np.concatenate((
                [self.clusters_entries[i]],
                [self.clusters_energy[i]],
                [self.clusters_energy_unc[i]],
                utils.vec_as_np(self.clusters_position[i]),
                np.abs(utils.vec_as_np(self.clusters_position_unc[i])),
            ))
            l_clusters.append(cluster)
            
        features = np.concatenate(l_clusters)
        return features
    
    def get_targets(self):
        '''Generates and return the event targets. 
        Targets are of dimension 1x11 and the format is:
        [
            event type (is ideal Compton or not),
            e energy,
            p energy,
            e position (x,y,z),
            p position (x,y,z),
            e cluster index,
            p cluster index,
        ]'''

        # return the features only if the event is an ideal compton
        # otherwise return 0s
        if self.is_distributed_clusters and self.is_ideal_compton:
            
            # find cluster index of both e & p
            if self.e_clusters_count == 1:
                e_cluster_index = self._arg_matching_cluster(self.real_e_position)
            else:
                e_cluster_index = self._arg_closest_cluster(self.real_e_position)
            if self.p_clusters_count == 1:
                p_cluster_index = self._arg_matching_cluster(self.real_p_position)
            else:
                p_cluster_index = self._arg_closest_cluster(self.real_p_position)
                
            # build target features
            targets = np.concatenate((
                [1],
                [self.real_e_energy],
                [self.real_p_energy],
                utils.vec_as_np(self.real_e_position),
                utils.vec_as_np(self.real_p_position),
                [e_cluster_index],
                [p_cluster_index],
            ))
        else:
            targets = np.zeros(11)
            
        return targets