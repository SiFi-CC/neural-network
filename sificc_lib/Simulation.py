# Author: Awal Awal
# Date: Jul 2020
# Email: awal.nova@gmail.com

import sys
import uproot
from sificc_lib import Event, SiFiCC_Module
from tqdm import tqdm

class Simulation:
    '''Process a ROOT simulation for SiFi-CC detection'''
    def __init__(self, file_name, clusters_limit=6):
        root_file = uproot.open(file_name)
        self.__setup(root_file)
        self.tree = root_file[b'Events']
        self.num_entries = self.tree.numentries
        self.clusters_limit = clusters_limit
        
        self.clusters_count = self.tree['RecoClusterEnergies']
        self.clusters_position = self.tree['RecoClusterPositions.position']
        self.clusters_position_unc = self.tree['RecoClusterPositions.uncertainty']
        self.clusters_energy = self.tree['RecoClusterEnergies.value']
        self.clusters_energy_unc = self.tree['RecoClusterEnergies.uncertainty']
        self.clusters_entries = self.tree['RecoClusterEntries']
            
    def __setup(self, root_file):
        '''Extract scatterer and absorber modules setup from the ROOT file'''
        setup = root_file[b'Setup']
        self.scatterer = SiFiCC_Module(setup['ScattererThickness_x'].array()[0],
                                       setup['ScattererThickness_y'].array()[0],
                                       setup['ScattererThickness_z'].array()[0],
                                       setup['ScattererPosition'].array()[0],
                                      )
        self.absorber = SiFiCC_Module(setup['AbsorberThickness_x'].array()[0],
                                       setup['AbsorberThickness_y'].array()[0],
                                       setup['AbsorberThickness_z'].array()[0],
                                       setup['AbsorberPosition'].array()[0],
                                      )
        
    def iterate_events(self, basket_size=100000, desc='processing root file', 
                       bar_update_size=1000, entrystart=None):
        '''Iterate throughout all the events within the ROOT file. 
        Returns an event object on each step.
        '''
        total = self.num_entries if entrystart is None else self.num_entries - entrystart
        prog_bar = tqdm(total=total, ncols=100, file=sys.stdout, desc=desc)
        bar_step = 0
        for start, end, basket in self.tree.iterate(Event.l_leaves, entrysteps=basket_size, 
                                                    reportentries=True, namedecode='utf-8',
                                                    entrystart=entrystart, entrystop=None):
            length = end-start
            for idx in range(length):
                yield self.__event_at_basket(basket, idx)
                
                bar_step += 1
                if bar_step % bar_update_size == 0:
                    prog_bar.update(bar_update_size)
                
        prog_bar.update(self.num_entries % bar_update_size)
        prog_bar.close()
                
    def get_event(self, position):
        '''Return event object at a certain position within the ROOT file
        '''
        for basket in self.tree.iterate(Event.l_leaves, entrystart=position, entrystop=position+1, 
                                        namedecode='utf-8'):
            return self.__event_at_basket(basket, 0)
        
    def __event_at_basket(self, basket, position):
        '''Create and return event object at a certain position from a ROOT basket of data
        '''
        
        event = Event(real_primary_energy = basket['Energy_Primary'][position], 
                      real_e_energy = basket['RealEnergy_e'][position], 
                      real_p_energy = basket['RealEnergy_p'][position], 
                      real_e_positions = basket['RealPosition_e'][position], 
                      real_e_interactions = basket['RealInteractions_e'][position],
                      real_p_positions = basket['RealPosition_p'][position], 
                      real_p_interactions = basket['RealInteractions_p'][position],
                      real_src_pos = basket['RealPosition_source'][position], 
                      real_src_dir = basket['RealDirection_source'][position], 
                      real_compton_pos = basket['RealComptonPosition'][position], 
                      real_scatter_dir = basket['RealDirection_scatter'][position], 
                      identification_code = basket['Identified'][position], 
                      crossed = basket['PurCrossed'][position], 
                      clusters_count = basket['RecoClusterEnergies'][position],
                      clusters_position = basket['RecoClusterPositions.position'][position], 
                      clusters_position_unc = basket['RecoClusterPositions.uncertainty'][position], 
                      clusters_energy = basket['RecoClusterEnergies.value'][position], 
                      clusters_energy_unc = basket['RecoClusterEnergies.uncertainty'][position], 
                      clusters_entries = basket['RecoClusterEntries'][position],
                      event_type = basket['SimulatedEventType'][position],
                      scatterer = self.scatterer,
                      absorber = self.absorber,
                      clusters_limit = self.clusters_limit
                     )
        return event