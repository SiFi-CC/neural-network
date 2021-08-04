import numpy as np
import os
import csv

class Utils_new:
    def is_point_inside_cluster(point, cluster, cluster_unc):
        '''Checks if `point` is inside `cluster` within its uncertainties'''
        if np.abs(point.x - cluster.x) <= np.abs(cluster_unc.x) \
                and np.abs(point.y - cluster.y) <= np.abs(cluster_unc.y) \
                and np.abs(point.z - cluster.z) <= np.abs(cluster_unc.z):
            return True
        else:
            return False
    
    def is_energy_inside_cluster(energy, cluster_enrg, cluster_enrg_unc):
        '''Checks if `energy` is within `cluster_enrg` and its uncertainties'''
        if np.abs(energy - cluster_enrg) <= np.abs(cluster_enrg_unc):
            return True
        else:
            return False
    
    def euclidean_distance(a, b):
        '''Compute the euclidean distance between two victor points'''
        euclidean = np.sqrt(np.power(a.x-b.x,2)+np.power(a.y-b.y,2)+np.power(a.z-b.z,2))
        return euclidean
    
    def euclidean_distance_np(points_1, points_2, keepdims=False):
        '''Compute the euclidean distance between two numpy arrays representing 3D points'''
        dis = np.power(points_1 - points_2, 2)
        dis = np.sum(dis, axis=1, keepdims=keepdims)
        dis = np.sqrt(dis)
        return dis
    
    def vec_as_np(tvector):
        return np.array([tvector.x, tvector.y, tvector.z])
    
    def l_vec_as_np(l_tvector, flatten = False):
        array = [[tvector.x, tvector.y, tvector.z] for tvector in l_tvector]
        return np.array(array).reshape((-1,)) if flatten else np.array(array)
    
    def exp_ma(l_points, factor=.9):
        '''Applies smoothing to a sequence using exponential moving average'''
        l_smoothed = []
        for p in l_points:
            if l_smoothed:
                prev = l_smoothed[-1]
                l_smoothed.append(prev*factor + p * (1-factor))
            else:
                l_smoothed.append(p)
        return l_smoothed
    
    def show_root_file_analysis(simulation, only_valid=True):
        import matplotlib.pyplot as plt
        n_count = 0
        n_is_in_absorber = 0
        n_is_in_scatterer = 0
        n_is_in_volumes = 0
        n_mintwoclusters = 0
        n_oneclusters = 0
        n_realcoinc = 0
        n_randcoinc = 0
        n_realcoinc_pileup = 0
        n_randcoinc_pileup = 0
        n_distributed_clusters = 0
        n_compton = 0
        n_complete_compton = 0
        n_complete_distributed_compton = 0
        n_ideal_compton = 0
        n_ep = 0
        n_pe = 0
        n_ideal_realcoinc_compton = 0
        n_ideal_randcoinc_compton = 0
        n_distributed_clusters_randcoinc = 0
        n_distributed_clusters_realcoinc = 0
        n_ideal_realcoinc_compton_pileup = 0
        n_matching_ideal_compton = 0
        n_overlap_matching_ideal = 0
        l_matching_idx = []

        for event in simulation.iterate_events():
            n_count +=1
            n_is_in_absorber +=1 if event.is_in_absorber else 0
            n_is_in_scatterer +=1 if event.is_in_scatterer else 0
            n_is_in_volumes +=1 if event.is_in_volumes else 0
            n_mintwoclusters +=1 if event.is_mintwoclusters else 0
            n_oneclusters +=1 if event.is_onecluster else 0
            n_realcoinc += 1 if event.is_realcoinc else 0
            n_randcoinc += 1 if event.is_randcoinc else 0
            n_realcoinc_pileup += 1 if event.is_realcoinc_pileup else 0
            n_randcoinc_pileup += 1 if event.is_randcoinc_pileup else 0
            
            if not event.is_distributed_clusters and only_valid:
                continue
            
            n_distributed_clusters += 1 if event.is_distributed_clusters else 0
            n_compton += 1 if event.is_compton else 0
            n_complete_compton += 1 if event.is_complete_compton else 0
            n_complete_distributed_compton += 1 if event.is_complete_distributed_compton else 0
            n_ideal_compton += 1 if event.is_ideal_compton else 0
            n_ep += 1 if event.is_ep else 0
            n_pe += 1 if event.is_pe else 0
            n_ideal_realcoinc_compton += 1 if event.is_ideal_realcoinc_compton else 0
            n_ideal_randcoinc_compton += 1 if event.is_ideal_randcoinc_compton else 0
            n_ideal_realcoinc_compton_pileup += 1 if event.is_ideal_realcoinc_compton_pileup else 0
            
            n_distributed_clusters_realcoinc += 1 if event.is_distributed_clusters_realcoinc else 0
            n_distributed_clusters_randcoinc += 1 if event.is_distributed_clusters_randcoinc else 0
            
            if event.is_ideal_compton:
                if event.is_clusters_matching:
                    n_matching_ideal_compton += 1 
                    event._sort_clusters()
                    l_matching_idx.append(event._arg_matching_cluster(event.real_p_position))
                    l_matching_idx.append(event._arg_matching_cluster(event.real_e_position))
                n_overlap_matching_ideal += 1 if event.is_clusters_overlap else 0

        print('{:8,d} total entries'.format(simulation.num_entries))
        print('{:8,d} total entries count events'.format(n_count))
        print('{:8,d} total contains cluster in scatterer'.format(n_is_in_scatterer))
        print('{:8,d} total contains cluster in absorber'.format(n_is_in_absorber))
        print('{:8,d} total contains cluster in both volumes'.format(n_is_in_volumes))
        print('{:8,d} total real coinc'.format(n_realcoinc))
        print('{:8,d} total rand coinc'.format(n_randcoinc))
        print('{:8,d} total real coinc with pile-up'.format(n_realcoinc_pileup))
        print('{:8,d} total rand coinc with pile-up'.format(n_randcoinc_pileup))
        print('{:8,d} one cluster'.format(n_oneclusters))
        print('{:8,d} min two clusters'.format(n_mintwoclusters))
        print('{:8,d} valid entries with distributed clusters'.format(n_distributed_clusters))
        print('{:8,d} valid entries with distributed clusters real coinc'.format(n_distributed_clusters_realcoinc))
        print('{:8,d} valid entries with distributed clusters random coinc'.format(n_distributed_clusters_randcoinc))
        print('{:8,d} compton events'.format(n_compton))
        print('{:8,d} compton + second interaction'.format(n_complete_compton))
        print('{:8,d} compton + second in different module'.format(n_complete_distributed_compton))
        print('{:8,d} ideal compton events'.format(n_ideal_compton))
        print('\t{:8,d} ep'.format(n_ep))
        print('\t{:8,d} pe'.format(n_pe))
        print('{:8,d} ideal real coincidence compton events'.format(n_ideal_realcoinc_compton))
        print('{:8,d} ideal rand coincidence compton events'.format(n_ideal_randcoinc_compton))
        print('{:8,d} ideal real coincidence compton events with pileup'.format(n_ideal_realcoinc_compton_pileup))
        print('{:8,d} ideal compton events with matching clusters'.format(n_matching_ideal_compton))
        print('{:8,d} ideal compton events with overlapping clusters'.format(n_overlap_matching_ideal))
        n, bins, _ = plt.hist(l_matching_idx, np.arange(0,np.max(l_matching_idx)+2))
        plt.xticks(np.arange(0,np.max(l_matching_idx)+1), np.arange(1,np.max(l_matching_idx)+2))
        plt.xlabel('argmax of electron and photon clusters')
        plt.ylabel('count')
        plt.show()
        print('histogram bars\' count:', n)
        
    def show_simulation_setup(simulation):
        print('Scatterer:')
        print('\tPosition: ({:.1f}, {:.1f}, {:.1f})'.format(simulation.scatterer.position.x, 
                                                            simulation.scatterer.position.y, 
                                                            simulation.scatterer.position.z))
        print('\tThickness: ({:.1f}, {:.1f}, {:.1f})'.format(simulation.scatterer.thickness_x, 
                                                             simulation.scatterer.thickness_y, 
                                                             simulation.scatterer.thickness_z))
        print('\nAbsorber:')
        print('\tPosition: ({:.1f}, {:.1f}, {:.1f})'.format(simulation.absorber.position.x, 
                                                            simulation.absorber.position.y, 
                                                            simulation.absorber.position.z))
        print('\tThickness: ({:.1f}, {:.1f}, {:.1f})'.format(simulation.absorber.thickness_x, 
                                                             simulation.absorber.thickness_y, 
                                                             simulation.absorber.thickness_z))

    def calculate_normalizations(simulation, name = '', only_valid = True):
        
        # Create file for saving normalizations
        if not os.path.exists('Normalization_' + name + '.csv'):
            os.mknod('Normalization_' + name + '.csv')
        
        l_entries = []
        l_energies = []
        l_energies_unc = []
        l_positions = []
        l_positions_unc = []
        l_e_energy = []
        l_e_position = []
        l_p_energy = []
        l_p_position = []
        for event in simulation.iterate_events():
            if not event.is_distributed_clusters and only_valid:
                continue
            l_entries.append(event.clusters_entries)
            l_energies.append(event.clusters_energy)
            l_energies_unc.append(event.clusters_energy_unc)
            l_positions.append(Utils_new.l_vec_as_np(event.clusters_position, flatten=False))
            l_positions_unc.append(np.abs(Utils_new.l_vec_as_np(event.clusters_position_unc, flatten=False)))
            
            if event.is_ideal_compton:
                l_e_energy.append(event.real_e_energy)
                l_e_position.append(Utils_new.vec_as_np(event.real_e_position).reshape((1,-1)))
                l_p_energy.append(event.real_p_energy)
                l_p_position.append(Utils_new.vec_as_np(event.real_p_position).reshape((1,-1)))
                
        l_entries = np.concatenate(l_entries)
        l_energies = np.concatenate(l_energies)
        l_energies_unc = np.concatenate(l_energies_unc)
        l_positions = np.concatenate(l_positions)
        l_positions_unc = np.concatenate(l_positions_unc)
        l_e_position = np.concatenate(l_e_position)
        l_p_position = np.concatenate(l_p_position)
        
        print('Features normalization:')
        mean_entries = np.mean(l_entries)
        std_entries = np.std(l_entries)
        print('clusters entry')
        print('\tmean', mean_entries)
        print('\tstd', std_entries)
        
        mean_energies = np.mean(l_energies)
        std_energies = np.std(l_energies)
        print('\nclusters energy')
        print('\tmean', mean_energies)
        print('\tstd', std_energies)
        
        mean_energies_unc = np.mean(l_energies_unc)
        std_energies_unc = np.std(l_energies_unc)
        print('\nclusters energy uncertainty')
        print('\tmean', mean_energies_unc)
        print('\tstd', std_energies_unc)
        
        mean_positions = np.mean(l_positions, axis=0)
        std_positions = np.std(l_positions, axis=0)
        print('\nclusters position')
        print('\tmean', mean_positions)
        print('\tstd', std_positions)
        
        mean_positions_unc = np.mean(l_positions_unc, axis=0)
        std_positions_unc = np.std(l_positions_unc, axis=0)
        print('\nclusters position uncertainty')
        print('\tmean', mean_positions_unc)
        print('\tstd', std_positions_unc)
        
        print('\nTargets normalization')
        e_energy_mean = np.mean(l_e_energy)
        e_energy_std = np.std(l_e_energy)
        print('real e energy')
        print('\tmean', e_energy_mean)
        print('\tstd', e_energy_std)
        
        e_position_mean = np.mean(l_e_position, axis=0)
        e_position_std = np.std(l_e_position, axis=0)
        print('\nreal e position')
        print('\tmean', e_position_mean)
        print('\tstd', e_position_std)
        
        p_energy_mean = np.mean(l_p_energy)
        p_energy_std = np.std(l_p_energy)
        print('\nreal p energy')
        print('\tmean', p_energy_mean)
        print('\tstd', p_energy_std)
        
        p_position_mean = np.mean(l_p_position, axis=0)
        p_position_std = np.std(l_p_position, axis=0)
        print('\nreal p position')
        print('\tmean', p_position_mean)
        print('\tstd', p_position_std)
        
        file = open('Normalization_' + name + '.csv', 'w')
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Mean', 'Std'])
        writer.writerow(['Feature entries', mean_entries, std_entries])
        writer.writerow(['Feature energies', mean_energies, std_energies])
        writer.writerow(['Feature energies uncertainty', mean_energies_unc, std_energies_unc])
        writer.writerow(['Feature positions', mean_positions, std_positions])
        writer.writerow(['Feature position uncertainties', mean_positions_unc, std_positions_unc])
        writer.writerow(['Target e energies', e_energy_mean, e_energy_std])
        writer.writerow(['Target e position', e_position_mean, e_position_std])
        writer.writerow(['Target p energies', p_energy_mean, p_energy_std])
        writer.writerow(['Target p position', p_position_mean, p_position_std])
        
        file.close()
        
        
        
        