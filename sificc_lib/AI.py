import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from sificc_lib import utils
import pickle as pkl
import datetime as dt
from scipy.stats import gaussian_kde
import uproot
from scipy import constants

class MyCallback(keras.callbacks.Callback):
    def __init__(self, ai, file_name=None):
        self.ai = ai
        self.file_name = file_name
        
        if file_name is not None:
            with open('ModelsTrained/' + self.file_name + '.e', 'w') as f_epoch:
                f_epoch.write('')
        
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.ai.predict(self.ai.data.train_x)
        y_true = self.ai.data.train_row_y
        l_matches = self.ai._find_matches(y_true, y_pred, keep_length=False)
        logs['eff'] = np.mean(l_matches)
        logs['pur'] = np.sum(l_matches) / np.sum(y_pred[:,0])
        
        y_pred = self.ai.predict(self.ai.data.validation_x)
        y_true = self.ai.data.validation_row_y
        l_matches = self.ai._find_matches(y_true, y_pred, keep_length=False)
        logs['val_eff'] = np.mean(l_matches)
        logs['val_pur'] = np.sum(l_matches) / np.sum(y_pred[:,0])
        
        self.ai.append_history(logs)
        self.ai.save(self.file_name)
        
        if self.file_name is not None:
            with open('ModelsTrained/' + self.file_name + '.e', 'a') as f_epoch:
                now = dt.datetime.now()
                f_epoch.write('loss:{:5.3f} eff:{:5.3f}/{:5.3f} in epoch {:3d} at {} {}\n'.format(
                    logs['loss'], logs['eff'], logs['val_eff'], 
                    epoch, now.date().isoformat(), now.strftime('%H:%M:%S')))

        
class AI:
    def __init__(self, data_model, model_name=None):
        '''Initializing an instance of SiFi-CC Neural Network
        '''
        self.data = data_model
        
        self.history = {}
        self.model = None
        
        self.energy_factor_limit= .06 * 2
        self.position_absolute_limit = np.array([1.3, 5, 1.3]) * 2
        
        self.weight_type = 2
        self.weight_e_cluster = 1
        self.weight_p_cluster = 1
        self.weight_pos_x = 2.5
        self.weight_pos_y = 1
        self.weight_pos_z = 2
        self.weight_energy = 1.5
        
        self.penalty = ''
        self.savefigpath = ''
        
        self.callback = MyCallback(self, model_name)
        
    def train(self,*, epochs=100, verbose=0, shuffle=True, 
              shuffle_clusters=False, callbacks=None):
        '''Trains the AI for a fixed number of epoches
        '''
        if callbacks is None:
            callbacks = [self.callback]
        else:
            callbacks.append(self.callback)
            
        history = self.model.fit(self.data.generate_batch(shuffle=shuffle, augment=shuffle_clusters), 
                       epochs=epochs, steps_per_epoch=self.data.steps_per_epoch, 
                       validation_data=(self.data.validation_x, self.data.validation_y), 
                       verbose=verbose, callbacks = callbacks)
        #self.extend_history(history)
    
    def create_model(self, conv_layers=[], classifier_layers=[], dense_layers=[],
                     type_layers=[], pos_layers=[], energy_layers=[], 
                     base_l2=0, limbs_l2=0, 
                     conv_dropouts=[], activation='relu', 
                     pos_loss=None, energy_loss=None):
        if len(conv_dropouts) == 0:
            conv_dropouts = [0] * len(conv_layers)
        assert len(conv_dropouts) == len(conv_layers)
        
        ###### input layer ######
        feed_in = keras.Input(shape=self.data.get_features(0,1).shape[1:], name='inputs')
        cnv = feed_in
        
        ###### convolution layers ######
        for i in range(len(conv_layers)):
            cnv = keras.layers.Conv1D(conv_layers[i], 
                                    kernel_size = self.data.cluster_size if i == 0 else 1, 
                                    strides = self.data.cluster_size if i == 0 else 1, 
                                    activation = activation, 
                                    kernel_regularizer = keras.regularizers.l2(base_l2), 
                                    padding = 'valid', name='conv_{}'.format(i+1))(cnv)
            if conv_dropouts[i] != 0:
                cnv = keras.layers.Dropout(dropouts[i])(cnv)
                
        if len(conv_layers) >= 1:
            cnv = keras.layers.Flatten(name='flatting')(cnv)
            
        ###### clusters classifier layers ######
        cls = cnv
        for i in range(len(classifier_layers)):
            cls = keras.layers.Dense(classifier_layers[i], activation=activation, 
                                            kernel_regularizer=keras.regularizers.l2(base_l2), 
                                            name='dense_cluster_{}'.format(i+1))(cls)
            
        # e/p clusters classifiers
        e_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='e_cluster')(cls)
        p_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='p_cluster')(cls)
        
        # get the hardmax of clusters classifier
        e_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='e_hardmax')(e_cluster)
        p_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='p_hardmax')(p_cluster)
        
        ###### joining outputs ######
        base_layer = keras.layers.Concatenate(axis=-1, name='join_layer')(
                                            [cnv, e_cluster_1_hot, p_cluster_1_hot])
        
        
        ###### event type layers ######
        typ = base_layer
        for i in range(len(type_layers)):
            typ = keras.layers.Dense(type_layers[i], activation=activation, 
                                   kernel_regularizer = keras.regularizers.l2(limbs_l2), 
                                   name='dense_type_{}'.format(i+1))(typ)
            
        event_type = keras.layers.Dense(1, activation='sigmoid', name='type')(typ)
        
        
        ###### event position ######
        pos = base_layer
        for i in range(len(pos_layers)):
            pos = keras.layers.Dense(pos_layers[i], activation=activation, 
                                     kernel_regularizer= keras.regularizers.l2(limbs_l2), 
                                     name='dense_pos_{}'.format(i+1))(pos)
            
        pos_x = keras.layers.Dense(2, activation=None, name='pos_x')(pos)
        pos_y = keras.layers.Dense(2, activation=None, name='pos_y')(pos)
        pos_z = keras.layers.Dense(2, activation=None, name='pos_z')(pos)
        
        
        ###### event energy ######
        enrg = base_layer
        for i in range(len(energy_layers)):
            enrg = keras.layers.Dense(energy_layers[i], activation=activation, 
                                      kernel_regularizer= keras.regularizers.l2(limbs_l2), 
                                      name='dense_energy_{}'.format(i+1))(enrg)
            
        energy = keras.layers.Dense(2, activation=None, name='energy')(enrg)
        
        ###### building the model ######
        self.model = keras.models.Model(feed_in, [e_cluster, p_cluster, event_type, 
                                                  pos_x, pos_y, pos_z, energy])
        self.history = None
        self.model.summary()
        
    def compile_model(self, learning_rate=0.0003):
        # Losses: Final loss function is weighted sum of individual loss terms, weighted with loss_weights
        self.model.compile(optimizer= keras.optimizers.Nadam(learning_rate), 
                           loss = {
                               'type' : self._type_loss,
                               'e_cluster': self._e_cluster_loss,
                               'p_cluster': self._p_cluster_loss,
                               'pos_x': self._pos_loss,
                               'pos_y': self._pos_loss_y,
                               'pos_z': self._pos_loss,
                               'energy': self._energy_loss, 
                           }, 
                           metrics = {
                               'type' : [self._type_accuracy, self._type_tp_rate],
                               'e_cluster': [self._cluster_accuracy],
                               'p_cluster': [self._cluster_accuracy],
                               'pos_x': [],
                               'pos_y': [],
                               'pos_z': [],
                               'energy': [],
                           }, 
                           loss_weights = {
                               'type' : self.weight_type,
                               'e_cluster': self.weight_e_cluster,
                               'p_cluster': self.weight_p_cluster,
                               'pos_x': self.weight_pos_x,
                               'pos_y': self.weight_pos_y,
                               'pos_z': self.weight_pos_z,
                               'energy': self.weight_energy,
                           })
    
    def _type_loss(self, y_true, y_pred):
        # loss ∈ n
        return keras.losses.binary_crossentropy(y_true, y_pred)
    
    def _type_accuracy(self, y_true, y_pred):
        # return ∈ n
        return keras.metrics.binary_accuracy(y_true, y_pred) 
    
    def _type_tp_rate2(self, y_true, y_pred):
        y_pred = K.round(y_pred)
        matches= K.sum(y_true * y_pred)
        all_true=K.sum(y_true)
        
        # return ∈ 1
        return matches/all_true
    
    def _type_tp_rate(self, y_true, y_pred):
        y_pred = K.round(y_pred) # ∈ nx1
        event_filter = y_true[:,0] # ∈ n
        # y_pred, y_true ∈ nx1
        y_pred = tf.boolean_mask(y_pred, event_filter)
        y_true = tf.boolean_mask(y_true, event_filter)
        # return ∈ n
        return keras.metrics.binary_accuracy(y_true, y_pred)
    
    def _e_cluster_loss(self, y_true, y_pred):
        event_filter = y_true[:,0] # ∈ n
        e_cluster = K.reshape(y_true[:,1], (-1,1)) # ∈ nx1
        # loss ∈ n
        loss = keras.losses.sparse_categorical_crossentropy(e_cluster, y_pred)
        
        # composing _e_cluster_match ; a mask for the matched clusters of e
        y_pred_sparse = K.cast(K.argmax(y_pred), y_true.dtype) # ∈ n
        self._e_cluster_pred = y_pred_sparse # ∈ n
        self._e_cluster_match = K.cast(K.equal(y_true[:,1], y_pred_sparse), 'float32') # [float] ∈ n
        
        # return (n * n) ∈ n
        return event_filter * loss
    
    def _p_cluster_loss(self, y_true, y_pred):
        event_filter = y_true[:,0] # ∈ n
        p_cluster = K.reshape(y_true[:,1], (-1,1)) # ∈ nx1
        # loss ∈ n
        loss = keras.losses.sparse_categorical_crossentropy(p_cluster, y_pred)
        
        # composing _p_cluster_match; a mast for the matched clusters of p
        y_pred_sparse = K.cast(K.argmax(y_pred), y_true.dtype) # ∈ n
        self._p_cluster_pred = y_pred_sparse # ∈ n
        self._p_cluster_match = K.cast(K.equal(y_true[:,1], y_pred_sparse), 'float32') # [float] ∈ n
        
        # return (n*n) ∈ n
        return event_filter * loss
    
    def _cluster_accuracy(self, y_true, y_pred):
        event_filter = y_true[:,0] # ∈ n
        y_true = tf.boolean_mask(y_true, event_filter) # ∈ nx1
        y_pred = tf.boolean_mask(y_pred, event_filter) # ∈ nx1
        # return ∈ n
        return keras.metrics.sparse_categorical_accuracy(y_true[:,1], y_pred)
    
    def _pos_loss(self, y_true, y_pred):
        
        event_filter = y_true[:,0] # ∈ n  
        e_pos_true = K.reshape(y_true[:,2],(-1,1)) # ∈ nx1    # 
        e_pos_pred = K.reshape(y_pred[:,0],(-1,1)) # ∈ nx1    # 
        p_pos_true = K.reshape(y_true[:,4],(-1,1)) # ∈ nx1    # 
        p_pos_pred = K.reshape(y_pred[:,1],(-1,1)) # ∈ nx1    # 
        
        # e pos
        e_loss = keras.losses.logcosh(e_pos_true, e_pos_pred) # ∈ n
        e_loss = event_filter * self._e_cluster_match * e_loss # (n*n*n) ∈ n
        
        # p pos
        p_loss = keras.losses.logcosh(p_pos_true, p_pos_pred) # ∈ n
        p_loss = event_filter * self._p_cluster_match * p_loss # (n*n*n) ∈ n
        
        return e_loss + p_loss
    
    def _pos_loss_x(self, y_true, y_pred):
        
        event_filter = y_true[:,0] # ∈ n  
        e_pos_true = K.reshape(y_true[:,2],(-1,1)) # ∈ nx1    # 
        e_pos_pred = K.reshape(y_pred[:,0],(-1,1)) # ∈ nx1    # 
        p_pos_true = K.reshape(y_true[:,4],(-1,1)) # ∈ nx1    # 
        p_pos_pred = K.reshape(y_pred[:,1],(-1,1)) # ∈ nx1    #

        # e pos
        e_loss = keras.losses.logcosh(e_pos_true, e_pos_pred) # ∈ n
        e_loss = event_filter * self._e_cluster_match * e_loss # (n*n*n) ∈ n
        
        # p pos
        p_loss = keras.losses.logcosh(p_pos_true, p_pos_pred) # ∈ n
        p_loss = event_filter * self._p_cluster_match * p_loss # (n*n*n) ∈ n
        
        # Implement penalty if predictions lies out side of SiFi-CC volume (x direction)
        # e pos [209.63565735, -0.23477532, -5.38639807] x y z Normalization factors
        # p pos = [3.85999635e+02, 1.30259990e-01, 2.13816374e+00]
        # std e [41.08060207, 20.77702422, 27.19018651] / 10
        # std p [43.94193657, 27.44766386, 28.21021386] / 10
        
        def normalize_boundaries(x1,mean,std):
            return (x1 - mean)/std
        
        minX_eScat = normalize_boundaries(193.5, 209.63565735,   (41.08060207/10) )
        minX_pScat = normalize_boundaries(193.5, 3.85999635e+02, (43.94193657/10) )
        maxX_eScat = normalize_boundaries(206.5, 209.63565735,   (41.08060207/10) )
        maxX_pScat = normalize_boundaries(206.5, 3.85999635e+02, (43.94193657/10) )
        
        minX_eAbs = normalize_boundaries(380.5, 209.63565735,   (41.08060207/10) )
        minX_pAbs = normalize_boundaries(380.5, 3.85999635e+02, (43.94193657/10) )
        maxX_eAbs = normalize_boundaries(419.5, 209.63565735,   (41.08060207/10) )
        maxX_pAbs = normalize_boundaries(419.5, 3.85999635e+02, (43.94193657/10) )
        
        cond_tf = tf.ones_like(e_pos_pred)  # Mutliply with boundaries
        
        def isNotInVolumes(pos_pred, min_scat, max_scat, min_abs, max_abs):
            
            center_Scat = (min_scat+max_scat)/2.0
            center_Abs = (min_abs+max_abs)/2.0
            width_Scat = np.abs(center_Scat-min_scat)
            width_Abs = np.abs(center_Abs-min_abs)
            
            # Returns boolean tensor, true for which event a 
            greater_e    = tf.greater(pos_pred, cond_tf*max_scat ) # X between vol
            smaller_e    = tf.greater(cond_tf*min_abs, pos_pred)  # X between vol
            smallermin_e = tf.greater(cond_tf*min_scat, pos_pred)  # x too small
            largermax_e  = tf.greater(pos_pred, cond_tf*max_abs)   # x too large
           
            penalty_outsideX_1 = tf.logical_and(greater_e, smaller_e)      # Between vol
            penalty_outsideX_2 = tf.logical_or(smallermin_e, largermax_e)  # Too small and too large
            penalty_outsideX = tf.logical_or(penalty_outsideX_1, penalty_outsideX_2)   # Each three forbidden regions
            
            dist_Scat = tf.abs(tf.subtract(pos_pred,center_Scat))
            dist_Abs = tf.abs(tf.subtract(pos_pred,center_Abs))
            
            dist_Scat = (tf.subtract(dist_Scat,width_Scat))
            dist_Abs = (tf.subtract(dist_Abs,width_Abs))
            
            # quadratic
            penalty_Scat = tf.multiply(tf.cast(penalty_outsideX, float), dist_Scat**2)
            penalty_Abs = tf.multiply(tf.cast(penalty_outsideX, float), dist_Abs**2)
            
            return tf.minimum(penalty_Scat,penalty_Abs)
        
        penalty_outsideX_e = isNotInVolumes(e_pos_pred, minX_eScat, maxX_eScat, minX_eAbs, maxX_eAbs)
        penalty_outsideX_p = isNotInVolumes(p_pos_pred, minX_pScat, maxX_pScat, minX_pAbs, maxX_pAbs)
        
        penalty_strength = 1   # first strength 0.1 or 1
        pos_penalty_p = tf.multiply(tf.cast(penalty_outsideX_p, float), penalty_strength)
        pos_penalty_e = tf.multiply(tf.cast(penalty_outsideX_e, float), penalty_strength)
        
        #K.print_tensor(pos_penalty_e, message='Pos penalty e')
        
        pos_penalty_e = event_filter * self._e_cluster_match * pos_penalty_e
        pos_penalty_p = event_filter * self._p_cluster_match * pos_penalty_p
        
        return e_loss + p_loss + pos_penalty_p + pos_penalty_e
    
    
    def _pos_loss_y(self, y_true, y_pred):
        # Penalizes predictions outside of position in y - boundary 
        # Direction along fibers
        
        event_filter = y_true[:,0] # ∈ n  
        e_pos_true = K.reshape(y_true[:,2],(-1,1)) # ∈ nx1     
        e_pos_pred = K.reshape(y_pred[:,0],(-1,1)) # ∈ nx1     
        p_pos_true = K.reshape(y_true[:,4],(-1,1)) # ∈ nx1     
        p_pos_pred = K.reshape(y_pred[:,1],(-1,1)) # ∈ nx1    
        
        # e pos
        e_loss = keras.losses.logcosh(e_pos_true, e_pos_pred) # ∈ n
        e_loss = event_filter * self._e_cluster_match * e_loss # (n*n*n) ∈ n
        
        # p pos
        p_loss = keras.losses.logcosh(p_pos_true, p_pos_pred) # ∈ n
        p_loss = event_filter * self._p_cluster_match * p_loss # (n*n*n) ∈ n
        
        def normalize_y_boundaries(x1,mean,std):
            return (x1 - mean)/std
        
        # boundary are for scatterer and absorber the same
        minY_e = normalize_y_boundaries(-50.0,  -0.23477532,   (20.77702422/10) ) # electron
        minY_p = normalize_y_boundaries(-50.0, 1.30259990e-01, (27.44766386/10) ) # photon
        
        maxY_e = normalize_y_boundaries(50.0,  -0.23477532,   (20.77702422/10) ) 
        maxY_p = normalize_y_boundaries(50.0, 1.30259990e-01, (27.44766386/10) )
        
        cond_tf = tf.ones_like(e_pos_pred) 
        
        def isNotInVolumes(pos_pred, min_y, max_y):
            too_large = tf.greater(pos_pred, cond_tf*max_y ) # too large y position prediction
            too_small = tf.greater(cond_tf*min_y, pos_pred )  # too small y position prediction
            penalty_outsideY = tf.logical_or(too_small, too_large) # Boolean for activation of penalty
            
            cutted = tf.boolean_mask( pos_pred, too_large)
            cutted2 = tf.boolean_mask( pos_pred, too_small)
            #K.print_tensor(cutted, message ='too large')      # no too large values observed
            #K.print_tensor(cutted2, message ='too small')
            
            dist_min = tf.abs(tf.subtract(pos_pred,min_y))
            dist_max = tf.abs(tf.subtract(pos_pred,max_y))
                       
            # quadratic
            penalty_Scat = tf.multiply(tf.cast(penalty_outsideY, float), dist_min**2)
            penalty_Abs = tf.multiply(tf.cast(penalty_outsideY, float), dist_max**2)
            
            penalty_Dist = tf.minimum(penalty_Scat,penalty_Abs)
            #K.print_tensor(penalty_Dist, message ='penalty in Y')
            return penalty_Dist
        
        e_penalty = isNotInVolumes(e_pos_pred, minY_e, maxY_e)
        p_penalty = isNotInVolumes(p_pos_pred, minY_p, maxY_p)
        
        cut = tf.boolean_mask(e_pos_pred, p_penalty)
        K.print_tensor(cut, message ='penalty')
        
        pos_penalty_e = event_filter * self._e_cluster_match * e_penalty
        pos_penalty_p = event_filter * self._p_cluster_match * p_penalty
        
        return e_loss + p_loss + pos_penalty_e + pos_penalty_p
        
        
    def _energy_loss_penalty(self, y_true, y_pred):
        
        # y_true is [t, e_enrg, p_enrg] 
        # to find in DataModel under @property def _target_energy
        # Penalty for exceeding maximum electron energy
        
        event_filter = y_true[:,0] # ∈ n
        e_enrg_true  = K.reshape(y_true[:,1],(-1,1)) # ∈ nx1    
        e_enrg_pred  = K.reshape(y_pred[:,0],(-1,1)) # ∈ nx1    
        p_enrg_true  = K.reshape(y_true[:,2],(-1,1)) # ∈ nx1    
        p_enrg_pred  = K.reshape(y_pred[:,1],(-1,1)) # ∈ nx1    
        
        e_loss = keras.losses.logcosh(e_enrg_true, e_enrg_pred) # ∈ n
        e_loss = event_filter * e_loss
        
        p_loss = keras.losses.logcosh(p_enrg_true, p_enrg_pred) # ∈ n
        p_loss = event_filter * p_loss
        
        # New implementation with only prediction value for constraint
        # primary energy = pred. e energy + pred. photon energy
        total_energy = e_enrg_pred + p_enrg_pred
        
        me = constants.m_e
        c = constants.c
        e_el = constants.e
        e_max_energy_pred = total_energy / ( 1 + (me*c**2*10**(-6)) / (e_el*2*total_energy) ) 
        
        #thresh   = K.reshape(y_true[:,3],(-1,1))    # Real E max on 3 position
        strength = 0.0001  # 0.0001, 
        
        if self.penalty   == 'emax_linear':
            #e_energy_penalty = strength * tf.reduce_sum( tf.math.maximum(0.0, e_enrg_pred - thresh), axis=-1) 
            e_energy_penalty = strength * tf.reduce_sum( tf.math.maximum(0.0, e_enrg_pred - e_max_energy_pred), axis=-1) 
        elif self.penalty == 'emax_quadratic':
            #e_energy_penalty = strength * tf.reduce_sum( (tf.math.maximum(0.0, e_enrg_pred - thresh))**2 , axis=-1) 
            e_energy_penalty = strength * tf.reduce_sum( (tf.math.maximum(0.0, e_enrg_pred - e_max_energy_pred))**2 , axis=-1) 
        elif self.penalty == 'emax_logcosh':
            e_energy_penalty = strength * tf.reduce_sum( tf.math.log(tf.math.cosh(tf.math.maximum(0.0, e_enrg_pred - thresh))), axis=-1)
        else:
            return e_loss + p_loss
        
        e_energy_penalty = event_filter * e_energy_penalty    
        
        return e_loss + p_loss + e_energy_penalty
    
    
    def _energy_loss(self, y_true, y_pred):
        # y_true is [t, e_enrg, p_enrg] 
        # to find in DataModel under @property def _target_energy
        # Not used here, instead _energy_loss_penalty
        
        event_filter = y_true[:,0] # ∈ n
        e_enrg_true = K.reshape(y_true[:,1],(-1,1)) # ∈ nx1    
        e_enrg_pred = K.reshape(y_pred[:,0],(-1,1)) # ∈ nx1    
        p_enrg_true = K.reshape(y_true[:,2],(-1,1)) # ∈ nx1    
        p_enrg_pred = K.reshape(y_pred[:,1],(-1,1)) # ∈ nx1    
        
        e_loss = keras.losses.logcosh(e_enrg_true, e_enrg_pred) # ∈ n
        e_loss = event_filter * e_loss
        
        p_loss = keras.losses.logcosh(p_enrg_true, p_enrg_pred) # ∈ n
        p_loss = event_filter * p_loss
        
        if self.penalty == 'cone_quadratic':
            
            #K.print_tensor(e_enrg_true, message='y_pred El energy')
            cone_penalty = self.penaltyConeDistance(y_true, y_pred)
            print("cone", cone_penalty)
            
            return e_loss + p_loss + cone_penalty
        
        return e_loss + p_loss
    
    def penaltyConeDistance(self, y_true, y_pred):
        
        # Calculation of the penalty term
        # Penalty is added to the loss terms (Energy, Position X Y Z) if the reconstructed cone is
        #  too far away from soure axis
        
        event_filter = y_true[:,0] # ∈ n
        
        penalty_cone = 0
        
        return penalty_cone
   
    
    def predict(self, data, denormalize=False, verbose=0):
        pred = self.model.predict(data)
        pred = np.concatenate([np.round(pred[2]), 
                pred[6], 
                pred[3][:,[0]], pred[4][:,[0]], pred[5][:,[0]], 
                pred[3][:,[1]], pred[4][:,[1]], pred[5][:,[1]], 
               ], axis=1)
        if denormalize:
            pred = self.data._denormalize_targets(pred)
            
        return pred
    
    def _find_matches(self, y_true, y_pred, mask=None, keep_length=True):
        # Matches of prediction with real values within a certain range
        if mask is None:
            mask = np.ones(9)
        else:
            mask = np.asarray(mask)
            
        y_true = self.data._denormalize_targets(y_true)
        y_pred = self.data._denormalize_targets(y_pred)
        
        if y_true.shape[1] == 12:   # Changed for emax from 11  to 12
            y_true = y_true[:,:-3]  # changed from  -2 to -3
        
        assert y_true.shape == y_pred.shape
        assert mask.shape == (y_true.shape[1],)
        
        l_matches = []
        for i in range(y_true.shape[0]):
            if y_true[i,0] == 0:
                if keep_length:
                    l_matches.append(0)
                continue
                
            diff_limit = np.abs(np.concatenate((
                [.5],
                [y_true[i,1] * self.energy_factor_limit],
                [y_true[i,2] * self.energy_factor_limit],
                self.position_absolute_limit,
                self.position_absolute_limit
            )))
            assert (diff_limit >= 0).all()
            
            diff = np.abs(y_true[i]-y_pred[i])
            diff = diff * mask
            
            if np.all(diff <= diff_limit):
                l_matches.append(1)
            else:
                l_matches.append(0)
        
        return l_matches
            
    def extend_history(self, history):
        '''Extend the previous training history with the new training history logs'''
        if self.history is None or self.history=={}:
            self.history = history.history
        else:
            for key in self.history.keys():
                if key in history.history:
                    self.history[key].extend(history.history[key])
                    
    def append_history(self, logs):
        '''Append the existing training history with the training logs of a signle epoch'''
        if self.history is None or self.history=={}:
            self.history = {}
            for key in logs.keys():
                self.history[key] = [logs[key]]
        else:
            for key in self.history.keys():
                if key in logs.keys():
                    self.history[key].append(logs[key])
                    
    def plot_training_loss(self, mode='eff', skip=0, smooth=True, summed_loss=True):
        def plot_line(ax, key, label, style, color):
            metric = self.history[key][skip:]
            metric = utils.exp_ma(metric, factor=.8) if smooth else metric
            ax.plot(np.arange(1,len(metric)+1)+skip, metric, style, label=label, color=color)
            
        def plot_metric(ax, key, label, color):
            if key in self.history:
                plot_line(ax, key, label, '-', color)
                if 'val_' + key in self.history:
                    plot_line(ax, 'val_' + key, None, '--', color)

        fig, ax1 = plt.subplots(figsize=(12,4))
        ax2 = ax1.twinx()  

        color = 'tab:blue'
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss', color=color)
        ax2.set_ylim(bottom=0, top =8)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid()
        
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Precision', color=color)  
        ax1.tick_params(axis='y', labelcolor=color)
        
        if summed_loss:
            plot_metric(ax2, 'loss', 'Loss', 'tab:blue')
        
        if mode == 'acc':
            plot_metric(ax1, 'type__type_accuracy', 'Type accuracy', 'tab:pink')
            plot_metric(ax1, 'type__type_tp_rate', 'TP rate', 'tab:purple')
            plot_metric(ax1, 'e_cluster__cluster_accuracy', 'e cluster acc', 'tab:red')
            plot_metric(ax1, 'p_cluster__cluster_accuracy', 'p cluster acc', 'tab:brown')
        elif mode == 'eff':
            plot_metric(ax1, 'eff', 'Efficiency', 'tab:pink')
            plot_metric(ax1, 'pur', 'Purity', 'tab:orange')
        elif mode == 'loss':
            plot_metric(ax2, 'e_cluster_loss', 'Cluster e', 'tab:pink')
            plot_metric(ax2, 'p_cluster_loss', 'Cluster p', 'tab:orange')
            plot_metric(ax2, 'pos_x_loss', 'Pos x', 'tab:brown')
            plot_metric(ax2, 'pos_y_loss', 'Pos y', 'tab:red')
            plot_metric(ax2, 'pos_z_loss', 'Pos z', 'tab:purple')
            plot_metric(ax2, 'type_loss', 'Type', 'tab:orange')
            plot_metric(ax2, 'energy_loss', 'Energy', 'tab:cyan')
        elif mode == 'loss-cluster':
            plot_metric(ax2, 'e_cluster_loss', 'Cluster e', 'tab:pink')
            plot_metric(ax2, 'p_cluster_loss', 'Cluster p', 'tab:orange')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        elif mode == 'loss-pos':
            plot_metric(ax2, 'pos_x_loss', 'Pos x', 'tab:brown')
            plot_metric(ax2, 'pos_y_loss', 'Pos y', 'tab:red')
            plot_metric(ax2, 'pos_z_loss', 'Pos z', 'tab:purple')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        elif mode == 'loss-type':
            plot_metric(ax2, 'type_loss', 'Type', 'tab:orange')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        elif mode == 'loss-energy':
            plot_metric(ax2, 'energy_loss', 'Energy', 'tab:cyan')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        else:
            raise Exception('Invalid mode')
            
        ax1.plot([], '--', color='tab:gray', label='Validation')
                
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()

        ax2.yaxis.set_label_position('left')
        ax2.yaxis.tick_left()

        fig.legend(loc='upper left')
        fig.tight_layout()
        plt.show()
        
    def evaluate(self):
        [loss, e_cluster_loss, p_cluster_loss, type_loss, 
         pos_x_loss, pos_y_loss, pos_z_loss, energy_loss, 
         e_cluster__cluster_accuracy, p_cluster__cluster_accuracy, 
         type__type_accuracy, type__type_tp_rate] = self.model.evaluate(
            self.data.test_x, self.data.test_y, verbose=0)
        
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        l_matches = self._find_matches(y_true, y_pred, keep_length=False)
        # Effiency: No correctly recon. Compton events from Compton events in data
        effeciency = np.mean(l_matches)
        # Purity: no correctly recon. Compton events of all ident. Compton events
        purity = np.sum(l_matches) / np.sum(y_pred[:,0]) 
        
        identified_events = np.array(self._find_matches(y_true, y_pred, keep_length=True, mask=[1]+([0]*8))).astype(bool)
        y_pred = self.data._denormalize_targets(y_pred[identified_events])
        y_true = self.data._denormalize_targets(y_true[identified_events])
        euc = y_true[:,3:9] - y_pred[:,3:9]  # 
        euc = euc.reshape((-1,3))
        euc = np.power(euc, 2)
        euc = np.sqrt(np.sum(euc, axis=1))
        mean_euc = np.mean(euc)
        std_euc = np.std(euc)
                
        print('AI model')
        print('  Loss:       {:8.5f}'.format(loss))
        print('    -Type:        {:8.5f} * {:5.2f} = {:7.5f}'.format(type_loss, self.weight_type, 
                                                                 type_loss * self.weight_type))
        print('    -Pos X:       {:8.5f} * {:5.2f} = {:7.5f}'.format(pos_x_loss, self.weight_pos_x, 
                                                                 pos_x_loss * self.weight_pos_x))
        print('    -Pos Y:       {:8.5f} * {:5.2f} = {:7.5f}'.format(pos_y_loss, self.weight_pos_y, 
                                                                 pos_y_loss * self.weight_pos_y))
        print('    -Pos Z:       {:8.5f} * {:5.2f} = {:7.5f}'.format(pos_z_loss, self.weight_pos_z, 
                                                                 pos_z_loss * self.weight_pos_z))
        print('    -Energy:      {:8.5f} * {:5.2f} = {:7.5f}'.format(energy_loss, self.weight_energy, 
                                                                 energy_loss * self.weight_energy))
        print('    -Cls e:       {:8.5f} * {:5.2f} = {:7.5f}'.format(e_cluster_loss, self.weight_e_cluster, 
                                                                 e_cluster_loss * self.weight_e_cluster))
        print('    -Cls p:       {:8.5f} * {:5.2f} = {:7.5f}'.format(p_cluster_loss, self.weight_p_cluster, 
                                                                 p_cluster_loss * self.weight_p_cluster))
        print('  Accuracy:   {:8.5f}'.format(type__type_accuracy))
        print('    -TP rate:     {:8.5f}'.format(type__type_tp_rate))
        print('    -Cls e rate:  {:8.5f}'.format(e_cluster__cluster_accuracy))
        print('    -Cls p rate:  {:8.5f}'.format(p_cluster__cluster_accuracy))
        print('  Efficiency: {:8.5f}'.format(effeciency))
        print('  Purity:     {:8.5f}'.format(purity))
        print('  Euc mean:   {:8.5f}'.format(mean_euc))
        print('  Euc std:    {:8.5f}'.format(std_euc))
        
        # 
        y_pred = self.data.reco_test
        y_true = self.data.test_row_y
        l_matches = self._find_matches(y_true, y_pred, keep_length=False)
        effeciency = np.mean(l_matches)
        purity = np.sum(l_matches) / np.sum(y_pred[:,0])
        accuracy = self._type_accuracy(y_true[:,0], y_pred[:,0]).numpy()
        tp_rate = self._type_tp_rate2(y_true[:,0], y_pred[:,0]).numpy()
        
        identified_events = np.array(self._find_matches(y_true, y_pred, keep_length=True, mask=[1]+([0]*8))).astype(bool)
        y_pred = self.data._denormalize_targets(y_pred[identified_events])
        y_true = self.data._denormalize_targets(y_true[identified_events])
        euc = y_true[:,3:9] - y_pred[:,3:9]
        euc = euc.reshape((-1,3))
        euc = np.power(euc, 2)
        euc = np.sqrt(np.sum(euc, axis=1))
        mean_euc = np.mean(euc)
        std_euc = np.std(euc)
        
        print('\nReco')
        print('  Accuracy:   {:8.5f}'.format(accuracy))
        print('    -TP rate:     {:8.5f}'.format(tp_rate))
        print('  Efficiency: {:8.5f}'.format(effeciency))
        print('  Purity:     {:8.5f}'.format(purity))
        print('  Euc mean:   {:8.5f}'.format(mean_euc))
        print('  Euc std:    {:8.5f}'.format(std_euc))

    def save(self, file_name):
        self.model.save_weights('ModelsTrained/' + file_name + '.h5', save_format='h5')
        with open('ModelsTrained/' + file_name + '.hst', 'wb') as f_hist:
            pkl.dump(self.history, f_hist)
        with open('ModelsTrained/' + file_name + '.opt', 'wb') as f_hist:
            pkl.dump(self.model.optimizer.get_weights(), f_hist)
        
            
    def load(self, file_name, optimizer=False):
        self.model.load_weights('ModelsTrained/' + file_name+'.h5')
        with open('ModelsTrained/' + file_name + '.hst', 'rb') as f_hist:
            self.history = pkl.load(f_hist)
        if optimizer:
            with open('ModelsTrained/' + file_name + '.opt', 'rb') as f_hist:
                self.model.optimizer.set_weights(pkl.load(f_hist))
     
    
    def plot_electron_energy(self, mode='type-match', add_reco=True, focus=False):
        # Plot electron energies
        # Plot difference in predictions of electron energy
        # Plot difference of prediction to maximum allowed energy
 
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        y_reco = self.data.reco_test
        
        
        if mode == 'all-match':
            mask = None
        elif mode == 'pos-match':
            mask = [1,0,0,1,1,1,1,1,1]
        elif mode == 'type-match':
            mask = [1] + ([0] * 8)
        elif mode == 'miss':
            mask = None
        else:
            raise Exception('mode {} not recognized'.format(mode))        

        l_matches = np.array(self._find_matches(y_true, y_pred, mask, keep_length = True)).astype(bool)
        l_reco_matches = np.array(self._find_matches(y_true, y_reco, mask, keep_length = True)).astype(bool)
        if mode == 'pos-match':
            all_matches = np.array(self._find_matches(y_true, y_pred, keep_length = True)).astype(bool)
            all_reco_matches = np.array(self._find_matches(y_true, y_reco, keep_length = True)).astype(bool)
            l_matches = (l_matches * np.invert(all_matches)).astype(bool)
            l_reco_matches = (l_reco_matches * np.invert(all_reco_matches)).astype(bool)

        y_pred = self.data._denormalize_targets(y_pred)
        y_true = self.data._denormalize_targets(y_true)
        y_reco = self.data._denormalize_targets(y_reco)

        if mode == 'miss':
            l_matches = (np.invert(l_matches) * y_true[:,0]).astype(bool)
            l_reco_matches = (np.invert(l_reco_matches) * y_true[:,0]).astype(bool)

        y_emax = y_true[:,-1]   # Choose max electron energy real, last element of target list
        y_true = y_true[:,:-3]   # same length as others, cut away e and p cluster and e max
        
                
        # Recalculation to primary energy
        me = constants.m_e
        c = constants.c
        e_el = constants.e
        y_prim = np.asarray(y_emax)/2 + np.sqrt((np.asarray(y_emax)/2)**2 + (10**(-6)*me*c**2*np.asarray(y_emax))/(2*e_el))
        
        y_true = y_true[l_matches] # Matches of
        y_emax = y_emax[l_matches]
        y_prim = y_prim[l_matches]
        y_pred = y_pred[l_matches]
        y_reco = y_reco[l_reco_matches]
        
        data_etrue = y_true[:,1]  # true electron energy 
        data_emax = y_emax       # true max energy limit
        data_epred = y_pred[:,1]  #
        data_reco = y_reco[:,1]   # electron energy from reco
        data_epred_emax = np.asarray(data_emax) - np.asarray(data_epred)
        data_diff = np.asarray(data_etrue)-np.asarray(data_epred)
        
        fig_size = (10,4)

        def plot_hist(data, pos, title, width, x_min=None, x_max=None):
            plt.figure(figsize=fig_size)
            plt.title(title)

            if add_reco:
                reco_data = y_reco[:,pos]

            if x_min is None and x_max is None:
                if add_reco:
                    x_min = min(int(np.floor(data.min())), int(np.floor(reco_data.min())))
                    x_max = max(int(np.ceil(data.max())), int(np.ceil(reco_data.max())))

                else:
                    x_min = int(np.floor(data.min()))
                    x_max = int(np.ceil(data.max()))

            x_min = np.ceil(x_min / width) * width
            x_max = ((x_max // width)+1) * width
       
            n, bins, _ = plt.hist(data, np.arange(x_min, x_max, width), histtype='step', 
                                  label=str(len(data)), color='tab:blue')
           
            plt.ylabel('Count')
            plt.xlabel('Energy / MeV')
            plt.legend()
            plt.show()

        bin_width = 0.02 # usually 0.05
        if focus:
            plot_hist(data_etrue, 1, 'e- energy true', bin_width, -2, 2)
            plot_hist(data_epred, 1, 'e- energy predicted', bin_width, -2, 2)
            plot_hist(data_reco,  1,  'e- energy reco', bin_width, -2, 2)
            plot_hist(data_emax,  1,  'e- true maximum energy', bin_width, 3, 5)
            plot_hist(data_epred_emax, 1, 'e- energy prediction - maximum energy', .05, -2, 2)
            plot_hist(y_prim,  1,  'primary energy', bin_width, 4, 5)
            plot_hist(data_diff,  1,  'difference e- energy real - predicted', bin_width, -2, 2)
        else:
            plot_hist(data_etrue, 1, 'e- energy true',bin_width)
            plot_hist(data_epred, 1, 'e- energy predicted', bin_width)
            plot_hist(data_reco,  1,  'e- energy reco', bin_width)
            plot_hist(data_emax,  1,  'e- true maximum energy', bin_width)
            plot_hist(data_epred_emax, 1, 'maximum energy - e- energy prediction', bin_width)
            plot_hist(y_prim,  1,  'primary energy', bin_width)
            plot_hist(data_diff,  1,  'difference e- energy real - predicted', bin_width)
    
    '''
    def evaluationComptonConesOld(self, mask = 'type', events='True'):
        # To be deleted when not needed anymore for cross checks
        
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        y_reco = self.data.reco_test

        l_matches_all  = np.array(self._find_matches(y_true, y_pred, mask=None, keep_length = True)).astype(bool)   
        l_matches_type = np.array(self._find_matches(y_true, y_pred, mask= [1] + ([0] * 8), keep_length = True)).astype(bool)

        if mask == 'type':
            y_pred = y_pred[l_matches_type]
            y_true = y_true[l_matches_type]
        elif mask == 'all':
            y_pred = y_pred[l_matches_all]
            y_true = y_true[l_matches_all]

        y_pred = self.data._denormalize_targets(y_pred)
        y_true = self.data._denormalize_targets(y_true)

        e_energy = y_true[:,1]
        p_energy = y_true[:,2]
        me = 0.510999
        arc_base = np.abs(1 - me *(1/p_energy - 1/(e_energy + p_energy))) # Argument for arccos
        valid_arc = arc_base <= 1                                         # arccos(x), x goes from -1 to 1

        print("Check argument of arccos: Invalid events", np.sum(np.invert(valid_arc)), " from ", len(valid_arc))
        print("True: Invalid events", np.sum(np.invert(valid_arc)), " from ", len(valid_arc))
        
        y_pred = y_pred[valid_arc]
        y_true = y_true[valid_arc]

        def reconstructionCone(y, plotTitle):
            
            compton_angle, phi_angle = [], []
            array_y_OM, array_y_OC = np.array([]), np.array([])
            array_z_OM, array_z_OC = np.array([]), np.array([])
            countp1,countp2,countp3,countn1,countn2,countn3, count_theta=0,0,0,0,0,0,0
            for i in range(0, len(y)): 
                
                # Compton angle
                e_energy = y[:,1][i]
                p_energy = y[:,2][i]

                theta = np.arccos(1.0 - me *(1.0/p_energy - 1.0/( e_energy + p_energy )))
                theta = np.rad2deg(theta)
                                
                # Angle selection (~ 3% back-scattering)
                if theta >= 90:
                    # Case B.2: Backward scattering, electron in absorber
                    theta = 180 - theta
                    count_theta+=1
                else:
                    theta = theta
                    # Case B.1: Forward scattering, electron in scatterer
                                
                # Vector pointing to interaction point of electron (apex of cone)
                x_OA = y[:,3][i]
                y_OA = y[:,4][i]
                z_OA = y[:,5][i]

                # Vector pointing to interaction point of photon
                x_OP = y[:,6][i]
                y_OP = y[:,7][i]
                z_OP = y[:,8][i]

                # Vector from origin to point of intersection of cone axis with x=0-plane
                x_OC = x_OA - x_OA * (x_OA - x_OP) / (x_OA - x_OP)    #np.asarray([0.0]*len(x_OA))
                y_OC = y_OA - x_OA * (y_OA - y_OP) / (x_OA - x_OP)
                z_OC = z_OA - x_OA * (z_OA - z_OP) / (x_OA - x_OP)
                
                if y_OC >= 0:
                    compton_angle.append(theta)
                    # Vector pointing from apex of cone along cone axis to intersection with x=0 plane
                    # From point A to point C
                    # Plane at x=0 (parallel to the surface of the Compton camera), beam axis is along z direction 
                    x_AC = x_OC - x_OA
                    y_AC = y_OC - y_OA
                    z_AC = z_OC - z_OA

                    # Angle between cone axis and x axis, from radians to degree
                    distance_PA = np.sqrt( (x_OP - x_OA)**2 + (y_OP - y_OA)**2 + (z_OP - z_OA)**2 )  #######
                    phi = np.rad2deg( np.arcsin( np.abs(y_OA - y_OP) /  distance_PA ) )
                    # Distance between cone apex point and intersection point with x=0 plane, at z=z_OC,
                    distance_Aaxis = np.sqrt( (z_OA - z_OC)**2 + x_OA**2)    # Probably wrong assumption
                    phi_angle.append(phi)

                    # Cases: 
                    # 1. Positive y_OC
                    # 2. Negative y_OC
                    # A: Negative slope cone axis + theta < phi
                    # B: Negative slope cone axis + theta > phi 
                    # C: Positive slope cone axis
                    
                    if (y_OC >= 0):
                        # A: Negative slope cone axis
                        if (theta <= phi) and (y_OA >= y_OP):
                            a = +1
                            b = -1
                            c = +1
                            countp1+=1
                        elif (theta >= phi) and (y_OA >= y_OP):
                            a = -1
                            b = +1
                            c = -1
                            countp2+=1
                        else:
                            # Cases y_OA <= y_OP
                            a = +1
                            b = +1
                            c = -1
                            countp3+=1
                    if (y_OC <= 0):
                        if (theta <= phi) and (y_OA <= y_OP):
                            a = +1
                            b = -1
                            c = -1
                            countn1+=1
                        elif (theta >= phi) and (y_OA <= y_OP):
                            a = -1
                            b = +1
                            c = +1
                            countn2+=1
                        else:
                            # Cases y_OA >= y_OP
                            a = +1
                            b = +1
                            c = +1
                            countn3+=1

                    y_diff = distance_Aaxis * np.tan( np.deg2rad( a*phi + b*theta ) )
                    # Distance from source axis (z axis) along y (z constant, x = 0) to the surface of the Compton cone

                    y_OM   = y_OA + c*y_diff
                    z_OM   = z_OC

                    array_y_OM = np.append(array_y_OM, y_OM)
                    array_z_OM = np.append(array_z_OM, z_OM)
                    array_y_OC = np.append(array_y_OC, y_OC)
                    array_z_OC = np.append(array_z_OC, z_OC)
                
            compton_angle=np.asarray(compton_angle)
            phi_angle=np.asarray(phi_angle)
            print("Y OM larger y OC ", np.sum([array_y_OM>array_y_OC])) 
            
            print("Which hits yom larger y OC: yOM,yOC", array_y_OM[array_y_OM>array_y_OC], array_y_OC[array_y_OM>array_y_OC])
            print("Which hits yom larger y OC: theta,phi", compton_angle[array_y_OM>array_y_OC], phi_angle[array_y_OM>array_y_OC])
            print("\n Cone missed axis ,positive yoC  ", np.sum([array_y_OM[array_y_OC>0]>0]))
            print("Cone missed axis 5 deviation, post ", np.sum([array_y_OM[array_y_OC>0]>5]))
            
            print("Y OC", array_y_OC)
            print("Y OM", array_y_OM)
            print("Y OM", array_y_OM[10:30])
            
            print("Cases: ", countp1, countp2, countp3, countn1, countn2, countn3, count_theta)
            
            # Accepted cones which miss axis within 5 mm (2.5 mm width of proton beam)
            plt.figure()
            plt.title('Cone axis interception with x=0 plane')
            plt.scatter(array_z_OC, array_y_OC)
            plt.scatter(array_z_OC[array_y_OM>array_y_OC], array_y_OC[array_y_OM>array_y_OC], color = 'tab:orange')
            plt.xlabel("z")
            plt.ylabel("y")
            plt.show()
            
            plt.figure()
            plt.scatter(array_z_OM, array_y_OM)
            plt.scatter(array_z_OM[array_y_OM>array_y_OC], array_y_OM[array_y_OM>array_y_OC], color = 'tab:orange')
            plt.xlabel("z")
            plt.ylabel("y")
            plt.show()
            
            plt.figure()
            plt.scatter(array_z_OM, compton_angle)
            plt.xlabel("z (M)")
            plt.ylabel("Scatter angle")
            plt.show()
            
            plt.figure()
            plt.scatter(array_z_OC, compton_angle)
            plt.xlabel("z (C)")
            plt.ylabel("Compton cone angle")
            plt.show()
            
            plt.figure()
            plt.scatter(array_z_OC, phi_angle)
            plt.xlabel("z (C)")
            plt.ylabel("Cone axis to x axis angle")
            plt.show()
            
        if events=="True":
            reconstructionCone(y_true, 'True')    
        else:
            reconstructionCone(y_pred, 'Prediction')
        '''
            
    def isInsideVolume(self, mask = 'type', data = 'prediction'):
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        
        l_matches_all  = np.array(self._find_matches(y_true, y_pred, mask = None, keep_length = True)).astype(bool)   
        l_matches_type = np.array(self._find_matches(y_true, y_pred, mask = [1] + ([0] * 8), keep_length = True)).astype(bool)
        
        if mask == 'type':
            y_pred = y_pred[l_matches_type]
            y_true = y_true[l_matches_type]
        elif mask == 'all':
            y_pred = y_pred[l_matches_all]
            y_true = y_true[l_matches_all]
            
        y_pred = self.data._denormalize_targets(y_pred)
        y_true = self.data._denormalize_targets(y_true)
        
        if data == 'prediction':
            y = y_pred
        else:
            y = y_true
        
        eIsInsideScat = [ (193.5 < y[:,3]) & (y[:,3] < 206.5) & ( -50.< y[:,4]) & (y[:,4] < 50.) & (-49.4 < y[:,5]) & (y[:,5] < 49.4) ]
        eIsInsideAbs  = [ (380.5< y[:,3]) & (y[:,3] < 419.5) & ( -50.< y[:,4]) & (y[:,4] < 50.) & (-49.4 < y[:,5]) & (y[:,5] < 49.4) ]
        pIsInsideScat = [ (193.5 < y[:,6]) & (y[:,6] < 206.5) & ( -50.< y[:,7]) & (y[:,7] < 50.) & (-49.4 < y[:,8]) & (y[:,8] < 49.4) ]
        pIsInsideAbs  = [ (380.5< y[:,6]) & (y[:,6] < 419.5) & ( -50.< y[:,7]) & (y[:,7] < 50.) & (-49.4 < y[:,8]) & (y[:,8] < 49.4) ]
        
        eIsNotInsideScat = np.invert(eIsInsideScat)
        eIsNotInsideAbs  = np.invert(eIsInsideAbs)
        
        pIsNotInsideScat = np.invert(pIsInsideScat)
        pIsNotInsideAbs  = np.invert(pIsInsideAbs)
        
        # Checking single directions for photon and electron 
        pIsInsideScatX = [(193.5 < y[:,6]) & (y[:,6] < 206.5)]
        pIsInsideAbsX = [(380.5 < y[:,6]) & (y[:,6] < 419.5)]
        pIsInsideScatY = [(-50. < y[:,7]) & (y[:,7] < 50.)]
        pIsInsideScatZ = [(-49.4 <= y[:,8]) & (y[:,8] <= 49.4)]
        
        eIsInsideScatX = [(193.5 < y[:,3]) & (y[:,3] < 206.5)]
        eIsInsideAbsX = [(380.5 < y[:,3]) & (y[:,3] < 419.5)]
        eIsInsideScatY = [(-50. < y[:,4]) & (y[:,4] < 50.)]
        eIsInsideScatZ = [(-49.4 <= y[:,5]) & (y[:,5] <= 49.4)]
        
        #print("Wrong cases true values ", y[:,3][eIsNotInsideScat & eIsNotInsideAbs],y[:,4][eIsNotInsideScat & eIsNotInsideAbs],y[:,5][eIsNotInsideScat & eIsNotInsideAbs])
        
        print("{:10d} Events (matched)".format(len(y)))
        print("\n{:10d} Events e predicted inside scatterer".format(np.sum(eIsInsideScat)))
        print("{:10d} Events e predicted inside absorber".format(np.sum(eIsInsideAbs)))
        print("{:10d} Events e pred. outside of volumes".format(np.sum([eIsNotInsideScat & eIsNotInsideAbs])))
        print("{:8.4f} Percent of outside pred. from all (matched) events".format(100*np.sum([eIsNotInsideScat & eIsNotInsideAbs])/len(y)))
        
        print('\n    {:10d} ({:8.4f} %) x missed, e-'.format(np.sum([np.invert(eIsInsideScatX) & np.invert(eIsInsideAbsX)]), 100*np.sum([np.invert(eIsInsideScatX) & np.invert(eIsInsideAbsX)])/np.sum([eIsNotInsideScat & eIsNotInsideAbs])))
        print('    {:10d} ({:8.4f} %) y missed, e-'.format(np.sum([np.invert(eIsInsideScatY)]), 100*np.sum([np.invert(eIsInsideScatY)])/np.sum([eIsNotInsideScat & eIsNotInsideAbs])))
        print('    {:10d} ({:8.4f} %) z missed, e-'.format(np.sum(np.invert(eIsInsideScatZ)), 100*np.sum(np.invert(eIsInsideScatZ))/np.sum([eIsNotInsideScat & eIsNotInsideAbs])))
        
        print("\n{:10d} Events ph predicted inside scatterer,".format(np.sum(pIsInsideScat)))
        print("{:10d} Events ph predicted inside absorber".format(np.sum(pIsInsideAbs)))
        print("{:10d} Events ph outside of volumes".format(np.sum([pIsNotInsideScat & pIsNotInsideAbs])))
        print("{:8.4f} Percent of hits from all (matched) events".format(100*np.sum([pIsNotInsideScat & pIsNotInsideAbs]) / len(y)))
        
        print('\n    {:10d} ({:8.4f} %) x missed, photon'.format(np.sum([np.invert(pIsInsideScatX) & np.invert(pIsInsideAbsX)]), 100*np.sum([np.invert(pIsInsideScatX) & np.invert(pIsInsideAbsX)])/np.sum([pIsNotInsideScat & pIsNotInsideAbs])))
        print('    {:10d} ({:8.4f} %) y missed, photon'.format(np.sum([np.invert(pIsInsideScatY)]), 100*np.sum([np.invert(pIsInsideScatY)])/np.sum([pIsNotInsideScat & pIsNotInsideAbs])))
        print('    {:10d} ({:8.4f} %) z missed, photon'.format(np.sum([np.invert(pIsInsideScatZ)]), 100*np.sum(np.invert(pIsInsideScatZ) )/np.sum([pIsNotInsideScat & pIsNotInsideAbs])))
        
        
    def evaluationComptonCones(self, mask = 'type', events='True', save = False):
        
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        
        l_matches_all  = np.array(self._find_matches(y_true, y_pred, mask = None, keep_length = True)).astype(bool)   
        l_matches_type = np.array(self._find_matches(y_true, y_pred, mask = [1] + ([0] * 8), keep_length = True)).astype(bool)
        l_matches_pos  = np.array(self._find_matches(y_true, y_pred, mask = [1,0,0,1,1,1,1,1,1], keep_length = True)).astype(bool) # Position and type
        
        if mask == 'type':
            y_pred = y_pred[l_matches_type]
            y_true = y_true[l_matches_type]
        elif mask == 'all':
            y_pred = y_pred[l_matches_all]
            y_true = y_true[l_matches_all]
        elif mask == 'position':
            y_pred = y_pred[l_matches_pos]
            y_true = y_true[l_matches_pos]
        else:
            y_pred = y_pred[ y_pred[:,0]==1]
            y_true = y_true[ y_true[:,0]==1]
            
        y_pred = self.data._denormalize_targets(y_pred)
        y_true = self.data._denormalize_targets(y_true)

        if events == "True":
            e_energy = y_true[:,1]
            p_energy = y_true[:,2]
        else:
            e_energy = y_pred[:,1]
            p_energy = y_pred[:,2]
            
        me = 0.51099895        # electron mass in MeV
        arc_base = np.abs(1 - me *(1/p_energy - 1/(e_energy + p_energy))) # Argument for arccos
        valid_arc = arc_base <= 1                                         # arccos(x), x goes from -1 to 1
        y_pred = y_pred[valid_arc]
        y_true = y_true[valid_arc]
        
        print("Check argument of arccos: Invalid events", np.sum(np.invert(valid_arc)), " from ", len(valid_arc))
        
        if events == "True":
            y = y_true
        else:
            y = y_pred
            
        print('  Number of valid events (arccos): {:8.5f}'.format(len(y)))

        # Arrays for vectors with x,y,z components
        array_OM, array_OC = np.empty((0,3), float), np.empty((0,3), float)
        array_OA, array_OP = np.empty((0,3), float), np.empty((0,3), float)
        compton_angle, y_selection = [], []
       
        for i in range(0, len(y)): 

            e_energy = y[:,1][i]
            p_energy = y[:,2][i]
            
            # Compton scattering angle (not accounting for Doppler effect)
            theta = np.rad2deg( np.arccos(1.0 - me *(1.0/p_energy - 1.0/( e_energy + p_energy ))) )
            
            # Vector pointing to interaction point of electron (apex of cone)
            vector_OA = np.array([ y[:,3][i], y[:,4][i], y[:,5][i] ])
            
            # Vector pointing to interaction point of photon
            vector_OP = np.array([ y[:,6][i], y[:,7][i], y[:,8][i] ])
            
            # Vector from origin to point of intersection of cone axis with x=0-plane
            vector_OC = vector_OA - vector_OA[0] * ( vector_OA - vector_OP ) / ( vector_OA[0] - vector_OP[0] )
            
            if vector_OC[1] <= 0 and theta <= 90: # Apply selection    vector_OC[1] >= 0 and 

                 # Angle selection
                if theta >= 90:
                    # Case 2: Backward scattering, electron in absorber, (~ 3% back-scattering)
                    theta = 180 - theta
                else:
                    # Case 1: Forward scattering, electron in scatterer
                    theta = theta

                compton_angle.append(theta)

                # Vector pointing from apex of cone along cone axis to intersection with x=0 plane
                vector_AC = vector_OC - vector_OA

                # unit vector in AC
                norm_factor_AC = np.linalg.norm(vector_AC)
                norm_x_AC = vector_AC[0] / norm_factor_AC
                norm_z_AC = vector_AC[2] / norm_factor_AC

                # Define rotation axis, vectors in x and z plane (Rotation downwards along y)
                nrot_factor = 1 / (np.sqrt( 1 + ( norm_z_AC / norm_x_AC )**2 ))
                nx = nrot_factor * (- norm_z_AC / norm_x_AC)
                ny = nrot_factor * 0

                if vector_OC[1] >= 0:
                    nz = nrot_factor * 1
                else:
                    nx = - nx
                    nz = - nrot_factor * 1

                # Rotation matrix
                cos = np.cos(np.deg2rad(theta))
                sin = np.sin(np.deg2rad(theta))
                Rot = np.array([ [ nx**2*( 1 - cos ) + cos,   0 - nz*sin ,   nx*nz*( 1 - cos ) + 0   ], 
                               [   0 + nz*sin             ,   0 + cos    ,   0 - nx*sin              ], 
                               [   nz*nx*( 1 - cos ) -  0 ,   0 + nx*sin ,   nz**2*( 1 - cos ) + cos ] ])

                vector_AC_rot = Rot.dot(vector_AC)

                # Intersection vector_AC_rot with x=0 plane, fix point is A
                y_OM = vector_OA[1] - vector_OA[0]/vector_AC_rot[0] * vector_AC_rot[1]
                z_OM = vector_OA[2] - vector_OA[0]/vector_AC_rot[0] * vector_AC_rot[2]

                array_OM = np.append(array_OM, [[0.0, y_OM, z_OM]], axis=0)
                array_OC = np.append(array_OC, [vector_OC], axis=0)
                array_OA = np.append(array_OA, [vector_OA], axis=0)
                array_OP = np.append(array_OP, [vector_OP], axis=0)

        print('  Number of valid events (arccos): {:8.5f}       '.format(np.sum([array_OM[:,1]>array_OC[:,1]])))
        print('  Number of valid events (arccos): {:8.5f}       '.format(np.sum([array_OM[:,1]<array_OC[:,1]])))
        print('  Sanity check (No. y_OM larger y_OC): {:8.5f}   '.format(  np.sum([array_OM[:,1]>array_OC[:,1]]) ))
        print('  No. axis missed, pos y_OC, 0 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]>0]>0])  ))
        print('  No. axis missed, pos y_OC, 5 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]>0]>5])  ))
        print('  No. axis missed, pos y_OC, 8 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]>0]>8])  ))
        print('  No. axis missed, pos y_OC, 10 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]>0]>10])  ))
        print('  No. axis missed, pos y_OC, 12 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]>0]>12])  ))
        print('  No. axis missed, pos y_OC, 20 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]>0]>20])  ))
        #print('  No. axis missed, pos y_OC, %: {:8.5f}          '.format( 100*np.sum([array_OM[:,1][array_OC[:,1]>0]>5]) / len(array_OM[:,1][array_OC[:,1]>0]) ))  
        print('  No. axis missed, neg y_OC, 0 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]<0]<0])  ))
        print('  No. axis missed, neg y_OC, 5 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]<0]<-5]) ))
        print('  No. axis missed, neg y_OC, 8 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]<0]<-8]) ))
        print('  No. axis missed, neg y_OC, 10 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]<0]<-10]) ))
        print('  No. axis missed, neg y_OC, 12 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]<0]<-12]) ))
        print('  No. axis missed, neg y_OC, 20 tolerance: {:8.5f}'.format( np.sum([array_OM[:,1][array_OC[:,1]<0]<-20]) ))
        #print('  No. axis missed, neg y_OC, %: {:8.5f}          '.format( 100*np.sum([array_OM[:,1][array_OC[:,1]<0]<-5]) / len(array_OM[:,1][array_OC[:,1]<0]) ))      
        
        #print("Outliers in MCtruth", array_OM[array_OM[:,1]>1000], array_OA[array_OM[:,1]>1000])
        
        # Special selected events in plots:
        #selection = array_OM[:,1]>1000  # [array_OM[:,1]>5]
        
        selection = (array_OP[:,0]<300) & (array_OP[:,0]>100) # peculiar photon positions for forward scattering
        #selection = np.full_like(array_OM[:,1], False, dtype = bool)      
        #for i in range(0,len(array_OM[:,1])):
        #    if(array_OC[:,1][i]>0 and array_OM[:,1][i]>5):
        #        selection[i] = True
        #    if(array_OC[:,1][i]<0 and array_OM[:,1][i]<-5):
        #        selection[i] = True
        
        plt.figure()
        plt.title(events)
        plt.scatter(array_OM[:,2], array_OM[:,1])
        plt.scatter(array_OM[:,2][selection], array_OM[:,1][selection], marker='*', color='tab:red', label = 'Missed beam axis')
        plt.xlabel("z")
        plt.ylabel("y")
        plt.legend()
        plt.ylim(-70,800)
        plt.xlim(-170,170)
        if save == True: plt.savefig(self.savefigpath + 'Cone_' + events + '_' + mask +'_Mpoints_ZY')
        
        plt.figure()
        plt.title(events)
        plt.scatter(array_OA[:,0], array_OA[:,1], label= 'Electron')
        plt.scatter(array_OP[:,0], array_OP[:,1], label= 'Photon')
        plt.scatter(array_OC[:,0], array_OC[:,1], label= 'Cone axis intersection')
        plt.scatter(array_OA[:,0][selection], array_OA[:,1][selection], marker='*', color='tab:red', label = str( np.sum([selection]) ) + ' missed')
        plt.scatter(array_OP[:,0][selection], array_OP[:,1][selection], marker='*', color='tab:red')
        plt.scatter(array_OC[:,0][selection], array_OC[:,1][selection], marker='*', color='tab:red')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(-130,130)
        plt.xlim(-20,435)
        plt.legend()
        if save == True: plt.savefig(self.savefigpath + 'Cone_' + events + '_' + mask +'_EPCpoints_x-y')
        
        plt.figure()
        plt.title(events)
        plt.scatter(array_OA[:,0], array_OA[:,2], label= 'Electron')
        plt.scatter(array_OP[:,0], array_OP[:,2], label= 'Photon')
        plt.scatter(array_OC[:,0], array_OC[:,2], label= 'Cone axis intersection')
        plt.scatter(array_OA[:,0][selection], array_OA[:,2][selection], marker='*', color='tab:red', label = str( np.sum([selection]) ) + ' missed')
        plt.scatter(array_OC[:,0][selection], array_OC[:,2][selection], marker='*', color='tab:red')
        plt.scatter(array_OP[:,0][selection], array_OP[:,2][selection], marker='*', color='tab:red')
        plt.xlabel("x")
        plt.ylabel("z")
        plt.ylim(-165,165)
        plt.xlim(-20,435)
        plt.legend()
        if save == True: plt.savefig(self.savefigpath + 'Cone_' + events + '_' + mask +'_EPCpoints_x-z')
        
        plt.figure()
        plt.title(events)
        plt.scatter(array_OM[:,2], compton_angle)
        plt.scatter(array_OC[:,2][selection], np.asarray(compton_angle)[selection], marker='*', color='tab:red')
        plt.xlabel("z of cone axis intersection")
        plt.ylabel("Compton cone angle")
        plt.ylim(-5,95)
        plt.xlim(-170,170)
        plt.legend()
        if save == True: plt.savefig(self.savefigpath + 'Cone_' + events + '_' + mask + '_Z-Angle')
        # Plot as heat map
        
        xy = np.vstack([array_OM[:,2],compton_angle])
        z = gaussian_kde(xy)(xy)
        x2 = np.asarray(array_OM[:,2])
        y2 = np.asarray(compton_angle)
        idx = z.argsort()
        x3, y3, z = x2[idx], y2[idx], z[idx]
        plt.figure()
        plt.title(events)
        plt.scatter(x3, y3, c=z, marker='.', s=50)
        #plt.scatter(array_OC[:,2][selection], np.asarray(compton_angle)[selection], marker='*', color='tab:red')
        plt.xlabel("z of cone axis intersection")
        plt.ylabel("Compton cone angle")
        plt.ylim(-5,95)
        plt.xlim(-170,170)
        if save == True: plt.savefig(self.savefigpath + 'Cone_' + events + '_' + mask + '_Z-Angle_Heat')
        
        '''
        cone_open_plane = array_OC[:,1] - array_OM[:,1]
        plt.figure()
        plt.hist(cone_open_plane[cone_open_plane<3000], bins=1000)
        plt.xlim(-1000,200)
        if save == True: plt.savefig(self.savefigpath + 'Cone_' + events + '_' + mask + '_Hist_yOC-yOM_ConeOpening')

        plt.figure()
        plt.hist(compton_angle, bins=70)
        plt.xlabel("Cone angle")
        plt.ylabel("Counts")
        if save == True: plt.savefig(self.savefigpath + 'Cone_' + events + '_' + mask +'_Hist_ConeAngle')
        plt.show()
        '''

    def plot_diff(self, mode='type-match', add_reco=True, focus=False):
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        y_reco = self.data.reco_test

        if mode == 'all-match':
            mask = None
        elif mode == 'pos-match':
            mask = [1,0,0,1,1,1,1,1,1]
        elif mode == 'type-match':
            mask = [1] + ([0] * 8)
        elif mode == 'miss':
            mask = None
        else:
            raise Exception('mode {} not recognized'.format(mode))

        l_matches = np.array(self._find_matches(y_true, y_pred, mask, keep_length = True)).astype(bool)
        l_reco_matches = np.array(self._find_matches(y_true, y_reco, mask, keep_length = True)).astype(bool)
        if mode == 'pos-match':
            all_matches = np.array(self._find_matches(y_true, y_pred, keep_length = True)).astype(bool)
            all_reco_matches = np.array(self._find_matches(y_true, y_reco, keep_length = True)).astype(bool)
            l_matches = (l_matches * np.invert(all_matches)).astype(bool)
            l_reco_matches = (l_reco_matches * np.invert(all_reco_matches)).astype(bool)

        y_pred = self.data._denormalize_targets(y_pred)
        y_true = self.data._denormalize_targets(y_true)
        y_reco = self.data._denormalize_targets(y_reco)

        if mode == 'miss':
            l_matches = (np.invert(l_matches) * y_true[:,0]).astype(bool)
            l_reco_matches = (np.invert(l_reco_matches) * y_true[:,0]).astype(bool)
            

        diff = y_true[:,:-3] - y_pred        # change to -3
        reco_diff = y_true[:,:-3] - y_reco

        diff = diff[l_matches]
        reco_diff = reco_diff[l_reco_matches]

        #print('{:6.0f} total Compton events'.format(np.sum(y_true[:,0])))
        #print('{:6d} NN matched events'.format(np.sum(l_matches)))
        #print('{:6d} Reco matched events'.format(np.sum(l_reco_matches)))


        fig_size = (10,4)

        def plot_hist(pos, title, width, x_min=None, x_max=None):
            plt.figure(figsize=fig_size)
            plt.title(title)
            data = diff[:,pos]

            if add_reco:
                reco_data = reco_diff[:,pos]

            if x_min is None and x_max is None:
                if add_reco:
                    x_min = min(int(np.floor(data.min())), int(np.floor(reco_data.min())))
                    x_max = max(int(np.ceil(data.max())), int(np.ceil(reco_data.max())))

                else:
                    x_min = int(np.floor(data.min()))
                    x_max = int(np.ceil(data.max()))

            x_min = np.ceil(x_min / width) * width
            x_max = ((x_max // width)+1) * width

            if add_reco:
                n, bins, _ = plt.hist(reco_data, np.arange(x_min, x_max, width), histtype='step', 
                                      label='Cut-based reco', color='tab:orange')
                #plt.plot((bins[np.argmax(n)]+bins[np.argmax(n)+1])/2, n.max(), '.', color='tab:orange')
                
            n, bins, _ = plt.hist(data, np.arange(x_min, x_max, width), histtype='step', 
                                  label='SiFi-CC NN ' + str(len(data)), color='tab:blue')
            #plt.plot((bins[np.argmax(n)]+bins[np.argmax(n)+1])/2, n.max(), '.', color='tab:blue')
            
            plt.ylabel('Count')
            plt.legend()
            plt.show()

        if focus:
            plot_hist(1, 'electron energy difference', .05, -2, 2)
            plot_hist(2, 'photon energy difference', .05, -3, 5)
            plot_hist(3, 'electron position x difference', 1.3, -20, 20)
            plot_hist(4, 'electron position y difference', 1, -75, 75)
            plot_hist(5, 'electron position z difference', 1.3, -20, 20)
            plot_hist(6, 'photon position x difference', 1.3, -20, 20)
            plot_hist(7, 'photon position y difference', 1, -75, 75)
            plot_hist(8, 'photon position z difference', 1.3, -20, 20)

        else:
            plot_hist(1, 'electron energy difference', .05)
            plot_hist(2, 'photon energy difference', .05)
            plot_hist(3, 'electron position x difference', 1.3)
            plot_hist(4, 'electron position y difference', 1)
            plot_hist(5, 'electron position z difference', 1.3)
            plot_hist(6, 'photon position x difference', 1.3)
            plot_hist(7, 'photon position y difference', 1)
            plot_hist(8, 'photon position z difference', 1.3)

    def plot_scene(self, pos, is_3d=True):
        '''Plotting the scene for an event along with the original and 
        predicited positions of both e & p'''

        # initialize the data to be plotted
        y_true = self.data._targets[pos:pos+1]
        y_pred = self.predict(self.data.get_features(pos,pos+1))
        clusters = self.data._features[pos:pos+1]
        is_match = self._find_matches(y_true, y_pred)[0] == 1

        y_true = self.data._denormalize_targets(y_true)[:,:-2].ravel()
        y_pred = self.data._denormalize_targets(y_pred).ravel()
        clusters = self.data._denormalize_features(clusters)

        # if the event isn't an ideal compton event, then ignore
        if y_true[0]==0:
            print('Not an ideal compton')
            return False

        clusters = clusters.reshape((-1, self.data.cluster_size))
        valid_clusters = clusters[:,0] > .5
        l_clusters = [clusters[valid_clusters,3], clusters[valid_clusters,5], clusters[valid_clusters,4]]

        l_e_targets = [y_true[3], y_true[5], y_true[4]]
        l_p_targets = [y_true[6], y_true[8], y_true[7]]

        l_e_nn = [y_pred[3], y_pred[5], y_pred[4]]
        l_p_nn = [y_pred[6], y_pred[8], y_pred[7]]

        fig = plt.figure(figsize=(9,7))

        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Z axis')
            ax.set_zlabel('Y axis')
            plot_client = ax
        else:
            l_clusters.pop()
            l_e_targets.pop()
            l_p_targets.pop()
            l_e_nn.pop()
            l_p_nn.pop()
            plt.xlabel('X axis')
            plt.ylabel('Z axis')
            #plt.ylim((-50,50))
            plot_client = plt


        depthshade = {'depthshade':False} if is_3d else {}

        plot_client.scatter(*l_clusters, marker='+', **depthshade,
                            s=180, color='tab:blue', label='cluster center')
        plot_client.scatter(*l_e_targets, marker='*', **depthshade,
                            s=80, color='tab:red', label='e position')
        plot_client.scatter(*l_p_targets, marker='*', **depthshade,
                            s=80, color='tab:orange', label='p position')
        plot_client.scatter(*l_e_nn, marker='^', **depthshade,
                            s=60, color='tab:red', label='Network e position')
        plot_client.scatter(*l_p_nn, marker='^', **depthshade,
                            s=80, color='tab:orange', label='Network p position')

        plot_client.legend()
        plt.show()
        return is_match
    
  
    
    
    def export_predictions_root(self, root_name):
        # get the predictions and true values
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y          

        # filter the results with the identified events by the NN
        identified = y_pred[:,0].astype(bool)
        y_pred = y_pred[identified]
        y_true = y_true[identified,:-2]
        origin_seq_no = self.data._seq[self.data.test_start_pos:][identified]

        # find the event type of the identified events by the NN
        l_all_match = self._find_matches(y_true, y_pred, keep_length=True)
        l_pos_match = self._find_matches(y_true, y_pred, mask=[1,0,0,1,1,1,1,1,1], keep_length=True)

        # denormalize the predictions back to the real values
        y_pred = self.data._denormalize_targets(y_pred)

        # identify the events with invalid compton cones
        e = y_pred[:,1]
        p = y_pred[:,2]
        me = 0.510999
        arc_base = np.abs(1 - me *(1/p - 1/(e+p)))
        valid_arc = arc_base <= 1
        origin_seq_no = self.data._seq[self.data.test_start_pos:][identified]

        # filter out invalid events from the predictions and events types
        y_pred = y_pred[valid_arc]
        l_all_match = np.array(l_all_match)[valid_arc]
        l_pos_match = np.array(l_pos_match)[valid_arc]

        # create event type list (0:wrong, 1:only pos match, 2:total match)
        l_event_type = np.zeros(len(l_all_match))
        l_event_type += l_pos_match
        l_event_type += l_all_match

        # zeros list
        size = y_pred.shape[0]
        zeros = np.zeros(size)

        # required fields for the root file
        e_energy = y_pred[:,1]
        p_energy = y_pred[:,2]
        total_energy = e_energy + p_energy

        e_pos_x = y_pred[:,4] # 3,  y
        e_pos_y =-y_pred[:,5] # 4, -z
        e_pos_z =-y_pred[:,3] # 5, -x

        p_pos_x = y_pred[:,7] # 6,  y
        p_pos_y =-y_pred[:,8] # 7, -z
        p_pos_z =-y_pred[:,6] # 8, -x

        arc = np.arccos(1 - me *(1/p_energy - 1/total_energy))

        # create root file
        file = uproot.recreate(root_name, compression=None)

        # defining the branch
        branch = {
            'GlobalEventNumber': 'int32', # event sequence in the original simulation file
            'v_x': 'float32', # electron position
            'v_y': 'float32',
            'v_z': 'float32',
            'v_unc_x': 'float32',
            'v_unc_y': 'float32',
            'v_unc_z': 'float32',
            'p_x': 'float32', # vector pointing from e pos to p pos
            'p_y': 'float32',
            'p_z': 'float32',
            'p_unc_x': 'float32',
            'p_unc_y': 'float32',
            'p_unc_z': 'float32',
            'E0Calc': 'float32', # total energy
            'E0Calc_unc': 'float32',
            'arc': 'float32', # formula
            'arc_unc': 'float32',
            'E1': 'float32', # e energy
            'E1_unc': 'float32',
            'E2': 'float32', # p energy
            'E2_unc': 'float32',
            'E3': 'float32', # 0
            'E3_unc': 'float32',
            'ClassID': 'int32', #0
            'EventType': 'int32', # 2-correct  1-pos  0-wrong
            'EnergyBinID': 'int32', #0
            'x_1': 'float32', # electron position
            'y_1': 'float32',
            'z_1': 'float32',
            'x_2': 'float32', # photon position
            'y_2': 'float32',
            'z_2': 'float32',
            'x_3': 'float32', # 0
            'y_3': 'float32',
            'z_3': 'float32',
        }

        file['ConeList'] = uproot.newtree(branch, title='Neural network cone list')

        # filling the branch
        file['ConeList'].extend({
            'GlobalEventNumber': origin_seq_no,
            'v_x': e_pos_x, 
            'v_y': e_pos_y,
            'v_z': e_pos_z,
            'v_unc_x': zeros,
            'v_unc_y': zeros,
            'v_unc_z': zeros,
            'p_x': p_pos_x - e_pos_x, 
            'p_y': p_pos_y - e_pos_y,
            'p_z': p_pos_z - e_pos_z,
            'p_unc_x': zeros,
            'p_unc_y': zeros,
            'p_unc_z': zeros,
            'E0Calc': total_energy, 
            'E0Calc_unc': zeros,
            'arc': arc, 
            'arc_unc': zeros,
            'E1': e_energy, 
            'E1_unc': zeros,
            'E2': p_energy, 
            'E2_unc': zeros,
            'E3': zeros, 
            'E3_unc': zeros,
            'ClassID': zeros, 
            'EventType': l_event_type, 
            'EnergyBinID': zeros, 
            'x_1': e_pos_x, 
            'y_1': e_pos_y,
            'z_1': e_pos_z,
            'x_2': p_pos_x, 
            'y_2': p_pos_y,
            'z_2': p_pos_z,
            'x_3': zeros, 
            'y_3': zeros,
            'z_3': zeros,
        })

        # defining the settings branch
        branch2 = {
            'StartEvent': 'int32', 
            'StopEvent': 'int32',
            'TotalSimNev': 'int32'
        }

        file['TreeStat'] = uproot.newtree(branch2, title='Evaluated events details')

        # filling the branch
        file['TreeStat'].extend({
            'StartEvent': [self.data._seq[self.data.test_start_pos]], 
            'StopEvent': [self.data._seq[-1]],
            'TotalSimNev': [self.data._seq[-1]-self.data._seq[self.data.test_start_pos]+1]
        })

        # closing the root file
        file.close()
        
    def export_targets_root(self, root_name):
        # get the true values
        y_true = self.data.test_row_y

        # filter the results with the identified events
        identified = y_true[:,0].astype(bool)
        y_true = y_true[identified,:-2]

        # denormalize the predictions back to the real values
        y_true = self.data._denormalize_targets(y_true)

        # identify the events with invalid compton cones
        e = y_true[:,1]
        p = y_true[:,2]
        me = 0.510999
        arc_base = np.abs(1 - me *(1/p - 1/(e+p)))
        valid_arc = arc_base <= 1

        # filter out invalid events from the predictions and events types
        y_true = y_true[valid_arc]

        # zeros list
        size = y_true.shape[0]
        zeros = np.zeros(size)
        
        # create event type list (0:wrong, 1:only pos match, 2:total match)
        l_event_type = np.ones(size) * 2

        # required fields for the root file
        e_energy = y_true[:,1]
        p_energy = y_true[:,2]
        total_energy = e_energy + p_energy

        e_pos_x = y_true[:,4] # 3, y              along fiber
        e_pos_y =-y_true[:,5] # 4, -z
        e_pos_z =-y_true[:,3] # 5, -x             along SiFiCC axis

        p_pos_x = y_true[:,7] # 6, y
        p_pos_y =-y_true[:,8] # 7, -z
        p_pos_z =-y_true[:,6] # 8, -x

        arc = np.arccos(1 - me *(1/p_energy - 1/total_energy))

        # create root file
        file = uproot.recreate(root_name, compression=None)

        # defining the branch
        branch = {
            'v_x': 'float32', # electron position
            'v_y': 'float32',
            'v_z': 'float32',
            'v_unc_x': 'float32',
            'v_unc_y': 'float32',
            'v_unc_z': 'float32',
            'p_x': 'float32', # vector pointing from e pos to p pos
            'p_y': 'float32',
            'p_z': 'float32',
            'p_unc_x': 'float32',
            'p_unc_y': 'float32',
            'p_unc_z': 'float32',
            'E0Calc': 'float32', # total energy
            'E0Calc_unc': 'float32',
            'arc': 'float32', # formula
            'arc_unc': 'float32',
            'E1': 'float32', # e energy
            'E1_unc': 'float32',
            'E2': 'float32', # p energy
            'E2_unc': 'float32',
            'E3': 'float32', # 0
            'E3_unc': 'float32',
            'ClassID': 'int32', #0
            'EventType': 'int32', # 2-correct  1-pos  0-wrong
            'EnergyBinID': 'int32', #0
            'x_1': 'float32', # electron position
            'y_1': 'float32',
            'z_1': 'float32',
            'x_2': 'float32', # photon position
            'y_2': 'float32',
            'z_2': 'float32',
            'x_3': 'float32', # 0
            'y_3': 'float32',
            'z_3': 'float32',
        }

        file['ConeList'] = uproot.newtree(branch, title='Neural network cone list')

        # filling the branch
        file['ConeList'].extend({
            'v_x': e_pos_x, 
            'v_y': e_pos_y,
            'v_z': e_pos_z,
            'v_unc_x': zeros,
            'v_unc_y': zeros,
            'v_unc_z': zeros,
            'p_x': p_pos_x - e_pos_x, 
            'p_y': p_pos_y - e_pos_y,
            'p_z': p_pos_z - e_pos_z,
            'p_unc_x': zeros,
            'p_unc_y': zeros,
            'p_unc_z': zeros,
            'E0Calc': total_energy, 
            'E0Calc_unc': zeros,
            'arc': arc, 
            'arc_unc': zeros,
            'E1': e_energy, 
            'E1_unc': zeros,
            'E2': p_energy, 
            'E2_unc': zeros,
            'E3': zeros, 
            'E3_unc': zeros,
            'ClassID': zeros, 
            'EventType': l_event_type, 
            'EnergyBinID': zeros, 
            'x_1': e_pos_x, 
            'y_1': e_pos_y,
            'z_1': e_pos_z,
            'x_2': p_pos_x, 
            'y_2': p_pos_y,
            'z_2': p_pos_z,
            'x_3': zeros, 
            'y_3': zeros,
            'z_3': zeros,
        })

        # closing the root file
        file.close()

