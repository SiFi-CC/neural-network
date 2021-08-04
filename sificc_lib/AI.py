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
        self.position_absolute_limit = np.array([1.3, 5, 1.3]) * 2  # Old sim data set
        #self.position_absolute_limit = np.array([1.3, 10, 1.3]) * 2  # New sim data set
        
        self.weight_type = 2
        self.weight_e_cluster = 1
        self.weight_p_cluster = 1
        self.weight_pos_x = 2.5
        self.weight_pos_y = 1
        self.weight_pos_z = 2
        self.weight_energy = 1.5
        
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
        self.model.compile(optimizer= keras.optimizers.Nadam(learning_rate), 
                           loss = {
                               'type' : self._type_loss,
                               'e_cluster': self._e_cluster_loss,
                               'p_cluster': self._p_cluster_loss,
                               'pos_x': self._pos_loss,
                               'pos_y': self._pos_loss,
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
        
        return e_loss + p_loss
    
    def _energy_loss(self, y_true, y_pred):
        event_filter = y_true[:,0] # ∈ n
        e_enrg_true = K.reshape(y_true[:,1],(-1,1)) # ∈ nx1
        e_enrg_pred = K.reshape(y_pred[:,0],(-1,1)) # ∈ nx1
        p_enrg_true = K.reshape(y_true[:,2],(-1,1)) # ∈ nx1
        p_enrg_pred = K.reshape(y_pred[:,1],(-1,1)) # ∈ nx1
        
        e_loss = keras.losses.logcosh(e_enrg_true, e_enrg_pred) # ∈ n
        e_loss = event_filter * e_loss
        
        p_loss = keras.losses.logcosh(p_enrg_true, p_enrg_pred) # ∈ n
        p_loss = event_filter * p_loss
        
        return e_loss + p_loss
    
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
        if mask is None:
            mask = np.ones(9)
        else:
            mask = np.asarray(mask)
            
        y_true = self.data._denormalize_targets(y_true)
        y_pred = self.data._denormalize_targets(y_pred)
        
        if y_true.shape[1] == 11:
            y_true = y_true[:,:-2]
        
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
    
    def _find_matches_denorm(self, y_true, y_pred, mask=None, keep_length=True):
        # Input data is already denormalized
        if mask is None:
            mask = np.ones(9)
        else:
            mask = np.asarray(mask)
               
        if y_true.shape[1] == 11:
            y_true = y_true[:,:-2]
        
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
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
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
        
        #print("Matches in evaluate: ", l_matches)
        effeciency = np.mean(l_matches)
        purity = np.sum(l_matches) / np.sum(y_pred[:,0])
        
        identified_events = np.array(self._find_matches(y_true, y_pred, keep_length=True, mask=[1]+([0]*8))).astype(bool)
        y_pred = self.data._denormalize_targets(y_pred[identified_events])
        y_true = self.data._denormalize_targets(y_true[identified_events])
        euc = y_true[:,3:9] - y_pred[:,3:9]
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
        self.model.save_weights('ModelsTrained/' + file_name+'.h5', save_format='h5')
        with open('ModelsTrained/' + file_name + '.hst', 'wb') as f_hist:
            pkl.dump(self.history, f_hist)
        with open('ModelsTrained/' + file_name + '.opt', 'wb') as f_hist:
            pkl.dump(self.model.optimizer.get_weights(), f_hist)
        
            
    def load(self, file_name, optimizer=False):
        self.model.load_weights('ModelsTrained/' + file_name + '.h5')
        with open('ModelsTrained/' + file_name + '.hst', 'rb') as f_hist:
            self.history = pkl.load(f_hist)
        if optimizer:
            with open('ModelsTrained/' + file_name + '.opt', 'rb') as f_hist:
                self.model.optimizer.set_weights(pkl.load(f_hist))
        
            
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
            

        diff = y_true[:,:-2] - y_pred
        reco_diff = y_true[:,:-2] - y_reco

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
            plot_hist(1, 'e energy difference', .05, -2, 2)
            plot_hist(2, 'p energy difference', .05, -3, 5)
            plot_hist(3, 'e position x difference', 1.3, -20, 20)
            plot_hist(4, 'e position y difference', 1, -75, 75)
            plot_hist(5, 'e position z difference', 1.3, -20, 20)
            plot_hist(6, 'p position x difference', 1.3, -20, 20)
            plot_hist(7, 'p position y difference', 1, -75, 75)
            plot_hist(8, 'p position z difference', 1.3, -20, 20)

        else:
            plot_hist(1, 'e energy difference', .05)
            plot_hist(2, 'p energy difference', .05)
            plot_hist(3, 'e position x difference', 1.3)
            plot_hist(4, 'e position y difference', 1)
            plot_hist(5, 'e position z difference', 1.3)
            plot_hist(6, 'p position x difference', 1.3)
            plot_hist(7, 'p position y difference', 1)
            plot_hist(8, 'p position z difference', 1.3)

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
    
    
    def cluster_is_in_scatterer(self, cluster):
        # Used in events_prediction_analysis
        # Input is cluster feature data
        pos_x = cluster[3]
        pos_y = cluster[4]
        pos_z = cluster[5]
        
        if 193.5 < pos_x and pos_x < 206.5 and -50.< pos_y and pos_y < 50. and  -49.2 < pos_z and pos_z < 49.2:
            is_in_scatterer = 1
            is_in_absorber = 0
        elif 380.5 < pos_x < 419.5 and -50.< pos_y < 50. and  -49.2 < pos_z < 49.2:
            is_in_absorber = 1
            is_in_scatterer = 0
        else:
            is_in_scatterer = 0
            is_in_absorber = 0
        return is_in_scatterer, is_in_absorber
    
    def ep_is_in_scatterer(self, y, ep):  
        # Used in events_prediction_analysis
        # Input is target data from MC truth
        y=y[0] #all events
        if ep=='e':
            pos_x = y[3] 
            pos_y = y[4]
            pos_z = y[5]
        if ep=='p':
            pos_x = y[6] 
            pos_y = y[7]
            pos_z = y[8]
        
        if 193.5 < pos_x and pos_x < 206.5 and -50.< pos_y and pos_y < 50. and  -49.2 < pos_z and pos_z < 49.2:
            is_in_scatterer = 1
            is_in_absorber = 0
        elif 380.5 < pos_x < 419.5 and -50.< pos_y < 50. and  -49.2 < pos_z < 49.2:
            is_in_absorber = 1
            is_in_scatterer = 0
        else:
            is_in_scatterer = 0
            is_in_absorber = 0
         
        return is_in_scatterer, is_in_absorber
    
    def iterate_assign_clusters(self, clusters):
        # Used in events_prediction_analysis
        # Assign each cluster by position to absorber or scatterer
        
        clusters = np.array(clusters)
        clusters = np.reshape(clusters, (8,9))
        
        number_cluster_absorber = 0
        number_cluster_scatterer = 0
        number_clusters = 0
        is_in_scatterer = 0
        is_in_absorber = 0
        
        for cl in range(0, len(clusters)):
            is_in_scatterer, is_in_absorber = self.cluster_is_in_scatterer(clusters[cl])
            
            number_cluster_absorber += is_in_absorber
            number_cluster_scatterer += is_in_scatterer
            number_clusters += is_in_absorber
            number_clusters += is_in_scatterer
        
        return number_cluster_absorber, number_cluster_scatterer, number_clusters
     
    def events_prediction_analysis(self, plots = 'all-events', save = False):
        # Evaluate performance of predictions for different event types
        # All events, events with diff. cluster numbers, clusters in scatterer, e or p in scatterer
        
        y_pred = self.predict(self.data.test_x)  # Predictions for test data set
        y_true = self.data.test_row_y            # MC truth for test data set
        clusters = self.data.test_row_y_feat     # Features for test data set

        # Masks for selection of prediction matched
        mask_position = [1,0,0,1,1,1,1,1,1]
        mask_energy   = [1,1,1,0,0,0,0,0,0]
        mask_type     = [1] + ([0] * 8)
            
        l_matches_all = self._find_matches(y_true, y_pred, mask=None, keep_length=False) # All pred matched
        l_matches_type = np.array(self._find_matches(y_true, y_pred, mask_type, keep_length = True)).astype(bool) # Type
        
        numberComptonEvInData = np.sum(y_true[:,0])
        effeciency = np.sum(l_matches_all) / np.sum(y_true[:,0])
        purity = np.sum(l_matches_all) / np.sum(y_pred[:,0])
        
        print('{:8.5f} Total efficiency'.format(effeciency))
        print('{:8.5f} Total purity'.format(purity))
        print('{:6.0f} All valid events'.format(len(y_true)))
        print('{:6d} Correctly recon./ident. events'.format(np.sum(l_matches_all)))        
        print('{:6.0f} All true Compton events'.format(np.sum(y_true[:,0])))
        print('{:6.0f} NN pred Compton events'.format(np.sum(y_pred[:,0])))
        print('{:6d} NN type-matched events'.format(np.sum(l_matches_type)))
        
        y_true   = self.data._denormalize_targets(y_true)[:,:-2]  # Cut away cluster e/p as targets
        y_pred   = self.data._denormalize_targets(y_pred)         # Predictions denormalized as targets
        clusters = self.data._denormalize_features(clusters)

        list_no_cl_abs, list_no_cl_scat, list_no_cl = [],[],[]
        list_e_inAbsorber, list_p_inAbsorber        = [],[]
        list_e_inScatterer, list_p_inScatterer      = [],[]   # true positions
        
        for pos in range(0, len(clusters)):   # len(clusters)
            # iterate over all events
            # Count the number of clusters for each events
            no_cl_abs, no_cl_scat, no_cl = self.iterate_assign_clusters(clusters[pos:pos+1])
            list_no_cl_abs.append(no_cl_abs)
            list_no_cl_scat.append(no_cl_scat)
            list_no_cl.append(no_cl)
            
            # Where is electron, where is photon cluster from ground truth
            e_scat, e_abs = self.ep_is_in_scatterer(y_true[pos:pos+1], 'e')
            p_scat, p_abs = self.ep_is_in_scatterer(y_true[pos:pos+1], 'p')
            
            #e_scat_pred, e_abs_pred = self.ep_is_in_scatterer(y_pred[pos:pos+1], 'e')
            #p_scat_pred, p_abs_pred = self.ep_is_in_scatterer(y_pred[pos:pos+1], 'p')
            
            
            list_e_inAbsorber.append(e_abs)
            list_e_inScatterer.append(e_scat)
            list_p_inAbsorber.append(p_abs)
            list_p_inScatterer.append(p_scat)
        
        ##### internal definitions #####
        def select_ep_position(particle, volume):
            # Event selection: with e or p in scatterer or absorber
            # e is scattered electron, p is Compton scattered photon
            
            if particle=='e' and volume=='scatterer':
                mask = np.array(list_e_inScatterer).astype(bool)
            if particle=='e' and volume=='absorber': 
                mask = np.array(list_e_inAbsorber).astype(bool)
            if particle=='p' and volume=='scatterer':
                mask = np.array(list_p_inScatterer).astype(bool)
            if particle=='p' and volume=='absorber':
                mask = np.array(list_p_inAbsorber).astype(bool)   
            ep_y_true = y_true[mask]
            ep_y_pred = y_pred[mask]
            
            return ep_y_true, ep_y_pred
            
        def select_number_cluster(number_cluster):
            # Event selection: Events with certain number of clusters in total
            # Cluster number from 2 to 8 possible
            
            arr_no_cl = np.array(list_no_cl)
            cl_mask = np.where(arr_no_cl==number_cluster)[0] # def for masks
            
            if len(cl_mask)==0:
                return np.array([]),np.array([])
            
            cl_y_true = list(map(y_true.__getitem__, cl_mask))
            cl_y_pred = list(map(y_pred.__getitem__, cl_mask))
            cl_y_clusters = list(map(clusters.__getitem__, cl_mask))
            cl_y_true = np.vstack(cl_y_true)
            cl_y_pred = np.vstack(cl_y_pred)
            cl_y_clusters = np.vstack(cl_y_clusters)         
            
            return cl_y_true, cl_y_pred, cl_y_clusters
        
        def select_number_cluster_vol(number_cluster_volume, number_cluster):
            # Event selection: Select events with certain number of clusters in scatterer
            
            arr_no_cl_scat = np.array(list_no_cl_scat)
            arr_no_cl = np.array(list_no_cl)
            
            # Mask for total number of clusters and certain number of clusters in scatterer:
            cl_mask = np.where(np.logical_and(arr_no_cl==number_cluster,arr_no_cl_scat==number_cluster_volume))[0]
            
            if len(cl_mask)==0:
                return np.array([]),np.array([])
            
            cl_vol_y_true = list(map(y_true.__getitem__, cl_mask))
            cl_vol_y_pred = list(map(y_pred.__getitem__, cl_mask))
            cl_vol_y_true = np.vstack(cl_vol_y_true)
            cl_vol_y_pred = np.vstack(cl_vol_y_pred)
            
            return cl_vol_y_true, cl_vol_y_pred
        
        def plot_hist(data, title, width, x_min=None, x_max=None):
            # Plotting histograms
            
            plt.figure(figsize=(10,4))
            plt.title(title)

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
                                  label='NN ' + str(len(data)), color='tab:blue')
            plt.ylabel('Count')
            plt.legend()
        
        def select_matches_energy_deposition(l_matches, y_true, y_pred, y_clusters):
            # Event selection: According to given l_matches selection
            # Gives summed cluster energy, true summed energy, pred summed energy
            
            cl_y_true_match = y_true[l_matches]
            cl_y_pred_match = y_pred[l_matches]
            cl_y_clusters_match = y_clusters[l_matches]
            
            cl_true_e_energy = cl_y_true_match[:,1]
            cl_true_p_energy = cl_y_true_match[:,2]
            cl_pred_e_energy = cl_y_pred_match[:,1]
            cl_pred_p_energy = cl_y_pred_match[:,2]
            
            sum_energy_pred = np.array(cl_pred_p_energy) + np.array(cl_pred_e_energy) # Sum of electron and photon energy
            sum_energy_true = np.array(cl_true_p_energy) + np.array(cl_true_e_energy) 
            
            deposited_energy = [] 
            for i in range(0, len(cl_y_clusters_match)): 
                # Loop through all events, where the type is correctly predicted
                cl_y_cluster_ev = np.array(cl_y_clusters_match[i])
                cl_y_clusters_ev = np.reshape(cl_y_cluster_ev, (8,9))
                deposited_energy.append(np.sum(cl_y_clusters_ev[:,1],axis=0))  # Summed cluster energy (of 8 clusters)
                
            return deposited_energy, sum_energy_true, sum_energy_pred
        
        def scatter_plot_deposited_energy(cl_y_true, cl_y_pred, cl_y_clusters, title=''): 
            # Plot: Deposited energy

            l_matches_type = np.array(self._find_matches_denorm(cl_y_true, cl_y_pred, mask=mask_type, keep_length=True)).astype(bool)
           
            deposited_energy, sum_energy_true, sum_energy_pred = select_matches_energy_deposition(l_matches_type, cl_y_true, cl_y_pred, cl_y_clusters)
            
            fig_e = plt.figure(figsize=(10,5))
            ax_e = fig_e.add_subplot(1,2,1)
            ax_e.scatter(deposited_energy, sum_energy_true, marker='.', color ='tab:blue', label = 'True energy')
            ax_e.scatter(deposited_energy, sum_energy_pred, marker='.', color ='tab:cyan', label = 'Pred. energy')
            ax_e.set_title("Energy deposition " + title)
            ax_e.set_xlabel("Summed cluster energy / MeV")
            ax_e.set_ylabel("Electron + photon energy / MeV") 
            ax_e.set_aspect('equal', adjustable='box')
            ax_e.set_ylim(-0.2,17.7)
            ax_e.set_xlim(-0.2,15)
            #ax_e.grid()
            ax_e.legend()
            
            y_true_type = cl_y_true[l_matches_type]
            y_pred_type = cl_y_pred[l_matches_type]
            y_clus_type = cl_y_clusters[l_matches_type]
            
            l_matches_all = np.array(self._find_matches_denorm(y_true_type, y_pred_type, mask=None, keep_length=True)).astype(bool)
            l_matches_pos = np.array(self._find_matches_denorm(y_true_type, y_pred_type, mask=mask_position, keep_length=True)).astype(bool)
            l_matches_energy = np.array(self._find_matches_denorm(y_true_type, y_pred_type, mask=mask_energy, keep_length=True)).astype(bool)
            l_mismatches_all = np.invert(l_matches_all) # Only mismatches, where the type was predicted correctly
            
            print('{:8.0f} Type matched, pos energy mismatched '.format( np.sum(l_mismatches_all) ))
            print('{:8.0f} Type and pos matched '.format( np.sum(l_matches_pos) ))
            print('{:8.0f} Type and energy matched '.format( np.sum(l_matches_energy) ))
            
            deposited_energy_match, sum_energy_true_match, sum_energy_pred_match = select_matches_energy_deposition(l_matches_all, y_true_type,y_pred_type, y_clus_type)
            deposited_energy_mismatch, sum_energy_true_mismatch, sum_energy_pred_mismatch = select_matches_energy_deposition(l_mismatches_all, y_true_type, y_pred_type, y_clus_type)
            
            deposited_energy_Pmismatch, sum_energy_true_Pmismatch, sum_energy_pred_Pmismatch = select_matches_energy_deposition(np.invert(l_matches_pos), y_true_type, y_pred_type, y_clus_type)
            deposited_energy_Emismatch, sum_energy_true_Emismatch, sum_energy_pred_Emismatch = select_matches_energy_deposition(np.invert(l_matches_energy), y_true_type, y_pred_type, y_clus_type)

            plt.figure(figsize=(5,5))
            plt.scatter(deposited_energy, sum_energy_true, marker='.', color ='tab:blue', label = 'True ' + str(len(deposited_energy)) )
            plt.scatter(deposited_energy_mismatch, sum_energy_pred_mismatch, marker='.', color ='tab:orange', label = 'False pred. ' + str(len(deposited_energy_mismatch)))
            plt.scatter(deposited_energy_match, sum_energy_pred_match, color ='tab:cyan', marker='.', label = 'Matched pred. '+ str(len(deposited_energy_match)) )
            plt.title("Energy deposition " + title)
            plt.xlabel("Summed cluster energy / MeV")
            plt.ylabel("Electron + photon energy / MeV") 
            plt.axis('scaled')
            plt.ylim(-0.2,17.7)
            plt.xlim(-0.2,15)
            plt.legend()
            if save == True: plt.savefig(self.savefigpath + '_TrueFalsePredMatch.png')
            
            def xy_scatter_heat(deposited_energy, sum_energy_true):
                xy = np.vstack([deposited_energy,sum_energy_true])
                z = gaussian_kde(xy)(xy)
                deposited_energy_plot = np.asarray(deposited_energy)
                sum_energy_true_plot = np.asarray(sum_energy_true)
                idx = z.argsort()
                deposited_energy_plot, sum_energy_true_plot, z = deposited_energy_plot[idx], sum_energy_true_plot[idx], z[idx]
                return deposited_energy_plot, sum_energy_true_plot, z
            
            deposited_energy_plot, sum_energy_true_plot, z = xy_scatter_heat(deposited_energy, sum_energy_true)
            deposited_energy_mismatch, sum_energy_pred_mismatch, z_mismatch = xy_scatter_heat(deposited_energy_mismatch, sum_energy_pred_mismatch)
            deposited_energy_match, sum_energy_pred_match, z_match = xy_scatter_heat(deposited_energy_match, sum_energy_pred_match)
        
            plt.figure(figsize=(5,5))
            plt.title('True ' + str(len(deposited_energy)) )
            plt.scatter(deposited_energy_plot, sum_energy_true_plot, c=z, marker='.', s=50)
            plt.title("Energy deposition " + title)
            plt.xlabel("Summed cluster energy / MeV")
            plt.ylabel("Electron + photon energy / MeV") 
            plt.axis('scaled')
            plt.ylim(-0.2,17.7)
            plt.xlim(-0.2,15)
            if save == True: plt.savefig(self.savefigpath + '_True_Heat')
            
            plt.figure(figsize=(5,5))
            plt.title('False pred. ' + str(len(deposited_energy_mismatch)))
            plt.scatter(deposited_energy_mismatch, sum_energy_pred_mismatch, c=z_mismatch, marker='.', s=50)
            plt.title("Energy deposition " + title)
            plt.xlabel("Summed cluster energy / MeV")
            plt.ylabel("Electron + photon energy / MeV") 
            plt.axis('scaled')
            plt.ylim(-0.2,17.7)
            plt.xlim(-0.2,15)
            if save == True: plt.savefig(self.savefigpath + '_FalsePred_Heat.png')
            
            plt.figure(figsize=(5,5))
            plt.title('Matched pred. '+ str(len(deposited_energy_match)))
            plt.scatter(deposited_energy_match, sum_energy_pred_match, c=z_match, marker='.', s=50)
            plt.title("Energy deposition " + title)
            plt.xlabel("Summed cluster energy / MeV")
            plt.ylabel("Electron + photon energy / MeV") 
            plt.axis('scaled')
            plt.ylim(-0.2,17.7)
            plt.xlim(-0.2,15)
            if save == True: plt.savefig(self.savefigpath + '_Matched_Heat.png')
            plt.show()
            
            
            def plotContainedEnergy(x, y, text):
            
                fig_e = plt.figure(figsize=(10,5))
                
                ax_m = fig_e.add_subplot(1, 2, 1)
                ax_m.scatter(deposited_energy, sum_energy_true, marker='.', color ='tab:blue', label = 'True')
                ax_m.scatter(x, y, marker='.', color ='tab:orange', label = text + str(len(y)))
                ax_m.set_xlabel("Summed cluster energy / MeV")
                ax_m.set_ylabel("Electron + photon energy / MeV") 
                ax_m.set_aspect('equal', adjustable='box')
                ax_m.set_ylim(-0.2,17.7)
                ax_m.set_xlim(-0.2,15)
                ax_m.legend()
                fig_e.tight_layout()
                
                xy = np.vstack([x,y])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                deposited_energy_plot, sum_energy_true_plot, z = np.asarray(x)[idx], np.asarray(y)[idx], z[idx]
                
                ax1 = fig_e.add_subplot(1,2,2)
                ax1.scatter(deposited_energy_plot, sum_energy_true_plot, c=z,marker='.', s=50)
                ax1.set_title(text)
                ax1.set_xlabel("Summed cluster energy / MeV")
                ax1.set_ylabel("Electron + photon energy / MeV")
                ax1.set_aspect('equal', adjustable='box')
                ax1.set_ylim(-0.2,17.7)
                ax1.set_xlim(-0.2,15)
                if save == True: plt.savefig(self.savefigpath + 'truefalse_' + text + '.png')
                plt.show()
                
            plotContainedEnergy(deposited_energy_mismatch,  sum_energy_pred_mismatch,  text='Falsely pred. ')
            plotContainedEnergy(deposited_energy_Pmismatch, sum_energy_pred_Pmismatch, text='Falsely pred. position ')
            plotContainedEnergy(deposited_energy_Emismatch, sum_energy_pred_Emismatch, text='Falsely pred. energy ')
            
        def position_real_pred_plot(l_matches_type, y_true, y_pred, title=''):
            # Plot positions of events with matched type prediction
            
            y_true_type = y_true[l_matches_type]
            y_pred_type = y_pred[l_matches_type]
            
            l_matches_pos = np.array(self._find_matches_denorm(y_true, y_pred, mask_position, keep_length = True)).astype(bool)
            l_matches_pos_type = np.array(self._find_matches_denorm(y_true_type, y_pred_type, mask_position, keep_length = True)).astype(bool)
            
            y_true_pos = y_true[l_matches_pos]
            y_pred_pos = y_pred[l_matches_pos]            
            
            true_e_posx = y_true_type[:,3]
            true_p_posx = y_true_type[:,6]
            pred_e_posx = y_pred_type[:,3] 
            pred_p_posx = y_pred_type[:,6] 
            
            true_e_posy = y_true_type[:,4]
            true_p_posy = y_true_type[:,7]
            pred_e_posy = y_pred_type[:,4] 
            pred_p_posy = y_pred_type[:,7]
            
            # Define x boundaries of SiFi-CC volumes
            abs_max = 419.5
            abs_min = 380.5            
            scat_max = 206.5
            scat_min = 193.5
            
            # Electron: Sorting events to volumes
            bool_true_eInScat = np.logical_and(true_e_posx < scat_max, true_e_posx > scat_min)
            bool_pred_eInScat = np.logical_and(pred_e_posx < scat_max, pred_e_posx > scat_min)
            bool_true_eInAbs = np.logical_and(true_e_posx < abs_max, true_e_posx > abs_min)
            bool_pred_eInAbs = np.logical_and(pred_e_posx < abs_max, pred_e_posx > abs_min)
            
            # Photon
            bool_true_pInScat = np.logical_and(true_p_posx < scat_max, true_p_posx > scat_min)
            bool_pred_pInScat = np.logical_and(pred_p_posx < scat_max, pred_p_posx > scat_min)
            bool_true_pInAbs = np.logical_and(true_p_posx < abs_max, true_p_posx > abs_min)
            bool_pred_pInAbs = np.logical_and(pred_p_posx < abs_max, pred_p_posx > abs_min)
            
            # Position x correctly predicted
            bool_corr_pred_eInScat = np.logical_and(bool_true_eInScat, bool_pred_eInScat)
            bool_corr_pred_pInScat = np.logical_and(bool_true_pInScat, bool_pred_pInScat)
            bool_corr_pred_eInAbs  = np.logical_and(bool_true_eInAbs, bool_pred_eInAbs)
            bool_corr_pred_pInAbs  = np.logical_and(bool_true_pInAbs, bool_pred_pInAbs)
            
            print(' Prediction of x positions ')
            print('{:8.5f} Matched type number '.format(np.sum(l_matches_type)))
            print('{:8.5f} Matched position number '.format(np.sum(l_matches_pos)))
            print('{:8.5f} Matched position number and matched type'.format(np.sum(l_matches_pos_type)))
            print('{:8.5f} Matched position from matched type '.format(np.sum(l_matches_pos) / np.sum(l_matches_type)))
            
            print('{:8.0f} Pred e in scatterer '.format(np.sum(bool_pred_eInScat)))
            print('{:8.0f} Pred p in scatterer '.format(np.sum(bool_pred_pInScat)))
            print('{:8.0f} Matched e in scat '.format(np.sum(bool_corr_pred_eInScat)))
            print('{:8.0f} Matched p in scat '.format(np.sum(bool_corr_pred_pInScat)))
              
            print('{:8.0f} True e in scatterer '.format(np.sum(bool_true_eInScat)))
            print('{:8.0f} True p in scatterer '.format(np.sum(bool_true_pInScat)))
            print('{:8.0f} True e in abs '.format(np.sum(bool_true_eInAbs)))
            print('{:8.0f} True p in abs '.format(np.sum(bool_true_pInAbs)))
            
            print('{:8.0f} Pred e in abs '.format(np.sum(bool_pred_eInAbs)))
            print('{:8.0f} Pred p in abs '.format(np.sum(bool_pred_pInAbs)))
            print('{:8.0f} Matched e in abs '.format(np.sum(bool_corr_pred_eInAbs)))
            print('{:8.0f} Matched p in abs '.format(np.sum(bool_corr_pred_pInAbs)))          
            
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            ax.scatter(true_e_posx, pred_e_posx, marker = '.', color = 'tab:blue')
            ax.set_title("Distribution electron" + title)
            ax.set_xlabel("True position / mm")
            ax.set_ylabel("Predicted position / mm") 
            ax.set_aspect('equal', adjustable='box')
            ax.set_ylim(150,450)
            ax.set_xlim(150,450)
            
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(true_p_posx, pred_p_posx, marker = '.',color = 'tab:orange')
            ax2.set_title("Distribution photon" + title)
            ax2.set_xlabel("True position / mm")
            ax2.set_ylabel("Predicted position / mm") 
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_ylim(150,450)
            ax2.set_xlim(150,450)
            fig.tight_layout()
            fig0 = plt.figure()
            ax0 = fig0.add_subplot(1,2,1)
            ax0.scatter(true_e_posy, pred_e_posy, marker = '.',color = 'tab:blue')
            ax0.set_title("Distribution electron" + title)
            ax0.set_xlabel("True position / mm")
            ax0.set_ylabel("Predicted position / MeV") 
            ax0.set_aspect('equal', adjustable='box')
            ax0.set_ylim(-70,70)
            ax0.set_xlim(-70,70)
            
            ax1 = fig0.add_subplot(1, 2, 2)
            ax1.scatter(true_p_posy, pred_p_posy, marker = '.',color = 'tab:orange')
            ax1.set_title("Distribution photon" + title)
            ax1.set_xlabel("True position / mm")
            ax1.set_ylabel("Predicted position / MeV") 
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_ylim(-70,70)
            ax1.set_xlim(-70,70)
            fig0.tight_layout()
            
            fig0 = plt.figure()
            ax0 = fig0.add_subplot(1,1,1)
            ax0.set_title("Predicted electron position")
            ax0.scatter(pred_e_posx, pred_e_posy, marker = '.',color = 'tab:blue')
            ax0.set_xlabel("x / mm")
            ax0.set_ylabel("y / mm") 
            ax0.set_aspect('equal', adjustable='box')
            ax0.set_ylim(-120,120)
            ax0.set_xlim(170,470)
            fig0.tight_layout()
            #ax2.legend()
        
        def event_evaluation(event_type, y_true, y_pred, mode = ''):
            # Event evaluation on data sets with selection cuts
            # For denormalized data
            
            # find matches for denormalized y, list of 1s and 0s
            l_matches_length = self._find_matches_denorm(y_true, y_pred, keep_length=False) 
            
            # Efficiency: matched events wrt all Compton events in the data
            efficiency = np.sum(l_matches_length) / numberComptonEvInData
            # Efficiency wrt to the cuts/CE in selected data subset
            efficiency_selection = np.mean(l_matches_length)
            # Purity: Division by number of predicted Compton events within the selection cuts
            purity = np.sum(l_matches_length) / np.sum(y_pred[:,0])  
            
            if mode =='selection':
                return efficiency, efficiency_selection, purity, np.sum(l_matches_length)
            
            return efficiency, purity, np.sum(l_matches_length)
        ###### end internal definitions ######
        
         
        l_matches_type = np.array(self._find_matches_denorm(y_true, 
                                                            y_pred, 
                                                            mask_type, 
                                                            keep_length = True)).astype(bool) 
        
        if plots == 'all-events':
            scatter_plot_deposited_energy(y_true, y_pred, clusters)
            position_real_pred_plot(l_matches_type, y_true, y_pred)
            
        diff_all = y_true - y_pred    # No selection
        efficiency_all, purity_all, number_matches_total= event_evaluation('Total', y_true, y_pred) # No selection
        
        width=0.2 # for histogram bins
        eff_cl,eff_sel_cl,pur_cl,x_bar_cl = [],[],[],[]
        number_allEvents_cl,number_trueCE_cl,number_predCE_cl,number_typematch_cl,number_matches_cl = [],[],[],[],[]
        
        for cluster_index in range(2,9):
            # Loop over 2 to 8 clusters
            # Select event by number of clusters

            cl_y_true, cl_y_pred, cl_y_clusters = select_number_cluster(cluster_index) # Events with 2,3,... or 8 clusters
            diff_cl = cl_y_true - cl_y_pred

            t_eff_cl,t_eff_sel_cl,t_pur_cl,number_matches = event_evaluation(str(cluster_index), cl_y_true, cl_y_pred, 'selection')
            l_matches_type_cl = np.array(self._find_matches_denorm(cl_y_true, cl_y_pred, mask_type, keep_length = True)).astype(bool) 

            eff_cl.append(t_eff_cl)
            eff_sel_cl.append(t_eff_sel_cl)
            pur_cl.append(t_pur_cl)
            x_bar_cl.append(cluster_index)

            number_allEvents_cl.append(len(cl_y_pred))            # all events
            number_trueCE_cl.append(np.sum(cl_y_true[:,0]))       # true compton events
            number_predCE_cl.append(np.sum(cl_y_pred[:,0]))       # predicted compton events
            number_typematch_cl.append(np.sum(l_matches_type_cl)) # type-matched pred Compton events
            number_matches_cl.append(number_matches)

            if plots == 'cluster-numbers':
                scatter_plot_deposited_energy(cl_y_true, cl_y_pred, cl_y_clusters, title=str(cluster_index)+" clusters")
                position_real_pred_plot(l_matches_type_cl, cl_y_true, cl_y_pred, title=", " +str(cluster_index)+" clusters")

            if plots == 'cluster-distribution':
                # Select by number of clusters in scatterer
                
                y_eff_scat,y_eff_sel_scat,y_pur_scat,x_bar = [],[],[],[]
                y1_bar,y2_bar,y3_bar,number_typematch_cl_scat,y4_bar = [],[],[],[],[]
                
                for cluster_scat_index in range(1,cluster_index):

                    cl_scat_y_true, cl_scat_y_pred = select_number_cluster_vol(cluster_scat_index, cluster_index)
                    l_matches_type_cl_scat = np.array(self._find_matches_denorm(cl_scat_y_true, cl_scat_y_pred, mask_type, keep_length = True)).astype(bool)

                    if not np.any(cl_scat_y_pred):
                        print("No clusters")
                    else:
                        eff_scat, eff_selection_scat, pur_scat, number_matches_scat = event_evaluation(str(cluster_index)+str(cluster_scat_index), cl_scat_y_true, cl_scat_y_pred, 'selection')

                        y1_bar.append(len(cl_scat_y_pred))         # all valid events
                        y2_bar.append(np.sum(cl_scat_y_true[:,0])) # true compton events
                        y3_bar.append(np.sum(cl_scat_y_pred[:,0])) # predicted compton events
                        number_typematch_cl_scat.append(np.sum(l_matches_type_cl_scat)) # Correctly type predicted events
                        y4_bar.append(number_matches_scat)         # Compton events, correctly predicted and reconstructed

                        x_bar.append(cluster_scat_index)
                        y_eff_scat.append(eff_scat)
                        y_eff_sel_scat.append(eff_selection_scat)
                        y_pur_scat.append(pur_scat)
                        
                    if cluster_index == 2:
                        print("Eff scat", eff_scat)
                        print("Eff clus", t_eff_cl)
                        print("No Eff scat", np.sum(cl_scat_y_pred[:,0]))
                        print("No Eff cl",   np.sum(cl_y_pred[:,0]))
                        print("No Eff scat type match", np.sum(l_matches_type_cl_scat))
                        print("No Eff cl   type match", np.sum(l_matches_type_cl))                        

                # Efficency - purity plot
                plt.figure(figsize=(10,4))
                plt.axhline(t_eff_cl,             color='darkblue',   label = 'Cluster efficiency')
                plt.axhline(t_pur_cl, ls='--',    color='tab:orange', label = 'Cluster purity')
                plt.plot(x_bar,y_eff_scat, 'o',   color='darkblue',   label = 'Efficiency')
                plt.plot(x_bar,y_eff_sel_scat,'o',color='tab:blue',   label = 'Eff. on selection')
                plt.plot(x_bar,y_pur_scat, 'd',   color='tab:orange', label = 'Purity')
                plt.ylabel('Precision')
                plt.xlabel('Number of clusters in scatterer for ' + str(cluster_index) + " clusters")
                plt.ylim(-0.005,0.27)
                plt.xticks(x_bar)
                plt.legend()
                plt.grid()
                
                # Bar plot number of event type 
                plt.figure(figsize=(10,4))
                x_bar = np.array(x_bar)
                plt.bar(x_bar - 1.5*width,y2_bar,width,color='tab:blue', label='True Compton events')
                plt.bar(x_bar - width/2, y3_bar ,width,color='tab:orange', label='Pred. Compton events') 
                plt.bar(x_bar + width/2, number_typematch_cl_scat ,width,color='tab:cyan', label='Type-matched Compton events') 
                plt.bar(x_bar + 1.5*width,y4_bar,width,color='tab:green', label='Matched predictions')
                plt.ylabel('Number')
                plt.xlabel('Number of clusters in scatterer for ' + str(cluster_index) + " clusters")
                plt.xticks(x_bar)
                plt.grid()
                plt.legend()

        if plots == 'cluster-numbers':

            x_bar_cl=np.array(x_bar_cl)

            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(1, 1, 1)
            ax.axhline(efficiency_all, color='darkblue', label = 'Total efficiency')
            ax.axhline(purity_all, ls='--', color='tab:orange', label = 'Total purity',)
            ax.plot(x_bar_cl,eff_cl, 'o', color='darkblue', label = 'Cluster efficiency')
            ax.plot(x_bar_cl,eff_sel_cl,'o', color='tab:blue', label = 'Cluster eff. on selection')
            ax.plot(x_bar_cl,pur_cl, 'd', color='tab:orange', label = 'Cluster purity')
            ax.set_xticks(x_bar_cl)
            ax.set_ylabel('Precision')
            ax.set_xlabel('Number of clusters') 
            ax.set_ylim(-0.005,0.27)
            ax.grid()
            ax.legend()
            
            fig0 = plt.figure(figsize=(10,4)) 
            ax0 = fig0.add_subplot(1, 1, 1)   
            #ax0.bar(x_bar_cl, number_allEvents_cl,width,  color='tab:red', label='All valid events') # All
            ax0.bar(x_bar_cl - 1.5*width, number_trueCE_cl,    width, color='tab:blue',  label='True Compton events') 
            ax0.bar(x_bar_cl - 0.5*width, number_predCE_cl,    width, color='tab:orange',label='Pred. Compton events') 
            ax0.bar(x_bar_cl + 0.5*width, number_typematch_cl, width, color='tab:cyan',  label='Type-matched Compton events') 
            ax0.bar(x_bar_cl + 1.5*width, number_matches_cl,   width, color='tab:green', label='Matched predictions')
            ax0.set_ylabel('Number of events')
            ax0.set_xlabel('Number of clusters')
            ax0.grid()
            ax0.legend()
            plt.show()

        if plots == 'e-scatterer':
            
            # Select events by electron, photon position
            escat_y_true, escat_y_pred = select_ep_position('e','scatterer') # Choose events with e in scat
            eabs_y_true,  eabs_y_pred  = select_ep_position('e','absorber') 
            pscat_y_true, pscat_y_pred = select_ep_position('p','scatterer') 
            pabs_y_true,  pabs_y_pred  = select_ep_position('p','absorber') 

            diff_eInScat = escat_y_true - escat_y_pred
            diff_eInAbs  = eabs_y_true  - eabs_y_pred
            diff_pInScat = pscat_y_true - pscat_y_pred
            diff_pInAbs  = pabs_y_true  - pabs_y_pred

            plot_hist(diff_all[:,1],     'e energy difference all events', .05, -2, 2)
            plot_hist(diff_eInScat[:,1], 'e energy difference events with e in scatterer', .05, -2, 2)
            plot_hist(diff_eInAbs[:,1],  'e energy difference events with e in absorber', .05, -2, 2)
            plot_hist(diff_pInScat[:,1], 'photon energy difference events with p in scatterer', .05, -2, 2)
            plot_hist(diff_pInAbs[:,1],  'photon energy difference events with p in absorber', .05, -2, 2)

            # Plot purity efficiency for e/p in absorber scatterer
            eff_eInScat, pur_eInScat,number_matches_1  = event_evaluation('E in scatterer', escat_y_true, escat_y_pred)
            eff_eInAbs , pur_eInAbs, number_matches_2  = event_evaluation('E in absorber',  eabs_y_true, eabs_y_pred)
            eff_pInScat, pur_pInScat,number_matches_3  = event_evaluation('P in scatterer', pscat_y_true, pscat_y_pred)
            eff_pInAbs , pur_pInAbs, number_matches_4  = event_evaluation('P in absorber',  pabs_y_true, pabs_y_pred)

            types_pur = [pur_eInScat, pur_pInAbs, pur_pInScat, pur_eInAbs]
            types_eff = [eff_eInScat, eff_pInAbs, eff_pInScat, eff_eInAbs]
            types_x   = ['e- in scat','p in abs', 'p in scat', 'e- in abs']

            y_bar_ep_all   = [np.sum(escat_y_true[:,0]), np.sum(pabs_y_true[:,0]), np.sum(pscat_y_true[:,0]), np.sum(eabs_y_true[:,0])]
            y_bar_ep_pred  = [np.sum(escat_y_pred[:,0]), np.sum(pabs_y_pred[:,0]), np.sum(pscat_y_pred[:,0]), np.sum(eabs_y_pred[:,0])]
            y_bar_ep_match = [number_matches_1, number_matches_4, number_matches_3, number_matches_2]

            # Bar plot for number of data
            #print("Sums true / pred CE ", y_bar_ep_all[0]+y_bar_ep_all[2], y_bar_ep_pred[0]+y_bar_ep_pred[2])
            x_bar_ep = np.array([1,2,3,4])
            fig4 = plt.figure(figsize=(10,4))
            ax4 = fig4.add_subplot(1, 1, 1)
            ax4.bar(x_bar_ep - width, y_bar_ep_all  ,width, color='tab:blue', label='True Compton events') 
            ax4.bar(x_bar_ep ,        y_bar_ep_pred ,width, color='tab:red',  label='Type-matched Compton events')
            ax4.bar(x_bar_ep + width, y_bar_ep_match,width, color='tab:orange',label='Matched predictions')
            ax4.set_xticks(x_bar_ep)
            ax4.set_xticklabels(types_x)
            ax4.legend()
            
            # Plot eff/pur for e photon distribution
            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(types_x,types_eff, 'o', color='tab:blue', label = 'Efficiency')
            ax.plot(types_x,types_pur, 'd', color='tab:red', label = 'Purity (type-match)')
            ax.set_ylabel('Precision')
            ax.set_xlabel('Event types')
            ax.legend()
            ax.grid()
            plt.show()
            plt.close()
            
        print("")
        
        if plots == 'cluster-distribution':
            # Evaluate model: for events with all numbers of clusters, but only 1 cluster in scatter volume
            arr_no_cl_scat = np.array(list_no_cl_scat) # Array with events contained in scatterer    
            cl1_scat_mask = np.where(arr_no_cl_scat == 1)[0] # Mask to select event with one cluster in the scatterer
            cl1scat_y_true = list(map(y_true.__getitem__, cl1_scat_mask))
            cl1scat_y_pred = list(map(y_pred.__getitem__, cl1_scat_mask))
            cl1scat_y_true = np.vstack(cl1scat_y_true)
            cl1scat_y_pred = np.vstack(cl1scat_y_pred)
            eff_cl1_scat, pur_cl1_scat, number_matches_cl1_scat  = event_evaluation('1 Cluster in scatterer',  cl1scat_y_true, cl1scat_y_pred)

            print("{:6.0f} events for 1 cluster matches in scat from {:6.0f} events".format(len(cl1scat_y_true), len(y_true)))
            print("{:6.0f} matched events for 1 cluster matches in scat ".format(number_matches_cl1_scat))
            print("{:2.8f} efficiency, {:2.8f} purity fo 1 cluster matches in scat ".format(eff_cl1_scat, pur_cl1_scat))

            # Evaluate model: for events with all numbers of clusters, but only 2 cluster in scatter volume
            arr_no_cl_scat = np.array(list_no_cl_scat) # Array with events contained in scatterer    
            cl2_scat_mask = np.where(arr_no_cl_scat==2)[0] # Mask to select event with two cluster in the scatterer
            cl2scat_y_true = list(map(y_true.__getitem__, cl2_scat_mask))
            cl2scat_y_pred = list(map(y_pred.__getitem__, cl2_scat_mask))
            cl2scat_y_true = np.vstack(cl2scat_y_true)
            cl2scat_y_pred = np.vstack(cl2scat_y_pred)
            eff_cl2_scat, pur_cl2_scat, number_matches_cl2_scat  = event_evaluation('2 Cluster in scatterer',  cl2scat_y_true, cl2scat_y_pred)

            print("\n{:6.0f} events for 2 cluster matches in scat ".format(len(cl2scat_y_true)))
            print("{:6.0f} matched events for 2 cluster matches in scat ".format(number_matches_cl2_scat))
            print("{:2.8f} effiecency, {:2.8f} purity fo 2 cluster matches in scat ".format(eff_cl2_scat, pur_cl2_scat))
    
    def export_predictions_root(self, root_name):
        # get the predictions and true values
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y

        # filter the results with the identified events by the NN
        identified = y_pred[:,0].astype(bool)
        y_pred = y_pred[identified]
        y_true = y_true[identified,:-2]
        origin_seq_no = self.data._seq[self.data.test_start_pos:][identified]

        # find the real event type of the identified events by the NN
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

        e_pos_x = y_pred[:,4] # 3, y
        e_pos_y =-y_pred[:,5] # 4, -z
        e_pos_z =-y_pred[:,3] # 5, -x

        p_pos_x = y_pred[:,7] # 6, y
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

        e_pos_x = y_true[:,4] # 3, y
        e_pos_y =-y_true[:,5] # 4, -z
        e_pos_z =-y_true[:,3] # 5, -x

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

