import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from sificc_lib import utils
import pickle as pkl
import datetime as dt
import uproot

class MyCallback(keras.callbacks.Callback):
    def __init__(self, ai, file_name=None):
        self.ai = ai
        self.file_name = file_name
        
        if file_name is not None:
            with open(self.file_name + '.e', 'w') as f_epoch:
                f_epoch.write('')
        
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.ai.predict(self.ai.data.train_x)[:,:-4]
        y_true = self.ai.data.train_row_y
        l_matches = self.ai._find_matches(y_true, y_pred, keep_length=False)
        logs['eff'] = np.mean(l_matches)
        logs['pur'] = np.sum(l_matches) / np.sum(y_pred[:,0])
        
        y_pred = self.ai.predict(self.ai.data.validation_x)[:,:-4]
        y_true = self.ai.data.validation_row_y
        l_matches = self.ai._find_matches(y_true, y_pred, keep_length=False)
        logs['val_eff'] = np.mean(l_matches)
        logs['val_pur'] = np.sum(l_matches) / np.sum(y_pred[:,0])
        
        self.ai.append_history(logs)
        self.ai.save(self.file_name)
        
        if self.file_name is not None:
            with open(self.file_name + '.e', 'a') as f_epoch:
                now = dt.datetime.now()
                f_epoch.write('loss:{:5.3f} eff:{:5.3f}/{:5.3f} in epoch {:3d} at {} {}\n'.format(
                    logs['loss'], logs['eff'], logs['val_eff'], 
                    epoch, now.date().isoformat(), now.strftime('%H:%M:%S')))

        
class AIQlty:
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
        self.weight_qlty = 1
        
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
    
    def init_model(self, conv_layers=[], classifier_layers=[], dense_layers=[],
                   type_layers=[], pos_layers=[], energy_layers=[], base_l2=0, limbs_l2=0, 
                   conv_dropouts=[], activation='relu', plot_summary=True):
        if len(conv_dropouts) == 0:
            conv_dropouts = [0] * len(conv_layers)
        assert len(conv_dropouts) == len(conv_layers)
        
        ###### input layer ######
        self.feed_in = keras.Input(shape=self.data.get_features(0,1).shape[1:], name='inputs')
        cnv = self.feed_in
        
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
        self.e_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='e_cluster')(cls)
        self.p_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='p_cluster')(cls)
        
        # get the hardmax of clusters classifier
        e_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='e_hardmax')(self.e_cluster)
        p_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='p_hardmax')(self.p_cluster)
        
        ###### joining outputs ######
        self.base_layer = keras.layers.Concatenate(axis=-1, name='join_layer')(
                                            [cnv, e_cluster_1_hot, p_cluster_1_hot])
        
        
        ###### event type layers ######
        typ = self.base_layer
        for i in range(len(type_layers)):
            typ = keras.layers.Dense(type_layers[i], activation=activation, 
                                   kernel_regularizer = keras.regularizers.l2(limbs_l2), 
                                   name='dense_type_{}'.format(i+1))(typ)
            
        self.event_type = keras.layers.Dense(1, activation='sigmoid', name='type')(typ)
        
        
        ###### event position ######
        pos = self.base_layer
        for i in range(len(pos_layers)):
            pos = keras.layers.Dense(pos_layers[i], activation=activation, 
                                     kernel_regularizer= keras.regularizers.l2(limbs_l2), 
                                     name='dense_pos_{}'.format(i+1))(pos)
            
        self.pos_x = keras.layers.Dense(2, activation=None, name='pos_x')(pos)
        self.pos_y = keras.layers.Dense(2, activation=None, name='pos_y')(pos)
        self.pos_z = keras.layers.Dense(2, activation=None, name='pos_z')(pos)
        
        
        ###### event energy ######
        enrg = self.base_layer
        for i in range(len(energy_layers)):
            enrg = keras.layers.Dense(energy_layers[i], activation=activation, 
                                      kernel_regularizer= keras.regularizers.l2(limbs_l2), 
                                      name='dense_energy_{}'.format(i+1))(enrg)
            
        self.energy = keras.layers.Dense(2, activation=None, name='energy')(enrg)
        
        ###### building the model ######
        self.model = keras.models.Model(self.feed_in, [self.e_cluster, self.p_cluster, self.event_type, 
                                                  self.pos_x, self.pos_y, self.pos_z, self.energy])
        self.history = None
        if plot_summary:
            self.model.summary()
        
    def extend_model(self, quality_layers=[], limbs_l2=0, activation='relu', plot_summary=True):
        qlty = self.base_layer
        for i in range(len(quality_layers)):
            qlty = keras.layers.Dense(quality_layers[i], activation=activation, 
                                   kernel_regularizer = keras.regularizers.l2(limbs_l2), 
                                   name='dense_quality_{}'.format(i+1))(qlty)
            
        self.quality = keras.layers.Dense(4, activation=None, name='quality')(qlty)
        
        
        
        ###### building the model ######
        self.model = keras.models.Model(self.feed_in, [self.e_cluster, self.p_cluster, self.event_type, 
                                                  self.pos_x, self.pos_y, self.pos_z, self.energy, self.quality])
        self.history = None
        if plot_summary:
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
                               'quality': self._quality_loss,
                           }, 
                           metrics = {
                               'type' : [self._type_accuracy, self._type_tp_rate],
                               'e_cluster': [self._cluster_accuracy],
                               'p_cluster': [self._cluster_accuracy],
                               'pos_x': [],
                               'pos_y': [],
                               'pos_z': [],
                               'energy': [],
                               'quality': [],
                           }, 
                           loss_weights = {
                               'type' : self.weight_type,
                               'e_cluster': self.weight_e_cluster,
                               'p_cluster': self.weight_p_cluster,
                               'pos_x': self.weight_pos_x,
                               'pos_y': self.weight_pos_y,
                               'pos_z': self.weight_pos_z,
                               'energy': self.weight_energy,
                               'quality': self.weight_qlty,
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
    
    def _quality_loss(self, y_true, y_pred):
        event_filter = y_true[:,0] # ∈ n
        e_enrg_true = K.reshape(y_true[:,1],(-1,1)) # ∈ nx1
        e_enrg_pred = K.reshape(y_pred[:,0],(-1,1)) # ∈ nx1
        p_enrg_true = K.reshape(y_true[:,2],(-1,1)) # ∈ nx1
        p_enrg_pred = K.reshape(y_pred[:,1],(-1,1)) # ∈ nx1
        e_pos_true = K.reshape(y_true[:,3],(-1,1)) # ∈ nx1
        e_pos_pred = K.reshape(y_pred[:,2],(-1,1)) # ∈ nx1
        p_pos_true = K.reshape(y_true[:,4],(-1,1)) # ∈ nx1
        p_pos_pred = K.reshape(y_pred[:,3],(-1,1)) # ∈ nx1
        
        e_enrg_loss = keras.losses.logcosh(e_enrg_true, e_enrg_pred) # ∈ n
        e_enrg_loss = event_filter * e_enrg_loss
        
        p_enrg_loss = keras.losses.logcosh(p_enrg_true, p_enrg_pred) # ∈ n
        p_enrg_loss = event_filter * p_enrg_loss
        
        e_pos_loss = keras.losses.logcosh(e_pos_true, e_pos_pred) # ∈ n
        e_pos_loss = event_filter * e_pos_loss
        
        p_pos_loss = keras.losses.logcosh(p_pos_true, p_pos_pred) # ∈ n
        p_pos_loss = event_filter * p_pos_loss
        
        return e_enrg_loss + p_enrg_loss + e_pos_loss + p_pos_loss
    
    @property
    def predict_columns(self):
        return ['is_compton', 'e_energy', 'p_energy', 
                'e_pos_x', 'e_pos_y', 'e_pos_z', 
                'p_pos_x', 'p_pos_y', 'p_pos_z', 
                'e_energy_qlty', 'p_energy_qlty', 
                'e_pos_qlty', 'p_pos_qlty']
    
    def predict(self, data, denormalize=False, verbose=0):
        pred = self.model.predict(data)
        pred = np.concatenate([np.round(pred[2]), 
                pred[6], 
                pred[3][:,[0]], pred[4][:,[0]], pred[5][:,[0]], 
                pred[3][:,[1]], pred[4][:,[1]], pred[5][:,[1]], 
                pred[7][:,[0]], pred[7][:,[1]], pred[7][:,[2]], pred[7][:,[3]], 
               ], axis=1)
        if denormalize:
            pred = np.concatenate([self.data._denormalize_targets(pred[:,:-4]), 
                                   pred[:,-4:]], axis=1) 
            
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
            plot_metric(ax1, 'e_cluster_loss', 'Cluster e', 'tab:pink')
            plot_metric(ax1, 'p_cluster_loss', 'Cluster p', 'tab:orange')
            plot_metric(ax1, 'pos_x_loss', 'Pos x', 'tab:brown')
            plot_metric(ax1, 'pos_y_loss', 'Pos y', 'tab:red')
            plot_metric(ax1, 'pos_z_loss', 'Pos z', 'tab:purple')
            plot_metric(ax1, 'type_loss', 'Type', 'tab:orange')
            plot_metric(ax1, 'energy_loss', 'Energy', 'tab:cyan')
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
        elif mode == 'loss-quality':
            plot_metric(ax2, 'quality_loss', 'Quality', 'tab:cyan')
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
        
    def calc_eff_pur(self, quality_filter=[None, None, None, None]):
        y_pred = self.predict(self.data.test_x)
        
        for i in range(len(quality_filter)):
            if quality_filter[i] is None:
                quality_filter[i] = -np.inf
                
        qlty_filter = (y_pred[:,-4:] >= quality_filter).all(axis=1)
        y_pred[:,0] = y_pred[:,0] * qlty_filter
        y_pred = y_pred[:,:-4]
        
        y_true = self.data.test_row_y
        l_matches = self._find_matches(y_true, y_pred, keep_length=False)
        effeciency = np.mean(l_matches)
        purity = np.sum(l_matches) / np.sum(y_pred[:,0])
        
        return [effeciency, purity]
    
        
    def evaluate(self, quality_filter=[None, None, None, None]):
        [loss, e_cluster_loss, p_cluster_loss, type_loss, 
         pos_x_loss, pos_y_loss, pos_z_loss, energy_loss, qlty_loss,
         e_cluster__cluster_accuracy, p_cluster__cluster_accuracy, 
         type__type_accuracy, type__type_tp_rate] = self.model.evaluate(
            self.data.test_x, self.data.test_y, verbose=0)
        
        
        y_pred = self.predict(self.data.test_x)
        for i in range(len(quality_filter)):
            if quality_filter[i] is None:
                quality_filter[i] = -np.inf
        qlty_filter = (y_pred[:,-4:] >= quality_filter).all(axis=1)
        y_pred[:,0] = y_pred[:,0] * qlty_filter
        y_pred = y_pred[:,:-4]
        
        y_true = self.data.test_row_y
        l_matches = self._find_matches(y_true, y_pred, keep_length=False)
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
        print('    -Quality:     {:8.5f} * {:5.2f} = {:7.5f}'.format(qlty_loss, self.weight_qlty, 
                                                                 qlty_loss * self.weight_qlty))
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
        self.model.save_weights(file_name+'.h5', save_format='h5')
        with open(file_name + '.hst', 'wb') as f_hist:
            pkl.dump(self.history, f_hist)
        with open(file_name + '.opt', 'wb') as f_hist:
            pkl.dump(self.model.optimizer.get_weights(), f_hist)
        
            
    def load(self, file_name, optimizer=False):
        self.model.load_weights(file_name+'.h5')
        with open(file_name+'.hst', 'rb') as f_hist:
            self.history = pkl.load(f_hist)
        if optimizer:
            with open(file_name+'.opt', 'rb') as f_hist:
                self.model.optimizer.set_weights(pkl.load(f_hist))
        
            
    def plot_diff(self, mode='type-match', add_reco=True, focus=False, quality_filter=[None, None, None, None]):
        y_pred = self.predict(self.data.test_x)
        for i in range(len(quality_filter)):
            if quality_filter[i] is None:
                quality_filter[i] = -np.inf
        qlty_filter = (y_pred[:,-4:] >= quality_filter).all(axis=1)
        y_pred[:,0] = y_pred[:,0] * qlty_filter
        y_pred = y_pred[:,:-4]
        
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
                                  label='SiFi-CC NN', color='tab:blue')
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
        y_pred = self.predict(self.data.get_features(pos,pos+1))[:,:-4]
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
    
    def export_predictions_root(self, root_name, quality_filter=[None, None, None, None]):
        # get the predictions and true values
        y_pred = self.predict(self.data.test_x)
        for i in range(len(quality_filter)):
            if quality_filter[i] is None:
                quality_filter[i] = -np.inf
        qlty_filter = (y_pred[:,-4:] >= quality_filter).all(axis=1)
        y_pred[:,0] = y_pred[:,0] * qlty_filter
        y_pred = y_pred[:,:-4]
        
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

        # filter out invalid events from the predictions and events types
        y_pred = y_pred[valid_arc]
        l_all_match = np.array(l_all_match)[valid_arc]
        l_pos_match = np.array(l_pos_match)[valid_arc]
        origin_seq_no = origin_seq_no[valid_arc]

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