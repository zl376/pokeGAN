# Definition of
#   Wasserstein GAN
#
# Note:
#   Re-implement using native TensorFlow, 
#   since the Keras version seems to have slower convergence,
#   even with the SAME (need double-check) process
#
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-13

import os
import numpy as np
import tensorflow as tf
from .customized_layer import generator, discriminator, lrelu


class WGAN_2d:
    '''
    TODO: Multi-GPU version
    '''
    def __init__(self, img_size, len_code,
                       n_channel=3,
                       param_g=dict(), 
                       param_d=dict(),
                       dir_ckpt='./ckpt'):
        self.img_size = img_size
        self.len_code = len_code 
        self.n_channel = n_channel
        self.dir_ckpt = os.path.join(dir_ckpt)
        if not os.path.exists(self.dir_ckpt):
            os.makedirs(self.dir_ckpt)
            
        # Build graph
        self.build(param_g=param_g, param_d=param_d)
        
        
    def build(self, param_g=dict(), param_d=dict()):
        # Reset graph node
        # tf.reset_default_graph()

        with self.graph.as_default():
            # Prepare input placeholder
            with tf.variable_scope('input'):
                self.code = tf.placeholder(tf.float32, shape=(None,) + (self.len_code,), name='code')
                self.train = tf.placeholder(tf.bool, name='train')

            # Prepare (inter.) output placeholder
            self.real = tf.placeholder(tf.float32, shape=(None,) + self.img_size + (self.n_channel,), name='real')
            self.fake = generator(self.code, self.img_size, self.len_code, self.n_channel, self.train, **param_g)
            self.real_critic = discriminator(self.real, self.img_size, self.train, **param_d)
            self.fake_critic = discriminator(self.fake, self.img_size, self.train, reuse=True, **param_d)

            # Prepare save/load
            self.saver = tf.train.Saver(max_to_keep=10)
        
        
    def compile(self):
        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            self.d_vars = [var for var in t_vars if 'dis' in var.name]
            self.g_vars = [var for var in t_vars if 'gen' in var.name]

            self.d_loss = tf.reduce_mean(self.fake_critic) - tf.reduce_mean(self.real_critic)  # This optimizes the discriminator.
            self.g_loss = -tf.reduce_mean(self.fake_critic)  # This optimizes the generator.   

            self.optimizer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(self.d_loss, var_list=self.d_vars)
            self.optimizer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(self.g_loss, var_list=self.g_vars)
            # clip discriminator weights
            self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]
            
            # Initialization
            self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            
            # Prepare save/load
            self.saver = tf.train.Saver(max_to_keep=10)
        
        
    def fit(self, x=None,
                  y=None,
                  batch_size=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  validation_split=0.,
                  validation_data=None,
                  shuffle=True,
                  class_weight=None,
                  sample_weight=None,
                  initial_epoch=0,
                  steps_per_epoch=None,
                  validation_steps=None,
                  ncritic=5,
                  **kwargs):
        
        batch_num = int(x.shape[0] / batch_size)
        total_batch = 0        
        self.sess.run(self.init)
        
        losses_d = []
        losses_g = []        
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(x)
            print("Epoch: ", epoch)
            print("Number of batches: ", int(x.shape[0] // batch_size))
            
            i = 0
            while i < x.shape[0]:
                j = 0
                
                # z = np.random.uniform(-1.0, 1.0, size=(batch_size, self.len_code)).astype(np.float32)
                while i < x.shape[0] and j < ncritic:
                    # WGAN clip weights
                    self.sess.run(self.d_clip)
                    # Get new batch
                    batch_x = x[i:i+batch_size]
                    batch_size_true = batch_x.shape[0]
                    z = np.random.uniform(0, 1.0, size=(batch_size_true, self.len_code)).astype(np.float32)
                    # Update the discriminator
                    _, loss_d = self.sess.run([self.optimizer_d, self.d_loss],
                                               feed_dict={self.code: z[:batch_size_true, ...], self.real: batch_x, self.train: True})
                    
                    losses_d.append(-loss_d)
                    i += batch_size
                    j += 1
                z = np.random.uniform(0, 1.0, size=(batch_size, self.len_code)).astype(np.float32)
                _, loss_g = self.sess.run([self.optimizer_g, self.g_loss],
                                          feed_dict={self.code: z, self.train: True})
                losses_g.append(loss_g)
            
            # Plot loss
            from IPython import display
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16,4))
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.subplot(1, 3, 1)
            plt.plot(losses_g, label="loss_g")
            plt.legend()
            plt.subplot(1, 3, 2)
            plt.plot(losses_d, label="loss_d")
            plt.legend()
            plt.subplot(1, 3, 3)
            # Plot image
            z = np.random.uniform(0, 1.0, size=(16, self.len_code)).astype(np.float32)
            img_plot = self.predict(z)
            img_plot = img_plot.reshape((4,4) + self.img_size + (3,)).transpose(0,2,1,3,4).reshape((4*self.img_size[0], 4*self.img_size[1], 3))
            img_plot = (img_plot+1)/2
            plt.imshow(img_plot)
            plt.show();
        
        return losses_g[-1], losses_d[-1]
    
    
    def fit_generator(self, x=None,
                            y=None,
                            batch_size=None,
                            epochs=1,
                            verbose=1,
                            callbacks=None,
                            validation_split=0.,
                            validation_data=None,
                            shuffle=True,
                            class_weight=None,
                            sample_weight=None,
                            initial_epoch=0,
                            steps_per_epoch=None,
                            validation_steps=None,
                            ncritic=5,
                            freq_save=100,
                            **kwargs):
        
        batch_num = len(x)
        total_batch = 0
        
        if initial_epoch == 0:
            self.sess.run(self.init)
        
        if steps_per_epoch is None:
            steps_per_epoch = 1
        
        losses_d = []
        losses_g = []        
        for epoch in range(initial_epoch, initial_epoch+epochs):
            
            print("Epoch: ", epoch)
            print("Batches per epoch: ", steps_per_epoch)
            
            def should(freq):
                return freq > 0 and (epoch+1)%freq == 0
            # freq_save = 100
            
            i = 0
            while i < steps_per_epoch:
                
                # Trick from:
                #   https://github.com/lilianweng/unified-gan-tensorflow/blob/master/model.py
                # Increase critic update once in a while
                j = 100 if False and np.mod(epoch, 100) == 0 else ncritic
                
                # z = np.random.uniform(-1.0, 1.0, size=(batch_size, self.len_code)).astype(np.float32)
                while j > 0:
                    # WGAN clip weights
                    self.sess.run(self.d_clip)
                    # Get new batch
                    batch_x = next(x)
                    batch_size_true = batch_x.shape[0]
                    z = np.random.uniform(0, 1.0, size=(batch_size_true, self.len_code)).astype(np.float32)
                    # Update the discriminator
                    _, loss_d = self.sess.run([self.optimizer_d, self.d_loss],
                                              feed_dict={self.code: z, self.real: batch_x, self.train: True})
                    
                    losses_d.append(-loss_d)
                    
                    j -= 1
                z = np.random.uniform(0, 1.0, size=(batch_size, self.len_code)).astype(np.float32)
                _, loss_g = self.sess.run([self.optimizer_g, self.g_loss],
                                           feed_dict={self.code: z, self.train: True})
                losses_g.append(loss_g)
                
                i += 1
            
            # Plot loss
            if should(1):
                from IPython import display
                import matplotlib.pyplot as plt
                display.display(plt.gcf())
                display.clear_output(wait=True)
                plt.figure(figsize=(16,4))
                plt.subplot(1, 3, 1)
                plt.plot(losses_g, label="loss_g")
                plt.legend()
                plt.subplot(1, 3, 2)
                plt.plot(losses_d, label="loss_d")
                plt.legend()
                plt.subplot(1, 3, 3)
                # Plot image
                z = np.random.uniform(0, 1.0, size=(16, self.len_code)).astype(np.float32)
                img_plot = self.predict(z)
                img_plot = img_plot.reshape((4,4) + self.img_size + (3,)).transpose(0,2,1,3,4).reshape((4*self.img_size[0], 4*self.img_size[1], 3))
                img_plot = (img_plot+1)/2
                plt.imshow(img_plot)
                plt.show();
            
            # Save model
            if should(freq_save):
                print('Save model at epoch {}'.format(epoch+1))
                self.save_model('WGAN', epoch=epoch+1)
        
        return losses_g, losses_d
    
    
    def predict(self, x):
        pred = self.sess.run(self.fake, feed_dict={self.code: x, self.train: False})
        return pred

    
    def save_model(self, filename, epoch=0):
        filepath = os.path.join(self.dir_ckpt, filename)
        self.saver.save(self.sess, filepath, global_step=epoch)
        
        
    def load_model(self, filename, sess=None):
        sess = sess or self.sess
        try:
            self.saver.restore(sess, os.path.join(self.dir_ckpt, filename))
        except:
            self.saver.restore(sess, os.path.join(filename))
    
    
    @property
    def graph(self):
        if not hasattr(self, '_graph'):
            self._graph = tf.Graph()
        return self._graph
    
    
    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config, graph=self.graph)
        return self._sess