from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import numpy as np
import math
from ops import *
import time
from data_reader import data_reader
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave, imread
from data_reader import data_reader

class GAN(object):

    def __init__(self, sess, flag):
        self.conf = flag
        self.sess = sess
        self.chan_out_r = 2
        self.chan_out_r_s =3
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        if not os.path.exists(self.conf.sampledir):
            os.makedirs(self.conf.sampledir)
        print("Start building network=================")
        self.configure_networks()
        print("Finishing building network=================")
    
    def configure_networks(self):
        self.global_step  = tf.Variable(0, trainable=False)
        self.build_network()
        variables = tf.trainable_variables()

        self.var_gen = [var for var in variables if var.name.startswith('Generator')]
        self.var_disc = [var for var in variables if var.name.startswith('Discriminator')]
        
        self.train_disc = tf.contrib.layers.optimize_loss(self.dis_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_disc, update_ops=[])
        self.train_gen = tf.contrib.layers.optimize_loss(self.gen_loss, global_step = self.global_step, 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_gen, update_ops=[])    
  #      self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary()
    #    self.train_con_summary =self.config_con_summary()
    #    self.test_summary = self.config_test_summary()

    def build_network(self):

        self.sampled_z_s = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
        self.input_y = tf.placeholder(tf.int32,[None,self.conf.n_class])    
  #      self.input_latent_r = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
        self.input_x = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 3])
        self.input_x_r = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 2])
        self.input_y= tf.cast(self.input_y, tf.float32)
      #  self.iter_number = tf.placeholder(tf.int32)
        print("Start building the generator of the ConGAN========================")
        #build the conditional auto encoder
        with tf.variable_scope('Generator') as scope:
            self.output_r, self.down_outputs = encode_img(self.input_x_r, self.conf.hidden_size)
            print(self.output_r.get_shape())
            self.X_rec_s = generator(self.down_outputs, self.sampled_z_s, self.output_r, self.input_y, self.conf.batch_size) # only s channel
        print("=========================Now split and insert")
        self.ch1, self.ch2_, self.ch3 = tf.split(self.input_x, num_or_size_splits=3, axis= 3)
    #    print(self.X_rec.get_shape())
        print(self.ch1.get_shape())
        self.X_rec = tf.concat([self.ch1, self.X_rec_s, self.ch3], axis= 3) 
        print(self.X_rec.get_shape())

        with tf.variable_scope('Discriminator') as scope:
            self.out_real = discriminator(self.input_x, self.input_y, self.conf.batch_size)
            scope.reuse_variables()
            self.out_fake = discriminator(self.X_rec,  self.input_y, self.conf.batch_size)
        

        # the loss for the conditional auto encoder
        self.d_loss_real = self.get_bce_loss(self.out_real, tf.ones_like(self.out_real))
        self.d_loss_fake = self.get_bce_loss(self.out_fake, tf.zeros_like(self.out_fake))
        # Do we need to add the classification loss??????????????????????????
        self.g_loss = self.get_bce_loss(self.out_fake, tf.ones_like(self.out_fake))
        self.rec_loss = self.get_mse_loss(self.X_rec_s, self.ch2_)

        # build the model for the final conditional generation
        
        self.dis_loss= self.d_loss_fake+self.d_loss_real
        self.gen_loss= self.rec_loss + self.g_loss*self.conf.gamma_gen
    #    self.gen_loss= self.rec_loss*self.get_coefficient(self.global_step) + self.g_loss*self.conf.gamma_gen  ## this is for dynamic loss
    #    self.gen_loss = self.g_loss ### this is for no rec loss
        self.test_x_r = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 2])
        self.test_y = tf.placeholder(tf.int32,[None,self.conf.n_class])
   #     self.test_label = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 3])
        self.random_s_test= tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        fix_s_test = tf.zeros_like(self.random_s_test)
        self.test_y = tf.cast(self.test_y, tf.float32)
        with tf.variable_scope('Generator', reuse= True) as scope:
            inter_r, test_downs = encode_img(self.test_x_r, self.conf.hidden_size)
            self.test_out = generator(test_downs, self.random_s_test, inter_r, self.test_y, self.conf.batch_size)
        with tf.variable_scope('Generator', reuse= True) as scope:
            self.test_out2 = generator(test_downs, fix_s_test, inter_r, self.test_y, self.conf.batch_size)
        
        print("==================FINAL shape is ")
        print(self.test_out.get_shape())

        
       

    def config_summary(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/Rec_loss', self.rec_loss))
        summarys.append(tf.summary.scalar('/Global_step', self.global_step))
        summarys.append(tf.summary.scalar('/d_loss_real', self.d_loss_real))
        summarys.append(tf.summary.scalar('/d_loss_fake', self.d_loss_fake))
        summarys.append(tf.summary.scalar('/d_loss', self.dis_loss))
        summarys.append(tf.summary.scalar('/g_loss', self.g_loss)) 
        summarys.append(tf.summary.scalar('/generator_loss', self.gen_loss))
        summarys.append(tf.summary.image('input_X', self.input_x, max_outputs = 10))
        summarys.append(tf.summary.image('input_s', self.ch2_, max_outputs = 10))
    #    summarys.append(tf.summary.image('input_r', self.input_x, max_outputs = 10))
        summarys.append(tf.summary.image('rec_r', self.X_rec_s, max_outputs = 10))
        summarys.append(tf.summary.image('recon_X', self.X_rec, max_outputs = 10))        
        summary = tf.summary.merge(summarys)
        return summary

    # def config_test_summary(self):
    #     summarys= []
    #     input_ch1, input_ch2 = tf.split(self.test_input, num_or_size_splits=2, axis=3)
    #     test_input = tf.concat([input_ch1, tf.zeros_like(input_ch1), input_ch2], axis = 3)
    #     summarys.append(tf.summary.image('test_input', test_input, max_outputs = 10))
    #     summarys.append(tf.summary.image('test_label', self.test_label, max_outputs = 10))
    #     summarys.append(tf.summary.image('test_out', self.test_out, max_outputs = 10))
    #     summary = tf.summary.merge(summarys)
    #     return summary
    
    def get_coefficient(self, iter_number):
        boundaries= [50000,150000]
        values = [0.0, 0.5, 1.0]
        rate = tf.train.piecewise_constant(iter_number, boundaries, values)
        return rate
        

    def get_bce_loss(self, output_tensor, target_tensor, epsilon=1e-10):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= output_tensor, labels = target_tensor))
   #     return tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -(1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

    def get_log_softmax(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= x, labels = y))

    def get_mse_loss(self, x, y):
        return tf.losses.mean_squared_error(predictions= x, labels= y)

    def get_l1_loss(self,x, y):
        return tf.losses.absolute_difference(x, y, scope='l1_loss')

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def save_summary(self, summary, step):
         print('---->summarizing', step)
         self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.checkpoint >0:
            print('=======Now load the model===============')
            self.reload(self.conf.checkpoint)
        data = data_reader()
        iterations = 1
        max_epoch = int (self.conf.max_epoch - (self.conf.checkpoint)/ 500)

        for epoch in range(max_epoch):
            pbar = ProgressBar()
            for i in pbar(range(self.conf.updates_per_epoch)):
                inputs, labels, _ = data.next_batch(self.conf.batch_size)
                inputs_r = data.extract(inputs)
                sampled_zs = np.random.normal(size= (self.conf.batch_size,self.conf.hidden_size))
                feed_dict = {self.sampled_z_s: sampled_zs, self.input_y: labels, self.input_x_r:inputs_r, self.input_x: inputs}
                _ , d_loss = self.sess.run([self.train_disc,self.dis_loss], feed_dict= feed_dict)
                _ , g_loss, summary = self.sess.run([self.train_gen, self.gen_loss, self.train_summary], feed_dict = feed_dict)
                if iterations %self.conf.summary_step == 1:
                    self.save_summary(summary, iterations+self.conf.checkpoint)
                if iterations %self.conf.save_step == 0:
                    self.save(iterations+self.conf.checkpoint)
                iterations = iterations +1
           #     self.save_image(test_out, test_x, epoch)
            print("g_loss is ===================", g_loss, "d_loss is =================", d_loss)
            test_x, test_y, _ = data.next_test_batch(self.conf.batch_size)
            test_x_r = data.extract(test_x)
            test_out, test_out_2 = self.sess.run([self.test_out, self.test_out2], feed_dict= {self.test_x_r: test_x_r,  self.test_y: test_y})
      #      print(test_out.shape)
            self.save_image(test_out, test_out_2, test_x, epoch+int  (self.conf.checkpoint)/ 500)
    #           print("encd_s_loss is  ================", encd_s_loss, "decd_s_loss is =============", decd_s_loss)
     #       self.generate_con_image()
        self.evaluate(data)

    def save_image(self, imgs, imgs2, inputs, epoch):
        imgs_test_folder = os.path.join(self.conf.working_directory, 'imgs_unet_guide')
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for k in range(self.conf.batch_size):
            temp_test_dir= os.path.join(imgs_test_folder, 'epoch_%d_#img_%d.png'%(epoch,k))
            res = np.zeros((self.conf.height, self.conf.height*6+10, 3))
            res[:,0:self.conf.height,:]= inputs[k,:,:,:]
            res[:,self.conf.height+2:self.conf.height*2+2,0]=inputs[k,:,:,0]
            res[:,self.conf.height+2:self.conf.height*2+2,2]=inputs[k,:,:,2]
            res[:,self.conf.height*2+4:self.conf.height*3+4, 1]= inputs[k,:,:,1]
            res[:,self.conf.height*3+6:self.conf.height*4+6, 1]= imgs[k,:,:,0]
            res[:,self.conf.height*4+8:self.conf.height*5+8, 0]= inputs[k,:,:,0]
            res[:,self.conf.height*4+8:self.conf.height*5+8, 2]= inputs[k,:,:,2]
            res[:,self.conf.height*4+8:self.conf.height*5+8, 1]= imgs[k,:,:,0]
            res[:,self.conf.height*5+10:self.conf.height*6+10, 1]= imgs2[k,:,:,0]
            imsave(temp_test_dir, res)
        print("Evaluation images generated！==============================") 


    

    def generate_con_image(self):
        
        for i in range(self.conf.n_class):
            sampled_y = np.zeros((self.conf.batch_size, self.conf.n_class), dtype=np.float32)
            sampled_y[:,i]=1
            imgs = self.sess.run(self.generate_con_out, {self.generated_y: sampled_y})
            for k in range(imgs.shape[0]):
                imgs_folder = os.path.join(self.conf.working_directory, 'imgs_con_parallel', str(i))
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)   
                imsave(os.path.join(imgs_folder,'%d.png') % k,
                    imgs[k,:,:,:])
        print("conditional generated imgs saved!!!!==========================")               
    
    def evaluate(self, data):        
     #   data = data_reader()
        print("Now start Testing set evaluation ==============================")
        pbar = ProgressBar()
        imgs_original_folder = os.path.join(self.conf.working_directory, 'imgs_unet_guide_test')
        if not os.path.exists(imgs_original_folder):
            os.makedirs(imgs_original_folder)
        imgs_test_folder = os.path.join(self.conf.working_directory, 'imgs_unet_guide_test')
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for i in pbar(range(self.conf.max_test_epoch)):
            x, y, r = data.next_test_batch(self.conf.batch_size)
            x_extracted = data.extract(x)
            y_label  = np.argmax(y, axis= 1)
            for j in range (self.conf.max_generated_imgs):
                output_test = self.sess.run(self.test_out, feed_dict={self.test_x_r: x_extracted,  self.test_y: y})
                for k in range(output_test.shape[0]):                    
                    # res = np.ones([self.conf.height, self.conf.width*3 +4, 3])
                    # res[:,0:self.conf.width,:]= x[k,:,:,:]
                    # res[:,self.conf.width+2:self.conf.width*2+2,(0,2)] = x_extracted[k,:,:,:]
                    # res[:,self.conf.width+2:self.conf.width*2+2,1] = 0
                    # res[:,self.conf.width*2+4:, :] = output_test[k,:,:,:]
                #    print("============================",output_test[k,:,:,:].shape)
                    temp_test_dir = os.path.join(imgs_test_folder, 'epoch_%d_#img_%d'%(i,k))
                    if not os.path.exists(temp_test_dir):
                        os.makedirs(temp_test_dir)
                    imsave(os.path.join(imgs_original_folder,'epoch_%d_#img_%d_cls_%d.png') %(i,k,y_label[k]),
                        x[k,:,:,:])
                    res = np.zeros((self.conf.height, self.conf.height, 3))
                    res[:,:,1] = output_test[k,:,:,0]
                    imsave(os.path.join(temp_test_dir,'imgs_%d.png') %j,
                        res)
           #     self.save_summary(summary, i*10*50+k*50+j)
        print("Evaluation images generated！==============================")

    
    def generate_and_save(self):
        imgs = self.sess.run(self.generated_out)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(self.conf.working_directory, 'imgs_parallel')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)      
            res= np.zeros([imgs.shape[1],imgs.shape[2],3])         
            res[:,:,0]=imgs[k,:,:,0]
            res[:,:,1]= 0
            res[:,:,2]=imgs[k,:,:,1]                
            imsave(os.path.join(imgs_folder,'%d.png') % k,
                res)
            imsave(os.path.join(imgs_folder,'%d_ch0.png') % k,
                imgs[k,:,:,0]) 
            imsave(os.path.join(imgs_folder,'%d_ch1.png') % k,
                imgs[k,:,:,1])    
        print("generated imgs saved!!!!==========================")


    def test(self):        
        print("======Now the parzen window evaluation ============================ ")
        if self.conf.checkpoint >0:
            print('=======Now load the model===============')
            self.reload(self.conf.checkpoint)        
        else:
            print("===================We need a model to reload, please provide the checkpoint")
            return

        # Alpha actinin   Alpha tubulin   Beta actin   Desmoplakin  Fibrillarin 
        # Lamin B1   Myosin IIB  Sec61 beta  Tom20  ZO1
        # cls_to_evaluate = 9 #which class
        # cls_name = 'ZO1' 
        # print(cls_name,"======",cls_to_evaluate)



        samples, labels = self.generate_samples()
  #      np.savez('samples_guide',a = samples, b = labels)
        print('save done')
        

        np.random.shuffle(samples)
        print(samples.shape)
        data = data_reader()
        # sigmas = np.logspace(-1.0, 0.0, 10)
        # lls = []
        # for sigma in sigmas:
        #     print("sigma: ", sigma)
        #     nlls = []
        #     for i in range(1, 10 + 1):
        #         X, _, _ = data.next_test_batch(self.conf.batch_size)
        #         X = data.extract_label(X)
        #     #    print("===============", i)
        #    #     print(X.shape, "========================")
        #         nll = parzen_cpu_batch(
        #             X,
        #             samples,
        #             sigma=sigma,
        #             batch_size=self.conf.batch_size,
        #             num_of_samples=10700,
        #             data_size=65536)
        #         nlls.extend(nll)
        #     nlls = np.array(nlls).reshape(100)  # 1000 valid images
        # #    print("sigma: ", sigma)
        #     print("ll: %d" % (np.mean(nlls)))
        #     lls.append(np.mean(nlls))
        #     data.reset()
        # sigma = sigmas[np.argmax(lls)]

        data.reset()
        sigma = 0.1
        nlls = []
      
        for i in range(1, 107 + 1):  # number of test batches = 107
            print("===============", i)
            X, _, _  = data.next_test_batch(self.conf.batch_size)
            X = data.extract_label(X)
            nll = parzen_cpu_batch(
                X,
                samples,
                sigma=sigma,
                batch_size=self.conf.batch_size,
                num_of_samples=10700,
                data_size=65536)
            nlls.extend(nll)
        nlls = np.array(nlls).reshape(1070)  # 10000 test images
        print("sigma: ", sigma)
        print("ll: %d" % (np.mean(nlls)))
        print("se: %d" % (nlls.std() / np.sqrt(1070)))
        return


    def reload(self, epoch):
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        model_path = checkpoint_path +'-'+str(epoch)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return       
        self.saver.restore(self.sess, model_path)
        print("model load successfully===================")

   # def evaluate
    def log_marginal_likelihood_estimate(self):
        x_mean = tf.reshape(self.input_tensor, [self.conf.batch_size, self.conf.width*self.conf.height])
        x_sample = tf.reshape(self.out_put, [self.conf.batch_size,self.conf.width*self.conf.height])
        x_sigma = tf.multiply(1.0, tf.ones(tf.shape(x_mean)))
        return log_likelihood_gaussian(x_mean, x_sample, x_sigma)+\
                log_likelihood_prior(self.latent_sample)-\
                log_likelihood_gaussian(self.latent_sample, self.mean, self.stddev)        

    def evaluate_nll(self, test_input):
        sample_ll= []
        for j in range (1000):
            res= self.sess.run(self.lle,{self.input_tensor: test_input})
            sample_ll.append(res)
        sample_ll = np.array(sample_ll)
        m = np.amax(sample_ll, axis=1, keepdims=True)
        log_marginal_estimate = m + np.log(np.mean(np.exp(sample_ll - m), axis=1, keepdims=True))
        return np.mean(log_marginal_estimate)

    def generate_samples(self):
        data= data_reader()
        samples = []
        labels = []
        for k in range(self.conf.max_test_epoch): #generate 10*1070 images
            x, y, r =data.next_test_batch(self.conf.batch_size)
            x_extracted = data.extract(x)
            for i in range(10):
                output_test = self.sess.run(self.test_out, feed_dict={self.test_x_r: x_extracted,  self.test_y: y})
            #    self.save_image_parzen_window(output_test, k*10+i)
            #    print("output shape is ===============", output_test.shape)
                samples.extend(output_test)
                labels.extend(y)
        samples = np.array(samples)
        print (samples.shape)
        labels = np.array(labels)
        return samples, labels

    def save_image_parzen_window(self, imgs, epoch):
        imgs_test_folder = os.path.join(self.conf.working_directory, 'imgs_unet_prazen_windows')
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for k in range(self.conf.batch_size):
            temp_test_dir= os.path.join(imgs_test_folder, 'epoch_%d_#img_%d.png'%(epoch,k))
            imsave(temp_test_dir, imgs[k,:,:,0])
        print("Parzen windows images generated！==============================")   

        

