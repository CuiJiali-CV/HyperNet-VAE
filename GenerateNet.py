# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
from utils import *
import tensorflow as tf
import numpy as np
from params import *
from HyperNet import genWeight
from loadData import DataSet
import random


class GenNet(object):

    def __init__(self, category='Mnist', vis_step=10, Train_Epochs=200, z_batch_size=128, image_batch_size=128,
                 z_size_h=100, lr=0.001, z_size_i=100, history_dir='./', checkpoint_dir='./', logs_dir='./',
                 gen_dir='./', recon_dir='./', test_dir='./'):
        self.test = False

        self.category = category
        self.epoch = Train_Epochs
        self.img_size = 28 if (category == 'Fashion-Mnist' or category == 'Mnist') else 64
        self.z_batch_size = z_batch_size
        self.image_batch_size = image_batch_size
        self.batch_size = image_batch_size * z_batch_size
        self.z_size_h = z_size_h
        self.z_size_i = z_size_i

        self.vis_step = vis_step

        self.lr = lr
        self.channel = 1 if (category == 'Fashion-Mnist' or category == 'Mnist') else 3
        self.history_dir = history_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.gen_dir = gen_dir
        self.test_dir = test_dir
        self.recon_dir = recon_dir

        self.z_i = tf.placeholder(tf.float32, shape=[self.z_batch_size, self.image_batch_size, self.z_size_i],
                                  name='latent_img')
        self.z_h = tf.placeholder(tf.float32, shape=[self.z_batch_size, self.z_size_h], name='latent_hyper')

        self.x = tf.placeholder(tf.float32,
                                shape=[self.image_batch_size * self.z_batch_size, self.img_size, self.img_size,
                                       self.channel],
                                name='image')

    def build_Model(self):
        self.weights = self.HyperNet(self.z_h, reuse=False)
        self.weights_test = self.HyperNet(self.z_h, reuse=True)

        self.mu, self.logvar = self.encoder(self.x, reuse=False)
        self.reparms_z = self.reparameterize(self.mu, self.logvar)
        self.recon = self.decoder(self.reparms_z, self.weights, reuse=False)

        self.gen = self.decoder(self.z_i, self.weights_test, reuse=True)

        self.mu_test, self.logvar_test = self.encoder(self.x, reuse=True)
        self.reparms_z_test = self.reparameterize(self.mu_test, self.logvar_test)
        self.gen_test = self.decoder(self.reparms_z_test, self.weights_test, reuse=True)
        """
        Loss and Optimizer
        """
        self.loss = self.loss_func(self.recon, self.x, self.mu, self.logvar)
        self.var = [var for var in tf.trainable_variables() if var.name.startswith('Hyper')]
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        """
        Logs
        """
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.summary_op = tf.summary.merge_all()

    def decoder(self, z, weights, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            z = tf.reshape(z, [self.z_batch_size, self.image_batch_size, self.z_size_i])
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':

                fc1_w = weights['fc1_w']
                fc1_b = weights['fc1_b']
                fc2_w = weights['fc2_w']
                fc2_b = weights['fc2_b']
                fc3_w = weights['fc3_w']
                fc3_b = weights['fc3_b']


                fc1 = tf.nn.relu(tf.add(tf.reduce_sum(tf.expand_dims(z, -1) * tf.expand_dims(fc1_w, 1), axis=2),
                             tf.expand_dims(fc1_b, 1), name='fc1'))

                fc2 = tf.nn.relu(tf.add(tf.reduce_sum(tf.expand_dims(fc1, -1) * tf.expand_dims(fc2_w, 1), axis=2),
                             tf.expand_dims(fc2_b, 1), name='fc2'))

                fc3 = tf.nn.sigmoid(tf.add(tf.reduce_sum(tf.expand_dims(fc2, -1) * tf.expand_dims(fc3_w, 1), axis=2),
                             tf.expand_dims(fc3_b, 1), name='fc3'))

        return fc3

    def HyperNet(self, z, reuse=False):
        Hpyer_params = HyperNetParams(category=self.category, img_size=self.img_size, z_size_h=self.z_size_h, z_size_i=self.z_size_i)
        Gen_params = GeneratorParams(category=self.category, img_size=self.img_size,  z_size_i=self.z_size_i)
        weights = {}
        with tf.variable_scope('Hyper', reuse=reuse):
            if self.category == 'Mnist':
                # [1, z] - [1, 300] - [1, 300] - [1, prod+prod]
                with tf.variable_scope('code'):
                    weightSize = [Hpyer_params.extractor_w1, Hpyer_params.extractor_w2, Hpyer_params.extractor_w3]
                    codes = genWeight(layerSize=Hpyer_params.extractor_hiddenlayer_size, weightSize=weightSize,
                                      input=z)
                    # code1 128, 1024, 15
                    startIdx = 0
                    endIdx = startIdx + np.prod(Hpyer_params.code1_size)
                    code1 = tf.identity(tf.reshape(codes[:, startIdx:endIdx], [-1, ] + Hpyer_params.code1_size),
                                        name='code1')

                    # code2 128, 6272, 15
                    startIdx = endIdx
                    endIdx = startIdx + np.prod(Hpyer_params.code2_size)
                    code2 = tf.identity(
                        tf.reshape(codes[:, startIdx:endIdx], [-1, ] + Hpyer_params.code2_size, name='code2'))

                    # code3 128, 128, 15
                    startIdx = endIdx
                    endIdx = startIdx + np.prod(Hpyer_params.code3_size)
                    code3 = tf.identity(
                        tf.reshape(codes[:, startIdx:endIdx], [-1, ] + Hpyer_params.code3_size, name='code3'))


                # gen fc1 filter code[128, 1024 15] - [128, 1024, 40] - [128, 1024, 40] - [128, 1024, z+10+1] - [128, z+10+1, 1024]
                with tf.variable_scope('fc1_w', reuse=reuse):
                    weightSize = [Hpyer_params.fc1_w1_size, Hpyer_params.fc1_w2_size, Hpyer_params.fc1_w3_size]
                    w_b = genWeight(layerSize=Hpyer_params.w_gen_hiddenlayer_size, weightSize=weightSize,
                                    input=code1)
                    w = tf.identity(
                        tf.reshape(w_b[:, :, :Gen_params.fc1_filter_size[0]], [-1, ] + Gen_params.fc1_filter_size),
                        name='reshaped_w1')
                    b = tf.identity(tf.reshape(w_b[:, :, Gen_params.fc1_filter_size[0]:],
                                               [-1, ] + [Gen_params.fc1_filter_size[1]]), name='reshaped_b1')
                    weights['fc1_w'] = w
                    weights['fc1_b'] = b

                # gen fc2 filter code[128, 6272 15] - [128, 6272, 40] - [128, 6272, 40] - [128, 6272, 1024+10+1] - [128, 1024+10+1, 6272]
                with tf.variable_scope('fc2_w', reuse=reuse):
                    weightSize = [Hpyer_params.fc2_w1_size, Hpyer_params.fc2_w2_size, Hpyer_params.fc2_w3_size]
                    w_b = genWeight(layerSize=Hpyer_params.w_gen_hiddenlayer_size, weightSize=weightSize,
                                    input=code2)
                    w = tf.identity(
                        tf.reshape(w_b[:, :, :Gen_params.fc2_filter_size[0]], [-1, ] + Gen_params.fc2_filter_size),
                        name='reshaped_w1')
                    b = tf.identity(tf.reshape(w_b[:, :, Gen_params.fc2_filter_size[0]:],
                                               [-1, ] + [Gen_params.fc2_filter_size[1]]), name='reshaped_b1')
                    weights['fc2_w'] = w
                    weights['fc2_b'] = b

                # gen fc2 filter code[128, 128, 15] - [128, 128, 40] - [128, 128, 40] - [128, 128, 25*42+1] - [128, 5, 5, 42, 128]
                with tf.variable_scope('fc3_w', reuse=reuse):
                    weightSize = [Hpyer_params.fc3_w1_size, Hpyer_params.fc3_w2_size, Hpyer_params.fc3_w3_size]
                    w_b = genWeight(layerSize=Hpyer_params.w_gen_hiddenlayer_size, weightSize=weightSize,
                                    input=code3)
                    w = tf.identity(
                        tf.reshape(w_b[:, :, :Gen_params.fc3_filter_size[0]], [-1, ] + Gen_params.fc3_filter_size),
                        name='reshaped_w1')
                    b = tf.identity(tf.reshape(w_b[:, :, Gen_params.fc3_filter_size[0]:],
                                               [-1, ] + [Gen_params.fc3_filter_size[1]]), name='reshaped_b1')
                    weights['fc3_w'] = w
                    weights['fc3_b'] = b

        return weights

    def encoder(self, x, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':
                x = tf.reshape(x, [self.batch_size, -1])

                fc1 = tf.nn.relu(tf.layers.dense(inputs=x, units=500, name='fc1'))

                fc2 = tf.nn.relu(tf.layers.dense(inputs=fc1, units=500, name='fc2'))

                output = tf.layers.dense(inputs=fc2, units=self.z_size_i * 2, name='meanOfz')
                mu = output[:, :self.z_size_i]
                logvar = tf.sqrt(tf.exp(output[:, self.z_size_i:]))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # std = tf.exp(0.5*logvar)
        # eps = tf.random_uniform(shape=tf.shape(std))
        # return mu+eps*std
        return mu + logvar * tf.random_normal([tf.shape(mu)[0], self.z_size_i])

    def loss_func(self, recon, x, mu, logvar):
        # recon = tf.reshape(recon, [self.batch_size, -1])
        # recon = tf.clip_by_value(recon, 1e-8, 1 - 1e-8)
        # x = tf.reshape(x, [self.batch_size, -1])
        #
        # marginal_likelihood = tf.reduce_sum(x * tf.log(recon) + (1 - x) * tf.log(1 - recon), 1)
        # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(logvar) - tf.log(1e-8 + tf.square(logvar)) - 1, 1)
        #
        # marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        # KL_divergence = tf.reduce_mean(KL_divergence)
        #
        # ELBO = marginal_likelihood - KL_divergence
        #
        # loss = -ELBO

        # recon = tf.reshape(recon, [self.batch_size, 28, 28, 1])
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon - x), axis=(1,2,3)), axis=0)
        x = tf.reshape(x, [self.batch_size, -1])
        recon = tf.reshape(recon, [self.batch_size, -1])
        loss = tf.reduce_sum(0.5 * (tf.square(mu) + tf.square(logvar) -
                             2.0 * tf.log(logvar + 1e-8) - 1.0)) +\
                tf.reduce_sum(-x * tf.log(recon + 1e-8) -
                            (1.0 - x) * tf.log(1.0 - recon + 1e-8))
        return loss

    def train(self, sess):
        self.build_Model()

        data = DataSet(img_size=self.img_size, batch_size=self.batch_size, category=self.category)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        writer = tf.summary.FileWriter(self.logs_dir, sess.graph)

        start = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            latest_checkpoint.split('-')
            start = int(latest_checkpoint.split('-')[-1])
            saver.restore(sess, latest_checkpoint)
            print('Loading checkpoint {}.'.format(latest_checkpoint))

        tf.get_default_graph().finalize()

        for epoch in range(start + 1, self.epoch):
            num_batch = int(len(data) / self.batch_size)
            losses = []

            for step in range(num_batch):
                obs, labels = data.NextBatch(step)

                z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
                loss, summary, _ = sess.run([self.loss, self.summary_op, self.opt], feed_dict={self.x: obs, self.z_h:z_h})
                # print(epoch, ": Loss : ", loss)
                # print(epoch, ": marginal_likelihood : ", marginal_likelihood)
                # print(epoch, ": KL_divergence : ", KL_divergence)
                losses.append(loss)
                writer.add_summary(summary, global_step=epoch)


            print(epoch, ": Loss : ", np.mean(losses))
            if epoch % self.vis_step == 0:
                # print(epoch, ": Loss : ", np.mean(losses))
                self.visualize(saver, sess, epoch, data)
        self.visualize_test(sess, data)

    def visualize(self, saver, sess, epoch, data):
        saver.save(sess, "%s/%s" % (self.checkpoint_dir, 'model.ckpt'), global_step=epoch)
        idx = random.randint(0, int(len(data) / self.batch_size) - 1)
        """
            Recon
        """
        obs, labels = data.NextBatch(idx)
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
        sys = sess.run(self.recon, feed_dict={self.x: obs, self.z_h:z_h})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.recon_dir + 'epoch' + str(epoch) + 'recon.jpg'
        show_in_one(path, sys, column=16, row=8)

        """
        Generation
        """
        # obs = data.NextBatch(idx, test=True)
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size_i))
        # z = sess.run(self.langevin, feed_dict={self.z: z, self.x: obs})
        sys = sess.run(self.gen, feed_dict={self.z_h: z_h, self.z_i:z_i})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.gen_dir + 'epoch' + str(epoch) + 'gens.jpg'
        show_in_one(path, sys, column=16, row=8)

    def visualize_test(self, sess, data):

        """
        :param sess:
        :return:
        """

        """
            test a
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size_i))
        sys = sess.run(self.gen, feed_dict={self.z_h: z_h, self.z_i: z_i})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.test_dir + 'diff_weights_same_z' + '1.jpg'
        show_in_one(path, sys, column=16, row=8)


        """
            test b
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
        # z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size_i))
        sys = sess.run(self.gen, feed_dict={self.z_h: z_h, self.z_i: z_i})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.test_dir + 'diff_weights_same_z' + '2.jpg'
        show_in_one(path, sys, column=16, row=8)


        """
            test c
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size_i))
        sys = sess.run(self.gen, feed_dict={self.z_h: z_h, self.z_i: z_i})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.test_dir + 'diff_z_same_weight' + '1.jpg'
        show_in_one(path, sys, column=16, row=8)

        """
            test d
        """
        z_i = np.random.normal(size=(self.z_batch_size, self.image_batch_size, self.z_size_i))
        sys = sess.run(self.gen, feed_dict={self.z_h: z_h, self.z_i: z_i})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.test_dir + 'diff_z_same_weight' + '2.jpg'
        show_in_one(path, sys, column=16, row=8)


        idx = random.randint(0, int(len(data) / self.batch_size) - 1)
        obs, labels = data.NextBatch(idx)
        """
            Recon test e
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
        sys = sess.run(self.recon, feed_dict={self.x: obs, self.z_h:z_h})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.test_dir + 'diff_weights_same_img' + '1.jpg'
        show_in_one(path, sys, column=16, row=8)
        """
            Recon test f
        """
        z_h = np.random.normal(size=(self.z_batch_size, self.z_size_h))
        sys = sess.run(self.recon, feed_dict={self.x: obs, self.z_h:z_h})
        sys = np.reshape(sys, [self.batch_size, 28, 28, 1])
        sys = np.array(sys * 255.0, dtype=np.float)
        path = self.test_dir + 'diff_weights_same_img' + '2.jpg'
        show_in_one(path, sys, column=16, row=8)
