# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:29:01 2020

@author: 劉冠廷
"""


import numpy as np
import tensorflow as tf

def build_net(input_placeholder,
                 output_size,
                 scope,
                 n_layers = 3,
                 size = 512,
                 activation = tf.nn.relu,
                 output_activation = None,
                 name = 'actor'
                     ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for i in range(n_layers):
            #print(i)
            out = tf.layers.dense(out,size,activation = activation)
        out = tf.layers.dense(out,output_size,activation = output_activation)
    param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope)
    return out,param
class MF_Policy():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 batch_size,
                 iterations,
                 learning_rate,
                 scope
                 ):
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self._state_dim = env._get_obs().shape[0]
        self._action_dim = env.action_space.sample().shape[0]
        self._s = tf.placeholder(tf.float32,[None,self._state_dim])#input_placeholder
        self._a = tf.placeholder(tf.float32,[None,self._action_dim])#answer_placeholder
        self.pred_a,self.param = build_net(input_placeholder = self._s,#return output_placeholder and parameter
                                 output_size = self._action_dim,
                                 scope = scope,
                                 n_layers = self.n_layers,
                                 size = self.size)
        #print(self._state_dim)
        #print(self._action_dim)
        '''for DL part'''
        self.pred_error = tf.reduce_mean(tf.reduce_sum(tf.square(self.pred_a - self._a),reduction_indices = [1]))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.pred_error)
        #######################################################
        '''For RL part'''
        with tf.variable_scope('advantage'):
            self.advantage = tf.placeholder(shape=[None,1],dtype = tf.float32)
            self.pred_dis = tf.contrib.distributions.Normal(self.pred_a,tf.ones_like(self.pred_a))
    def fit(self,data,sess,saver):
        obs = []
        actions = []
        for path in data:
            obs += path['observations']
            actions += path['actions']
        obs = np.array(obs)
        actions = np.array(actions)
        for train_iter in range(self.iterations):
            indices = np.random.randint(0,obs.shape[0],self.batch_size)
            batch_obs = obs[indices,:]
            batch_acts = actions[indices,:]
            sess.run(self.train_op,
                          feed_dict = {self._s : batch_obs,
                                       self._a : batch_acts                                     
                                 })
    def predict(self,state,sess):         # assume I covariance matrix
        if state.ndim <2:
            state = state[np.newaxis,:]
        return sess.run(self.pred_a,feed_dict = {self._s:state})
class critic():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 learning_rate
                 
                 ):
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self._state_dim = env._get_obs().shape[0]
        self._s = tf.placeholder(tf.float32,[None,self._state_dim])
        self.y = tf.placeholder(tf.float32,[None,1])
        self.value,_= build_net(input_placeholder = self._s,
                               output_size = 1,
                               scope = 'critic',
                               n_layers = self.n_layers,
                               output_activation = self.output_activation,
                               size = self.size,
                               name = 'critic')
        self.adv = self.y - self.value
        self.critic_loss = tf.reduce_mean(tf.square(self.adv))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.critic_op = optimizer.minimize(self.critic_loss)
    def get_v(self,state,sess):
        if state.ndim <2:
            state = state[np.newaxis,:]
        return sess.run(self.value,feed_dict = {self._s : state})
        
class PPO():
    def __init__(self,                 
                 env,               
                 n_layers = 3,
                 size = 512,
                 activation = tf.nn.relu,
                 output_activation = None,
                 batch_size = 256,
                 iterations = 100,
                 a_learning_rate = 0.001,
                 c_learning_rate = 0.002,
    ):
        ''''''
        self.gamma = 0.99
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.batch_size = batch_size
        self.iterations = iterations
        self.a_learning_rate = a_learning_rate
        self.c_learning_rate = c_learning_rate
        self.eps = 0.2
        '''actor'''
            
        self.actor = MF_Policy(
                     env = self.env,
                     n_layers = self.n_layers,
                     size = self.size,
                     activation = self.activation,
                     output_activation = self.output_activation,
                     batch_size = self.batch_size,
                     iterations = self.iterations,
                     learning_rate = self.a_learning_rate,
                     scope = 'actor'
                     )
        self.para = self.actor.param
        '''target_actor'''
        self.target_actor = MF_Policy(
                     env = self.env,
                     n_layers = self.n_layers,
                     size = self.size,
                     activation = self.activation,
                     output_activation = self.output_activation,
                     batch_size = self.batch_size,
                     iterations = self.iterations,
                     learning_rate = self.a_learning_rate,
                     scope = 'target_actor')
        self.target_para = self.target_actor.param
        '''actor_loss for PPO'''
        with tf.variable_scope('actor_loss'):
            ratio = self.actor.pred_dis.prob(self.actor._a) /(self.target_actor.pred_dis.prob(self.actor._a) + 1e-5)
            pg_losses1 = self.actor.advantage * ratio
            pg_losses2 = self.actor.advantage * tf.clip_by_value(ratio,1.0-self.eps,1.0+self.eps)
            self.actor_loss = - tf.reduce_mean(tf.minimum(pg_losses1,pg_losses2))
            
        '''target_net replace_ment'''            
        self.syn_old_pi = [oldp.assign(p) for p,oldp in zip(self.para,self.target_para)]
        '''critic'''
        self.critic = critic(
                     env = self.env,
                     n_layers = self.n_layers,
                     size = self.size,
                     activation = self.activation,
                     output_activation = None,
                     learning_rate = self.c_learning_rate)
        self.saver = tf.train.Saver(max_to_keep = 5)
    '''let actor to be initialized as a true policy for modle-based method'''
    def init_mf_agent(self,data,sess):
        self.actor.fit(data,sess,self.saver)
        self.save_model('final',sess,'init')
        print('fit done')
    '''use the actor to get the action'''
    def choose_action(self,state,sess):
        return self.actor.predict(state,sess)
    '''use the critic to get the value'''
    def get_value(self,state,sess):
        return self.critic.get_v(state,sess)
    def process(self,sess):
        state = self.env.reset()
        next_state = 0
        total_rewards = 0
        plot_reward =[]
        plot_episode = []
        episode_length = 0
        self.training_epi = 0
        max_epi = 1000
        max_tra_len = 1000
        rw_reward = 0
        rw_rewards = []
        while self.training_epi < max_epi:
            states_buff = []
            actions_buff = []
            rewards_buff = []
            for i in range (max_tra_len):
                action = self.choose_action(state,sess)
                next_state,reward,done,_ = self.env.step(action)
                total_rewards += reward
                episode_length +=1
                states_buff.append(state)
                actions_buff.append(action)            
                rewards_buff.append(reward/10)
                state = next_state
                if done:
                    break
            bootstrap_value = self.get_value(state,sess)
            if states_buff:
                discounted_r = []
                v_s_ = bootstrap_value
                for r in rewards_buff[::-1]:
                    v_s_ = r + self.gamma * v_s_
                    discounted_r.append(v_s_[0])
                discounted_r.reverse()

                bs,ba,br = np.vstack(states_buff),np.vstack(actions_buff),np.array(discounted_r)
                self.train(sess,bs,ba,br,self.training_epi)
                print("[EPISODE]:",self.training_epi," ","[REWARD]:"," ",total_rewards)
                if(self.training_epi and self.training_epi%10 == 0):
                    print("[EPISODE:]",self.training_epi," ","RW_REWARD"," ",rw_reward)
                plot_reward.append(total_rewards)
                plot_episode.append(episode_length)
                total_rewards = 0
                episode_length = 0
                rw_reward = 0.95 * rw_reward + 0.05 * total_rewards
                rw_rewards.append(rw_reward)
                plot_reward_ = np.array(plot_reward)
                plot_episode_ = np.array(plot_episode)
                rw_rewards_ = np.array(rw_rewards)
                np.save('reward'+str(self.training_epi),plot_reward_)
                np.save('episode_length'+str(self.training_epi),plot_episode_)
                np.save('rw_rewards'+str(self.training_epi),rw_rewards_)
            self.training_epi += 1
        self.save_model('final',sess,'mbmf')
        plot_reward = np.array(plot_reward)
        plot_episode = np.array(plot_episode)
        rw_reward = np.array(rw_rewards)
        np.save('reward',plot_reward)
        np.save('episode_length',plot_episode)
        np.save('rw_rewards',rw_rewards)
    def save_model(self,name,sess,mode = 'mbmf'):
        self.saver.save(sess,'model/'+ mode +'_'+str(name)+'_.ckpt')
    def load_model(self,sess,name='model/init_final_.ckpt'):
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
        if checkpoint:
            self.saver.restore(sess,checkpoint.model_checkpoint_path)
            print('load the model successfully')
        else:
            print("model not found.")
    def train(self,sess,bs,ba,br,epi):
        '''copy param'''
        if epi %10 and epi :
            print("Copy the param to the target_Net")
            sess.run(self.syn_old_pi)            
        #print('adv')
        adv = sess.run(self.critic.adv,feed_dict={self.critic._s:bs,
                                                  self.critic.y:br})
        #print('al')
    
    
        sess.run(self.actor_loss,feed_dict={self.actor._s:bs,
                                            self.target_actor._s:bs,
                                            self.actor._a:ba,
                                            self.target_actor._s:bs,
                                            self.actor.advantage : adv})
        #print('co')
        sess.run(self.critic.critic_op,feed_dict = {self.critic._s:bs,
                                                    self.critic.y:br})
        if epi and epi % 100 == 0:
            self.save_model(epi,sess,'mbmf')
            