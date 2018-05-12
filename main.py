import argparse
import numpy as np
import tensorflow as tf
import model.prior_factory as prior
import model.aae as aae
import model.utils as utils
import pickle
import collections
from sklearn import svm
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior_type', type=str, default='mixGaussian',
                        choices=['mixGaussian', 'swiss_roll', 'normal'],
                        help='The type of prior')
    parser.add_argument('--n_hidden', type=int, default=1000, help='Number of hidden units in MLP')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=300, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=385, help='Batch size')

    return parser.parse_args()

def extract_code_vector(args,idx):
    """ prepare IEMOCAP data """

    path_X_Uttr_train  = "datasets/0%s/X_stat_Utter_train"%idx
    path_y_Uttr_train  = "datasets/0%s/y_Utter_train"%idx

    path_X_Uttr_test   = "datasets/0%s/X_stat_Utter_test"%idx
    path_y_Uttr_test   = "datasets/0%s/y_Utter_test"%idx
    
    train_total_data    = np.load('%s.npy' % path_X_Uttr_train)
    test_data           = np.load('%s.npy'  % path_X_Uttr_test)

    train_size      =  train_total_data.shape[0]
    n_samples       = train_size

    y_train         = np.load('%s.npy' % path_y_Uttr_train)
    y_test          = np.load('%s.npy'  % path_y_Uttr_test)
    num_classes     = np.max(y_train)+1
    train_labels    = utils.dense_to_one_hot(y_train,num_classes) #tf.one_hot(y_train, num_classes) 
    test_labels     = utils.dense_to_one_hot(y_test,num_classes) #tf.one_hot(y_test, num_classes) #

    # network architecture
    n_hidden = args.n_hidden
    dim_features = train_total_data.shape[1]
    dim_z = 2                      # to visualize learned manifold
    nDrop_out= 0.5
    display_step=100
    
    # train
    n_epochs = args.num_epochs
    batch_size = n_samples
    learn_rate = args.learn_rate

    """ build graph """
    tf.reset_default_graph()  
    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_features], name='input')
    x = tf.placeholder(tf.float32, shape=[None, dim_features], name='target')
    x_id = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_label')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # samples drawn from prior distribution
    z_sample = tf.placeholder(tf.float32, shape=[None, dim_z], name='prior_sample')
    z_id = tf.placeholder(tf.float32, shape=[None, num_classes], name='prior_sample_label')

    # network architecture
    y, z, neg_marginal_likelihood, D_loss, G_loss = aae.adversarial_autoencoder(x_hat, x, x_id, z_sample, z_id, dim_features,
                                                                                dim_z, n_hidden, keep_prob)
    z_train=aae.encoder(x, n_hidden, dim_z)
    z_test=aae.encoder(x, n_hidden, dim_z)

    # optimization
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    g_vars = [var for var in t_vars if "MLP_encoder" in var.name]
    ae_vars = [var for var in t_vars if "MLP_encoder" or "MLP_decoder" in var.name]

    train_op_ae = tf.train.AdamOptimizer(learn_rate).minimize(neg_marginal_likelihood, var_list=ae_vars)
    train_op_d = tf.train.AdamOptimizer(learn_rate/5).minimize(D_loss, var_list=d_vars)
    train_op_g = tf.train.AdamOptimizer(learn_rate).minimize(G_loss, var_list=g_vars)

    """ training """
    # train
    total_batch = int(n_samples / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : nDrop_out})

        for epoch in range(n_epochs):
            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_total_data[offset:(offset + batch_size), :]
                batch_ids_input = train_labels[offset:(offset + batch_size), :]
                batch_xs_target = batch_xs_input

                # draw samples from prior distribution
                if args.prior_type == 'mixGaussian':
                    z_id_ = np.random.randint(0, num_classes, size=[batch_size])
                    samples = prior.gaussian_mixture(batch_size, dim_z, label_indices=z_id_)
                elif args.prior_type == 'swiss_roll':
                    z_id_ = np.random.randint(0, num_classes, size=[batch_size])
                    samples = prior.swiss_roll(batch_size, dim_z, label_indices=z_id_)
                elif args.prior_type == 'normal':
                    samples, z_id_ = prior.gaussian(batch_size, dim_z, use_label_info=True)
                else:
                    raise Exception("[!] There is no option for " + args.prior_type)

                z_id_one_hot_vector = np.zeros((batch_size, num_classes))
                z_id_one_hot_vector[np.arange(batch_size), z_id_] = 1

                # reconstruction loss
                _, loss_likelihood = sess.run(
                    (train_op_ae, neg_marginal_likelihood),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, x_id: batch_ids_input, z_sample: samples,
                               z_id: z_id_one_hot_vector, keep_prob:nDrop_out})

                # discriminator loss
                _, d_loss = sess.run(
                    (train_op_d, D_loss),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, x_id: batch_ids_input, z_sample: samples,
                               z_id: z_id_one_hot_vector, keep_prob: nDrop_out})

                # generator loss
                for _ in range(2):
                    _, g_loss = sess.run(
                        (train_op_g, G_loss),
                        feed_dict={x_hat: batch_xs_input, x: batch_xs_target, x_id: batch_ids_input, z_sample: samples,
                                   z_id: z_id_one_hot_vector, keep_prob: nDrop_out})
    
            tot_loss = loss_likelihood + d_loss + g_loss
            # print cost every epoch
            if (epoch % display_step) == 0:
                print("epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (epoch, tot_loss, loss_likelihood, d_loss, g_loss))
        
        """ generate code-vectors """
        X_train = sess.run((z_train), feed_dict={x: train_total_data})
        X_test = sess.run((z_test), feed_dict={x: test_data})
        
        data={'X_Utter_train':X_train, 'X_Utter_test':X_test, 'y_Utter_train':y_train, 'y_Utter_test':y_test}
        filename   = "datasets/0%s/Z.pickle"     %idx
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate(idx):
    filename  = "datasets/0%d/Z.pickle"  %idx
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        
    X_train         = data['X_Utter_train']
    X_test          = data['X_Utter_test']
    y_train         = data['y_Utter_train']
    y_test          = data['y_Utter_test']
        
    #X_train, X_test= utils.normalize_Zscore(X_train, X_test)
    #X_train, X_test, norm = utils.normalize_MinMax(X_train, X_test)

    from sklearn.model_selection import GridSearchCV 
    '''
    parameters = {'C':[np.power(2,4), np.power(2,5), np.power(2,6),np.power(2,7), 100,], 'gamma': 
              [ 1/np.power(2,9),1/np.power(2,10),0.0001]}

    svr = svm.SVC(kernel='rbf')
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    '''
    clf = svm.SVC(kernel='rbf',gamma=0.001, C=100,cache_size=20000)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    


    test_weighted_accuracy=clf.score(X_test, y_test)

    uar=0
    cnf_matrix = confusion_matrix(y_test, y_pred)
    diag=np.diagonal(cnf_matrix)
    for index,i in enumerate(diag):
        uar+=i/collections.Counter(y_test)[index]
    test_unweighted_accuracy=uar/len(cnf_matrix)

    accuracy=[]
    accuracy.append(test_weighted_accuracy*100)
    accuracy.append(test_unweighted_accuracy*100)
        # Compute confusion matrix
    cnf_matrix = np.transpose(cnf_matrix)
    cnf_matrix = cnf_matrix*100 / cnf_matrix.astype(np.int).sum(axis=0)
    cnf_matrix = np.transpose(cnf_matrix).astype(float)
    cnf_matrix = np.around(cnf_matrix, decimals=1)

    #accuracy per class 
    conf_mat = (cnf_matrix.diagonal()*100)/cnf_matrix.sum(axis=1)
    conf_mat = np.around(conf_mat, decimals=2)

    print('Feature Dimension: %d'%X_train.shape[1])
    print('Confusion Matrix:\n%s'%cnf_matrix)
    print('Accuracy per classes:\n%s'%conf_mat)
    print("WAR\t\t\t:\t%.2f %%" %(test_weighted_accuracy*100))
    print("UAR\t\t\t:\t%.2f %%" %(test_unweighted_accuracy*100))
    return np.around(np.array(accuracy),decimals=1)
    
if __name__ == '__main__':
    args = parse_args()
    
    acc_stat=np.zeros(2)
    for idx in range(10):
        extract_code_vector(args,idx) 
        acc_stat += evaluate(idx)
    print('[ %s ]'%(acc_stat/10))
    
    
    