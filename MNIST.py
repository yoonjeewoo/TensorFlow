from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#getShape
import tensorflow as tf

X = tf.placeholder("float32",[None,784])
Y = tf.placeholder("float32",[None,10])

W1 = tf.Variable(tf.random_normal([784,256]))
W2 = tf.Variable(tf.random_normal([256,256]))
W3 = tf.Variable(tf.random_normal([256,10]))

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))
hypothesis = (tf.add(tf.matmul(L2,W3),b3))

#cross_entropy = -tf.reduce_sum(Y*tf.log(hypothesis))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)
    
    for step in xrange(2001):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X:batch_xs,Y:batch_ys})
        #if step % 20 == 0:
            #print "batch_ys"
            #print batch_ys
            #print "hypothesis"
            #print sess.run(hypothesis, feed_dict={X:batch_xs,Y:batch_ys})
    correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels})
        
