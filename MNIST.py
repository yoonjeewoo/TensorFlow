from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#getShape
import tensorflow as tf

X = tf.placeholder("float",[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

mat = tf.matmul(X,W)
hypothesis = tf.nn.softmax(mat + b)
Y = tf.placeholder("float",[None,10])

cross_entropy = -tf.reduce_sum(Y*tf.log(hypothesis))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)
    
    for step in xrange(2001):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X:batch_xs,Y:batch_ys})
        if step % 20 == 0:
            print "batch_ys"
            print batch_ys
            print "hypothesis"
            print sess.run(hypothesis, feed_dict={X:batch_xs,Y:batch_ys})
    correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels})
        
