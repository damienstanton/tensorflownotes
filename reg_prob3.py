"""
Simple Autoencoder with ReLu and L2 Loss Regularization
written for Google TensorFlow DL Course on Udacity
Assignment Set 3, Problem 4
Based on model by github.com/kcbighuge
"""
# Params
batch_size = 128
num_hidden_nodes = 1024
# How strong the regularization factor is
reg_lambda = 0.001

# Helper methods give finer-grain control over hyperparams. Kudos to kcbighuge
# for this one
# Traditional method:
# bias1 = tf.Variable(tf.zeros([num_hidden_nodes]))
# bias2 = tf.Variable(tf.zeros([num_labels]))
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, constant=0.1):
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)

# Define the graph
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size,
                                             image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables
    weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
    biases1 = bias_variable([num_hidden_nodes])

    weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes, num_labels]))
    biases2 = bias_variable([num_labels])

    logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
    a1 = tf.nn.relu(logits1)

    logits2 = tf.matmul(a1, weights2) + biases2
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits2, tf_train_labels))

    # L2 regularization
    regularizers = (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) +
                    tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2))
    # Add the regularization term to the loss.
    loss += reg_lambda * 0.5 * regularizers

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits2)
    valid_a1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_a1, weights2) + biases2)
    test_a1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_prediction = tf.nn.softmax(
        tf.matmul(test_a1, weights2) + biases2)

# Run the model
num_steps = 10000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
