import tensorflow as tf


class DeepQNetwork:
    def __init__(self, state_shape, num_units, num_actions, name):
        # input size
        input_size = 1
        for i in range(len(state_shape)):
            input_size *= state_shape[i]

        # inner product 1
        self.inner1_inputs = input_size
        self.inner1_outputs = num_units
        self.inner1_weights, self.inner1_biases = self.create_inner_net(
            [self.inner1_inputs, self.inner1_outputs], name=name + 'inner1')

        # hybrid reward network (inner product 2)
        self.inner2_inputs = self.inner1_outputs
        self.inner2_outputs = num_actions
        self.inner2_weights, self.inner2_biases = self.create_inner_net(
            [self.inner2_inputs, self.inner2_outputs], name=name + 'inner2')

        # Network name
        self.name = name

    def forward(self, data):
        inner1 = tf.nn.relu(tf.matmul(data, self.inner1_weights) +
                            self.inner1_biases, name=self.name + 'relu')

        inner2 = tf.matmul(inner1, self.inner2_weights) + self.inner2_biases

        return inner2

    def q_values(self, data):
        return self.forward(data)

    def filtered_q_values(self, data, q_value_filter):
        return tf.multiply(self.q_values(data), q_value_filter)

    def loss(self, data, target, q_value_filter):
        filtered_qs = self.filtered_q_values(data, q_value_filter)
        return tf.reduce_mean(tf.nn.l2_loss(target - filtered_qs))

    def clipped_loss(self, data, target, q_value_filter):
        filtered_qs = self.filtered_q_values(data, q_value_filter)
        error = tf.abs(target - filtered_qs)
        quadratic = tf.clip_by_value(error, 0.0, 1.0)
        linear = error - quadratic
        return tf.reduce_sum(0.5 * tf.square(quadratic) + linear)

    def create_conv_net(self, shape, name):
        weights = tf.Variable(tf.truncated_normal(
            shape=shape, stddev=0.01), name=name + 'weights')
        biases = tf.Variable(tf.constant(
            0.01, shape=[shape[3]]), name=name + 'biases')
        return weights, biases

    def create_inner_net(self, shape, name):
        weights = tf.Variable(tf.truncated_normal(
            shape=shape, stddev=0.01), name=name + 'weights')
        biases = tf.Variable(tf.constant(
            0.01, shape=[shape[1]]), name=name + 'biases')
        return weights, biases

    def weights_and_biases(self):
        return [self.inner1_weights, self.inner1_biases, self.inner2_weights, self.inner2_biases]

    def copy_network_to(self, target, session):
        copy_operations = [target.assign(origin)
                           for origin, target in zip(self.weights_and_biases(), target.weights_and_biases())]
        session.run(copy_operations)
