import tensorflow as tf

def crps(actual, pred):

   total = 0

   for j in range(-99, 100, 1):
       prob = 0
       j = tf.convert_to_tensor(j, dtype='float32')
       prob = tf.cast(tf.less_equal(pred, j), dtype='int32')
       H = tf.cast(tf.greater_equal(j, actual), dtype='int32')
       total += (prob - H) ** 2

   return total / 199