import tensorflow as tf

def getPairs(objects_array):
    """A function that compute all the possible non-ordered pairs of elements without repetition
    given an input sequence of elements objects_array. Very fast!"""

    pairs = tf.meshgrid(objects_array, objects_array)
    pairs = tf.transpose(pairs)
    pairs = tf.reshape(pairs, shape=[-1,2])

    column_1, column_2 = tf.unstack(pairs, axis=1)
    interesting_pairs = tf.boolean_mask(pairs, column_1 < column_2)

    return interesting_pairs