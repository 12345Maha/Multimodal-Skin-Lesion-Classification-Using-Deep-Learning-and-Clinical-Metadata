from keras.layers import *
from keras.models import *
from keras import backend as K

#defining attention layer
class attention(Layer):
    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(attention,self).__init__(name=name)
        self.return_sequences = return_sequences
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")

        super(attention,self).build(input_shape)

    def call(self, x):
        '''
        linearly projecting the input feature x into Query (Q), Key (K) and Value (V) vector, followed by multiplying V by the attention weight obtained from
        dot-product of Q and K to get the weighted feature x. This attention X contains height weight for relevant features and low weight for irrelevant features
        '''
        e = K.tanh(K.dot(x,self.W)+self.b)
        #softmax to get probabilties of classes based on weight e
        a = K.softmax(e, axis=1)
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)
