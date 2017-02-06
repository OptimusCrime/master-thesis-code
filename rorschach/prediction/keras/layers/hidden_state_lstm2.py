from keras import backend as K
from keras.layers.recurrent import LSTM

class HiddenStateLSTM2(LSTM):
    """LSTM with input/output capabilities for its hidden state.
    This layers behaves just like an LSTM, except that it accepts further inputs
    to be used as its initial states, and returns additional outputs,
    representing the layers's final states.
    See Also:
        https://github.com/fchollet/keras/issues/2995
    """
    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape, *hidden_shapes = input_shape
            for shape in hidden_shapes:
                assert shape[0]  == input_shape[0]
                assert shape[-1] == self.output_dim
        super().build(input_shape)

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        # Hidden
        if isinstance(x, (tuple, list)):
            x, *custom_initial = x
        else:
            custom_initial = None

        input_shape = K.int_shape(x)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layers. If your '
                             'first layers is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layers.')

        # Hidden
        if self.stateful and custom_initial:
            raise Exception(('Initial states should not be specified '
                             'for stateful LSTMs, since they would overwrite '
                             'the memorized states.'))

        if self.stateful:
            initial_states = self.states
        elif custom_initial: # Hidden
            initial_states = custom_initial
        else:
            initial_states = self.get_initial_states(x)

        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        # Hidden
        if isinstance(mask, list):
            mask = mask[0]

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, x)

        # Hidden
        if self.return_sequences:
            return [outputs] + list(states)
        else:
            return [last_output] + list(states)

    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape = input_shape[0]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.output_dim)
        else:
            output_shape = (input_shape[0], self.output_dim)
        state_output = (input_shape[0], self.output_dim)
        return [output_shape, state_output, state_output]

    def compute_mask(self, input, mask=None):
        if isinstance(mask, list) and len(mask) > 1:
            return mask
        elif self.return_sequences:
            return [mask, None, None]
        else:
            return [None] * 3
