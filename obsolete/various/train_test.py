#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from obsolete.various import Model, TextLoader


seq_length = 20

args = {
    'num_epochs': 1,
    'batch_size': 1,
    'learning_rate': 0.002,
    'decay_rate': 0.97,
    'model': 'lstm',
    'seq_length': seq_length,
    'num_layers': 2,
    'rnn_size': 128,
    'vocab_size': None, # Number of words (?)
    'grad_clip': 5,
    'data_dir': 'derp',
}

data_loader = TextLoader(args['data_dir'], args['batch_size'], args['seq_length'])
args['vocab_size'] = data_loader.vocab_size

model = Model(args)

with tf.Session() as session:
    tf.initialize_all_variables().run()

    # TODO restore state
    for e in range(args['num_epochs']):
        session.run(tf.assign(model.lr, args['learning_rate'] * (args['decay_rate'] ** e)))
        data_loader.reset_batch_pointer()
        state = session.run(model.initial_state)
        for b in range(data_loader.num_batches):
            x, y = data_loader.next_batch()
            '''
            print('x = ')
            print(x)
            print(x.shape)
            print(x[0])
            print('y = ')
            print(y)
            print(y.shape)
            print(y[0])
            print('--')
            '''

            feed = {model.input_data: x, model.targets: y}
            for i, (c, h) in enumerate(model.initial_state):
                feed[c] = state[i].c
                feed[h] = state[i].h
            print(feed)
            train_loss, state, _ = session.run([model.cost, model.final_state, model.train_op], feed)
            print("{}/{} (epoch {}), train_loss = {:.3f}" \
                  .format(e * data_loader.num_batches + b,
                          args['num_epochs'] * data_loader.num_batches,
                          e, train_loss))

    x, y = data_loader.next_batch()
    for i, (c, h) in enumerate(model.initial_state):
        feed[c] = state[i].c
        feed[h] = state[i].h
    print(feed)
    prediction = session.run([model.cost, model.final_state, model.train_op], feed)
    print('prediction = ')
    print(prediction)

    #result = session.run(y, feed_dict={x: [True, False, False, False, True, False, False, False, False, True]})
    #print(result)
