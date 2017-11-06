# Bangla_char_rnn_tensorflow
Author: Shuvendu Roy Bikash
Dept of CSE, KUET

Predicting next Bangla char or word from the previous text written using LSTM in tensorflow

Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow. 

Inspired by [Sherjil Ozair's char-nn](https://github.com/sherjilozair/char-rnn-tensorflow) which is a tensorflow implementations of the project of from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

## Requirements
- [Tensorflow 1.0](http://www.tensorflow.org) or greater

## Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`. To access all the parameters use `python train.py --help`.
you might want to specify the data directory using --data_dir "directory"

To sample from a checkpointed model, `python sample.py --prime "input_text"`.

To get one word instead of a sentence `python word_predict.py "input_text"`

## Datasets
You can use any plain text file as input. But is must be it utf-8 format to represent bangla with out breakdown

Then start train from the top level directory using `python train.py --data_dir=./data/sherlock/`

## Tensorboard
To visualize training progress, model graphs, and internal state histograms:  fire up Tensorboard and point it at your `log_dir`.  E.g.:
```bash
$ tensorboard --logdir=./logs/
```

Then open a browser to [http://localhost:6006](http://localhost:6006) or the correct IP/Port specified.
