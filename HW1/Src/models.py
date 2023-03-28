# models.py
from torch.autograd import Variable

from preprocessing import Preprocess
from sentiment_data import *
from evaluator import *

import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


######################################
# IMPLEMENT THE SENTIMENT CLASSIFIER #
######################################

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """

    def __init__(self, n_classes, vocab_size, emb_dim, n_hidden_units):
        """
        In the __init__ function, you will define modules in FFNN.
        :param n_classes: number of classes in this classification problem
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units

        # TODO: create a randomly initialized embedding matrix, and set padding_idx as 0
        # PAD's embedding will not be trained and by default is initialized as zero
        self.word_embeddings = torch.nn.EmbeddingBag(self.vocab_size, self.emb_dim, padding_idx=0)

        # replace None with the correct implementation

        # TODO: implement the FFNN architecture
        # when you build the FFNN model, you will need specify the embedding size using self.emb_dim, the hidden size using self.n_hidden_units,
        # and the output class size using self.n_classes        

        # Linear function
        self.fc1 = torch.nn.Linear(self.emb_dim, self.n_hidden_units)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.n_hidden_units, self.n_classes)

    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return the logits outputs of FFNN (i.e., the unnormalized hidden units before softmax)
        """
        # TODO: implement the forward function, which returns the logits
        embeddings = self.word_embeddings(batch_inputs)
        fc1 = self.fc1(embeddings)
        relu = self.relu(fc1)
        out = self.fc2(relu)
        out = F.softmax(out, dim=1)
        return out

    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)

        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        # TODO: implement the prediction function, which could reuse the forward function
        out = self.forward(batch_inputs, batch_lengths)
        pred_label = out.data.max(1)[1].numpy()
        return list(pred_label)


##################################
# IMPLEMENT THE TRAINING METHODS #
##################################

def train_feedforward_neural_net(
        args,
        train_exs: List[SentimentExample],
        dev_exs: List[SentimentExample],
        requires_grad=True,
        optimizer_type="Adam",
        learning_rate=None
) -> FeedForwardNeuralNetClassifier:
    """
    Main entry point for your modifications. Trains and returns a FFNN model (whose architecture is configured based on args)

    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # TODO: read in all training examples and create a vocabulary (a List-type object called `vocab`)
    preprocess = Preprocess(train_exs, args.emb_dim, args.glove_path)
    vocab = preprocess.vocab  # replace None with the correct implementation

    # add PAD and UNK as the first two tokens
    # DO NOT CHANGE, PAD must go first and UNK next (as their indices have been hard-coded in several places)
    vocab = ["PAD", "UNK"] + vocab
    print("Vocab size:", len(vocab))
    # write vocab to an external file, so the vocab can be reloaded to index the test set
    with open("data/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    # indexing the training/dev examples
    indexing_sentiment_examples(train_exs, vocabulary=vocab, UNK_idx=1)
    indexing_sentiment_examples(dev_exs, vocabulary=vocab, UNK_idx=1)

    # TODO: create the FFNN classifier
    model = FeedForwardNeuralNetClassifier(2, len(vocab), args.emb_dim,
                                           args.n_hidden_units)  # replace None with the correct implementation
    model.word_embeddings.weight = nn.Parameter(torch.FloatTensor(preprocess.embedding), requires_grad=requires_grad)
    model.avg_loss_per_epochs = []

    # TODO: define an Adam optimizer, using default config
    # optimizer = optim.SGD(model.parameters(), lr=0.001)  # replace None with the correct implementation

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "Adagrad":
        if learning_rate:
            optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adagrad(model.parameters())
    elif optimizer_type == "AdamW":
        if learning_rate:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.AdamW(model.parameters())
    else:
        if learning_rate:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters())

    loss_function = nn.CrossEntropyLoss()

    # create a batch iterator for the training data
    batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=args.batch_size, PAD_idx=0, shuffle=True)

    # training
    best_epoch = -1
    best_acc = -1
    for epoch in range(args.n_epochs):
        print("Epoch %i" % epoch)

        batch_iterator.refresh()  # initiate a new iterator for this epoch

        # turn on the "training mode"
        model.train()
        batch_loss = 0.0
        batch_example_count = 0
        batch_data = batch_iterator.get_next_batch()

        while batch_data is not None:
            batch_inputs, batch_lengths, batch_labels = batch_data
            # TODO: clean up the gradients for this batch
            optimizer.zero_grad()

            # TODO: call the model to get the logits
            probs = model(batch_inputs, batch_lengths)

            # print(probs)
            # print(probs.view(1,-1), batch_labels)
            # print(torch.tensor(probs).shape(), batch_labels.shape())

            # TODO: calculate the loss (let's name it `loss`, so the follow-up lines could collect the stats)
            loss = loss_function(probs, batch_labels)
            # loss = Variable(loss, requires_grad=True)

            # record the loss and number of examples, so we could report some stats
            batch_example_count += len(batch_labels)
            batch_loss += loss.item() * len(batch_labels)

            # TODO: backpropagation (backward and step)
            loss.backward()
            optimizer.step()
            # get another batch
            batch_data = batch_iterator.get_next_batch()

        model.avg_loss_per_epochs.append(batch_loss / batch_example_count)
        print("Avg loss: %.5f" % (batch_loss / batch_example_count))

        # evaluate on dev set
        model.eval()  # turn on the "evaluation mode"
        acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))

            # save the current best model parameters
            print("Save the best model checkpoint as `best_model.ckpt`!")
            torch.save(model.state_dict(), "best_model.ckpt")
        print("-" * 10)

    # load back the best checkpoint on dev set
    model.load_state_dict(torch.load("best_model.ckpt"))

    model.eval()  # switch to the evaluation mode
    return model
