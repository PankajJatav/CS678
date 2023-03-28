import os
import matplotlib
import random, torch, numpy as np
import matplotlib.pyplot as plt
from evaluator import evaluate
from models import train_feedforward_neural_net
from sentiment_data import read_sentiment_examples, read_blind_sst_examples


def get_column(matrix, i):
    return [row[i] for row in matrix]

def save_loss_fig(data, file_loc):

    plt.xlabel('Epochs')
    plt.ylabel('Avg Loss Per Epochs')
    plt.title("Graph: Average Loss Per Epochs")
    plt.plot(data, label="Graph")
    plt.xticks(np.arange(len(data)), np.arange(1, len(data) + 1))
    plt.savefig(file_loc, dpi=500)
    plt.close()

def save_comp_fig(x, y1, y2, x_label, file_loc):

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 2, constrained_layout=True)

    # Accuracy
    axis[0, 0].plot(x, get_column(y1, 0), label="train")
    axis[0, 0].plot(x, get_column(y2, 0), label="dev", color="green")
    axis[0, 0].legend()
    axis[0, 0].set_xlabel(x_label)
    axis[0, 0].set_ylabel("accuracy")
    axis[0, 0].set_title("Accuracy")

    # Precision
    axis[0, 1].plot(x, get_column(y1, 1), label="train")
    axis[0, 1].plot(x, get_column(y2, 1), label="dev", color="green")
    axis[0, 1].legend()
    axis[0, 1].set_xlabel(x_label)
    axis[0, 1].set_ylabel("precision")
    axis[0, 1].set_title("Precision")

    # Recall
    axis[1, 0].plot(x, get_column(y1, 2), label="train")
    axis[1, 0].plot(x, get_column(y2, 2), label="dev", color="green")
    axis[1, 0].legend()
    axis[1, 0].set_xlabel(x_label)
    axis[1, 0].set_ylabel("recall")
    axis[1, 0].set_title("Recall")

    # F1 Score
    axis[1, 1].plot(x, get_column(y1, 3), label="train")
    axis[1, 1].plot(x, get_column(y2, 3), label="dev", color="green")
    axis[1, 1].legend()
    axis[1, 1].set_xlabel(x_label)
    axis[1, 1].set_ylabel("F1 score")
    axis[1, 1].set_title("F1 Score")

    # Combine all, save and display
    figure.suptitle("Train and Dev Model Comparison")
    plt.savefig(file_loc, dpi=500)
    figure.clear(True)

def save_comp_bar_fig(x, y1, y2, x_label, file_loc):

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 2, constrained_layout=True)
    r = np.arange(len(x))

    # Accuracy
    axis[0, 0].bar(r, get_column(y1, 0), label="train", width = 0.25)
    axis[0, 0].bar(r + 0.25, get_column(y2, 0), label="dev", width = 0.25, color="green")
    axis[0, 0].set_xticks(r + 0.25 / 2, x)
    axis[0, 0].legend()
    axis[0, 0].set_xlabel(x_label)
    axis[0, 0].set_ylabel("accuracy")
    axis[0, 0].set_title("Accuracy")

    # Precision
    axis[0, 1].bar(r, get_column(y1, 1), label="train", width = 0.25)
    axis[0, 1].bar(r + 0.25, get_column(y2, 1), label="dev", width = 0.25, color="green")
    axis[0, 1].set_xlabel(x_label)
    axis[0, 1].set_xticks(r + 0.25 / 2, x)
    axis[0, 1].legend()
    axis[0, 1].set_ylabel("precision")
    axis[0, 1].set_title("Precision")

    # Recall
    axis[1, 0].bar(r, get_column(y1, 2), label="train", width = 0.25)
    axis[1, 0].bar(r + 0.25, get_column(y2, 2), label="dev", width = 0.25, color="green")
    axis[1, 0].set_xticks(r + 0.25 / 2, x)
    axis[1, 0].legend()
    axis[1, 0].set_xlabel(x_label)
    axis[1, 0].set_ylabel("recall")
    axis[1, 0].set_title("Recall")

    # F1 Score
    axis[1, 1].bar(r, get_column(y1, 3), label="train", width = 0.25)
    axis[1, 1].bar(r + 0.25, get_column(y2, 3), label="dev", width = 0.25, color="green")
    axis[1, 1].set_xticks(r + 0.25 / 2, x)
    axis[1, 1].legend()
    axis[1, 1].set_xlabel(x_label)
    axis[1, 1].set_ylabel("F1 score")
    axis[1, 1].set_title("F1 Score")

    # Combine all, save and display
    figure.suptitle("Train and Dev Model Comparison")
    plt.savefig(file_loc, dpi=500)
    figure.clear(True)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



default_parameter = dotdict({
    'train_path': 'data/train.txt',
    'dev_path': 'data/dev.txt',
    'blind_test_path': 'data/test-blind.txt',
    'test_output_path': 'test-blind.output.txt',
    'glove_path': None,
    'no_run_on_test': True,
    'n_epochs': 10,
    'batch_size': 32,
    'emb_dim': 300,
    'n_hidden_units': 300
})

print(default_parameter)

# Set up overall seed
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Load train, dev, and test exs and index the words.
train_exs = read_sentiment_examples(default_parameter.train_path)
dev_exs = read_sentiment_examples(default_parameter.dev_path)
test_exs_words_only = read_blind_sst_examples(default_parameter.blind_test_path)

# Tuning Epochs
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots"):
    os.makedirs("plots")
if not os.path.exists("plots/epochs_tuning"):
    os.makedirs("plots/epochs_tuning")
if not os.path.exists("plots/epochs_tuning/loss"):
    os.makedirs("plots/epochs_tuning/loss")
epochs_arr = [5, 10, 15, 20, 25, 30]
for epochs in epochs_arr:
    default_parameter.n_epochs = epochs
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/epochs_tuning/loss/' + str(epochs) + '_epochs.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_fig(
    epochs_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'no of epochs',
    'plots/epochs_tuning/result.png')

default_parameter.n_epochs = 10

# Tuning Batch size
eval_train_matrix = []
eval_dev_matrix = []

if not os.path.exists("plots/batch_tuning"):
    os.makedirs("plots/batch_tuning")
if not os.path.exists("plots/batch_tuning/loss"):
    os.makedirs("plots/batch_tuning/loss")
batch_arr = [16, 32, 64, 128, 256]
for batch in batch_arr:
    default_parameter.batch_size = batch
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/batch_tuning/loss/' + str(batch) + '_batch.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_fig(
    batch_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'batch size',
    'plots/batch_tuning/result.png')
default_parameter.batch_size = 32


# Tuning Embedding  size
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots/emb_dim"):
    os.makedirs("plots/emb_dim")
if not os.path.exists("plots/emb_dim/loss"):
    os.makedirs("plots/emb_dim/loss")
emb_dim_arr = [50, 100, 300, 500, 1000]
for emb_dim in emb_dim_arr:
    default_parameter.emb_dim = emb_dim
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/emb_dim/loss/' + str(emb_dim) + '_emb_dim.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_fig(
    emb_dim_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'Embedding size',
    'plots/emb_dim/result.png')
default_parameter.emb_dim = 300

# Tuning Embedding  size
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots/hidden_units"):
    os.makedirs("plots/hidden_units")
if not os.path.exists("plots/hidden_units/loss"):
    os.makedirs("plots/hidden_units/loss")
n_hidden_units_arr = [50, 100, 300, 500, 1000]
for n_hidden_units in n_hidden_units_arr:
    default_parameter.n_hidden_units = n_hidden_units
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/hidden_units/loss/' + str(n_hidden_units) + '_hidden_units.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_fig(
    n_hidden_units_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'Hidden Unit size',
    'plots/hidden_units/result.png')

default_parameter.n_hidden_units = 300

# SGD
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots/SGD"):
    os.makedirs("plots/SGD")
if not os.path.exists("plots/SGD/loss"):
    os.makedirs("plots/SGD/loss")
learning_rate_arr = [1, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rate_arr:
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs, optimizer_type="SGD", learning_rate=learning_rate)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/SGD/loss/' + str(learning_rate) + '_SGD.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_bar_fig(
    learning_rate_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'Learning Rate',
    'plots/SGD/result.png')


# Adam
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots/Adam"):
    os.makedirs("plots/Adam")
if not os.path.exists("plots/Adam/loss"):
    os.makedirs("plots/Adam/loss")
learning_rate_arr = [None, 1, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rate_arr:
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs, optimizer_type="Adam", learning_rate=learning_rate)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/Adam/loss/' + str(learning_rate) + '_Adam.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_bar_fig(
    learning_rate_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'Learning Rate',
    'plots/Adam/result.png')

# Adagrad
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots/Adagrad"):
    os.makedirs("plots/Adagrad")
if not os.path.exists("plots/Adagrad/loss"):
    os.makedirs("plots/Adagrad/loss")
learning_rate_arr = [None, 1, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rate_arr:
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs, optimizer_type="Adagrad", learning_rate=learning_rate)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/Adagrad/loss/' + str(learning_rate) + '_Adagrad.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_bar_fig(
    learning_rate_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'Learning Rate',
    'plots/Adagrad/result.png')

# AdamW
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots/AdamW"):
    os.makedirs("plots/AdamW")
if not os.path.exists("plots/AdamW/loss"):
    os.makedirs("plots/AdamW/loss")
learning_rate_arr = [None, 1, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rate_arr:
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs, optimizer_type="AdamW", learning_rate=learning_rate)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/AdamW/loss/' + str(learning_rate) + '_AdamW.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_bar_fig(
    learning_rate_arr,
    eval_train_matrix,
    eval_dev_matrix,
    'Learning Rate',
    'plots/AdamW/result.png')


# Glove
eval_train_matrix = []
eval_dev_matrix = []
if not os.path.exists("plots/GloveT"):
    os.makedirs("plots/GloveT")
if not os.path.exists("plots/GloveT/loss"):
    os.makedirs("plots/GloveT/loss")
req_grads = [True, False]
for req_grad in req_grads:
    default_parameter.glove_path = 'glove.6B.300d.txt'
    model = train_feedforward_neural_net(default_parameter, train_exs, dev_exs, requires_grad=req_grad)
    save_loss_fig(model.avg_loss_per_epochs, 'plots/GloveT/loss/' + str(req_grad) + '_Glove.png')
    eval_train_matrix.append(list(evaluate(model, train_exs, return_metrics=True)))
    eval_dev_matrix.append(list(evaluate(model, dev_exs, return_metrics=True)))
print(eval_train_matrix, eval_dev_matrix)
save_comp_bar_fig(
    req_grads,
    eval_train_matrix,
    eval_dev_matrix,
    'Require Grads',
    'plots/GloveT/result.png')