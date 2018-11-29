import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import utils1 as utils
import sys

# Hyper-parameters
BATCH_SIZE = 1024
LEARN_RATE = 0.01
EPOCHS = 15
FIRST_HIDDEN_LAYER_SIZE = 150
SECOND_HIDDEN_LAYER_SIZE = 50
NUMBER_OF_CLASSES = 10
EMBEDDING_VEC_SIZE = 50
WIN_SIZE = 5
INPUT_SIZE = 250

SUFFIX_SIZE = 3
PREFIX_SIZE = 3


class ModelTrainer(object):
    """
    Trains a given model on a given set of data for train and validation.
    """

    def __init__(self, train_loader, validation_loader, test_loader, model, optimizer):
        """
        initializes the ModelTrainer.
        :param train_loader: training set
        :param validation_loader: validation set
        :param test_loader: test set
        :param model: neural network model
        :param optimizer: optimizer
        """
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer

    def run(self, tagger_type):
        """
        calls train and validation methods as the number of epochs.
        calls to method that draw the results graph (avg loss per epoch)
        finally, model is passing on the test set.
        :return: None
        """
        avg_train_loss_per_epoch_dict = {}
        avg_validation_loss_per_epoch_dict = {}
        validation_accuracy_per_epoch_dict = {}
        for epoch in range(1, EPOCHS + 1):
            print str(epoch)
            self.train(epoch, avg_train_loss_per_epoch_dict)
            self.validation(epoch, avg_validation_loss_per_epoch_dict,
                            validation_accuracy_per_epoch_dict, tagger_type)
        plotTrainAndValidationGraphs(avg_validation_loss_per_epoch_dict, validation_accuracy_per_epoch_dict)
        self.test(tagger_type)

    def train(self, epoch, avg_train_loss_per_epoch_dict):
        """
        go through all examples on the validation set, calculates perdiction, loss
        , accuracy, and updating the model parameters.
        :param epoch: number of epochs
        :param avg_train_loss_per_epoch_dict: avg loss per epoch dictionary
        :return: None
        """
        self.model.train()
        train_loss = 0
        correct = 0

        for data, labels in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            # negative log likelihood loss
            loss = F.nll_loss(output, labels)
            train_loss += loss
            # calculating gradients
            loss.backward()
            # updating parameters
            self.optimizer.step()

        train_loss /= (len(self.train_loader))
        avg_train_loss_per_epoch_dict[epoch] = train_loss
        print("Epoch: {} Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(epoch, train_loss, correct, len(self.train_loader) * BATCH_SIZE,
                                                                                            100. * correct / (len(self.train_loader) * BATCH_SIZE)))

    def validation(self, epoch_num, avg_validation_loss_per_epoch_dict,
                   validation_accuracy_per_epoch_dict, tagger_type):
        """
        go through all examples on the validation set, calculates perdiction, loss
        and accuracy
        :param epoch: number of epochs
        :param avg_train_loss_per_epoch_dict: avg loss per epoch dictionary
        :return: None
        """
        self.model.eval()
        validation_loss = 0
        correct = 0
        total = 0
        for data, target in self.validation_loader:
            output = self.model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            if tagger_type == 'ner':
                if utils.INDEX_TO_TAG[pred.cpu().sum().item()] != 'O' or utils.INDEX_TO_TAG[target.cpu().sum().item()] != 'O':
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    total+=1
            else:
                total+=1
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()


        validation_loss /= len(self.validation_loader)
        avg_validation_loss_per_epoch_dict[epoch_num] = validation_loss
        accuracy =  100. * correct / total
        validation_accuracy_per_epoch_dict[epoch_num] = accuracy

        print('\n Epoch:{} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_num, validation_loss, correct, total,
            accuracy))

    def test(self, tagger_type):
        """
        writes all the model predictions on the test set to test.pred file.
        :return:  None
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        pred_list = []
        for data in self.test_loader:
            output = self.model(torch.LongTensor(data))
            # get the predicted class out of output tensor
            pred = output.data.max(1, keepdim=True)[1]
            # add current prediction to predictions list
            pred_list.append(pred.item())

        pred_list = self.convert_tags_indices_to_tags(pred_list)
        self.write_test_results_file(tagger_type + "/test", "test3." + tagger_type, pred_list)

    def convert_tags_indices_to_tags(self, tags_indices_list):
        return [utils.INDEX_TO_TAG[index] for index in tags_indices_list]

    def write_test_results_file(self, test_file_name, output_file_name, predictions_list):
        with open(test_file_name, 'r') as test_file, open(output_file_name, 'w') as output:
            content = test_file.readlines()
            i = 0
            for line in content:
                if line == '\n':
                    output.write(line)
                else:
                    output.write(line.strip('\n') + " " + predictions_list[i] + "\n")
                    i += 1


class NeuralNet(nn.Module):
    """
    First model version.
    two hidden layers.
    activation function betweeb the layers: Relu.
    using batch normalization.
    """
    def __init__(self, use_pre_trained,input_size):
        super(NeuralNet, self).__init__()
        self.use_pre_trained = use_pre_trained
        if not use_pre_trained:
            self.E = nn.Embedding(len(utils.WORDS_SET), EMBEDDING_VEC_SIZE)
        else:
            self.E = nn.Embedding(utils.E.shape[0], utils.E.shape[1])
            self.E.weight.data.copy_(torch.from_numpy(utils.E))

        self.input_size = WIN_SIZE * EMBEDDING_VEC_SIZE
        self.fc0 = nn.Linear(input_size, FIRST_HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, len(utils.TAGS_SET) )
        #initialize prefixes and suffixes
        self.prefixes = {word[:PREFIX_SIZE] for word in utils.WORDS_SET}
        self.suffixes = {word[:-SUFFIX_SIZE] for word in utils.WORDS_SET}
        self.prefixes = list(self.prefixes)
        self.suffixes = list(self.suffixes)
        self.prefix_to_index = {suff : i for i, suff in enumerate(self.prefixes)}
        self.suffix_to_index = {suff : i for i, suff in enumerate(self.suffixes)}
        self.E_pref = nn.Embedding(len(self.prefixes), EMBEDDING_VEC_SIZE)
        self.E_suff = nn.Embedding(len(self.suffixes), EMBEDDING_VEC_SIZE)


    def forward(self, x):
        """
        forward pass
        :param x: input
        :return: prediction
        """
        windows_pref = x.data.numpy().copy()
        windows_suff = x.data.numpy().copy()
        windows_pref = windows_pref.reshape(-1)
        windows_suff = windows_suff.reshape(-1)
        # get lists of prefixes/suffixes for the words in the window
        windows_pref = [self.prefixes[self.prefix_to_index[utils.INDEX_TO_WORD[index][:PREFIX_SIZE]]]
                        for index in windows_pref]

        windows_suff = [self.suffixes[self.suffix_to_index[utils.INDEX_TO_WORD[index][:-SUFFIX_SIZE]]]
                        for index in windows_suff]
        # get lists of the indices for the prefixes/suffixes
        windows_pref = [self.prefix_to_index[pref] for pref in windows_pref]
        windows_suff = [self.suffix_to_index[suff] for suff in windows_suff]

        #convert to np array
        windows_pref = np.asanyarray(windows_pref)
        windows_suff = np.asanyarray(windows_suff)
        #reshape
        windows_pref = torch.from_numpy(windows_pref.reshape(x.data.shape)).type(torch.LongTensor)
        windows_suff = torch.from_numpy(windows_suff.reshape(x.data.shape)).type(torch.LongTensor)
        #prefix_vectors, suffix_vectors = prefix_vectors.type(tr.LongTensor), suffix_vectors.type(tr.LongTensor)

        x = (self.E(x) + self.E_pref(windows_pref) + self.E_suff(windows_suff)).view(-1, self.input_size)
        x = F.tanh(self.fc0(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def plotTrainAndValidationGraphs(avg_validation_loss_per_epoch_dict, validation_accuracy_per_epoch_dict):
    """
    plot two graphs:
    1. avg loss per epoch on train set
    2. avg loss per epoch on validation set
    :param avg_train_loss_per_epoch_dict: avg train loss per epoch dictionary
    :param avg_validation_loss_per_epoch_dict: avg validation loss per epoch dictionary
    :return: None
    """
    # line1, = plt.plot(avg_train_loss_per_epoch_dict.keys(), avg_train_loss_per_epoch_dict.values(), "orange",
    #                   label='Train average loss')
    line1, = plt.plot(avg_validation_loss_per_epoch_dict.keys(), avg_validation_loss_per_epoch_dict.values(), "purple",
                      label='Validation average loss')
    # drawing name of the graphs
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()
    plt.savefig('vld_avg_loss.png)
    line2, = plt.plot(validation_accuracy_per_epoch_dict.keys(), validation_accuracy_per_epoch_dict.values(),
                      label='Validation average accuracy')
    # drawing name of the graphs
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})
    plt.show()
    plt.savefig('vld_acc.png')



def make_data_loader_with_tags(file_name, is_dev = False):
    x, y = utils.get_tagged_data(file_name,is_dev)
    # x, y = torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))
    # x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)\
    x, y = np.asarray(x, np.float32), np.asarray(y, np.int32)
    x, y = torch.from_numpy(x) , torch.from_numpy(y)
    x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(x, y)
    if is_dev:
        return torch.utils.data.DataLoader(dataset, 1, shuffle=True)
    return torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

def make_test_data_loader(file_name):
    x = utils.get_not_tagged_data(file_name)
    return x
    # x = torch.from_numpy(np.array(x))
    # x = x.type(torch.LongTensor)
    # return torch.utils.data.TensorDataset(x)
    # return torch.utils.data.DataLoader(dataset, 1, shuffle=False)
is_pre_trained_embeddings_needed = bool(int(sys.argv[2]))
if is_pre_trained_embeddings_needed:
    import utils2 as utils
else:
    import utils1 as utils
def main(argv):
    tagger_type = argv[0]
    # 0 - don't use pre trained word embeddings, 1 - use pre trained word embeddings

    # Create the train_loader
    train_loader = make_data_loader_with_tags(tagger_type + '/train')

    validation_loader = make_data_loader_with_tags(tagger_type +'/dev',is_dev = True)

    test_loader = make_test_data_loader(tagger_type + '/test')
    ## done splitting

    model = NeuralNet(is_pre_trained_embeddings_needed,input_size=INPUT_SIZE)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARN_RATE)

    trainer = ModelTrainer(train_loader, validation_loader, test_loader, model, optimizer)
    trainer.run(tagger_type)

if __name__ == "__main__":
    main(sys.argv[1:])
