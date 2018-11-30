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
import utils1 as ut1
import sys

# Hyper-parameters
BATCH_SIZE = 1024
LEARN_RATE = 0.01
EPOCHS = 3
FIRST_HIDDEN_LAYER_SIZE = 100
SECOND_HIDDEN_LAYER_SIZE = 50
NUMBER_OF_CLASSES = 10
EMBEDDING_VEC_SIZE = 50
WIN_SIZE = 5
INPUT_SIZE = 250

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
        print("Epoch: {} Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)".format(epoch, train_loss, correct, len(self.train_loader) * BATCH_SIZE,
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
                if ut1.INDEX_TO_TAG[pred.cpu().sum().item()] != 'O' or ut1.INDEX_TO_TAG[target.cpu().sum().item()] != 'O':
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
        pred_list = []
        for data in self.test_loader:
            output = self.model(torch.LongTensor(data))
            # get the predicted class out of output tensor
            pred = output.data.max(1, keepdim=True)[1]
            # add current prediction to predictions list
            pred_list.append(pred.item())

        pred_list = self.convert_tags_indices_to_tags(pred_list)
        self.write_test_results_file(tagger_type + "/test", "test1." + tagger_type, pred_list)

    def convert_tags_indices_to_tags(self, tags_indices_list):
        """
        Converts list of tags indices to tags list (string representation).
        :param tags_indices_list: tags indices list
        :return: tags list (string representation)
        """
        return [ut1.INDEX_TO_TAG[index] for index in tags_indices_list]

    def write_test_results_file(self, test_file_name, output_file_name, predictions_list):
        """
        writes test predictions to output file.
        :param test_file_name: test file name
        :param output_file_name: output file name
        :param predictions_list: predictions for every word in the test data.
        """
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
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()

        self.E = nn.Embedding(len(ut1.WORDS_SET), EMBEDDING_VEC_SIZE)  # Embedding matrix
        self.input_size = WIN_SIZE * EMBEDDING_VEC_SIZE
        self.fc0 = nn.Linear(input_size, FIRST_HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_SIZE, len(ut1.TAGS_SET))
        # self.bn1 = nn.BatchNorm1d(FIRST_HIDDEN_LAYER_SIZE)
        # self.bn2 = nn.BatchNorm1d(SECOND_HIDDEN_LAYER_SIZE)

    def forward(self, x):
        """
        forward pass
        :param x: input
        :return: prediction
        """
        x = self.E(x).view(-1, self.input_size)
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
    line2, = plt.plot(validation_accuracy_per_epoch_dict.keys(), validation_accuracy_per_epoch_dict.values(),
                      label='Validation average accuracy')
    # drawing name of the graphs
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})
    plt.show()



def make_data_loader_with_tags(file_name, is_dev = False):
    """
    make_data_loader_with_tags functions.
    make data loader for dev or train.
    :param file_name: dev or train file name
    :param is_dev: boolean indicates if the data is for validation.
    :return: new data loader.
    """
    x, y = ut1.get_tagged_data(file_name,is_dev)
    x, y = np.asarray(x, np.float32), np.asarray(y, np.int32)
    x, y = torch.from_numpy(x) , torch.from_numpy(y)
    x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(x, y)
    if is_dev:
        return torch.utils.data.DataLoader(dataset, 1, shuffle=True)
    return torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

def make_test_data_loader(file_name):
    """
    make_test_data_loader function.
    make data loader for test.
    :param file_name: test file name.
    :return: new data loader.
    """
    x = ut1.get_not_tagged_data(file_name)
    return x

def main(argv):
    """
    main function.
    runs the program.
    :param argv: args[0] indicates if its ner or pos
    :return:
    """
    global LEARN_RATE
    tagger_type = argv[0]
    if tagger_type == 'ner':
        LEARN_RATE = 0.05
    # Create the train_loader
    train_loader = make_data_loader_with_tags(tagger_type + '/train')

    validation_loader = make_data_loader_with_tags(tagger_type +'/dev',is_dev = True)

    test_loader = make_test_data_loader(tagger_type + '/test')
    # done splitting
    model = NeuralNet(input_size=INPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    trainer = ModelTrainer(train_loader, validation_loader, test_loader, model, optimizer)
    trainer.run(tagger_type)

if __name__ == "__main__":
    main(sys.argv[1:])