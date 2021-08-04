import torch
import torch.nn as nn
import torch.optim as torch_optimizer
from torchvision import datasets, models, transforms
import time
import copy
import gin


@gin.configurable
class TorchTest:

    def __init__(self, num_classes=90, feature_extract=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Setup the loss function, here we use cross entropy loss
        self.criterion = nn.CrossEntropyLoss()
        # Init model object
        self.model_ft = None
        self.optimizer_ft = None
        """
        Input size will be confirmed after model generation.
        It should be 224 for "VGG16" and "ResNet50" but 229 for "InceptionV3"
        """
        self.input_size = 0
        """
        If we are doing fine-tuning or want to train the model from scratch
        then feature_extract should be false. However, if we are using feature
        extract method, it should be True.
        """
        self.feature_extract = feature_extract
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_inception = False

    def initialize_model(self, model_name, use_pretrained=True):

        if model_name == "resnet50":
            """ Resnet50
            """
            self.model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif model_name == "vgg16":
            """ VGG16
            """
            self.model_ft = models.vgg16(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif model_name == "inceptionv3":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad()
            # Handle the auxilary net
            num_ftrs = self.model_ft.AuxLogits.fc.in_features
            self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 299
            self.is_inception = True

        else:
            print("Invalid model name, exiting...")
            exit()

    def set_optimizer(self, optimizer="SGD", learning_rate=0.001, momentum=0.9):
        self.model_ft = self.model_ft.to(self.device)
        # Gather the parameters to be optimized/updated in this run. If we are
        # fine tuning we will be updating all parameters.
        # However, if we are doing feature extract method, we will only update the parameters
        # that we have just initialized, i.e. the parameters with requires_grad is True.
        params_to_update = self.model_ft.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        # Observe that all parameters are being optimized
        if optimizer == "SGD":
            self.optimizer_ft = torch_optimizer.SGD(params_to_update, learning_rate, momentum)
        elif optimizer == "Adam":
            self.optimizer_ft = torch_optimizer.Adam(params_to_update, learning_rate)
        else:
            print("For the time being, only SGD and Adam optimizers are supported.")
            exit()

    def torch_train_model(self, dataloaders, num_epochs=10):
        """
        The function trains for the specified number of epochs and after each epoch runs a full validation step.
        :param dataloaders:     dictionary of dataloaders
        :param num_epochs:      number of epochs
        :return: None
        """

        # Training
        since = time.time()
        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model_ft.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model_ft.train()  # Set model to training mode
                else:
                    self.model_ft.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer_ft.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if self.is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model_ft(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model_ft(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer_ft.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model_ft.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model_ft.load_state_dict(best_model_wts)
        return val_acc_history

    def set_parameter_requires_grad(self):
        """
        This helper function sets the <tt>.requires_grad</tt> attribute of the parameters in the model
        to False when we are feature extracting.
        """
        if self.feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
