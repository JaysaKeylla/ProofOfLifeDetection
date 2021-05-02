# importing laibrary
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# Enable GPU if available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available! Training on GPU')
torch.cuda.is_available()
# Data Augmentation
train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomRotation(20),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
# Data Preprocessing
train_data = datasets.ImageFolder('dataset/train/', transform=train_transforms)
valid_data = datasets.ImageFolder('dataset/valid/', transform=valid_transforms)
test_data = datasets.ImageFolder('dataset/test/', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=128, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=128)
testloader = torch.utils.data.DataLoader(test_data, batch_size=128)
print(valid_data.class_to_idx)
x, y = next(iter(trainloader))
x.shape
model = models.densenet121(pretrained=True)
model
y.shape
# importing the model densenet121
# Use GPU if it is availabel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.densenet121(pretrained=True)

# Freeze the parameters except last fully connected layer so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Now unfreeze the fully connected layer
for param in list(model.parameters())[-15:]:
    param.requires_grad = True


# Add custom layer at end of the model
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(512, 8))
# Defining the loss function
criterion = nn.CrossEntropyLoss()
# Defining the optimizer
optimizer = optim.Adam(model.classifier.parameters(),
                       lr=0.001, weight_decay=0.0001)
if train_on_gpu:
    model.cuda()
# Frist time training :
# Learning rate is set 0.001 , batch size = 128 and run for 22 epochs, weight_decay = 0.0001 , the best result is :
# Epoch :9/100 Training Loss : 0.028698 Validation Loss : 0.072107 Validation Accuracy : 0.974649
# Test Loss : 0.135589 Test Accuracy : 0.950160
# Defining the training process
epochs = 100
valid_loss_min = np.Inf
train_his, valid_his = [], []
model.train()
for epoch in range(epochs):
    training_loss = 0
    valid_loss = 0
    for inputs, labels in trainloader:
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()

        optimizer.step()
        training_loss += loss.item()

    valid_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            torch.cuda.empty_cache()
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            valid_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    training_loss = training_loss / len(trainloader)
    valid_loss = valid_loss / len(validloader)
    valid_accuracy = valid_accuracy / len(validloader)
    train_his.append(training_loss)
    valid_his.append(valid_loss)

    print('Epoch :{}/{} \tTraining Loss : {:.6f} \tValidation Loss : {:.6f} '
          '\tValidation Accuracy : {:.6f}'.format(epoch+1, epochs, training_loss, valid_loss, valid_accuracy))

    # Save the model if the validation loss is decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ............'.format(
            valid_loss_min, valid_loss))

        torch.save(model, 'liveliness_detection_2.pt')
        valid_loss_min = valid_loss
plt.plot(train_his, label='Training loss')
plt.plot(valid_his, label='Validation loss')
plt.title('Training Vs Validation No. 01')
plt.legend(frameon=False)
plt.savefig('Train_vs_Valid_1.png')
test_his = []
test_accuracy = 0
test_loss = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the loss function
criterion = nn.CrossEntropyLoss()
# Defining the optimizer
#optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, weight_decay = 0.0001)
if train_on_gpu:
    model.cuda()


model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        test_loss += loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_loss = test_loss / len(testloader)
    test_accuracy = test_accuracy / len(testloader)
    test_his.append(test_loss)


print('Test Loss : {:.6f} \tTest Accuracy : {:.6f} '.format(
    test_loss, test_accuracy))

# Training for 2nd time
# 2nd time training with 0.001 learning rate

model = torch.load('liveliness_detection_2.pt')
model

# List of training after 1st time :
# Use GPU if it is availabel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Freeze the parameters except last fully connected layer so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Now unfreeze the fully connected layer
for param in list(model.parameters())[-15:]:
    param.requires_grad = True


# Defining the loss function
criterion = nn.CrossEntropyLoss()
# Defining the optimizer
optimizer = optim.Adam(model.classifier.parameters(),
                       lr=0.0000001, weight_decay=0.001)
if train_on_gpu:
    model.cuda()

# Defining the training process
epochs = 10
valid_loss_min = 0.072107
train_his, valid_his = [], []
model.train()
for epoch in range(epochs):
    training_loss = 0
    valid_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()

        optimizer.step()
        training_loss += loss.item()

    valid_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            valid_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    training_loss = training_loss / len(trainloader)
    valid_loss = valid_loss / len(validloader)
    valid_accuracy = valid_accuracy / len(validloader)
    train_his.append(training_loss)
    valid_his.append(valid_loss)

    print('Epoch :{}/{} \tTraining Loss : {:.6f} \tValidation Loss : {:.6f} '
          '\tValidation Accuracy : {:.6f}'.format(epoch+1, epochs, training_loss, valid_loss, valid_accuracy))

    # Save the model if the validation loss is decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ............'.format(
            valid_loss_min, valid_loss))

        torch.save(model, 'liveliness_detection_2.pt')
        valid_loss_min = valid_loss
for i in range(len())
# Plot the train and validation loss
plt.plot(train_his, label='Training loss2')
plt.plot(valid_his, label='Validation loss2')
plt.title('Training Vs Validation No. 02')
plt.legend(frameon=False)
plt.savefig('liveliness_train_vs_valid_2.png')
# Test the 2nd time pretrained model
test_accuracy = 0
test_loss = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        test_loss += loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_loss = test_loss / len(testloader)
    test_accuracy = test_accuracy / len(testloader)


print('Test Loss : {:.6f} \tTest Accuracy : {:.6f} '.format(
    test_loss, test_accuracy))

type(testloader.shape)

model = torch.load('liveliness_detection_3.pt')
model

# Defining the training process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
# Defining the optimizer
optimizer = optim.Adam(model.classifier.parameters(),
                       lr=0.0001, weight_decay=0.0001)
if train_on_gpu:
    model.cuda()

epochs = 20
valid_loss_min = 0.074786
train_his, valid_his = [], []
model.train()
for epoch in range(epochs):
    training_loss = 0
    valid_loss = 0
    for inputs, labels in trainloader:
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()

        optimizer.step()
        training_loss += loss.item()

    valid_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            torch.cuda.empty_cache()
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            valid_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    training_loss = training_loss / len(trainloader)
    valid_loss = valid_loss / len(validloader)
    valid_accuracy = valid_accuracy / len(validloader)
    train_his.append(training_loss)
    valid_his.append(valid_loss)

    print('Epoch :{}/{} \tTraining Loss : {:.6f} \tValidation Loss : {:.6f} '
          '\tValidation Accuracy : {:.6f}'.format(epoch+1, epochs, training_loss, valid_loss, valid_accuracy))

    # Save the model if the validation loss is decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ............'.format(
            valid_loss_min, valid_loss))

        torch.save(model, 'liveliness_detection_3.pt')
        valid_loss_min = valid_loss

# Plot the train and validation loss
plt.plot(train_his, label='Training loss2')
plt.plot(valid_his, label='Validation loss2')
plt.title('Training Vs Validation No. 02')
plt.legend(frameon=False)
plt.savefig('liveliness_train_vs_valid_3.png')

nb_classes = 5

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(testloader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

print(confusion_matrix.diag()/confusion_matrix.sum(1))
