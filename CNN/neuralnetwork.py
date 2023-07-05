import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd
from sklearn.metrics import confusion_matrix

# Release all unoccupied cached memory currently held by the caching allocator
torch.cuda.empty_cache()
# File path is dynamic to use for other datasets
# Some variables are dynamic to change fastly and use for other datasets
file_path = "Vegetable Images"
resize_shape = 256
input_shape = 224
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
train_per, test_per, valid_per = 15, 3, 3
batch_size = 32
EPOCH_COUNT = 10
# Transform.Compose is used to chain several transforms images together
# We can resize, normalize and convert our images to tensor with Compose
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'validation': transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'all': transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# We can take our image data from our file path and apply the  transform process to them
# And obtain our own train validation and test data through ImageFolder.
train_data = ImageFolder(root=file_path, transform=data_transforms["train"])
validation_data = ImageFolder(root=file_path, transform=data_transforms["validation"])
test_data = ImageFolder(root=file_path, transform=data_transforms["test"])
all_data = ImageFolder(root=file_path, transform=data_transforms["all"])
train_size = int(len(train_data) * train_per / (train_per + test_per + valid_per))
val_size = int(len(validation_data) * valid_per / (train_per + test_per + valid_per))
test_size = int(len(test_data) * test_per / (train_per + test_per + valid_per))
train_data, val_data, test_data = random_split(all_data, [train_size, val_size, test_size])

# We can get data loader with the data we created above
# And the batch size we set with shuffle feature, through DataLoader
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
# Determine the device with torch.device if cuda is available we use cuda else we use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, optimizer, criterion, epoch_count):
    # We created train and validation lists to make loss and accuracy plots.
    train_losses, train_epochs, train_acc = [], [], []
    val_losses, val_epochs, val_acc = [], [], []
    for epoch in range(epoch_count):
        # number of total and correct variables to calculate accuracy
        n_correct = 0
        n_total = 0
        for i, (words, labels) in enumerate(train_data_loader):
            # Get the inputs
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            with torch.set_grad_enabled(True):
                outputs = model(words)
                loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Feedforward tutorial solution
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()
            n_total += labels.shape[0]
        # Calculate accuracy
        accuracy = 100 * n_correct / n_total
        print("Epoch", epoch + 1, "/", epoch_count)
        print("Train Loss: ", loss.item())
        print("Train Accuracy: ", accuracy)
        # Append the values to make plot
        train_losses.append(loss.item())
        train_epochs.append(epoch + 1)
        train_acc.append(accuracy)
        # We use no_grad to do not calculate gradients
        with torch.no_grad():
            n_correct = 0
            n_total = 0
            for i, (words, labels) in enumerate(validation_data_loader):
                # Get the inputs
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)

                with torch.set_grad_enabled(True):
                    outputs = model(words)
                    loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Feedforward tutorial solution
                _, predicted = torch.max(outputs, 1)
                n_correct += (predicted == labels).sum().item()
                n_total += labels.shape[0]
            # Calculate accuracy
            accuracy = 100 * n_correct / n_total
            print("Validation Loss: ", loss.item())
            print("Validation Accuracy: ", accuracy)
        # Append the values to make plot
        val_losses.append(loss.item())
        val_epochs.append(epoch)
        val_acc.append(accuracy)
    print('Training Done')
    train_loss = np.array(train_losses)
    valid_loss = np.array(val_losses)
    train_acc = np.array(train_acc)
    valid_acc = np.array(val_acc)
    epoch = np.array(train_epochs)
    # Make plot of train and valid accuracy
    # Set all features for plot
    plt.plot(epoch, train_acc)
    plt.plot(epoch, valid_acc)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Train Accuracy", "Valid Accuracy"], loc="lower right")
    plt.show()
    # Make plot of train and valid loss
    # Set all features for plot
    plt.plot(epoch, train_loss)
    plt.plot(epoch, valid_loss)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train Loss", "Valid Loss"], loc="lower right")
    plt.show()
    return model


model = torchvision.models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Modify last layer
number_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]
features.extend([torch.nn.Linear(number_features, EPOCH_COUNT)])
model.classifier = torch.nn.Sequential(*features)
model = model.to(device)
# As the loss function we used CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss()
# As the optimizer we used SGD
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Apply training
train_model(model, optimizer, criterion, EPOCH_COUNT)
# Creating lists to make predict and find precision recall and f1 score
predicted_labels, ground_truth_labels = [], []
model.eval()
for i, (words, labels) in enumerate(test_data_loader):
    words = words.to(device)
    labels = labels.to(dtype=torch.long).to(device)

    # Forward pass
    outputs = model(words)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # feedforward tutorial solution
    _, predicted = torch.max(outputs, 1)
    predicted_labels.append(predicted.cpu().detach().numpy())
    ground_truth_labels.append(labels.cpu().detach().numpy())
pred_list = np.concatenate(predicted_labels).ravel().tolist()
truth_list = np.concatenate(ground_truth_labels).ravel().tolist()

# cm = confusion_matrix(truth_list, pred_list)
# df_cm = pd.DataFrame(cm, index=[i for i in "ABC"], columns=[i for i in "ABC"])
# plt.figure(figsize=(10, 7))
# plt.title("Confusion Matrix")
# seaborn.heatmap(df_cm, annot=True)
# plt.show()

# Find the precision score with precision_score, recall score with recall_score, f1score score with f1_score
# We used zero division to prevent any errors and warnings and specify "weighted" the average
prec = precision_score(truth_list, pred_list, zero_division=1, average="weighted")
recall = recall_score(truth_list, pred_list, zero_division=1, average="weighted")
fscore = f1_score(truth_list, pred_list, zero_division=1, average="weighted")
print("Average Precision Value: ", prec)
print("Average Recall Value: ", recall)
print("Average F1 Score Value: ", fscore)

######### TWO LAST FULLY CONNECTED LAYERS #########
print("####### Fine tune the weights of only two last fully connected (FC1 and FC2) layers #######")

# Modify last layer
model_2 = torchvision.models.vgg19(pretrained=True)
number_features = model_2.classifier[6].in_features
features = list(model_2.classifier.children())[:-1]
features.extend([torch.nn.Linear(number_features, 10)])
model_2.classifier = torch.nn.Sequential(*features)
model_2 = model_2.to(device)
# Freeze all the layers
for param in model_2.parameters():
    param.requires_grad = False
# Just train last 2 fully connected layer
for layer in model_2.classifier[3:]:
    for param in layer.parameters():
        param.requires_grad = True
# As the loss function we used CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss()
# As the optimizer we used SGD
optimizer = optim.SGD(model_2.parameters(), lr=0.001, momentum=0.9)
# Apply training
train_model(model_2, optimizer, criterion, EPOCH_COUNT)
model_2.eval()
# Creating lists to make predict and find precision recall and f1 score
predicted_labels, ground_truth_labels = [], []
for i, (words, labels) in enumerate(test_data_loader):
    words = words.to(device)
    labels = labels.to(dtype=torch.long).to(device)

    # Forward pass
    outputs = model_2(words)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # feedforward tutorial solution
    _, predicted = torch.max(outputs, 1)
    predicted_labels.append(predicted.cpu().detach().numpy())
    ground_truth_labels.append(labels.cpu().detach().numpy())
pred_list = np.concatenate(predicted_labels).ravel().tolist()
truth_list = np.concatenate(ground_truth_labels).ravel().tolist()

# cm = confusion_matrix(truth_list, pred_list)
# df_cm = pd.DataFrame(cm, index=[i for i in "ABC"], columns=[i for i in "ABC"])
# plt.figure(figsize=(10, 7))
# plt.title("Confusion Matrix")
# seaborn.heatmap(df_cm, annot=True)
# plt.show()

# Find the precision score with precision_score, recall score with recall_score, f1score score with f1_score
# We used zero division to prevent any errors and warnings and specify "weighted" the average
prec = precision_score(truth_list, pred_list, zero_division=1, average="weighted")
recall = recall_score(truth_list, pred_list, zero_division=1, average="weighted")
fscore = f1_score(truth_list, pred_list, zero_division=1, average="weighted")
print("Average Precision Value: ", prec)
print("Average Recall Value: ", recall)
print("Average F1 Score Value: ", fscore)

################ VISUALIZATION ################
print("Visualization")


def visualize_weights(layer):
    weight = []

    # All convolution layers and append their corresponding filters in a list
    for w in model.features.children():
        if isinstance(w, torch.nn.modules.conv.Conv2d):
            weight.append(w.cpu().weight.data)

    print("Size of Filter:", weight[layer].shape)
    print("Number of Filters:", weight[layer].shape[0])
    filter = []
    for i in range(weight[layer].shape[0]):
        filter.append(weight[layer][i, :, :, :].sum(dim=0))
        filter[i].div(weight[layer].shape[1])
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.title("Visualization of Weights")
    plt.axis('off')
    for i in range(int(weight[layer].shape[0])):
        fig.add_subplot(int(np.sqrt(weight[layer].shape[0])), int(np.sqrt(weight[layer].shape[0])), i + 1)
        plt.imshow(filter[i])
        plt.axis('off')
    plt.show()


def layer_images(image):
    outputs = []
    names = []

    # Feed forward the image through the network and store the outputs
    for layer in modules:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    # Convert the output into a 2D image by averaging across the filters.
    output_im = []
    for temp in outputs:
        temp = temp.squeeze(0)
        temp = torch.sum(temp, dim=0)
        temp = torch.div(temp, temp.shape[0])
        output_im.append(temp.data.numpy())

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (25, 30)
    plt.title("Layers of the Image")
    plt.axis('off')
    for i in range(len(output_im)):
        a = fig.add_subplot(8, 5, i + 1)
        plt.imshow(output_im[i])
        plt.axis('off')
        a.set_title(str(i + 1) + ". " + names[i].partition('(')[0], fontsize=8)
    plt.show()


def filter_image(image, layer_to_visualize, num_filters=64):
    output = None
    name = None
    # Take outputs corresponding to the layer
    for count, layer in enumerate(modules):
        image = layer(image)
        if count == layer_to_visualize:
            output = image
            name = str(layer)

    filters = []
    output = output.data.squeeze()

    # Visualize all the filters, if num_filters = -1
    num_filters = min(num_filters, output.shape[0])
    if num_filters == -1:
        num_filters = output.shape[0]

    for i in range(num_filters):
        filters.append(output[i, :, :])

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.title("Applying " + str(num_filters) + " different filters to the " + str(layer_to_visualize + 1) + ". layer")
    plt.axis('off')
    for i in range(int(np.sqrt(len(filters))) * int(np.sqrt(len(filters)))):
        fig.add_subplot(int(np.sqrt(len(filters))), int(np.sqrt(len(filters))), i + 1)
        plt.imshow(filters[i])
        plt.axis('off')
    plt.show()


img_raw = Image.open("Vegetable Images/train/Bean/0029.jpg")
plt.imshow(img_raw)
plt.title("Original Image")
plt.axis("off")
plt.show()
# Transform.Compose is used to chain several transforms images together
# We can resize, normalize and convert our images to tensor with Compose
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = np.array(img_raw)
img = transform(img)
img = img.unsqueeze(0)
modules = list(model.features.modules())
modules = modules[1:]
visualize_weights(1)
layer_images(img)
# Applying 16 filters to [1,8] layers
filter_image(img, 0, 16)
filter_image(img, 1, 16)
filter_image(img, 2, 16)
filter_image(img, 3, 16)
filter_image(img, 4, 16)
filter_image(img, 5, 16)
filter_image(img, 6, 16)
filter_image(img, 7, 16)

print("Process Completed Successfully")
