from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import PIL
from PIL import Image
import matplotlib.pyplot as plt




use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


imsize = 500 # desired size of output image


loader = transforms.Compose([
        transforms.Scale(imsize), # scale imported image to specified size
        transforms.ToTensor()])   # transform loaded image into a torch tensor



def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image



style = image_loader("images/pic/texture5.jpg").type(dtype)
content = image_loader("images/pic/content5.jpg").type(dtype)

assert style.size() == content.size(), "Need style and content images to be of the same size."

unloader = transforms.ToPILImage() # reconvert into PIL image


def imshow(tensor):
    image = tensor.clone().cpu() # we clone the tensor as as not to make changes to it
    image = image.view(3, imsize, imsize) # remove fake batch dimension
    image = unloader(image)
    plt.imshow(image)

%matplotlib inline
fig = plt.figure()
plt.subplot(221)
imshow(style.data)
plt.subplot(222)
imshow(content.data)




class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we "detach" the target content from the tree used
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()


    def forward(self, input):
        self.loss = self.criterion.forward(input * self.weight, self.target)
        self.output = input
        return self.output


    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss





class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a = batch size (1), b = number of feature maps, (c, d) = dimensions of feature map


        # define \hat{F} as the reshaped version of F into a K x N matrix, where K = number of feature maps, N = c x d
        features = input.view(a * b, c * d)


        G = torch.mm(features, features.t())  # compute the Gram matrix for the given layer

        # we normalize the values of the Gram matrix by dividing by the total number of elements in feature map stack
        return G.div(a * b * c * d)




class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()


    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram.forward(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion.forward(self.G, self.target)
        return self.output


    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss




# load pretrained VGG model
cnn = models.vgg19(pretrained=True).features

# move it to GPU if possible
if use_cuda:
    cnn = cnn.cuda()



# define which layers to use for content extraction, and which layers to use for style transfer
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# for convenience
content_losses = []
style_losses = []



model = nn.Sequential() # the new network with GramMatrix, ContentLoss and StyleLoss modules included
gram = GramMatrix() # we need a Gram module in order to compute the style targets

# move these modules to the GPU if possible
if use_cuda:
    model = model.cuda()
    gram = gram.cuda()



# define weights associated with style and content losses
content_weight = 1
style_weight = 1000



i = 1

for layer in list(cnn):
    if isinstance(layer, nn.Conv2d):
        name = "conv_" + str(i)
        model.add_module(name, layer)



        if name in content_layers:
            # add content loss:
            target = model.forward(content).clone()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)


        if name in style_layers:
            # add style loss:
            target_feature = model.forward(style).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)



    if isinstance(layer, nn.ReLU):
        name = "relu_" + str(i)
        model.add_module(name, layer)



        if name in content_layers:
            # add content loss:
            target = model.forward(content).clone()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)


        if name in style_layers:
            # add style loss:
            target_feature = model.forward(style).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)

        i += 1


    if isinstance(layer, nn.MaxPool2d):
        name = "pool_" + str(i)
        model.add_module(name, layer)



input = image_loader("images/pic/content5.jpg").type(dtype)

# choose if we want to start with white noise or with content image directly
input.data = content.data # torch.randn(input.data.size()).type(dtype)


# add original input image to plot
plt.subplot(223)
imshow(input.data)


# tell pytorch that the input image is a parameter that requires a gradient update
input = nn.Parameter(input.data)
optimizer = optim.LBFGS([input])


run = [0]

while run[0] <= 400:
    
    def closure():
        optimizer.zero_grad()
        model.forward(input)
        style_score = 0
        content_score = 0
        

        for sl in style_losses:
            style_score += sl.backward()
        for cl in content_losses:
            content_score += cl.backward()
            

        run[0] += 1
        

        if run[0] % 10 == 0:
            print("Run " + str(run) + ":")
            print(style_score.data[0])
            print(content_score.data[0])
            

        return content_score + style_score
    
    optimizer.step(closure)


input.data.clamp_(0, 1)

plt.subplot(224)
imshow(input.data)
plt.show()

