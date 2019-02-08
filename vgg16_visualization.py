import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class FeatureVisualization():
    def __init__(self,img_path,selected_layers):
        self.img_path = img_path
        self.selected_layers = selected_layers
        self.pretrained_model = models.vgg16(pretrained=True).features
        self.pretrained_kenel = models.vgg16(pretrained=True)

    def process_image(self):
        img=cv2.imread(self.img_path)
        img=preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input = self.process_image()
#        print(input.shape)
        index_layer = 0

        x = input
        # print the feature maps of selected layers
        for index,layer in enumerate(self.pretrained_model):
            if index > selected_layers[index_layer]:
                break
            x,y = layer(x), layer(x)

            if (index == selected_layers[index_layer]):
                y = self.get_single_feature(y)

                self.save_feature_to_img(y,index_layer)

                if index_layer < len(selected_layers) - 1:
                    index_layer += 1

    def get_single_feature(self, input):
#        features = self.get_feature(input)
#        print(features.shape)

        feature = input[:,10,:,:]
#        print(feature.shape)

        feature = feature.view(feature.shape[1],feature.shape[2])
#        print(feature.shape)

        return feature

    def save_feature_to_img(self,feature,selected_layer):
        #to numpy
        feature = feature.data.numpy()

        #use sigmod to [0,1]
        feature = 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature = np.round(feature*255)
#        print(feature[0])

        plt.subplot(121)
        plt.title('feature maps of conv' + str(selected_layer + 1))
        plt.imshow(feature, cmap=plt.get_cmap('gray'))
#        plt.imshow(feature)


        cv2.imwrite('./feature_maps of conv' + str(selected_layer + 1) + '.png',feature)

        plt.subplot(122)
        arr = feature.flatten()
        plt.title('histogram image of conv' + str(selected_layer + 1))
        plt.hist(arr, bins=224, normed=1, edgecolor='None', facecolor='red')

        plt.savefig('feature map and histogram image of conv' + str(selected_layer + 1)+'.png')

        plt.show()

    def kernel_calling(self):
        body_model = [i for i in self.pretrained_kenel.children()][0]

        for index, selected_layer in enumerate(self.selected_layers):
            kernel = body_model[selected_layer]
            tensor = kernel.weight.data.numpy()
            self.plot_kernels(tensor, index)
            self.plot_kernels_hist(tensor,index)

    def plot_kernels(self, tensor, selected_layer, num_cols=6):
        if not tensor.ndim == 4:
            raise Exception("assumes a 4D tensor")
        if not tensor.shape[-1] == 3:
            raise Exception("last dim needs to be 3 to plot")

        if selected_layer == 0:
            num_kernels = tensor.shape[0]
            num_rows = 1 + num_kernels // num_cols
            fig = plt.figure(figsize=(num_cols, num_rows))

            for i in range(tensor.shape[0]):
                ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
                ax1.imshow(tensor[i], cmap=plt.get_cmap('gray'))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.title('kernel of conv' + str(selected_layer + 1))
            plt.savefig('kernel of conv' + str(selected_layer + 1) + '.png')
            plt.show()

        else:
            num_kernels = tensor.shape[1]
            num_rows = 1 + num_kernels // num_cols
            fig = plt.figure(figsize=(num_cols, num_rows))

            for i in range(tensor.shape[1]):
                ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
                #            tensor = tensor.view(tensor.shape[1],tensor.shape[1],tensor.shape[2])
                ax1.imshow(tensor[0][i], cmap=plt.get_cmap('gray'))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.title('kernel of conv' + str(selected_layer + 1))
            plt.savefig('kernel of conv' + str(selected_layer + 1) + '.png')
            plt.show()

    def plot_kernels_hist(self, tensor, selected_layer, num_cols=6):
        if not tensor.ndim == 4:
            raise Exception("assumes a 4D tensor")
        if not tensor.shape[-1] == 3:
            raise Exception("last dim needs to be 3 to plot")

        if selected_layer == 0:
            num_kernels = tensor.shape[0]
            num_rows = 1 + num_kernels // num_cols
            fig = plt.figure(figsize=(num_cols, num_rows))

            for i in range(tensor.shape[0]):
                ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
                ax1.hist(tensor[i][0].flatten(), bins=3, normed=1, edgecolor='None', facecolor='red')
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.title('histogram image of kernel in conv' + str(selected_layer + 1))
            plt.savefig('histogram image of kernel in conv' + str(selected_layer + 1) + '.png')
            plt.show()

        else:
            num_kernels = tensor.shape[1]
            num_rows = 1 + num_kernels // num_cols
            fig = plt.figure(figsize=(num_cols, num_rows))

            for i in range(tensor.shape[1]):
                ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
                ax1.hist(tensor[0][i].flatten(), bins=3, normed=1, edgecolor='None', facecolor='red')
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.title('histogram image of kernel in conv' + str(selected_layer + 1))
            plt.savefig('histogram image of kernel in conv' + str(selected_layer + 1) + '.png')
            plt.show()


if __name__=='__main__':
    # get class
    selected_layers = [0,2,5,7,10]
    image_path = './input_images/snake.jpg'
    feature_vis = FeatureVisualization(image_path,selected_layers)
    print (feature_vis.pretrained_model)

    feature_vis.get_feature()
    feature_vis.kernel_calling()

