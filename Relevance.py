import numpy
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import cv2
import sys
import numpy as np
import torch
import copy

#relevance based cam - based on https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Relevance-CAM_Your_Model_Already_Knows_Where_To_Look_CVPR_2021_paper.pdf
#official implementation - https://github.com/mongeoroo/Relevance-CAM
class Relevance_CAM:
    def __init__(self,model,class_dict,device="cpu"):
        """
        model: torch model 
        """
        self.imgclasses = class_dict
        self.device = device
        self.model = model
        self.eps = 1e-9
    
    def convert_to_conv(self,layers:list):
        """
        This function transforms the linear layers like classifier layer to convolutional layers
        [Input]:
        layers : List of all the layers in the network
        
        [Returns]:
        output_layer (List) : List of layers with Linear layers converted to Conv2D 
        
        """
        output_layers = []
        for l in layers:
            if isinstance(l,nn.Sequential):
                for i,layer in enumerate(l):
                    if isinstance(layer,nn.Linear):
                        m,n = layer.weight.shape[1],layer.weight.shape[0]
                        print(m,n)
                        substitute_layer = nn.Conv2d(m,n,1)
                        substitute_layer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))
                        substitute_layer.bias = nn.Parameter(layer.bias)
                        output_layers += [substitute_layer]
                    else:
                        output_layers += [layer]
            elif isinstance(l,nn.Linear):
                layer = l
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                substitute_layer = nn.Conv2d(m,n,1)
                substitute_layer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))
                substitute_layer.bias = nn.Parameter(layer.bias)
                output_layers += [substitute_layer]
            else:
                output_layers += [layer]
                
        return output_layers

    def get_activations(self,input:torch.Tensor,layers:list):
        """
        This function returns output activations at each layer as a list
        [Input]:
        layers [list] : List of all the layers in the network
        
        [Returns] :
        activations [list]: Activation outputs for each layer
        
        """
        Number_of_layers = len(layers)
        activations = [input] + [None]*Number_of_layers
        for i in range(Number_of_layers): 
            activations[i+1]=layers[i].forward(activations[i])
        return activations

    def get_prediction(self,act:list):
        """
        Compute predicted class
        
        [input]:
        act [list] : list of  activation outputs for each layer
        
        [Returns]:
        scores [tensor]: Prediction scores wrt class
        
        """
        scores = (torch.softmax(act[-1].data.view(-1),dim=0))
        ind = torch.argsort(scores.cpu()).numpy()[::-1]
#         for i in ind:
#             print('%20s (%3d): %6.8f'%(self.imgclasses[i],i,scores[i]))
        return scores

    def construct_target(self,act:list,target_index:int):
        
        """
        computes the target layer relevances
        [input]
        act [list]: list of activations
        target_index [int] : Class index
        
        [Returns]:
        relevances_last_layer: relevance vector for the last layer
        
        """
        Target_vector = act[-1].detach()
        z_val = Target_vector[0,target_index]
        Target_vector = -Target_vector/(len(self.imgclasses)-1)
        Target_vector[0,target_index] = z_val
        relevances_last_layer = [(Target_vector).data]
        
        return relevances_last_layer
    
    def clone_layer(self,layer:int):
        
        """
        This function clones a given layer, copies it's weights
        
        [Input]:
        layer[torch.nn] : Layer to be cloned
        
        [Returns]:
        layer_new[torch.nn]: cloned layer
        
        """
        layer_new = copy.deepcopy(layer)
        try: layer_new.weight = nn.Parameter(layer_new.weight)
        except AttributeError: pass
        try: layer_new.bias   = nn.Parameter(layer_new.bias)
        except AttributeError: pass
        return layer_new
    
    def relevance_propagation(self,act:list,relevance:list,layers:list):
        """
        Performs relevance propagation
        
        [Input]:
        act[list] : list of activation outputs from all the layers
        relevance[list] : list containing the relevance scores of the last layer  
        layers[list]:List of all the layers in the network
        
        [Returns]:
        relevances_per_layer[list] : relevance scores for each layer
        
        """
        
        number_of_layers = len(layers)
        
        relevances_per_layer = [None]*number_of_layers + relevance
        
        #propagating layers
        
        for i in np.arange(number_of_layers-1,0,-1):
            #setting req_grad = True
            
            act[i] = (act[i].data).requires_grad_(True)
            
            if isinstance(layers[i],torch.nn.MaxPool2d): layers[i] = torch.nn.AvgPool2d(2,2)
            if isinstance(layers[i],torch.nn.ReLU): layers[i] = torch.nn.ReLU(inplace=False)
                
            #skipping dropout and Relu - from relevance propagation
            if (not isinstance(layers[i],torch.nn.Dropout)) or  (not isinstance(layers[i],torch.nn.ReLU)):
                
                cloned_layer = self.clone_layer(layers[i])
                #Passing the activation from n-1 layer to nth layer via  forward pass (here len of activations is = len of layers + 1)
                act[i].to(self.device)
                forward_pass = cloned_layer.forward(act[i]) + self.eps
                (forward_pass * (relevances_per_layer[i+1].to(self.device)/forward_pass).data).sum().backward()
                relevances_per_layer[i] = (act[i]*act[i].grad).data   
                
            else:
                #simply copying relevances
                relevances_per_layer[i] = relevances_per_layer[i+1]
        return relevances_per_layer
    def generate_relevance_cam(self,layer_number,relevances,activations,input_image):
        """
        Generates CAM using the relevancy scores
        
        [input]:
        layer_number[int] : layer number at which the CAM is to be generated
        relevances [list] : Calculated relevances for each layer
        activations [list] : Activations for each layer
        input_image [tensor] : Input image on which the relevance MAP is to be overlayed
        
        [Returns]:
        relevance_cam[np.ndarry] : Relevance CAM
        
        """
        
        
        weights = nn.AdaptiveAvgPool2d((1,1))(relevances[layer_number])
        output = torch.zeros((activations[layer_number].shape[2],activations[layer_number].shape[3]))
        for i,r in enumerate(activations[layer_number][0]):
            output+=r.cpu()*weights[0][i].cpu()
        relevance_cam = cv2.resize(output.detach().numpy(), (input_image.shape[3],input_image.shape[2]))
        return relevance_cam
    
    def relevancy(self,layer_number:list,img:torch.Tensor,cam_class:int,plot=True):
        """
        Generates Relevance CAM for a given target class
        [input]:
        layer_number: layer number at which you want to visualize the Relevance CAM
        img : Image tensor
        cam_class : Target Class index for the relevance cam
        
        [Returns]:
        Relevance_CAM
        """
        
        layers = []
#         layers = list(self.model._modules['features']) + self.convert_to_conv(list(self.model._modules['classifier']))

        for key in self.model._modules.keys():
            if (key == "classifier" or key == "fc") :
                layers+=[nn.AdaptiveAvgPool2d((1,1))]
                layers+=self.convert_to_conv([self.model._modules[key]])
            else:
                if "avgpool" not in key:
                    layers+=[(self.model._modules[key])]
        
                    
        
                
        activations = self.get_activations(img,layers)
        #predict class
        score = self.get_prediction(activations)
        #construct target
        target_vector = self.construct_target(activations,cam_class)
        
        #relevance propagation
        relevancy_per_layer = self.relevance_propagation(activations,target_vector,layers)
        
        
        relevancy_cam = self.generate_relevance_cam(layer_number,relevancy_per_layer,activations,img)
        if plot:
            plt.imshow(relevancy_cam)
            plt.plot()

        
        return relevancy_cam,score
            
    