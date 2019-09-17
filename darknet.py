 
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *

def get_test_input():
    img=cv2.imread("dog-cycle-car.png")
    img=cv2.resize(img,(416,416))
    img_=img[:,:,::-1].transpose((2,0,1))
    img_=img_[np.newaxis,:,:,:]/255.0
    img_=torch.from_numpy(img_).float()
    img_= Variable(img_)
    return img_

def cfg_parser(cfgfile):                    #Reading the config file

    file=open(cfgfile,'r')
    lines=file.read().split('\n')           #storing lines by splitting using \n
    lines=[line for line in lines if len(line)>0]    #filtering out lines that has values
    lines=[line for line in lines if line[0]!='#']   #filtering out comments
    lines=[line.rstrip().lstrip() for line in lines]  #stripping the lines to remove spaces

    block={}
    blocks=[]
    for item in lines:                              #loop through each blocks in config file
        if item[0]=="[":
            if len(block)!=0:
                blocks.append(block)
                block={}
            block["type"]=item[1:-1].rstrip()
        else:
            key,value=item.split("=")
            block[key.rstrip()]=value.lstrip()
    blocks.append(block)

    return blocks
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info=blocks[0]   #Information about the network, whicjh is provided in the begginnig of cfg file
    module_list=nn.ModuleList()
    previousFilters=3
    outputFilters=[]
    for idx,x in enumerate(blocks[1:]):
        module=nn.Sequential()  #Sequentially execute a number of nn.Module Objects
        if (x["type"]=="convolutional"):
            activation=x["activation"]
            try:
                batchNormalize=int(x["batch_normalize"])
                bias =False
            except:
                batchNormalize=0
                bias =True
            filters=int(x["filters"])
            padding=int(x["pad"])
            kernelSize=int(x["size"])
            stride=int(x["stride"])
            if padding:
                pad=(kernelSize-1)//2
            else:
                pad=0

            conv=nn.Conv2d(previousFilters,filters,kernelSize,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(idx),conv)

            if batchNormalize:
                bN=nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(idx),bN)

            if activation=="leaky":
                activatn=nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky{0}".format(idx),activatn)    

        elif(x["type"]=="upsample"):
            stride=int(x["stride"])
            upsample=nn.Upsample(scale_factor=2,mode="nearest")
            module.add_module("upsample{}".format(idx),upsample)

        elif(x["type"]=="route"):
            x["layers"]=x["layers"].split(',')
            start=int(x["layers"][0])

            try:
                end=int(x["layers"][1])
            except:
                end=0

            if start>0:
                start=start-idx
            if end>0:
                end=end-idx

            route=EmptyLayer()
            module.add_module("route_{0}".format(idx),route)
            if end<0:
                filters=outputFilters[idx+start]+outputFilters[idx+end]
            else:
                filters=outputFilters[idx+start]

        elif x["type"]=="shortcut":
            shortcut=EmptyLayer()
            module.add_module("shortcut_{}".format(idx),shortcut)

        elif x["type"]=="yolo":
            mask=x["mask"].split(",")
            mask=[int(x) for x in mask]

            anchors=x["anchors"].split(",")
            anchors=[int(a) for a in anchors]
            anchors=[(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors=[anchors[i] for i in mask]
            detection=DetectionLayer(anchors)
            module.add_module("Detection_{}".format(idx),detection)
        module_list.append(module)
        previousFilters=filters
        outputFilters.append(filters)
    return(net_info,module_list)



class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks=cfg_parser(cfgfile)
        #self.netInfo,self.moduleList=create_modules(self.blocks)
        self.net_info, self.module_list = create_modules(self.blocks)
    def forward(self,x,CUDA):
        modules=self.blocks[1:]
        outputs={}
        write=0
        for i, module in enumerate(modules):
            moduleType=(module["type"])
            if moduleType=="convolutional" or moduleType=="upsample":
                x=self.module_list[i](x)
            elif moduleType=="route":
                layers=module["layers"]
                layers=[int(a) for a in layers]

                if (layers[0])>0:
                    layers[0]=layers[0]-i
                if len(layers)==1:
                    x=outputs[i+(layers[0])]
                else:
                    if(layers[1])>0:
                        layers[1]=layers[1]-i

                    map1=outputs[i+layers[0]]
                    map2=outputs[i+layers[1]]
                    x=torch.cat((map1,map2),1)
            elif  moduleType == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif moduleType=='yolo':
                anchors=self.module_list[i][0].anchors
                inpDim=int(self.net_info["height"])
                numClasses=int(module["classes"])

                x=x.data
                x=predictTransform(x,inpDim,anchors,numClasses,CUDA)
                if not write:
                    detections=x
                    write=1
                else:
                    detections=torch.cat((detections,x),1)
            outputs[i]=x
        return detections


    def load_weights(self,weightfile):
        fp=open(weightfile,"rb")
        header=np.fromfile(fp,dtype=np.int32,count=5)
        self.header=torch.from_numpy(header)
        self.seen=self.header[3]
        weights=np.fromfile(fp,dtype=np.float32)
        ptr=0
        for i in range(len(self.module_list)):
            moduleType=self.blocks[i+1]["type"]
            if moduleType=="convolutional":
                model=self.module_list[i]
                try:
                    batchNormalize=int(self.blocks[i+1]["batch_normalize"])
                except:
                    batchNormalize=0

                conv=model[0]
                if batchNormalize:
                    bn=model[1]
                    num_bn_biases=bn.bias.numel()

                    bn_biases=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases

                    bn_weights=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases

                    bn_running_mean=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases

                    bn_running_var=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases

                
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases=conv.bias.numel()
                    conv_biases=torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr=ptr+num_biases
                    conv_biases=conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights=conv.weight.numel()
                conv_weights=torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr=ptr+num_weights

                conv_weights=conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


