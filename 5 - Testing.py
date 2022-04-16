# This code defines our Network (same as in "4 - Data Loader, model and training.py" on which we trained) and loads the best model weights from our training
# We do one feedforward pass of our test data through the network to test our model generalisation

# Import libraries, methods and functions
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from plotly.subplots import make_subplots
import seaborn as sns
import plotly.graph_objs as go
import copy
import os
import os.path
import torch
import torchvision
from PIL import Image
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torchsummary import summary
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
import plotly.express as px
import matplotlib
from ignite.metrics import EpochMetric


#Define path of the patches
test_tum=os.path.join(r'<<Insert_your_path>>') 
test_norm=os.path.join(r'<<Insert_your_path>>') 

#Print path
print(test_tum)
print(test_norm)

file_names_tumor = os.listdir(test_tum)
file_names_norm = os.listdir(test_norm)

# Create dataframe with tumor patches
i=0
for i in range (len(test_tum)):
    path_files_tum = [os.path.join(test_tum, i) for i in file_names_tumor]

df_tum = pd.DataFrame({"Patch":file_names_tumor, "Path":path_files_tum})
df_tum['label']=1

#Create dataframe with normal patches
i=0
for i in range (len(test_norm)):
    path_files_norm = [os.path.join(test_norm, i) for i in file_names_norm]

df_norm = pd.DataFrame({"Patch":file_names_norm, "Path":path_files_norm})
df_norm['label']=0

# Merge dataframes and create test df
df_test=pd.concat([df_tum, df_norm], ignore_index=True)
df_test.shape

# Function to get the size of the images
def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width, height

# Create list to save the size of the images and save values in it
size=[]
for i in range (len(df_test)):
    size.append((get_num_pixels(df_test['Path'][i])))

# Add column size 
df_test['size']=size

# Only keep images which size is (299,299) - incomplete patches can appear after creating patches from other image level than image level 0
# We drop all patches which do not have the requried 299x299 size and reset the index of our dataframe afterwards
df_test = df_test[df_test['size'] == (299,299)]
df_test.reset_index()

# Drop column which contains the size - not necessary anymore
df_test=df_test.drop(['size'], axis=1)

# drop duplicated patches
df_test.drop_duplicates()

# Shuffle dataframe and keep only keep certain fraction X of all rows by setting frac=X
df_test=df_test.sample(frac=1)
df_test

# idea for this pytorch transformer from https://www.kaggle.com/shtrausslearning/binary-cancer-image-classification-w-pytorch and adapted

class pytorch_data(Dataset):
    
    def __init__(self, df, transform): 
        self.file_path=[df['Path'][i] for i in range (len(df))]
        
        self.labels=[df['label'][i] for i in range (len(df))]

        self.transform = transform
        
        
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.file_path[idx])  # Open Image with PIL
        image = self.transform(image) # Apply Specific Transformation to Image
        return image, self.labels[idx]

pytorch_transformer = transforms.Compose([transforms.ToTensor()])

#Create patch dataset
patch_dataset=pytorch_data(df_test, pytorch_transformer)

print(len(patch_dataset))

test=patch_dataset

#Show 10 instances from our dataset
#copied from: https://www.kaggle.com/shtrausslearning/binary-cancer-image-classification-w-pytorch#kln-112 and just slightly adapted
# Define method for plotting
def plot_img(x,y):

    npimg = x.numpy() # convert tensor to numpy array
    npimg_tr=np.transpose(npimg, (1,2,0)) # Convert to H*W*C shape
    fig = px.imshow(npimg_tr)
    fig.update_layout(coloraxis_showscale=False,title=str(y_grid_test))
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(template='plotly_white',height=400);fig.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
    fig.update_layout(title={'text': str(y),'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    
    fig.show()

#Create grid of sample images and show the respective labels 
grid_size=8
rnd_inds=np.random.randint(0,len(test),grid_size)
print("image indices:",rnd_inds)

x_grid_test=[test[i][0] for i in rnd_inds]
y_grid_test=[test[i][1] for i in rnd_inds]

x_grid_test=utils.make_grid(x_grid_test, nrow=4, padding=2)
print(x_grid_test.shape)
    
plot_img(x_grid_test,y_grid_test)

#Define our CNN
#The code for the model is copied from https://www.programmersought.com/article/50056762794/ and adapted

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class InceptionResNetA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetA, self).__init__()
        #branch1: conv1*1(32)
        self.b1 = BasicConv2d(in_channels, 32, kernel_size=1)
        
        #branch2: conv1*1(32) --> con3*3(32)
        self.b2_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b2_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        
        #branch3: conv1*1(32) --> conv3*3(32) --> conv3*3(32)
        self.b3_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b3_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        
        #totalbranch: conv1*1(256)
        self.tb = BasicConv2d(96, 256, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(x)
        b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b_out3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))
        b_out = torch.cat([b_out1, b_out2, b_out3], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)
                           
        return out

class InceptionResNetB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetB, self).__init__()
        #branch1: conv1*1(128)
        self.b1 = BasicConv2d(in_channels, 128, kernel_size=1)
        
        #branch2: conv1*1(128) --> con1*7(128) --> conv7*1(128)
        self.b2_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.b2_2 = BasicConv2d(128, 128, kernel_size=(1,7), padding=(0,3))
        self.b2_3 = BasicConv2d(128, 128, kernel_size=(7,1), padding=(3,0))
    
        #totalbranch: conv1*1(896)
        self.tb = BasicConv2d(256, 896, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(x)
        b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_3(F.relu(self.b2_2(F.relu(self.b2_1(x))))))
        b_out = torch.cat([b_out1, b_out2], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)
                           
        return out

class InceptionResNetC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetC, self).__init__()
        #branch1: conv1*1(192)
        self.b1 = BasicConv2d(in_channels, 192, kernel_size=1)
        
        #branch2: conv1*1(192) --> con1*3(192) --> conv3*1(192)
        self.b2_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.b2_2 = BasicConv2d(192, 192, kernel_size=(1,3), padding=(0,1))
        self.b2_3 = BasicConv2d(192, 192, kernel_size=(3,1), padding=(1,0))
    
        #totalbranch: conv1*1(1792)
        self.tb = BasicConv2d(384, 1792, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(x)
        b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_3(F.relu(self.b2_2(F.relu(self.b2_1(x))))))
        b_out = torch.cat([b_out1, b_out2], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)
                           
        return out

class ReductionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(ReductionA, self).__init__()
        #branch1: maxpool3*3(stride2 valid)
        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        #branch2: conv3*3(n stride2 valid)
        self.b2 = BasicConv2d(in_channels, n, kernel_size=3, stride=2)
        
        #branch3: conv1*1(k) --> conv3*3(l) --> conv3*3(m stride2 valid)
        self.b3_1 = BasicConv2d(in_channels, k, kernel_size=1)
        self.b3_2 = BasicConv2d(k, l, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(l, m, kernel_size=3, stride=2)
        
    def forward(self, x):
        y1 = self.b1(x)
        y2 = F.relu(self.b2(x))
        y3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))
        
        outputsRedA = [y1, y2, y3]
        return torch.cat(outputsRedA, 1)

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        #branch1: maxpool3*3(stride2 valid)
        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        #branch2: conv1*1(256) --> conv3*3(384 stride2 valid)
        self.b2_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.b2_2 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        
        #branch3: conv1*1(256) --> conv3*3(256 stride2 valid)
        self.b3_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.b3_2 = BasicConv2d(256, 256, kernel_size=3, stride=2)
        
        #branch4: conv1*1(256) --> conv3*3(256) --> conv3*3(256 stride2 valid)
        self.b4_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.b4_2 = BasicConv2d(256, 256, kernel_size=3, padding=1)
        self.b4_3 = BasicConv2d(256, 256, kernel_size=3, stride=2)
        
    def forward(self, x):
        y1 = self.b1(x)
        y2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        y3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        y4 = F.relu(self.b4_3(F.relu(self.b4_2(F.relu(self.b4_1(x))))))
        
        outputsRedB = [y1, y2, y3, y4]
        return torch.cat(outputsRedB, 1)

class StemForIR1(nn.Module):
    def __init__(self, in_channels):
        super(StemForIR1, self).__init__()
        #conv3*3(32 stride2 valid)
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        #conv3*3(32 valid)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)
        #conv3*3(64)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        #maxpool3*3(stride2 valid)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        #conv1*1(80)
        self.conv4 = BasicConv2d(64, 80, kernel_size=1)
        #conv3*3(192 valid)
        self.conv5 = BasicConv2d(80, 192, kernel_size=3)
        #conv3*3(256, stride2 valid)
        self.conv6 = BasicConv2d(192, 256, kernel_size=3, stride=2)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.maxpool1(out)
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        
        return out

class InceptionResNetv1(nn.Module):
    def __init__(self):
        super(InceptionResNetv1, self).__init__()
        self.stem = StemForIR1(3)
        self.irA = InceptionResNetA(256)
        self.redA = ReductionA(256, 192, 192, 256, 384)
        self.irB = InceptionResNetB(896)
        self.redB = ReductionB(896)
        self.irC = InceptionResNetC(1792)
        self.avgpool = nn.MaxPool2d(kernel_size=8)
        self.dropout = nn.Dropout(p=0.5) # Dropout changed to 0.5
        self.linear = nn.Linear(1792, 2) # changed from 1000 to 2, because we have only two classes
        
    def forward(self, x):
        n = [5, 10, 5]
        out = self.stem(x)
        
        if n[0] > 0:
            out = self.irA(out)
            n[0] -= 1
        out = self.redA(out)
        
        if n[1] > 0:
            out = self.irB(out)
            n[1] -= 1
        out = self.redB(out)
        
        if n[2] > 0:
            out = self.irC(out)
            n[2] -= 1
            
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out


#Send data and model to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Set our CNN as model
model = InceptionResNetv1()

cnn_model=model.to(device)

#Show our model
print(cnn_model)

#Show summary of our model
summary(cnn_model, input_size=(3, 299, 299))

# Define method to show the number of correct predictions (for Tensors) - copied from https://deeplizard.com/learn/video/p1xZ2yWU1eo
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

#Set batch size
batch_Size=16

#Define the Data Loader
test_loader=torch.utils.data.DataLoader(test, batch_size=batch_Size, shuffle=True, num_workers=2)

# Define function to calculate ROC AUC score
# See documentation: https://pytorch.org/ignite/_modules/ignite/contrib/metrics/roc_auc.html

def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    y_true = y_targets.cpu().detach().numpy()
    y_pred = y_preds.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)


# Set path to load weights
weight_path=(r"<<Insert_your_path>>/weights_best.pt")

# Class Probabilites and labels
best_prob = []
label_best = []


print("--------------- Testing started ---------------") 
# Load the best model weights, see documentation https://pytorch.org/tutorials/beginner/saving_loading_models.html
cnn_model.load_state_dict(torch.load(weight_path), map_location=torch.device('cpu')) # load best model weights from weight file

total_loss = 0
total_correct = 0
total_roc_auc = 0

    
# Implemented for metric and confusion matrix
all_preds = [] # Create empty list to store predictions
all_labels = [] # Create empty list to store respective labels
all_probs = [] # Create empty list to store the probabilities of the prediction
all_softmax = [] # Create empty list to store the probabilites


with torch.no_grad():
    for batch_val in test_loader: # Get Batch
        cnn_model.eval() # Set model to evaluation mode: Disable dropout layers and normalisation layers use running statistics 
        images, labels = batch_val
        images = images.to(device)
        labels = labels.to(device)
        preds = cnn_model(images) # Pass Batch            
        loss_val = F.cross_entropy(preds, labels, reduction="sum") # Define loss function and calculate loss

        pred_softmax = F.softmax(preds, dim=-1) # Get the probabilities of the predictions

        all_softmax.extend(pred_softmax.cpu().detach().numpy().tolist())

        pred_prob = pred_softmax[:, 1:2] # Get the probabilities of the of the class with the label 1
        pred_prob = pred_prob.to(device)
        
        all_preds.extend(preds.cpu().detach().numpy().tolist()) # Convert Pytorch tensor to numpy array to avoid running out of GPU memory space
                    
        all_labels.extend(labels.cpu().detach().numpy().tolist())
                    
        all_probs.extend(pred_prob.cpu().detach().numpy().tolist())
            
        total_loss += loss_val.item()
        total_correct += get_num_correct(preds, labels)

# Get ROC AUC score
roc_auc = roc_auc_score(all_labels, all_probs)
print(
        "Test:",
        "Correct predictions:", total_correct, 
        "| Accuracy:", (total_correct/len(test)),
        "| ROC AUC:", roc_auc,
        "| Loss:", total_loss,
         )


# Get the predicted class (can't call our method from above, because predictions are saved in numpy arrays)
i=0
best_pred=[]
while i<len(all_softmax):
    best_pred.append(np.argmax(all_softmax[i]))
    i=i+1

# Get the number of correct predictions (can't call our method from above, because predictions are saved in numpy arrays)
i=0
preds_correct = 0
while i < len(best_pred):
    if best_pred[i] == all_labels[i]:
        preds_correct = preds_correct+1
    else:
        None
    i=i+1

# Export predictions as Excel file
results=pd.DataFrame({'label_best':all_labels,'best_pred':best_pred})
results.to_excel("TEST_lvl0.xlsx")

# Create confusion matrix
cm = confusion_matrix(all_labels, best_pred)

# Show confusion matrix
print(cm)