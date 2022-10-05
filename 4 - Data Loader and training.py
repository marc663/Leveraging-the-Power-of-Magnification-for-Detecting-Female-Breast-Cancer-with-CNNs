# This code creates our training and validation data sets out of the normalized patches
# We also define our model, train our model and measure the performance per epoch and determine the best performance and save the respective model weights
# We export the most important results as Excel files 

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
from ignite.metrics import EpochMetric
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
# Import CNN model
import model


#Define path of the patches
train_tum=os.path.join(r'<<Insert_your_path>>')
train_norm=os.path.join(r'<<Insert_your_path>>')

#Print path
print(train_tum)
print(train_norm)

file_names_tumor = os.listdir(train_tum)
file_names_norm = os.listdir(train_norm)

# Create dataframe with tumor patches
i=0
for i in range (len(train_tum)):
    path_files_tum = [os.path.join(train_tum, i) for i in file_names_tumor]

df_tum = pd.DataFrame({"Patch":file_names_tumor, "Path":path_files_tum})
df_tum['label']=1

#Create dataframe with normal patches
i=0
for i in range (len(train_norm)):
    path_files_norm = [os.path.join(train_norm, i) for i in file_names_norm]

df_norm = pd.DataFrame({"Patch":file_names_norm, "Path":path_files_norm})
df_norm['label']=0

# Get the fraction of normal samples to match tumor samples
frac_df=(len(df_tum)/len(df_norm))

# Randomly sample normal instances to match number of tumor patches
df_norm=df_norm.sample(frac=frac_df, ignore_index=True)

# Merge dataframes and create trainining df
df_train=pd.concat([df_tum, df_norm], ignore_index=True)
df_train.shape

# Get dataframe which contains a certain 3034 instances
# Comment the following two lines out if we run our data on the whole data set
frac_train=3034/len(df_train)
df_train=df_train.sample(frac=frac_train, ignore_index=True)

# Function to get the size of the images
def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width, height

# Create list to save the size of the images and save values in it
size=[]
for i in range (len(df_train)):
    size.append((get_num_pixels(df_train['Path'][i])))

# Add column size 
df_train['size']=size

# Only keep images which size is (299,299) - incomplete patches can appear after creating patches from other image level than image level 0 (we might have overlooked some incomplete patches when we manually removed them)
# We drop all patches which do not have the requried 299x299 size and reset the index of our dataframe afterwards
df_train = df_train[df_train['size'] == (299,299)]
df_train.reset_index()

# Drop column which contains the size - not necessary anymore
df_train=df_train.drop(['size'], axis=1)

# drop duplicated patches
df_train.drop_duplicates()

# Shuffle dataframe and keep only keep certain fraction X of all rows by setting frac=X
df_train=df_train.sample(frac=1, ignore_index=True)
df_train

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
patch_dataset=pytorch_data(df_train, pytorch_transformer)

print(len(patch_dataset))

# Creating training and validation set
len_img=len(patch_dataset)
len_train=int(0.8*len_img)
len_val=len_img-len_train

# Split Pytorch tensor into train and validation
train,val=random_split(patch_dataset,[len_train,len_val]) # random split 80/20

print("train dataset size:", len(train))
print("validation dataset size:", len(val))

#Show 10 instances from our dataset
#copied from: https://www.kaggle.com/shtrausslearning/binary-cancer-image-classification-w-pytorch#kln-112 and just slightly adapted
# Define method for plotting
def plot_img(x,y):

    npimg = x.numpy() # convert tensor to numpy array
    npimg_tr=np.transpose(npimg, (1,2,0)) # Convert to H*W*C shape
    fig = px.imshow(npimg_tr)
    fig.update_layout(coloraxis_showscale=False,title=str(y_grid_train))
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(template='plotly_white',height=400);fig.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
    fig.update_layout(title={'text': str(y),'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    
    fig.show()

#Create grid of sample images and show the respective labels 
grid_size=8
rnd_inds=np.random.randint(0,len(train),grid_size)
print("image indices:",rnd_inds)

x_grid_train=[train[i][0] for i in rnd_inds]
y_grid_train=[train[i][1] for i in rnd_inds]

x_grid_train=utils.make_grid(x_grid_train, nrow=4, padding=2)
print(x_grid_train.shape)
    
plot_img(x_grid_train,y_grid_train)


#Send data and model to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Import our model and set as our CNN as model
model = model.InceptionResNetv1()

cnn_model=model.to(device)

#Show our model
print(cnn_model)

#Show summary of our model
summary(cnn_model, input_size=(3, 299, 299))

# Define method to show the number of correct predictions (for Tensors) - copied from https://deeplizard.com/learn/video/p1xZ2yWU1eo
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

# Define method to get learning rate 
# See PyTorch Documentation: https://pytorch.org/docs/stable/optim.html
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# Reset weights to avoid weight leakage
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

cnn_model.apply(reset_weights)

# Set a manual seed to have the same random weights for each initialization
torch.manual_seed(4)

# define function to randomly initialize the weights before training the model 
# With the manual seed the same random weights will be initialized
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

cnn_model.apply(weights_init)

#Set number of epochs and batch size
num_epochs=100
batch_Size=16

#Define the Data Loader
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_Size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_Size, shuffle=True, num_workers=2)

# Define optimizer with initial learning rate of 0.045 and weight decay=0.0005 to regularize the model
optimizer = optim.Adam(cnn_model.parameters(), lr=0.045 , weight_decay=0.0005)

#Define learning rate scheduler - Learning rate halfed when there is no improvement for 5 epochs
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5,verbose=1)

# Define function to calculate ROC AUC score
# See documentation: https://pytorch.org/ignite/_modules/ignite/contrib/metrics/roc_auc.html

def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:

    y_true = y_targets.cpu().detach().numpy()
    y_pred = y_preds.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)


# Set path to save weights
weight_path=("weights_IL1.pt")

#Creating training loop
print("######### Training started #########")
best_loss=float('inf')
# best_model_wts = copy.deepcopy(cnn_model.state_dict()) # a deep copy of weights for the best performing model
start_total = time.time()

# Class Probabilites and labels for the best epoch
best_prob = []
label_best = []

# Store the history of each loss, learning rate and the metric values 
loss_history={"train": [],"val": []} # history of loss values in each epoch
metric_history={"val": [] } #,"train": []} # histroy of metric values in each epoch
learning_history=[]

#Define loop for every epoch
for epoch in range(num_epochs):

    cnn_model.apply(weights_init)

    print("--------------- Epoch", epoch, "---------------") # Line to separate the results of each epoch
    start = time.time()
    
    total_loss_train = 0
    total_correct_train = 0
    total_loss = 0
    total_correct = 0
    total_roc_auc = 0

    
    # Implemented for metric and confusion matrix
    all_preds = [] # Create empty list to store predictions
    all_labels = [] # Create empty list to store respective labels
    all_probs = [] # Create empty list to store the probabilities of the prediction
    all_softmax = [] # Create empty list to store the probabilites
   
    ########### Training the model ###############
    for batch in train_loader: # Get Batch
        cnn_model.train() # Set layers in training mode: activates Dropout layers, normalisation layers use per-batch statistics
        images_train, labels_train = batch 
        images_train = images_train.to(device)
        labels_train = labels_train.to(device)
        preds_train = cnn_model(images_train) # Pass Batch
        loss_train = F.cross_entropy(preds_train, labels_train, reduction="sum") # Define loss function and calculate Loss
        
        total_loss_train += loss_train.item()
        total_correct_train += get_num_correct(preds_train, labels_train)
            
        optimizer.zero_grad() #Set gradients to zero before starting backpropagation
        loss_train.backward() # Calculate Gradients 
        optimizer.step() # Update Weights

    # Get the current learning rate
        current_lr = get_lr(optimizer)

     # Update learning rate
    lr_scheduler.step(total_loss_train)

    loss_history["train"].append(total_loss_train)
    learning_history.append(current_lr)
    
    print(
            "Training:", 
            "Correct predictions:", total_correct_train, 
            "| Accuracy:", (total_correct_train/len(train)), 
            "| Loss:", total_loss_train
           # '| Learning rate:', current_lr
            )
        
    ########## Validation of the model: ##############
    with torch.no_grad():
        for batch_val in val_loader: # Get Batch
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

    # Get ROC AUC score for whole epoch
    roc_auc = roc_auc_score(all_labels, all_probs)

    loss_history["val"].append(total_loss)
    metric_history["val"].append(roc_auc)

    print(
        "Validation:",
        "Correct predictions:", total_correct, 
        "| Accuracy:", (total_correct/len(val)),
        "| ROC AUC:", roc_auc,
        "| Loss:", total_loss,
         )

    # Store best model and update saved values
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_wts = copy.deepcopy(cnn_model.state_dict())
        
        # store weights into a local file to store them permanently
        torch.save(cnn_model.state_dict(), weight_path)

        # Copy the probabilites and labels of validation process for confusion matrix
        best_prob = all_softmax
        label_best = all_labels      
        best_roc_auc = roc_auc
        best_acc = total_correct/len(val)
        print("Loss improved --> saving model weights")

    end = time.time()
    minutes = round((end - start) / 60 , 2)
    print ("Time needed for Epoch", epoch,  minutes, "min")

end_total = time.time()

minutes_total = round((end_total - start_total) / 60 , 2)
average_min_epoch = round((minutes_total/num_epochs),2)
print("Total time needed for training:", minutes_total, "min")
print("Average time per epoch:", average_min_epoch, "min")
print("######### Training finished #########")

print('Maximum occupied GPU memory:', round((torch.cuda.max_memory_allocated(device)/1073741824),2), 'GB') # Get the maximum occupied GPU memory, convert Bytes in GB

# Get the predicted class of the best epoch (can't call our method from above, because predictions are saved in numpy arrays)
i=0
best_pred=[]
while i<len(best_prob):
    best_pred.append(np.argmax(best_prob[i]))
    i=i+1

# Get the number of correct predictions (can't call our method from above, because predictions are saved in numpy arrays)
i=0
preds_correct = 0
while i < len(best_pred):
    if best_pred[i] == label_best[i]:
        preds_correct = preds_correct+1
    else:
        None
    i=i+1

print('total correct best prediction:', preds_correct)
print('accuracy best prediction:', best_acc)
print ('roc auc best prediction:', best_roc_auc)

# Export predictions as Excel file
results=pd.DataFrame({'label_best':label_best,'best_pred':best_pred})
results.to_excel("results_IL1.xlsx")

# Export loss, learning rate and metric history as Excel file
history=pd.DataFrame({'training loss':loss_history['train'],'validation loss':loss_history['val'], 'valdiation roc auc':metric_history['val'], 'learning rate':learning_history})
history.to_excel("history_IL1.xlsx")

# Create confusion matrix
cm = confusion_matrix(label_best, best_pred)

# Show confusion matrix
print(cm)
