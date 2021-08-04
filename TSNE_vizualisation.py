import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
# IEEE Format for Figures
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'weight' : 400,
        'size'   : 19}

plt.rc('font', **font)
img_save_path   = "/itet-stor/sfurrer/net_scratch/UNITER/ViLT/attacks_analysis/TSNE"


def TSNE_projection(neg_img, neg_txt,nbr_element,batch_idx,img_save_path) : 

    #TO DO : Add the possibility to choose The number of element to show 
    
    neg_img = neg_img[:, :nbr_element]
    neg_txt = neg_txt[:, :nbr_element]
    pairs   = np.concatenate((np.arange(nbr_element),np.arange(nbr_element))) 
    
    embeddings_array = torch.cat((neg_img,neg_txt),1)
    """ 2D Plot"""
    tsne = TSNE(2, verbose=False)
    tsne_proj = tsne.fit_transform(embeddings_array.T.cpu())
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(16,10))
    
    for lab in range(nbr_element):

        indices = pairs==lab
        ax.scatter(tsne_proj[indices,0],
                   tsne_proj[indices,1],
                   c=np.array(cmap(lab)).reshape(1,4), 
                   #label = objects[lab] ,
                   alpha=0.5)    

    plt.title('TSNE : Embedded representation Text/Images', fontsize=20)  
    
    plt.savefig(os.path.join(img_save_path,"TSNE_{}.png".format(batch_idx)), 
        #This is simple recomendation for publication plots
        dpi=1000, 
        # Plot will be occupy a maximum of available space
        bbox_inches='tight', 
        )   