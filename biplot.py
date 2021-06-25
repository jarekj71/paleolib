import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def crosscor(raw,transformed):
    corr = np.corrcoef(raw,transformed, rowvar=False)
    cc = corr[raw.shape[1]:,:raw.shape[1]]
    return cc

def scatter3D(coords,axislabels=None,axislabelsize=5,color=None,ax=None,**kwargs):
    grey = (0.99,0.99,0.99)
    ax = ax or plt.gca(projection='3d')
    ax.scatter(coords[:,0],coords[:,1],coords[:,2],color=color,**kwargs)

    try:
        axlabels = coords.columns
    except AttributeError:
        axlabels = axislabels

    try:
        nlabels = len(axlabels)
    except TypeError:
        nlabels = 0

    if nlabels != 3:
        axislabels = None

    if axislabels:
        ax.set_xlabel(axlabels[0])
        ax.set_ylabel(axlabels[1])
        ax.set_zlabel(axlabels[2])
    ax.tick_params(axis='both', which='major', labelsize=axislabelsize)
    ax.w_xaxis.set_pane_color(grey)
    ax.w_yaxis.set_pane_color(grey)
    ax.w_zaxis.set_pane_color(grey)
    ax.view_init(azim=-60)
    return ax

#%%
def biplot(score,coeff,components=(1,2),colors=None,markers=None,labels=None,ax=None,title=None,**kwargs):
    ax = ax or plt.gca() # pobranie aktywnego axes
    c0 = components[0]-1
    c1 = components[1]-1
    xs = score[:,c0]
    ys = score[:,c1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    if markers is not None:
        for marker in np.unique(markers):
            slr = (markers == marker) # selector
            ax.scatter(xs[slr] * scalex,ys[slr] * scaley, c = colors[slr],marker=marker,cmap="brg",**kwargs)
    else:
        ax.scatter(xs * scalex,ys * scaley, c = colors,cmap="brg",**kwargs)
    for i in range(n):
        ax.arrow(0, 0, coeff[i,c0], coeff[i,c1],color = 'r',alpha = 0.5)
        if labels is None:
            ax.text(coeff[i,c0]* 1.15, coeff[i,c1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            ax.text(coeff[i,c0]* 1.15, coeff[i,c1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    #ax.set_xlim(-1,1)
    #ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(components[0]))
    ax.set_ylabel("PC{}".format(components[1]))
    if title is not None:
        ax.set_title(title)

#%%
def biplot3D(score,coeff,colors=None,labels=None,ax=None,title=None,markers=None,**kwargs):
    grey = (0.99,0.99,0.99)
    ax = ax or plt.gca() # pobranie aktywnego axes
    c0 = 0
    c1 = 1
    c2 = 2
    xs = score[:,c0]
    ys = score[:,c1]
    zs = score[:,c2]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    scalez = 1.0/(zs.max() - zs.min())
    if markers is not None:
        for marker in np.unique(markers):
            slr = (markers == marker)
            ax.scatter(xs[slr] * scalex,ys[slr] * scaley,zs[slr] * scalez, c = colors[slr],marker=marker,cmap="brg",**kwargs)
    else:
        ax.scatter(xs * scalex,ys * scaley,zs * scalez, c = colors,cmap="brg",**kwargs)

    for i in range(n):
        ax.plot([0, coeff[i,c0]],[0, coeff[i,c1]],[0, coeff[i,c2]], color = 'r',alpha = 0.5)
        if labels is None:
            ax.text(coeff[i,c0]* 1.15, coeff[i,c1] * 1.15,coeff[i,c2] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            ax.text(coeff[i,c0]* 1.15, coeff[i,c1] * 1.15, coeff[i,c2] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    #ax.set_xlim(-1,1)
    #ax.set_ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))
    ax.set_ylabel("PC{}".format(3))
    ax.w_xaxis.set_pane_color(grey)
    ax.w_yaxis.set_pane_color(grey)
    ax.w_zaxis.set_pane_color(grey)
    if title is not None:
        ax.set_title(title)

#%%

site_color = "#1155AA"
feature_color = "#AA4411"
arrow_color = "#AA0011"

def _shrink_scaler(column):
    return column * 1.0/(column.max() - column.min())

def ordination2D(ordination,fontsize=12,extent=None,ax=None):
    '''
    wyświetla ordination
    '''
    ax = ax or plt.gca() # pobranie aktywnego axes
    
    xs = _shrink_scaler(ordination.samples.iloc[:,0])
    ys = _shrink_scaler(ordination.samples.iloc[:,1])
    xf = _shrink_scaler(ordination.features.iloc[:,0])
    yf = _shrink_scaler(ordination.features.iloc[:,1])
    

    ax.scatter(xf,yf,c=feature_color,s=150)
    ax.scatter(xs,ys,c=site_color,s=90)
    
    font0=FontProperties()
    font0.set_size(fontsize)
    for x,y,txt in zip(xs,ys,ordination.samples.index):
        ax.text(x,y,txt,verticalalignment='top',fontproperties=font0,color=site_color)
    for x,y,txt in zip(xf,yf,ordination.features.index):
        ax.text(x,y,txt,verticalalignment='bottom',fontproperties=font0,color=feature_color)

    font1=FontProperties()
    font1.set_size(fontsize-2)   
    arrow_width = (xs.max()-xs.min())/300
    
    if ordination.biplot_scores is not None:
        
        ord_min_x = ordination.biplot_scores.iloc[:,0].min()
        ord_max_x = ordination.biplot_scores.iloc[:,0].max()
        ord_min_y = ordination.biplot_scores.iloc[:,1].min()
        ord_max_y = ordination.biplot_scores.iloc[:,1].max()
        for x,y,label in zip(ordination.biplot_scores.iloc[:,0],ordination.biplot_scores.iloc[:,1],ordination.biplot_scores.index):
            ax.arrow(0, 0, x, y,color = arrow_color,alpha = 0.5,width=arrow_width)
            ax.text(x* 1.1, y * 1.1, label, color=arrow_color, ha = 'center', va = 'center',fontproperties=font1)
    else:
        ord_min_x, ord_max_x, ord_min_y, ord_max_y = (1,-1,1,-1) # eliminujemy
    
    if extent is not  None:
        xmin,xmax,ymin,ymax = extent
    else:
        xmin = min(xs.min(),xf.min(),ord_min_x)*1.1
        xmax = max(xs.max(),xf.max(),ord_max_x)*1.1
        ymin = min(ys.min(),yf.min(),ord_min_y)*1.1
        ymax = max(ys.max(),yf.max(),ord_max_y)*1.1
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Component 1: " + str(round(ordination.proportion_explained[0]*100,1))+ "% explained",fontsize=16)
    ax.set_ylabel("Component 2: " + str(round(ordination.proportion_explained[1]*100,1))+ "% explained",fontsize=16)

