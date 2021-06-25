#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 08:28:45 2021

@author: jarekj
"""

import math, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import pdist,squareform, euclidean
from scipy.interpolate import interp1d
from shap import explainers as Exp
from sklearn.inspection import partial_dependence
from sklearn.decomposition import PCA

class clustershap:
    def __init__(self,model,X,y,yorg=None,labelrows=False,labelcols=False,sizes=None):
        self._clear()
        self._load(model,X,y,yorg,labelrows,labelcols,sizes)

    def _clear(self):
        self._explainer = None
        self._pdp = None
        self._model = None
        
        self._X = None
        self._columns = None
        self._index = None
        self._nvars = None
        self._ncases = None
        self._shapvalues = None
        self._varimportance = None
        self._y = None
        self._yhat = None
        self._yorg = None
        self._sizes = None
        self._palette = None
        self._dendrogram = None
        self._clusters = None

    def _load(self,model,X,y,yorg=None,labelrows=False,labelcols=False,sizes=None):
        if X.shape[0] != y.shape[0]:
            print("X must be data frame")
            self._clear()
            return

        # labelrows and labels cols must be a dict
        if labelrows or labelcols:
            self._X = X.rename(index=labelrows,columns=labelcols)
        else:
            self._X = X

        self._columns = self._X.columns
        self._index = self._X.index
        self._nvars = len(self._columns)
        self._ncases = len(self._index)
        self._model = model
        self._y = y
        self._yorg = yorg if yorg is not None else self._y
        if sizes is None:
            self._sizes = (self._y-self._y.min()+1)**2
        else:
            self._sizes = sizes**2

        # model must be fitted and tuned outside
        try:
            self._yhat = self._model.predict(self._X) 
        except:
            print("model not compatibile with data")
            self._clear()
            return
        explainer = Exp.Tree(model)
        shapvalues = explainer.shap_values(X)
        self._shapvalues = pd.DataFrame(shapvalues,columns=self._columns,index=self._index)
        self._varimportance = pd.Series(self._model.feature_importances_,
                                        index=self._columns).sort_values(ascending=False)

    def get_influence(self):
        if self._shapvalues is None:
            return None
        shap_matrix = self._shapvalues.loc[:,self._varimportance.index]
        shap_matrix = shap_matrix.assign(variable= self._y.values,
                                            variableorg= self._yorg.values,
                                            prediction = self._yhat)
        if self._clusters is not None:
            shap_matrix = shap_matrix.assign(clusters = self._clusters)
        return shap_matrix

    
    def build_dendrogram(self,metric='euclidean',link='ward',file=None):
        self._dst = pdist(self._shapvalues,metric=metric)
        self._linkage = linkage(self._dst,link)
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        self._dendrogram = dendrogram(self._linkage,labels=self._shapvalues.index.values,
                           color_threshold=100,ax=ax)
        y = np.repeat(0.05,self._ncases)
        x =  np.arange(self._ncases)*10+5
        leaves =np.array(self._dendrogram['leaves'])
        ax.scatter(x,y,s=self._sizes[leaves],c='#003300',zorder=100)
        
        
        if file is not None:
            plt.subplots_adjust(top=0.99,bottom=0.25)
            fig.savefig(file)

    def cluster_dendrogram(self,threshold=None):
        if self._linkage is None:
            print("show_dendrogram first")
            return
        self._clusters = fcluster(self._linkage,threshold,'distance')
        self.set_color_palette()
        print("Number of clusters: {}".format(self._clusters.max()))

    def cluster_kmeans(self,nclusters):
        pass

    def set_color_palette(self,color_palette=None):
        if self._clusters is None:
            print("Create cluster first")
            return
        
        if color_palette is None:
            colors = sns.color_palette("bright",n_colors=self._clusters.max())
            self._palette = dict(zip(np.unique(self._clusters),colors))
        else:
            self._palette = color_palette
        self._cluster_colors = pd.Series(self._clusters, 
                                         index=self._index).map(self._palette)

    def plot_3D(self,method='pca',file=None):
        colors = '#FFCCAA' if self._cluster_colors is None else self._cluster_colors
        
        if method != 'pca':
            print("not implemented yet")
            return

        pca = PCA().fit(self._shapvalues)
        expl = pca.explained_variance_ratio_
        pca_values = pca.transform(self._shapvalues)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(*pca_values.T[:3],c=colors,s=self._sizes)
        ax.set_xlabel("PC1: {} %".format(round(100*expl[0],2)))
        ax.set_ylabel("PC2: {} %".format(round(100*expl[1],2)))
        ax.set_zlabel("PC3: {} %".format(round(100*expl[2],2)))
        if file is not None:
            fig.savefig(file)

    def plot_influence(self,file=None,plot_pdp=False):
        if self._shapvalues is None:
            return None

        intercept = self._yhat.mean()
        rows = math.ceil(math.sqrt(self._nvars))
        cols = math.ceil(self._nvars/rows)
        fig,axes = plt.subplots(rows,cols,figsize=(cols*4,rows*3))

        for i,var,ax in zip(range(self._nvars),self._varimportance.index,axes.flatten()):
            sns.scatterplot(x=self._X.loc[:,var],y=self._shapvalues.loc[:,var],
                    hue=self._clusters,
                    palette = self._palette,
                    size=self._sizes.values,
                    ax=ax)
            if plot_pdp:
                index = self._columns.get_loc(var)
                ys,x = partial_dependence(self._model,self._X,[index],kind='average').values()
                sns.lineplot(x=x[0],y=(ys[0]-intercept),ax=ax,color="#999999")
            ax.set_title(var,y=1.0, pad=-11)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.legend([],[], frameon=False)
        ax.legend(bbox_to_anchor=(1.05, 1),loc="upper left")
        for ax in axes.flatten()[i+1:]:
            ax.remove()
        if file is not None:
            fig.savefig(file)

    def plot_cluster_map(self,file=None,vmin=-0.8,vmax=0.4):
        if self._clusters is None:
            return

        shap_matrix = self._shapvalues.loc[:,self._varimportance.index]
        #order by clusters after custom clustering
        g = sns.clustermap(shap_matrix,cmap='vlag',vmin=vmin,vmax=vmax,center=0,
                           figsize=(8,12),cbar_pos=(.03, .03, .03, .2),
                           yticklabels=1,xticklabels=1,
                           row_colors = self._cluster_colors,
                           dendrogram_ratio=(0.3,0.4),
                           row_linkage=self._linkage,col_cluster=False)
        g.ax_col_dendrogram.remove()
        x = np.repeat(0.5,len(self._y))
        y =  np.arange(len(self._y))+0.5
        leaves =np.array(self._dendrogram['leaves'])
        g.ax_row_colors.scatter(x,y,s=self._sizes[leaves],c='#000000')
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), 
                                     fontsize = 11)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), 
                                     fontsize = 8)

        if file is not None:
            g.savefig(file)

    def plot_cluster_boxplots(self,nvars=6,varname="",file=None):
        if self._shapvalues is None:
            return
        shap_matrix = self._shapvalues.loc[:,self._varimportance.index]
        selection = shap_matrix.iloc[:,:nvars]
        nplots = self._clusters.max()
        fig,axes = plt.subplots(1,nplots,figsize=(2*nplots,4),sharey=True)


        fp = dict(markerfacecolor = '0.50', markersize = 1)
        for i,ax in enumerate(axes):
            sns.boxplot(data=selection.loc[self._clusters==i+1],ax=ax,linewidth=0.1,flierprops=fp)
            ax.set_xlim(-1,nvars+2)
            if nvars % 2:
                box_x = (nvars+1)/10
            else:
                box_x = (nvars+2)/10
            axins = ax.inset_axes([box_x, 0, 1-box_x, 1])
            #axins.set_yscale("log")
            y = self._yorg
            axins.set_ylim(y.min(),y.max())
            if i == nplots-1:
                axins.yaxis.tick_right()
            else:
                axins.set_yticks([])

            axins.axhline(0,lw=0.5)
            axins.spines['left'].set_linewidth(0.5)

            sns.boxplot(y=y[self._clusters==i+1],
                        ax=axins,linewidth=0.1,color=self._palette[i+1],flierprops=fp)
            axins.set_ylabel("")
            axins.set_xticklabels([varname],fontdict={'fontsize':12,'rotation':'vertical'})
            plt.sca(ax) 
            plt.xticks(rotation=90)
            plt.subplots_adjust(left=0.05, bottom=0.35, right=0.9, top=0.99,wspace=0.2)
            if file is not None:
                fig.savefig(file)

    def plot_cluster_strips(self,nvars=8,varname="",file=None):
        if self._shapvalues is None:
            return

        shap_matrix = self._shapvalues.loc[:,self._varimportance.index]
        fig,ax = plt.subplots(2,1,figsize=(8,8),gridspec_kw={'height_ratios':[1,3]})
        spaw = pd.DataFrame({'x':self._yorg.values,'y':varname,'clusters':self._clusters})
        d = sns.stripplot(y='y',x='x',hue='clusters',data=spaw,
                          orient='h',size=5,palette=self._palette,
                          ax=ax[0],dodge=True,jitter=0)
        d.get_legend().remove()
        ax[0].set_yticklabels([])
        ax[0].set_ylabel("groups")
        ax[0].set_xlabel(varname)
        #ticks
        #f0 = interp1d(self._yorg,self._y,fill_value="extrapolate")
        #f1 = interp1d(self._y,self._yorg,fill_value="extrapolate")
        #ticks = f0(np.quantile(self._yorg,[0.05,0.1,0.4,0.75,1]))
        #ax[0].set_xticks(ticks)
        #labels = np.round(f1(ticks),2)
        #ax[0].set_xticklabels(labels)
        #ax[0].set_xlabel("{} value".format(varname))
        #ax[0].set_ylabel("")
        
        selection = shap_matrix.iloc[:,:nvars].assign(clustering=self._clusters)
        melted = selection.melt(id_vars=['clustering'])
        
        g= sns.stripplot(y='variable',x='value',data=melted,orient='h',
              hue='clustering',ax=ax[1],dodge=True,s=4,jitter=0,palette=self._palette)
        g.get_legend().remove()
        ax[1].hlines(y=np.arange(nvars-1)+0.5,xmin=-0.5,xmax=1,lw=0.2,color='#CCCCCC')
        ax[1].set_xlabel("influence")
        ax[1].set_ylabel("explanatory variables")
        plt.subplots_adjust(left=0.2, bottom=0.1, right=0.99, top=0.99,wspace=0.2)
        if file is not None:
            fig.savefig(file)
