#!/usr/bin/env python
# coding: utf-8
# Author: Licheng ZHANG [licheng.zhang@cern.ch]
# # Demonstration of distribution reweighting
# 
#     requirements:
#     xgboost
#     numpy
#     matplotlib
#     sklearn
#     pandas
#     ROOT 6
#     IF ROOT 6.20+ RDataFrame can directly generate Numpy Array with AsNumpy()
#     More general, following packages can be used.
#     uproot (using python 3, download root_numpy from official web and install by hand)
#     root_numpy (using python 2.7, download root_numpy from official web and install by hand) if uproot is not avaliable

# In[1]:

from __future__ import division
import xgboost as xgb 
from xgboost import plot_importance
from xgboost import plot_tree
from xgboost import XGBClassifier

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=4)

import uproot
# dihiggs = uproot.concatenate(f"{basedir}/hh2b2w.root:tree",library="pd")
# ttbar = uproot.concatenate(f"{basedir}/ttbar.root:tree",library="pd")
# import root_numpy
# from array import array
import ROOT as R
from ROOT import TCut
import time as timer

from util import *
from analysis_branch import *
import sys
import os
import argparse

from multiprocessing import Pool

def plotter(_arg):
    print('\033[1;33m Variable :: {} Checked \033[0m'.format(_arg[0]))
    plot_vars_as_ROOT_ML('LC','HC',_arg[0],_arg[1],_arg[2])
    
R.gROOT.SetBatch(True)
time_start=timer.time()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--MultThread",dest="MT",    type=int, default=16,   help="How many threads will be used (default = 16)" )
parser.add_argument("-d", "--Direction", dest="dir",   type=str, default='./Application_Samples/', help="Direction of the training samples (default = ./Application_Samples/)")
args = parser.parse_args()

if args.MT != 0:
    R.EnableImplicitMT(args.MT)
    print('using ' + str(args.MT) + ' threads in MT Mode!')
else:
    print('Disable MT Mode!')

path = os.path.abspath(args.dir)
_file_pool = os.listdir(path)
file_pool = []
for _file in _file_pool:
    #if _file in file_list_all:
        #for _root in os.listdir(path+'/'+_file):
            #if '.root' in _root:
                #file_pool.append(path+'/'+_file+'/'+_root)
                file_pool.append(path+'/'+_file)
    #endif
#endloop

XGBmodel = './XGB_Model/RwT_bdt.json'

xgbc = XGBClassifier()
xgbc.load_model(XGBmodel)
print('\033[1;33m {} models loaded \033[0m'.format(XGBmodel))

#def apply_c2v_bdt(file):
#    print("check0")
#    rdf_loop = R.RDataFrame('Events',file)
#    print("check1")
#    np_rdf_loop = rdf_loop.AsNumpy()
#    print("check2")
#    pd_rdf_loop = pd.DataFrame(np_rdf_loop)
#    print("check3")
#    
#    train_var = ['deta_2mu2bj', \
#                 'pt2bj_ov_m2bj', \
#                 'pt2bj_ov_mbbmm', \
#                 'pt2mu_ov_m2mu', \
#                 'scalar_hh_pt', \
#                ]
#    X = pd_rdf_loop[train_var]
#    print("check4")
#    score_xgb = xgbc.predict_proba(X)[:,1]
#    print('check5')
#        
#    data={"xgb_bdt_score" : score_xgb}
#    pd_score_xgb = pd.DataFrame(data)
#
#    post_pd_rdf_loop = pd.concat([pd_rdf_loop,pd_score_xgb],axis=1)
#    
#    data = post_pd_rdf_loop.to_dict('list')
#    print('check6')
#    for _key,_list in data.items():
#        data[_key] = np.array(_list)
#        print('check7')
#    print('check8')
#            
#    post_rdf_loop = R.RDF.MakeNumpyDataFrame(data)
#    post_rdf_loop.Snapshot('Events',file+'.dat')
#end
print(file_pool)

#with Pool(args.MT) as p:
#    print("check")
#    p.map(apply_c2v_bdt, file_pool)

for file in file_pool:
     rdf_loop = R.RDataFrame('Events',file)
     np_rdf_loop = rdf_loop.AsNumpy()
     pd_rdf_loop = pd.DataFrame(np_rdf_loop)
     print("check1")

     train_var = ['m_2mu', 'm_2bj', \
                  'ptmu0_ov_m2mu', 'pt2mu_ov_m2mu', 'pt2mu_ov_mbbmm', \
                  'ptbj0_ov_m2bj', 'pt2bj_ov_m2bj', 'pt2bj_ov_mbbmm', \
                  'mumu_pt', 'bjbj_pt', 'scalar_hh_pt', \
                  'deta_2mu2bj', 'deta_mubj_max', \
                  'mumu_eta', 'bjbj_eta', \
                  'dr_2mu', 'dr_2bj', \
                  'dphi_2mu', 'dphi_2bj', \
                  'ptmu1_ov_m2mu', 'ptbj1_ov_m2bj', \
                  'mu0_pt', 'mu1_pt', 'bj0_pt', 'bj1_pt', \
                  'mu0_eta', 'mu1_eta', 'bj0_eta', 'bj1_eta', \
                  'deta_2bj', 'deta_mubj', 'deta_mubj_other', \
                  'dr_2mu2bj', 'dphi_2mu2bj', 'dphi_mubj', 'dphi_vbfjj', \
                  'mumu_costheta', 'bjbj_costheta', \
                  'dr_mubj', 'dr_mubj_other', 'pt2mu_ov_pt2bj', \
                  'mu0_e', 'mu1_e', 'bj0_e', 'bj1_e', \
                  'mumu_e', 'bjbj_e', 'miss_MET', 'miss_eta', \
                  'deta_2mu', 'dr_mubj_min', 'dr_vbfjj', \
                  'hh_pt', 'm_bbmm', 'm_bbmm_corr', \
                 ]
    
     X = pd_rdf_loop[train_var]
     print("check2")
     score_xgb = xgbc.predict_proba(X)[:,1]
       
     data={"xgb_bdt_score" : score_xgb}
     pd_score_xgb = pd.DataFrame(data)

     post_pd_rdf_loop = pd.concat([pd_rdf_loop,pd_score_xgb],axis=1)
     print("check3")

     data = post_pd_rdf_loop.to_dict('list')
     print("check4")
     for _key,_list in data.items():
         data[_key] = np.array(_list)
            
     post_rdf_loop = R.RDF.MakeNumpyDataFrame(data)
     post_rdf_loop.Snapshot('Events',file+'.dat')
     print("check5")
 #endloop

time_end=timer.time()
print('\033[1;34m totally cost {} seconds. \033[0m'.format(str(time_end-time_start)))
exit(0)
