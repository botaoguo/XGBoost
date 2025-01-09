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

# import uproot
# dihiggs = uproot.concatenate(f"{basedir}/hh2b2w.root:tree",library="pd")
# ttbar = uproot.concatenate(f"{basedir}/ttbar.root:tree",library="pd")
# import root_numpy
# from array import array
import ROOT as R
from ROOT import TCut
import time as timer

from util import *
import sys
import os
import argparse

from multiprocessing import Pool

def plotter(_arg):
    print('\033[1;33m Variable :: {} Checked \033[0m'.format(_arg[0]))
    plot_vars_as_ROOT_ML('signal','background', _arg[0], _arg[1], _arg[2])
    
R.gROOT.SetBatch(True)
time_start=timer.time()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--MultThread" ,dest="MT",     type=int, default=16,   help="How many threads will be used (default = 16)" )
parser.add_argument("-d", "--Direction"  ,dest="dir",    type=str, default='./',   help="working direction (default = ./)" )
parser.add_argument("-i", "--inputFile"  ,dest="inFile", type=str, default='./Sample/train.root', help="Low Coupling value (default = ./Sample/zmerged.root)")
args = parser.parse_args()

if args.MT != 0:
    R.EnableImplicitMT(args.MT)
    print('using ' + str(args.MT) + ' threads in MT Mode!')
else:
    print('Disable MT Mode!')

path = os.path.abspath(args.dir)
os.system('mkdir -p SvB_Variables_Plots')
plot_path = os.path.abspath('./SvB_Variables_Plots')
file = path+'/'+args.inFile
print('\033[1;34m Reading Data... \033[0m')

rdf = R.RDataFrame('Events',file)

rdf_dict['signal'] = rdf.Filter('(dsid==002) && ((find_vbfj!=1 || (find_vbfj==1 && m_vbfjj<880)) && m_bbmm_corr>400)')
#rdf_dict['signal'] = rdf.Filter('dsid==002')

hi = rdf_dict['signal'].Histo1D('m_2mu','weight')
c = R.TCanvas()
hi.Draw()
c.SaveAs('test.png')
#exit(0)

rdf_dict['background'] = rdf.Filter('(dsid>215&&dsid<220) && ((find_vbfj!=1 || (find_vbfj==1 && m_vbfjj<880)) && m_bbmm_corr>400)')
#rdf_dict['background'] = rdf.Filter('dsid>215&&dsid<220')

variable("m_2mu", ";m(mm) [GeV];",     [20, 100., 180.])
variable("m_2bj", ";m(bb) [GeV];",     [20, 70., 190.])

variable("ptmu0_ov_m2mu",       ";pT(m0)/m(mm);",   [20, 0., 5.])
variable("ptmu1_ov_m2mu",       ";pT(m1)/m(mm);",   [20, 0., 5.])
variable("pt2mu_ov_m2mu",       ";pT(mm)/m(mm);",   [20, 0., 5.])
variable("pt2mu_ov_mbbmm", ";pT(mm)/m(bbmm);",     [20, 0., 1])

variable("ptbj0_ov_m2bj", ";pT(b0)/m(bb);",     [20, 0., 5.])
variable("ptbj1_ov_m2bj", ";pT(b1)/m(bb);",     [20, 0., 5.])
variable("pt2bj_ov_m2bj", ";pT(bb)/m(bb);",     [20, 0., 5.])
variable("pt2bj_ov_mbbmm", ";pT(bb)/m(bbmm);",     [20, 0., 1])

variable("mumu_pt",       ";p_{T}_mm [GeV];",             [30, 0., 600.])
variable("bjbj_pt",       ";p_{T}_bb [GeV];",             [30, 0., 600.])
variable("mumu_e",       ";E_mm [GeV];",             [30, 0., 600.])
variable("bjbj_e",       ";E_bb [GeV];",             [30, 0., 600.])
variable("mumu_eta",       ";#eta(mm);",             [20, 0., 5.])
variable("bjbj_eta",       ";#eta(bb);",             [20, 0., 5.])
variable("mumu_costheta",       ";#costheta(mm);",             [20, -1., 1.])
variable("bjbj_costheta",       ";#costheta(bb);",             [20, -1., 1.])

variable("scalar_hh_pt",       ";H_{T}_HH [GeV];",             [30, 0., 1200.])
variable("hh_pt",       ";p_{T}_HH [GeV];",             [30, 0., 300.])
variable("pt2mu_ov_pt2bj",       ";pT(mm)/pT(bb);",   [20, 0., 5.])
variable("m_bbmm_corr",       ";m_HH_corr [GeV];",         [20, 400., 1000.])
variable("m_bbmm",       ";m_HH [GeV];",         [20, 200., 1000.])
variable("miss_MET", ";miss_MET [GeV];",  [20,0,200])
variable("miss_eta", ";miss-#eta ;",  [20,0,10])

variable("mu0_pt",       ";p_{T}_m0 [GeV];",             [30, 0., 300.])
variable("mu1_pt",       ";p_{T}_m1 [GeV];",             [30, 0., 200.])
variable("bj0_pt",       ";p_{T}_b0 [GeV];",             [30, 0., 300.])
variable("bj1_pt",       ";p_{T}_b1 [GeV];",             [30, 0., 200.])
variable("mu0_e",       ";E_m0 [GeV];",             [30, 0., 300.])
variable("mu1_e",       ";E_m1 [GeV];",             [30, 0., 200.])
variable("bj0_e",       ";E_b0 [GeV];",             [30, 0., 300.])
variable("bj1_e",       ";E_b1 [GeV];",             [30, 0., 200.])
variable("mu0_eta",       ";#eta(m0);",             [20, 0., 2.4])
variable("mu1_eta",       ";#eta(m1);",             [20, 0., 2.4])
variable("bj0_eta",       ";#eta(b0);",             [20, 0., 2.4])
variable("bj1_eta",       ";#eta(b1);",             [20, 0., 2.4])

variable("deta_2mu2bj", ";#Delta #eta (HH) ;",     [20, 0., 5.])
variable("deta_mubj_max", ";#Delta #eta(mb)^{max} ;",     [20, 0., 5.])
variable("deta_2bj", ";#Delta #eta (bb) ;",     [20, 0., 5.])
variable("deta_2mu", ";#Delta #eta (mm) ;",     [20, 0., 5.])
variable("deta_mubj", ";#Delta #eta(mb) ;",     [20, 0., 5.])
variable("deta_mubj_other", ";#Delta #eta(mb)^{other} ;",     [20, 0., 5.])

variable("dr_mubj", ";#Delta R (b#mu) ;",     [20, 0., 5.])
variable("dr_mubj_other", ";#Delta R (b#mu)^{other} ;",     [20, 0., 5.])
variable("dr_mubj_min", ";#Delta R (b#mu)^{min} ;",     [20, 0., 5.])
variable("dr_2mu", ";#Delta R (mm) ;",     [20, 0., 5.])
variable("dr_2bj", ";#Delta R (bb) ;",     [20, 0., 5.])
variable("dr_2mu2bj", ";#Delta R (bb) ;",     [20, 0., 8.])
variable("dr_vbfjj", ";#Delta R (vbfjj) ;",     [20, 0., 8.])

variable("dphi_2mu", ";#Delta #phi (mm) ;",     [20, 0., 5.])
variable("dphi_2bj", ";#Delta #phi (bb) ;",     [20, 0., 5.])
variable("dphi_2mu2bj", ";#Delta #phi (HH) ;",     [20, 0., 5.])
variable("dphi_mubj", ";#Delta #phi (b#mu) ;",     [20, 0., 5.])
variable("dphi_vbfjj", ";#Delta #phi (vbfjj) ;",     [20, 0., 5.])

print('\033[1;34m Plotting Variables... \033[0m')
#exit(0)
arg_list = []
for key, items in variables.items():
    arg_list.append([key, items, plot_path])

with Pool(args.MT) as p:
    p.map(plotter, arg_list)
    
#exit(0)

print('\033[1;34m Preparing training matrix... \033[0m')

np_rdf_dict['signal'] = rdf_dict['signal'].AsNumpy()
np_rdf_dict['background'] = rdf_dict['background'].AsNumpy()

pd_rdf_dict['signal'] = pd.DataFrame(np_rdf_dict['signal'])
pd_rdf_dict['background'] = pd.DataFrame(np_rdf_dict['background'])

pd_rdf_dict['signal'].loc[:, "isSig"] = True
pd_rdf_dict['signal'].loc[:, "isBkg"] = False
pd_rdf_dict['background'].loc[:, "isSig"] = False
pd_rdf_dict['background'].loc[:, "isBkg"] = True

all_data = pd.concat([pd_rdf_dict['signal'],pd_rdf_dict['background']],axis=0)

#train_var = ['m_2mu','m_2bj','ptmu0_ov_m2mu','pt2mu_ov_m2mu','pt2mu_ov_mbbmm','weight']
#corr_test_var = ['m_2mu','m_2bj','ptmu0_ov_m2mu','pt2mu_ov_m2mu','pt2mu_ov_mbbmm','isSig']

train_var = [    'm_2mu', 'm_2bj', \
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
                 'weight'
]

corr_test_var = ['m_2mu', 'm_2bj', \
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
                 'isSig'
]

corr_test = all_data[corr_test_var]
_bkg = corr_test.loc[corr_test["isSig"] == False]
_sig = corr_test.loc[corr_test["isSig"] == True]

correlations(_bkg.drop('isSig',axis=1), 'bkg')
correlations(_sig.drop('isSig',axis=1), 'sig')
del corr_test, _bkg, _sig

X = all_data[train_var]
y = all_data[['isSig','isBkg']]

test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
w_train = X_train['weight']
w_test = X_test['weight']

pos_scale_count = dict(y_train['isSig'].value_counts())
print(pos_scale_count)
pos_scale = pos_scale_count[False]/pos_scale_count[True]
print('pos_scale = '+str(pos_scale))

X_train.drop('weight',axis=1,inplace=True)
X_test.drop('weight',axis=1,inplace=True)

print('\033[1;33m Splitting train : test = {0} : {1} \033[0m'.format(str(1-test_size),str(test_size)))
print('\033[1;33m Start training : \033[0m')

xgbc = XGBClassifier(
    booster='gbtree',
    colsample_bytree=0.7, #prevent over training (0.5~1)
    colsample_bylevel=0.7,
    colsample_bynode=0.7,
    reg_alpha=3, #accelerate at high dimention 0
    reg_lambda=5, #prevent over training 1
    subsample=0.7, #prevent over training
    max_depth=3,
    n_estimators=2500,
    learning_rate =0.08,
    nthread=args.MT,
    scale_pos_weight=pos_scale,
)

eval_set = [(X_train, y_train), (X_test, y_test)]
xgbc.fit(
    X=X_train, y=y_train['isSig'], 
#    sample_weight=w_train.map(lambda x: x if x>0 else 0),
#     eval_set = eval_set,
    eval_metric=["error", "logloss"], 
    verbose=True
)

test_score_xgb = xgbc.predict_proba(X_test)
test_score_xgb_sig = test_score_xgb[y_test['isSig']][:,1]
test_score_xgb_bkg = test_score_xgb[y_test['isBkg']][:,1]

train_score_xgb = xgbc.predict_proba(X_train)
train_score_xgb_sig = train_score_xgb[y_train['isSig']][:,1]
train_score_xgb_bkg = train_score_xgb[y_train['isBkg']][:,1]

print('\033[1;33m Plotting supporting figures \033[0m')
w_train_score_sig = w_train[y_train['isSig']].values
w_train_score_bkg = w_train[y_train['isBkg']].values
w_test_score_sig = w_test[y_test['isSig']].values
w_test_score_bkg = w_test[y_test['isBkg']].values

decisions = []
decisions += [test_score_xgb_sig,test_score_xgb_bkg,train_score_xgb_sig,train_score_xgb_bkg]

low = min(np.min(d) for d in decisions)
high = max(np.max(d) for d in decisions)
low_high = (low,high)
bins = 15

sig_train,bins,patches = plt.hist(train_score_xgb_sig,weights=w_train_score_sig,
         color='r', alpha=0.5, range=low_high, bins=bins,
         histtype='stepfilled', density=True,
         label='signal (train)'
)
print(sig_train)

bkg_train,bins,patches = plt.hist(train_score_xgb_bkg,weights=w_train_score_bkg,
         color='b', alpha=0.5, range=low_high, bins=bins,
         histtype='stepfilled', density=True,
         label='background (train)'
)
print(bkg_train)

hist, bins = np.histogram(test_score_xgb_sig, weights=w_test_score_sig, bins=bins, range=low_high, density=True)
scale = len(test_score_xgb_sig) / sum(hist)
err   = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='signal (test)')
sig_test = hist
print(sig_test)

hist, bins = np.histogram(test_score_xgb_bkg, weights=w_test_score_bkg, bins=bins, range=low_high, density=True)
scale = len(test_score_xgb_bkg) / sum(hist)
err   = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='background (test)')
bkg_test = hist
print(bkg_test)

plt.legend(loc='best',frameon=True,edgecolor='blue',facecolor='blue') 
plt.legend()
plt.xlabel("BDT output")
plt.ylabel("Arbitrary units")
plt.text(0.3,np.max(bkg_test)*0.9,'K-S prob sig(bkg) = {:0.3f}({:0.3f})'.format(R.TMath.KolmogorovTest(len(sig_train),sig_train,len(sig_test),sig_test,"D"),R.TMath.KolmogorovTest(len(bkg_train),np.array(bkg_train[::-1],dtype=float),len(bkg_test),np.array(bkg_test[::-1],dtype=float),"D")),fontsize=10)
# plt.xlim(0,1)
#plt.yscale('log')
save_plot_batch('./Test_Distribution.png')

plot_importance(xgbc)
save_plot_batch('./Variable_importance.png')

fpr_xgb, tpr_xgb, _thres = metrics.roc_curve(y_test['isSig'],test_score_xgb[:,1])

AUC = 0
for i in range(len(fpr_xgb)-1):
    AUC=AUC+(fpr_xgb[i+1]-fpr_xgb[i])*(tpr_xgb[i]+tpr_xgb[i+1])
auc=0.5*AUC

f, ax = plt.subplots(figsize=(8, 8))
ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), '--', label='random')
ax.plot(fpr_xgb, tpr_xgb, label='XGBoost Classifier')
plt.text(0.6,0.1,'AUC = {:0.3f}'.format(auc),fontsize=18)
ax.set_yscale('linear'); ax.set_xlim(0, 1); ax.set_ylim(1e-3, 1)
ax.set_xlabel('target efficiency', ha='right', x=1.0); ax.set_ylabel('sea efficiency', ha='right', y=1.0)
ax.legend()
save_plot_batch('./ROC_curve.png')

os.system('mkdir -p XGB_Model')
xgbc.save_model('XGB_Model/RwT_bdt.json')
print('\033[1;33m Saving models to XGB_Model \033[0m')

time_end=timer.time()
print('\033[1;34m totally cost {} seconds. \033[0m'.format(str(time_end-time_start)))
exit(0)
