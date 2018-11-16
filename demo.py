#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: demo.py
Author: leowan(leowan)
Date: 2018/11/16 16:14:36
"""

import os
import shutil

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from tffmclassifier import TFFMClassifier

def test_dense(dataset):
    """
        Unittest dense predict
    """
    tffm = TFFMClassifier(factor_num=10, batch_size=1, print_step=100, input_type='dense')
    X = dataset[0]
    y = dataset[1]
    tffm.fit(X, y)
    logits = tffm.decision_function(X)
    pred = tffm.predict(X)
    print(accuracy_score(y, pred))
    print(confusion_matrix(y, pred))
    print(roc_auc_score(y, logits))

def test_sparse(dataset):
    """
        Unittest dense predict
    """
    tffm = TFFMClassifier(factor_num=10, batch_size=1, print_step=100, input_type='sparse')
    X = sparse.csr_matrix(dataset[0])
    y = dataset[1]
    tffm.fit(X, y)
    logits = tffm.decision_function(X)
    pred = tffm.predict(X)
    print(accuracy_score(y, pred))
    print(confusion_matrix(y, pred))
    print(roc_auc_score(y, logits))

def test_load_chkpt(dataset):
    """
        Unittest load checkpoint
    """
    chkpt_dir = './tmp'
    if os.path.exists(chkpt_dir):
        shutil.rmtree(chkpt_dir)

    tffm = TFFMClassifier(factor_num=10, batch_size=1, input_type='dense',
        chkpt_dir=chkpt_dir)
    X = dataset[0]
    y = dataset[1]
    tffm.fit(X, y)
    logits = tffm.decision_function(X)
    pred = tffm.predict(X)
    print('train acc: {}'.format(accuracy_score(y, pred)))
    lr_load = TFFMClassifier(input_type='dense')
    lr_load.load_checkpoint(feature_num=X.shape[1], chkpt_dir=chkpt_dir)
    print('loaded acc: {}'.format(accuracy_score(y, pred)))

def test_export(dataset, input_type='dense'):
    """
        Unittest import / export Pb model
    """
    export_dir = './tmp'
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    tffm = TFFMClassifier(factor_num=10, batch_size=1, input_type=input_type)
    X = dataset[0]
    if input_type == 'sparse':
        X = sparse.csr_matrix(X)
    y = dataset[1]
    tffm.fit(X, y)
    logits = tffm.decision_function(X)
    print('train auc: {}'.format(roc_auc_score(y, logits)))
    tffm.export_model(export_dir, input_type)
    logits = tffm.decision_function_imported(X, export_dir, input_type)
    print('loaded auc: {}'.format(roc_auc_score(y, logits)))

if __name__ == "__main__":
    # iris data
    from sklearn import datasets
    dataset_ori = datasets.load_iris(return_X_y=True)
    y_label = map(lambda x: x == 0, dataset_ori[1])
    dataset = []
    dataset.append(dataset_ori[0])
    dataset.append(np.array(list(y_label)).astype(int))

    test_dense(dataset)
    test_sparse(dataset)
    test_load_chkpt(dataset)
    test_export(dataset, input_type='dense')
    test_export(dataset, input_type='sparse')
