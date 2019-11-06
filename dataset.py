# -*- coding: utf-8 -*-

#%% IMPORTS
import numpy as np
import time
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Transforms
import random
from tqdm import tqdm 

from utils import Paths
import sys
paths = Paths()

NORMAL_PATH = paths.normal_images_path_50
ANOM_PATH = paths.anomalous_images_path_50
#%%
#def loadData(opt):
#    
#    train_data = []
#    train_label = []
#    validation_data = []
#    validation_label = []
#    
##    patches_per_image = opt.patch_per_im
#    
#    n_anomaly = 0
#    
#    NORMAL_LABEL = np.float64(0)
#    ANOMALY_LABEL = np.float64(1)
#    
#    sequence = [i for i in range(0, opt.endFolder)]    
#    folder_indexes = random.sample(sequence, len(sequence))
#    
#    valid_perc = 1 - opt.split
#    anomaly_perc = valid_perc * 0.5
#    valid_perc = anomaly_perc + opt.split
#
#    
#    countFolders = 0
#    
#    start = time.time()
#    
#    for index in folder_indexes:
#        print('Loading images')
#        countFolders +=1
#        countSplit = 0
##        print(1)
#        if(countFolders <= opt.nFolders):
#            print('\nFolder n.{}\n'.format(countFolders))
#            path_files = paths.normal_images_path + str(index) + '/'
#            n_files = len(os.listdir(path_files))
#                        
#            for filename in tqdm(os.listdir(path_files)):
#                countSplit += 1
#                path_filename = path_files + filename
#                
##                image = Image.open(path_filename)
#                image = cv2.imread(path_filename)
##                image = image.flatten()
#
#                # TRAINING : split-0.1      VALIDATION : 1-(split-0.1)
#                if(countSplit / n_files < opt.split):
#                    train_data.append(image)
#                    train_label.append(NORMAL_LABEL)
#                elif(countSplit / n_files > opt.split and countSplit / n_files < valid_perc):
#                    validation_data.append(image)
#                    validation_label.append(NORMAL_LABEL)
#                else:
#                    n_anomaly += 1
##                    pass
#                       
#            print('\n{} images loaded'.format(countSplit)) 
#            print('Val images: ', len(validation_data))        
#        else:
##            print(anomaly_perc)
##            print(n_files)
##            n_anomaly = int(n_files * anomaly_perc)
#            print('Anomalous Images: ', n_anomaly)
#            break
#        
#    print('Normal Images Loaded')    
#    count = 0
#    for index in range(0, n_anomaly):
#        
#        path_files = paths.anomalous_images_path + str(index) + '/'
#        countFile = 0
#        for filename in os.listdir(path_files):
#            count += 1
#            countFile += 1
#            if(count <= n_anomaly):
#                    
#                path_filename = path_files + filename
#            
#    #            image = Image.open(path_filename)
#                image = cv2.imread(path_filename)
##                image = image.flatten()
#
#                validation_data.append(image)
#                validation_label.append(ANOMALY_LABEL)
#            else:
#                print('Anomalous Images Loaded') 
#                
#                end = time.time()
#                print('{} anomalous images loaded in {} sec'.format(countFile, end-start))
#                
#                return train_data, train_label, validation_data, validation_label
#            
def newLoadData(opt):
    
    train_data = []
    train_label = []
    validation_data = []
    validation_label = []
    
    patches_per_image = opt.patch_per_im
    
    n_anomaly = 0
    
    NORMAL_LABEL = np.float64(0)
    ANOMALY_LABEL = np.float64(1)
    
    sequence = [i for i in range(0, opt.endFolder)]    
    folder_indexes = random.sample(sequence, len(sequence))
    
    training_index = int(patches_per_image * opt.split)
    normal_val_index = training_index + int(patches_per_image*(1-opt.split)*0.5)
    anom_val_index = int(patches_per_image * (1-opt.split) * 0.5)

    counter = 0
    start = time.time()
    sequence = [i for i in range(0, opt.endFolder)]    
    folder_indexes = random.sample(sequence, len(sequence))
    
    print('\nPatches per image: ', patches_per_image)
    for index in folder_indexes[0: opt.nFolders]:
        counter += 1
        print('\n')
        print('Image n.', counter)
        
        path_normal = NORMAL_PATH + str(index) + '/'
        path_anom = ANOM_PATH + str(index) + '/'
        
        norm_filename = os.listdir(path_normal)
        random.shuffle(norm_filename)

        anom_filename = os.listdir(path_anom)
        random.shuffle(anom_filename)
                    
        # TRAINING SET
        for filename in tqdm(norm_filename[0 : training_index], leave=True, desc='Training-set', file=sys.stdout):

            image = cv2.imread(path_normal + filename)

            train_data.append(image)
            train_label.append(NORMAL_LABEL)
        
        # VALIDATION(NORMAL)
        for filename in tqdm(norm_filename[training_index : normal_val_index], leave=True, desc='Val_norm-set',
                             file=sys.stdout):
            
            image = cv2.imread(path_normal + filename)
        
            validation_data.append(image)
            validation_label.append(NORMAL_LABEL)
        
        # VALIDATION(ANOMALY) 
        for filename in tqdm(anom_filename[0 : anom_val_index], leave=True, desc='Val_anom-set',file=sys.stdout):
            n_anomaly += 1
            image = cv2.imread(path_anom + filename)
        
            validation_data.append(image)
            validation_label.append(ANOMALY_LABEL)
        
    print('\n')
    print('Training set: {} images'.format(len(train_data)))
    print('Validation set: {} images [{} normal and {} anomalous]'.format(len(validation_data) ,
                                                                          len(validation_data)-n_anomaly,
                                                                          n_anomaly))
    end = time.time()
    print('Spent time : {} sec'.format(end - start))
    return train_data, train_label, validation_data, validation_label
#%%
#def toNumpy(array):
#    
#    npArray = np.copy(array)
#    
#    return npArray

def generateDataloader(opt):
    print('\n>Loading Steel Dataset')
    
    if(opt.loadedData == False):
        opt.loadDatasets()
    
    dataset = {}
    dataset['train']       = SteelDataset(opt, train=True)
    dataset['validation']  = SteelDataset(opt, train=False)
    
    shuffle = {'train':True, 'validation':False}
    
    dataloader = {x: DataLoader(dataset    = dataset[x],
                                batch_size = opt.batch_size,
                                drop_last  = True,
                                shuffle = shuffle[x],
                                num_workers= opt.n_workers
                                )
                  
                  for x in ['train', 'validation']}
    
    return dataloader

def collectNormalSamples(nImages, normal_per_img=None):
    n_normal = 0
    counterImage = 0
    limit = 0

    normalTest = []
    
    listNormImages = os.listdir(NORMAL_PATH)
    
    sequence = [i for i in range(len(listNormImages))]    
    folder_indexes = random.sample(sequence, len(sequence))
    
    for index in folder_indexes[0 : nImages]:
        counterImage += 1
        path_anom = NORMAL_PATH + str(index) + '/'
        normImages = os.listdir(path_anom)
        
        print('\nImage n.', counterImage)
        
        if(normal_per_img is None):
            normal_per_img = len(normImages)
        
        limit = 0
#        print(len(normImages))
        for filename in tqdm(normImages, leave=True, desc='Anom. Images:\t', 
                             total=normal_per_img, file=sys.stdout):
            
            cond1 = normal_per_img is not None and limit < normal_per_img
            cond2 = normal_per_img is None
            
            if(cond1 or cond2):
                n_normal += 1
                limit += 1
                image = cv2.imread(path_anom + filename)
                normalTest.append(image)
                
        
            
    
    print('> {} anomalous images loaded'.format(n_normal))
    
    return normalTest
    
    
def collectAnomalySamples(nImages, anom_per_img=None):
    n_anomaly = 0
    counterImage = 0
    limit = 0

    anomalyTest = []
    
    listAnomImages = os.listdir(ANOM_PATH)
    
    sequence = [i for i in range(len(listAnomImages))]    
    folder_indexes = random.sample(sequence, len(sequence))
    
    for index in folder_indexes[0 : nImages]:
        counterImage += 1
        path_anom = ANOM_PATH + str(index) + '/'
        anomImages = os.listdir(path_anom)
        
        print('\nImage n.', counterImage)
        
        if(anom_per_img is None):
            anom_per_img = len(anomImages)
        
        limit = 0
        
        for filename in tqdm(anomImages, leave=True, desc='Anom. Images:\t', 
                             total=anom_per_img, file=sys.stdout):
            
            cond1 = anom_per_img is not None and limit < anom_per_img
            cond2 = anom_per_img is None
            
            if(cond1 or cond2):
                n_anomaly += 1
                limit += 1
                image = cv2.imread(path_anom + filename)
                anomalyTest.append(image)
            else:
                pass
    
    print('> {} anomalous images loaded'.format(n_anomaly))
    
    return anomalyTest

class SteelDataset(Dataset):
    
    def __init__(self, opt, train=True):
        
        if(train):
            self.data = opt.train_data
            self.targets = opt.train_targets
        else:
            self.data = opt.validation_data
            self.targets = opt.validation_targets
        
#        print(self.data[0].shape)
#        print(np.vstack(self.data).shape)

        self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
        print(self.data.shape)
#        self.data = self.data.transpose((0, 2, 3, 1))
           
#        plt.imshow(prova)
#        plt.imshow(self.data[0])
#        return
        
        self.transforms = self._initTransforms(opt)

    def _initTransforms(self, opt):
        if(opt.transforms is None):
            transforms = Transforms.Compose(
                                [
#                                    transforms.Resize(32, interpolation=Image.BILINEAR),
                                    Transforms.Grayscale(num_output_channels = opt.in_channels),
                                    Transforms.ToTensor(),
                                    Transforms.Normalize((0.5,),
                                                         (0.5,)),
#                                    Transforms.Grayscale(num_output_channels = opt.in_channels)
#                                    transforms.ToPILImage()
                                ]
                            )
                                
        else:
            transforms = opt.transforms
        
        return transforms
    
    def __getitem__(self, index):
       
        if torch.is_tensor(index):
            index = index.tolist()
            
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        
        # GRAYSCALE
#        image = image.convert('LA')
           
        image = self.transforms(image)
#        print(image.shape)
        return image, target
    
    def __len__(self):
        return len(self.data)


#%%
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

def getCifar10(opt):
    
    print('>Loading Cifar Dataset')
    
    splits = ['train', 'test']
    shuffle = {'train': True, 'test': False}
    drop_last = {'train': True, 'test': True}
    transform = transforms.Compose(
        [
    #                transforms.Resize(opt.isize),
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    classes = {
        'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    }
    
    dataset = {}
    dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    #a = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    dataset['train'].data, dataset['train'].targets, \
    dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
        trn_img=dataset['train'].data,
        trn_lbl=dataset['train'].targets,
        tst_img=dataset['test'].data,
        tst_lbl=dataset['test'].targets,
    #            abn_cls_idx=classes[opt.abnormal_class],
        abn_cls_idx=classes['car'],
    #            manualseed=opt.manualseed
    )
    
    dataset['train'].targets = dataset['train'].targets.astype('float64')
    dataset['test'].targets = dataset['test'].targets.astype('float64')
    
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batch_size,
                                                 shuffle=shuffle[x],
    #                                                     num_workers=int(opt.workers),  
                                                 drop_last=drop_last[x],
    #                                                     worker_init_fn=(None if opt.manualseed == -1
    #                                                     else lambda x: np.random.seed(opt.manualseed))
                                                )
                  for x in splits}
    
    return dataloader


def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

