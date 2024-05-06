#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:29:36 2024

@author: rasaneno

This script calculates the proportion of utterances in the generated text data 
(or original CHILDES) that are unique compared to (other) CHILDES.
"""


import numpy as np
import os, glob,csv


# Function to read all .txt files from a directory and its subdirectories
def read_txt_files(directory):
    all_text = ""
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                with open(os.path.join(root, filename), 'r') as file:
                    all_text += file.read().replace('\n', ' ')
    return all_text


# Function to count occurrences of a substring
def count_occurrences(substring, text):
    return text.count(substring)

# Paths

refdatadir = '/Users/rasaneno/speechdb/aochildes_2021/CHILDES_ao_dataset_1word/'

source_basename = '/Users/rasaneno/rundata/GILES_CogSci/txt_out/agetests_cogsci_final_1word/GILES_CHILDES_agecond_final_1word_5layers_512emb_512ff_8head_8000voc_5do_500k_100t_'

outputpath = '/Users/rasaneno/rundata/GILES_CogSci/overlaps_500k_1word.csv'

pattern = source_basename + '*.txt'

file_list = glob.glob(pattern)

file_list.sort()

all_text_data = read_txt_files(refdatadir)
all_text_data = all_text_data.lower()


outputpath_newwords = '/Users/rasaneno/rundata/GILES_CogSci/newwords/'


os.remove(outputpath_newwords + '/uttlen1.txt')
os.remove(outputpath_newwords + '/uttlen4.txt')
os.remove(outputpath_newwords + '/uttlen8.txt')
    

ages_to_analyze = [3,6,9,12,15,18,21,24,36,48]


UL = []
UC = []
UQ = []
UA = []

for source_file in file_list:
    
    
    # Read all text data from .txt files
    
    agebin=np.int32(source_file[-6:-4])
    
    if(sum(ages_to_analyze == agebin)):
    
    
        with open(outputpath_newwords + '/uttlen1.txt', 'a', newline='') as file:
            file.write("{}".format(agebin) + ' months \n')   
    
        with open(outputpath_newwords + '/uttlen4.txt', 'a', newline='') as file:
            file.write("{}".format(agebin) + ' months \n')   
    
        with open(outputpath_newwords + '/uttlen8.txt', 'a', newline='') as file:
            file.write("{}".format(agebin) + ' months \n')   
            
    
           
        
        
        # Read and split data from source file
        with open(source_file, 'r') as file:
            data = file.read().replace('\n', '').split('.')
        
        L = len(data)
        N = 10000
        order = np.random.permutation(L)
        result = []
        N_words = []
        
        for i in range(N):
            string_to_search = data[order[i]]
            string_to_search = string_to_search.strip()
            string_to_search = string_to_search.lower()
            N_words.append(len(string_to_search.split()))
            
            # Count occurrences in the loaded text data
            count = count_occurrences(string_to_search, all_text_data)
            result.append(count)
            
            if count == 0 and len(string_to_search.split()) == 1:
                print(string_to_search)
                with open(outputpath_newwords + '/uttlen1.txt', 'a', newline='') as file:
                    file.write(string_to_search + '\n')   
            if count == 0 and len(string_to_search.split()) == 4:
                print(string_to_search)
                with open(outputpath_newwords + '/uttlen4.txt', 'a', newline='') as file:
                    file.write(string_to_search + '\n')   
            if count == 0 and len(string_to_search.split()) == 8:
                print(string_to_search)
                with open(outputpath_newwords + '/uttlen8.txt', 'a', newline='') as file:
                    file.write(string_to_search + '\n')   
                
    
    
                
        
        # Now 'result' contains the count of each string occurrence
        
               
        r = np.int32(result)
        N_w = np.int32(N_words)
        
        prop_unique = sum(r == 0)/len(r)
        
        prop_uq = np.zeros(np.max(N_w)) * np.NAN
        count_uq = np.zeros(np.max(N_w))
        
        for uttlen in range(1,np.max(N_w)):
            tmp = r[N_w == uttlen]
            count_uq[uttlen] = len(tmp)
            if(len(tmp)):
                prop_uq[uttlen] = sum(tmp == 0)/len(tmp)
            
            print("Length " + str(uttlen) + ": " + "{:.2f}%".format(prop_uq[uttlen]*100)+ " (N = " + "{:.0f}".format(count_uq[uttlen]) + ")")
            
            UL.append(uttlen)
            UQ.append(prop_uq[uttlen]*100)
            UC.append(count_uq[uttlen])
            UA.append(agebin)
            
           
        print("Overall: " + "{:.2f}".format(prop_unique) + " (" + "{}".format(agebin) +" mo)" )
    
    

with open(outputpath, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header (optional)
    writer.writerow(['Age', 'Uttlen', 'Unique', 'Count'])

    # Write the data
    for row in zip(UA, UL, UQ, UC):
        writer.writerow(row)        
 

# Check self-reps of CHILDES against CHILDES



outputpath = '/Users/rasaneno/rundata/GILES_CogSci/overlaps_CHILDES_1word_fixed.csv'
refdatadir = '/Users/rasaneno/speechdb/aochildes_2021/CHILDES_ao_dataset_1word/'


ages_to_test = [6,9,12,15,18,21,24,36,48]

UL = []
UC = []
UQ = []
UA = []
  

for i in range(0,len(ages_to_test)):
    
    source_basename = '/Users/rasaneno/rundata/GILES_CogSci/reference_data_CHILDES_1word/childes_age_0' + "{:02d}/".format(ages_to_test[i])
    

    
    pattern = source_basename + '*.txt'
    
    file_list = glob.glob(pattern)
    
    file_list.sort()
    
    all_text_data = read_txt_files(refdatadir)
    all_text_data = all_text_data.lower()
    
    
  
    for source_file in file_list:
        
               
        # Read all text data from .txt files
        
        agebin = np.int32(os.path.split(source_file)[0][-2:])
        
        if(sum(ages_to_analyze == agebin)):       
            
            # Read and split data from source file
            with open(source_file, 'r') as file:
                data = file.read().replace('\n', '').split('.')
            
            L = len(data)
            N = np.min([5000,L])
            order = np.random.permutation(L)
            result = []
            N_words = []
            
            for i in range(N):
                string_to_search = data[order[i]]
                string_to_search = string_to_search.strip()
                string_to_search = string_to_search.lower()
                N_words.append(len(string_to_search.split()))
                
                # Count occurrences in the loaded text data
                count = count_occurrences(string_to_search, all_text_data)
                result.append(count-1)
                
                if count == 1 and len(string_to_search.split()) == 1:
                    print(string_to_search)
                    with open(outputpath_newwords + '/uttlen1_CHILDES.txt', 'a', newline='') as file:
                        file.write(string_to_search + '\n')   
                if count == 1 and len(string_to_search.split()) == 4:
                    print(string_to_search)
                    with open(outputpath_newwords + '/uttlen4_CHILDES.txt', 'a', newline='') as file:
                        file.write(string_to_search + '\n')   
                if count == 1 and len(string_to_search.split()) == 8:
                    print(string_to_search)
                    with open(outputpath_newwords + '/uttlen8_CHILDES.txt', 'a', newline='') as file:
                        file.write(string_to_search + '\n')   
              
                    
            # Now 'result' contains the count of each string occurrence
                   
            r = np.int32(result)
            N_w = np.int32(N_words)
            
            prop_unique = sum(r == 0)/len(r)
            
            prop_uq = np.zeros(np.max(N_w)) * np.NAN
            count_uq = np.zeros(np.max(N_w))
            
            for uttlen in range(1,np.max(N_w)):
                tmp = r[N_w == uttlen]
                count_uq[uttlen] = len(tmp)
                if(len(tmp)):
                    prop_uq[uttlen] = sum(tmp == 0)/len(tmp)
                
                print("Length " + str(uttlen) + ": " + "{:.2f}%".format(prop_uq[uttlen]*100)+ " (N = " + "{:.0f}".format(count_uq[uttlen]) + ")")
                
                UL.append(uttlen)
                UQ.append(prop_uq[uttlen]*100)
                UC.append(count_uq[uttlen])
                UA.append(agebin)
                
               
            print("Overall: " + "{:.2f}".format(prop_unique) + " (" + "{}".format(agebin) +" mo)" )
            
        
    
with open(outputpath, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header (optional)
    writer.writerow(['Age', 'Uttlen', 'Unique', 'Count'])

    # Write the data
    for row in zip(UA, UL, UQ, UC):
        writer.writerow(row)        
 

