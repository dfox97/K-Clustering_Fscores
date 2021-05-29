K Means and K Medians algorithms

Libaries used:
import numpy as np
import matplotlib.pyplot as plt
import csv  

Note: Used pandas to print out a nice clear table for comparing different k clusters for question 7 (This is commented out in the code):
import pandas as pd

Everything should be setup and run automatically:

To change questions there is a varibale called SELECT_QUESTION which you can set the variable between 1-6 for each question and then an additional function called "Panda-Compare" which will loop through questions 3-6 printing BCUBED scores and plots for each 
I have commented out the pandas dataframe at the end of the "Panda-Compare" loop incase it was not allowed, it would print out a table comparing the f-score results which helped draw comparisons for question 7.

Selecting questions 3 to 6 will just print out the BCUBED results for that question and its related graph. 
Selecting "Panda-Comapre" will print out questions 3-6 presenting BCUBED and graphs one by one.

##################################################################################################
Param Manual setup:

set the method to "mean" or "median"
set the norm to True or False

Example:
BCUBED(kClusteringAlgorithm(data, k=4,method="mean",norm=False),ani=a,country=c,fruits=f,veg=v) #NO GRAPH 
#use the loopResults function for graphs examples can be seen below.

##################################################################################################

##################################################################################################
The program default will be set at Panda-Compare which compares questions 3-6 (printing 1 question at a time along with the graph of the question):
SELECT_QUESTION = "Panda-Compare"

if SELECT_QUESTION = "Panda-Compare":
        norm=False
        method="mean"
        list_f=loopResults(data,norm=norm,method=method)

        norm=True
        method="mean"
        list_f_norm=loopResults(data,norm=norm,method=method)

        norm=False
        method="median"
        median_list=loopResults(data,norm=norm,method=method)

        norm=True
        method="median"
        median_list_norm=loopResults(data,norm=norm,method=method)

	#UNCOMMENT THIS AND IMPORT PANDAS TO REPLICATE THE K F-SCORE COMPARISON TABLE USED IN QUESTION 7
        # counter=[]
        # for i in range(1,10):
        #     counter.append(i)
        
        # data=list(zip(list_f,list_f_norm,median_list,median_list_norm))
        # print("---Seed 10--------------")
        # print("----Mean F-score--------Median F-Scores-----")
        # print(pd.DataFrame(data,index=counter,columns=['Mean','Normalised','Median','Normalised']))

if SELECT_QUESTION==1:
        K_VAL=4
        method="mean"
        norm=False
        P,R,F = BCUBED(kClusteringAlgorithm(data, k=K_VAL,method=method,norm=norm,).run(),ani=a,country=c,fruits=f,veg=v)

if SELECT_QUESTION==2:
        K_VAL=4
        method="median"
        norm=False
        P,R,F=BCUBED(kClusteringAlgorithm(data, k=K_VAL,method=method,norm=norm,).run(),ani=a,country=c,fruits=f,veg=v)
        
if SELECT_QUESTION == 3:
        norm=False
        method="mean"
        loopResults(data,norm=norm,method=method)
if SELECT_QUESTION == 4:
        norm=True
        method="mean"
        loopResults(data,norm=norm,method=method)
if SELECT_QUESTION == 5:
        norm=False
        method="median"
        loopResults(data,norm=norm,method=method)
if SELECT_QUESTION == 6:  
        norm=True
        method="median"
        loopResults(data,norm=norm,method=method)
##################################################################################################


