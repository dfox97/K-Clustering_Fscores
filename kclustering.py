"""
Name: Daniel Fox
StudentID: 201278002
Assignment 2 : K-Means and K-Medians 
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

#Uncomment to print out nice table for "Panda-Compare" F-scores
#import pandas as pd 

def conversion(list_name):
    """
    reads data using csv import 
    convert the data into an array.
    add a new column caled category to track all the data 
    combines the new_data (category) to the dataset

    """
    data = open(list_name, 'rt')
    data = csv.reader(data, delimiter=' ')
    data = list(data)
    data = np.array(data)
  
    # #add new column for a category
    category = np.array([list_name])
    new_data = np.tile(category[np.newaxis,:], (data.shape[0],1))
    dataset = np.concatenate((data,new_data), axis=1)
   
    return dataset

def parseData(lists):
    """
    Main function to read data and sort 
    lists=['animals','countries','fruits','veggies']
    conversion is a function called for each file name in list.
    concatenate the dataset and combine all data and return it.

    """
    animals=conversion(lists[0])
    countries = conversion(lists[1])
    fruits = conversion(lists[2])
    veggies = conversion(lists[3])

    #combining
    dataset = np.concatenate((animals,countries), axis=0)
    dataset = np.concatenate((dataset,fruits), axis=0)
    dataset = np.concatenate((dataset,veggies), axis=0)

    return dataset


def maxCategoryIndex(dataset,category):
    """
    Used for B-CUBED to find the total amount in index catergory
    """
    maxIndex= np.where(dataset[:,-1]==category)[0][-1]
    return maxIndex    


class kClusteringAlgorithm():
    def __init__(self,data,k,method,norm):
        """
        K Clustering algorithm which can operate as mean or medians
        args:
        data=full dataset for k clustering 
        k=k the amount of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.
        method= choose between means or medians for k-method clustering algorithm.
        randomise=call randomise function which shuffles the dataset 
        norm=default (false) is normalisation l2. If True perform l2 normalisation for the data.Normalise each row so that the sum of each row is equal 1.

        kmeans:
            k-means unsupervised  learning  algorithm  that  solve  the clustering problem. The procedure follows a simple way  to classify a given data set  through a certain number of  clusters (k-clusters). It defines k number of centers, one for each cluster. Take each point belonging  to a  given data set and associate it to the nearest center. 
            Assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more similar the data points are within the same cluster thus assuming that those points belong to the same label.

            K means methos involves using the Euclidean Distance to find the distance between two datapoints
        
        kmedians:
            Instead of calculating the mean for each cluster to determine its centroid, one instead calculates the median. This has the effect of minimizing error over all clusters with respect to the 1-norm distance metric, as opposed to the squared 2-norm distance metric (which k-means does.)

            Kmedians method involves using the Manhattan Distance to solve the l1 normalisation 

        """
        self.data=data
        self.method=method
        self.norm=norm
        self.k=k

    def run(self):
        """
        Runs the algorithm
        first: Append randmoised dataet to centroids
        oldCentroids=makes a copy of centroids to work out the error
        
        make centroids an array

        Work out error = Centroids - old centroids
        numError=0 counts errors
        While error not equal 0 "Train"

        dist=create an array to hold distanes, set as zeros of datset shape [0] (rows), with k amount (columns) 

        The run will then decide on mean or median , depending on the method

        if method == mean:
            Pass the dist variable into the Euclidean to normalise data, np.linalg.norm(self.data-self.centroids[i]) and update the dist varible
            clusters= find the argmin of the new dist of columns(axis=1)
            oldcentroids=update the varibale to the new centroids 
            work out the mean for each centroid (at k)
            np.mean(data[clusters==i],axis=0)row 
            clusters is found from minimum distances and for each clusters where they are equal k and work out the mean and then sort the values into centroid groups. 

        if method==median:
            Pass the dist variable into the Manhattan to normalise data, np.sum(np.abs(self.data-self.centroids[i]) and update the dist varible 
            clusters= find the argmin of the new dist of columns(axis=1)
            oldcentroids=update the varibale to the new centroids (clustered group)
            work out the median for each centroid (at k).np.median(data[clusters==i],axis=0)row 
            clusters is found from minimum distances and for each clusters where they are equal k and work out the median and then sort the values into centroid groups. 
        
        error is updated once clusters have been calcualted when error=0 break out of while loop
        return clusters for B-CUBED calcs 
        """
        l2=""
        if norm==True: #perform l2 normalisation
            #gives better results from https://macnux.medium.com/normalization-using-numpy-norm-simple-examples-like-geeks-b079bc4ea06b
            l2="with L2 Normalization"
            for i in range(len(self.data)):
                x_norm_col = np.linalg.norm(self.data[i], axis=0)
                self.data[i]=self.data[i]/x_norm_col
                #PROOF that data is normalised
                #l=np.linalg.norm(self.data[i])
                ##print(l) #PRINTS 1 as its normalised each row
            #ANOTHER METHOD
                # length = len(self.data)
                # for i in range(length):
                #     norm = np.sqrt(np.sum(self.data[i] * self.data[i]))
                #     self.data[i] = self.data[i] / norm

        np.random.seed(10)#1,2,6,7,8 #10 works best
        self.centroids=[]#acts as the label/group of clusters
        self.randomise=self.randomiseDataset()#ranmdomise dataset   
        for i in self.randomise:
            self.centroids.append(data[i])

        oldCentroids=np.zeros(np.shape(self.centroids))  #used for error
        self.centroids=np.array(self.centroids)#make nparray

        error=np.linalg.norm(self.centroids-oldCentroids)#determine starting error taking self.centroids vs old self.centroids which is currently set at zeros.
        numError=0#counter for errors
    
        while error != 0:
            dist=np.zeros([self.data.shape[0],self.k])#determine distances for working out clusters all are equal zero at first , taking the row of the dataset by the amount of k self.centroids.
            numError+=1
            if self.method=="mean":
                method_name="Euclidean Distance"    
                dist=self.Euclidean(dist)

                clusters=np.argmin(dist, axis = 1)#clusters determined from distance variable, take the min values for finding closest points to each other.
                oldCentroids=np.array(self.centroids)#reupdate old self.centroids after checking error
                for i in range(self.k):
                    self.centroids[i] = np.mean(self.data[clusters==i],axis=0)#finding mean range in k
                
            elif self.method == "median":
                method_name="Manhattan Distance"
                dist=self.Manhattan(dist)

                clusters=np.argmin(dist,axis=1)
                oldCentroids=np.array(self.centroids)#reupdate old self.centroids after checking error
                for i in range(self.k):
                    self.centroids[i] = np.median(self.data[clusters == i],axis=0)#calc median
            #update error until error=0 then break out of while loop
            error=np.linalg.norm(self.centroids-oldCentroids)
        final_centroids=np.array(self.centroids)
        predicted_clusters = clusters
        print("----------------------------------------------------------------")
        print("Final Results of K-%s Clustering with %s while k=%s %s"%(self.method,method_name,self.k,l2))
        #uncomment for further info on algorithm.
        # print("Num clusters:", self.k)
        # print("Num updates:", numError)
        # print ("Final Clusters:\n", predicted_clusters)
        # print ("Final Centroid Locations:\n", final_centroids)
        
        return predicted_clusters
    
    
    def randomiseDataset(self):
        """
            randomise the data before passing into kclustering algorithm        
        """
        randomise=np.random.randint(self.data.shape[0],size=self.k)#randomise each row
        return randomise

    def Euclidean(self,dist):
            """
                The mean is a least squares estimator of location. It is appropriate to use with squared deviations
                find distance between two points/rows of data based on pythagoream theorem
                dist=distance to each cluster values
            """
            for i in range(len(self.centroids)):
                    dist[:,i]=np.linalg.norm(self.data-self.centroids[i],axis=1)

            return dist
        #used for median
    def Manhattan(self,dist):
        """
        The main method for median
        The median is the best absolute deviation estimator or location. It is appropriate to use with absolute deviations 
        find distance between two points measured along axes at  right angles
        sum of absolute differences between each vector
        When distances is updated its done by taking slice of ith column vector of the matrix where i is the len of self.centroids 
        """
        for i in range(len(self.centroids)):
                dist[:,i] = np.sum(np.abs(self.data-self.centroids[i]), axis=1)
        return dist
        

def BCUBED(predictedClusters,ani,country,fruits,veg):
    """
    Function to work out B-CUBED values: Percision , Recall, F-Score

    Args:
        clusters=predicted clusters from the run function in class Kclustering
        ani=Animals size (49)
        country=countries (210)
        fruits=fruits (268)
        veg=veggies (328)
        k=k for printing to terminal

    """
    #clusters.size == 329
    #creating objects of the index position of the different classes
    a = predictedClusters[:ani+1] #+1 for 0th 0-50 elements
    c = predictedClusters[ani+1:country+1] #50-211 elements
    f = predictedClusters[country+1:fruits+1]#211-269
    v = predictedClusters[fruits+1:veg+1]#269-329
    #print(a)
    TP = 0 #true positives
    TN = 0#true negatives
    FP = 0#false positives
    FN = 0#false negatives
    
#check animals
    for i,_ in enumerate(a):
        #print(i)
        for j,_ in enumerate(a):#animals
            #print(j)
            if j>i and i!=j:#iterate through and count true positives 
                if(a[i]==a[j]): TP+=1
                else: FN+=1 #false negative
        for j,_ in enumerate(c):#through countries
            if(a[i]==c[j]): FP+=1 #check against countries if a[i]==c[j] then false positive
            else: TN+=1 #anything else is a true negative
        for j,_ in enumerate(f): #iterate through fruits
            if(a[i]==f[j]):FP+=1#check against countries if a[i]==f[j] then false positive
            else:TN+=1
        for j,_ in enumerate(v):  #veggies
            if(a[i]==v[j]): FP+=1
            else:TN+=1

    #countries
    for i,_ in enumerate(c):#start at countries and do the same to check for true positives 
        #dont need to do animals since its alread done.
        for j,_ in enumerate(c):
            if j>i and i!=j:
                if(c[i]==c[j]):TP+=1
                else: FN+=1
        for j,_ in enumerate(f):
            if(c[i]==f[j]): FP+=1
            else:TN+=1
        for j,_ in enumerate(v):  
            if(c[i]==v[j]): FP+=1
            else:TN+=1

    #fruits
    for i,_ in enumerate(f):#check fruits
        for j,_ in enumerate(f):
            if j>i and i!=j:
                if(f[i]==f[j]): TP+=1
                else: FN+=1
        for j,_ in enumerate(v):
            if(f[i]==v[j]): FP+=1
            else:TN+=1

    #veggie
    for i,_ in enumerate(v):#finally check true positives and false negatives for veggies.
        for j,_ in enumerate(v):
            if j>i and i!=j:
                if(v[i]==v[j]): TP+=1
                else: FN+=1

    #calcs for B-CUBED
    P = round((TP / (TP + FP)),2) #Percision round to 2 decimal places
    R = round((TP / (TP + FN)),2)  #Recall round to 2 decimal places
    F = round((2 * (P * R) / (P + R)),2) #F-score round to 2 decimal places
   
    print("B-CUBED Results: Percision:", P, ", Recall:", R, ", F-Score:", F)
    print("----------------------------------------------------------------")
    
    #for plotting values
    return P, R, F
          
def plot(k,p,r,f,method,l2):
    """
    Plot the B-Cubed results for all of K (1-9) 
    k=1-9 
    p=percision from B-CUBED
    r=recall from B-CUBED
    f=F-score from B-cubed
    method=Name of method (mean or median)
    Plot results 

    """
    plt.plot(k,p,label="Percision")
    plt.plot(k,r,label="Recall")
    plt.plot(k,f,label="F-Score")
    plt.title("K-%s Clustering %s" %(str(method),str(l2)))
    plt.xlabel('Number of Clusters')
    plt.ylabel("Scores")
    plt.legend()
    # Display the plot
    plt.show()   

def loopResults(x,norm,method):
    list_k,list_p,list_r,list_f=[],[],[],[] #FOR plotting
    for k in range(1,10):
        list_k.append(k)
        P,R,F=BCUBED(kClusteringAlgorithm(x,k=k,norm=norm,method=method).run(),a,c,f,v)
        list_p.append(P)
        list_r.append(R)
        list_f.append(F)
    if method=="mean":
        l2=""
        if norm==True:
            l2="With L2 Normalization"
        plot(list_k,list_p,list_r,list_f,method="mean",l2=l2)
    if method=="median":
        l2=""
        if norm==True:
            l2="With L2 Normalization"
        plot(list_k,list_p,list_r,list_f,method="median",l2=l2)
    return list_f

if __name__ == '__main__':
    fNames=["animals","countries","fruits","veggies"]
    dataset=parseData(fNames)

    a=maxCategoryIndex(dataset,fNames[0]) #used for B-CUBED find max index values.
    c=maxCategoryIndex(dataset,fNames[1])
    f=maxCategoryIndex(dataset,fNames[2])##find max index by using the category column added to identify data
    v=maxCategoryIndex(dataset,fNames[3])

    dataset=np.delete(dataset,0,axis=1)#delete the names, first column
    data=np.delete(dataset,-1,axis=1)#delete the categories added earlier to label data (animals,countries..)
    data=np.array(data).astype(np.float) #convert array values from strings to floats

    
    #BCUBED(kClusteringAlgorithm(data, 4,method="mean",norm=False),ani=a,country=c,fruits=f,veg=v)


#****************************************************************************************************************************
#**********************************                                     ****************************************************
#***************************>>>>>>>>>>>   RUN Questions !!!            <<<<<<<<<<<<<<<<<<<<<<<<<****************************
#**********************************                                     *****************************************************
#****************************************************************************************************************************
    SELECT_QUESTION="Panda-Compare"# #SELECT QUESTION BETWEEN 1-6 or choose "Panda-Compare" to run questions 3-6 then print out F-score comparrison using pandas
    if SELECT_QUESTION==1:
        K_VAL=4
        method="mean"
        norm=False
        P,R,F = BCUBED(kClusteringAlgorithm(data, k=K_VAL,method=method,norm=norm,).run(),ani=a,country=c,fruits=f,veg=v)

    elif SELECT_QUESTION==2:
        K_VAL=4
        method="median"
        norm=False
        P,R,F=BCUBED(kClusteringAlgorithm(data, k=K_VAL,method=method,norm=norm,).run(),ani=a,country=c,fruits=f,veg=v)
        
    elif SELECT_QUESTION == 3:
        norm=False
        method="mean"
        loopResults(data,norm=norm,method=method)
    elif SELECT_QUESTION == 4:
        norm=True
        method="mean"
        loopResults(data,norm=norm,method=method)
    elif SELECT_QUESTION == 5:
        norm=False
        method="median"
        loopResults(data,norm=norm,method=method)
    elif SELECT_QUESTION == 6:  
        norm=True
        method="median"
        loopResults(data,norm=norm,method=method)

    elif SELECT_QUESTION =="Panda-Compare":
        #PRINTS ALL QUESTIONS 3-6 ONE BY ONE WITH BCUBED RESULTS AND THE GRAPH FOR EACH QUESTION.  TO PROCEED TO THE NEXT QUESTION CLOSE THE GRAPH AND THE LOOP WILL CONTINUE TO THE NEXT QUESTION
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
    else:
        print("No valid question, please choose value between 1-6 for SELECT_QUESTION")
