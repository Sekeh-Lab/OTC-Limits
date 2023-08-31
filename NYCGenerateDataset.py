import pandas as pd
import math
from datetime import datetime,timedelta
import GenerateDatasets as GD
import numpy as np;
import networkx as nx
#NYC borders locations
MaxLongitude = -73.69
MinLongitude = -74.27
MaxLatitude = 40.93
MinLatitude = 40.48
#if i=j then returns node attributes
#if i!=j then returns an edge between two nodes
def FindNodeAttribute(Dataframe,i,j,LatRange,LongRange):
    #find the location that we want to find the needed information of
    LatcountPickup=int(i/int(math.sqrt(NodeNum)))
    longcountPickup=i%(int(math.sqrt(NodeNum)))
    LatcountDropoff=int(j/int(math.sqrt(NodeNum)))
    longcountDropoff=j%(int(math.sqrt(NodeNum)))
    try:
        #find the samples that have trips between the two selected regions
        NewDF=Dataframe[(Dataframe['pickup_latitude']>=((LatcountPickup*LatRange)+MinLatitude))& \
                        (Dataframe['pickup_latitude']<((LatcountPickup+1)*LatRange)+MinLatitude) & \
                        (Dataframe['pickup_longitude']>=((longcountPickup*LongRange)+MinLongitude))& \
                        (Dataframe['pickup_longitude']<(((longcountPickup+1)*LongRange)+MinLongitude)) & \
            (Dataframe['dropoff_latitude']>=((LatcountDropoff*LatRange)+MinLatitude))& \
                        (Dataframe['dropoff_latitude']<((LatcountDropoff+1)*LatRange)+MinLatitude) & \
                        (Dataframe['dropoff_longitude']>=((longcountDropoff*LongRange)+MinLongitude))& \
                        (Dataframe['dropoff_longitude']<(((longcountDropoff+1)*LongRange)+MinLongitude))]
    except Exception as e:
        print(e)
    #find the summation of the trips' times
    return sum(np.array(NewDF['dropoff_datetime']-NewDF['pickup_datetime']))
def Str2Datetime(Input):
    return datetime.strptime(Input, "%Y-%m-%d %H:%M:%S")

NodeNum=9
#generate dataset
dataset = GD.Dataset()

for iterat in range(1,13):
    #read the trips file
    df=pd.read_csv("dataset/trip_data_"+str(iterat)+".csv")
    df.columns = df.columns.str.replace(' ', '')
    print("Data loaded!\n")
    #separate the trips with different vendorsId( that is the label here)
    #eliminate trips with errors, they should be in a range of NYC borders:
    CMTdf=df[df['vendor_id']=="CMT"][((df[df['vendor_id']=="CMT"]['pickup_longitude'])>-74.27) \
                                     & ((df[df['vendor_id']=="CMT"]['pickup_longitude'])<-73.69)  \
                                     &  ((df[df['vendor_id']=="CMT"]['pickup_latitude'])>40.48) \
                                     & ((df[df['vendor_id']=="CMT"]['pickup_latitude'])<40.93)  \
                                     & ((df[df['vendor_id']=="CMT"]['dropoff_longitude'])>-74.27) \
                                     & ((df[df['vendor_id']=="CMT"]['dropoff_longitude'])<-73.69)  \
                                     &  ((df[df['vendor_id']=="CMT"]['dropoff_latitude'])>40.48) \
                                     & ((df[df['vendor_id']=="CMT"]['dropoff_latitude'])<40.93)]

    VTSdf=df[df['vendor_id']=="VTS"][((df[df['vendor_id']=="VTS"]['pickup_longitude'])>-74.27) \
                                     & ((df[df['vendor_id']=="VTS"]['pickup_longitude'])<-73.69) \
                                     &  ((df[df['vendor_id']=="VTS"]['pickup_latitude'])>40.48) \
                                     & ((df[df['vendor_id']=="VTS"]['pickup_latitude'])<40.93) \
                                     & ((df[df['vendor_id']=="VTS"]['dropoff_longitude'])>-74.27) \
                                     & ((df[df['vendor_id']=="VTS"]['dropoff_longitude'])<-73.69) \
                                     &  ((df[df['vendor_id']=="VTS"]['dropoff_latitude'])>40.48) \
                                     & ((df[df['vendor_id']=="VTS"]['dropoff_latitude'])<40.93)]
    #extract dates to have samples base on them
    CMTdf['pickup_datetime']=CMTdf['pickup_datetime'].apply(Str2Datetime)
    CMTdf['dropoff_datetime']=CMTdf['dropoff_datetime'].apply(Str2Datetime)
    i=min(CMTdf['pickup_datetime'])
    samples = []

    OneDay=timedelta(days=1)
    #Generate each sample as the trips of one day:
    while i<(max(CMTdf['pickup_datetime'])-OneDay):
        samples.append(CMTdf[(CMTdf['pickup_datetime']>=i) & (CMTdf['pickup_datetime']<i+OneDay)])
        i+=OneDay
    VTSdf['pickup_datetime']=VTSdf['pickup_datetime'].apply(Str2Datetime)
    VTSdf['dropoff_datetime']=VTSdf['dropoff_datetime'].apply(Str2Datetime)
    i=min(VTSdf['pickup_datetime'])
    while i<(max(VTSdf['pickup_datetime'])-OneDay):
        samples.append(VTSdf[(VTSdf['pickup_datetime']>=i) & (VTSdf['pickup_datetime']<i+OneDay)])
        i+=OneDay


    #each grid longitude and latitude length:
    LongRange=(MaxLongitude-MinLongitude)/int(math.sqrt(NodeNum))
    LatRange=(MaxLatitude-MinLatitude)/int(math.sqrt(NodeNum))

    print("About to start building the dataset")
    for item in samples:
        if item.empty:
            continue
        try:
            G = nx.Graph()
            [G.add_node(it) for it in range(1, NodeNum + 1)]  ###
            #FindNodeAttribute finds the total time of the trips from i to j
            attributes=[FindNodeAttribute(item,i,i,LatRange,LongRange) for i in range(NodeNum)]
            #convert nanosecond to seconds, also have attributes as np.array for ease of use
            attributes=np.array([float(it)/(1e+9) for it in attributes])
        except Exception as e:
            print(e)
        try:
            for node1 in range(NodeNum):
                for node2 in range(NodeNum):
                    if node2==node1:
                        continue
                    try:
                        #Add edge between i and j if the total time of the trips from i to j is greater than zero
                        if ((FindNodeAttribute(item, node1, node2, LatRange, LongRange))>0):
                            G.add_edge(node1+1, node2+1, weight=1)
                            G.add_edge(node2+1, node1+1, weight=1)
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)

        try:
            #add the generated sample to the dataset
            dataset.add(G, attributes, 0 if (np.array(item.head(1)['vendor_id'])[0]=='CMT') else 1)
        except Exception as e:
            print(e)

testsize = int(dataset.getDatasetSize() * (1/6))
trainingDataset, testDataset = dataset.getTrainTest(testsize)
print("writing to file:")
GD.writeDataset(trainingDataset, "real-datasets/NYC-training-"+str(NodeNum)+"5-1-new.pkl" )
GD.writeDataset(testDataset, "real-datasets/NYC-testing-"+str(NodeNum)+"5-1-new.pkl" )





