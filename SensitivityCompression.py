
import numpy as np
from statistics import mean
def Rho(Input,nodeSize):
    print("\nEntered Rho\n")
    features = np.squeeze(Input.getFeatures())

    ar = np.array(Input.points)
    label = ar[:, 2]
    feat = features
    if np.ndim(np.array(feat))==2:
        P_Xv = [np.average(np.histogram(item, density=True)[0]) for item in np.array(feat).T]
    else:
        P_Xv = [[np.average(np.histogram(it, density=True)[0]) for it in item] for item in np.array(feat).T]
    numerator =0
    for i in range(Input.getNumClasses()):
        print("\nEntered Rho loop...\n ")
        feati=feat[(label == i)]
        if np.ndim(np.array(feat)) == 2:
            P_Xv_ci = [np.average(np.histogram(item, density=True)[0]) for item in np.array(feati).T]
        else:
            P_Xv_ci = [[np.average(np.histogram(it, density=True)[0]) for it in item] for item in np.array(feati).T]
        numerator += np.multiply(P_Xv_ci , (len(feati) / len(features)))# +np.multiply(P_Xv_c1 ,(len(feat1) / len(features)))
    print("Rho loop ended")
    Output = numerator / P_Xv
    if np.ndim(np.array(feat)) != 2:
        Output=np.average(Output,axis=0)
    indxs=np.argsort(Output)
    # Output[indxs[Output.shape[0]-nodeSize:Output.shape[0]]] = 1#200/100
    # Output[indxs[0:Output.shape[0] - nodeSize ]]=0
    Output[indxs[0:nodeSize]] = 1#200/100
    Output[indxs[nodeSize:]]=0
    return Output
def Rho_Prime(Input,mask,nodesize):#
    print("entered Rho prime")
    features = np.squeeze(Input.getFeatures())[:,mask == 1]
    numNodes = np.shape(features)[1]
    ar=np.array(Input.points)
    label=ar[:,2]
    feat=features
    P_Xv = [np.average(np.histogram(item, density=True)[0]) for item in np.array(feat).T]
    new_adj = []
    adj = Input.getAdjacencies();
    adj[:, mask == 0]=0
    adj[:,:, mask == 0]=0
    for item in adj:
        item = item[mask == 1].T[mask == 1]
        new_adj.append(item)
    new_adj = np.array(new_adj)
    numerator=np.zeros(shape=(feat.T.shape[0],numNodes));
    P_Gv_Xv = np.zeros(shape=(numNodes));
    for i in range(Input.getNumClasses()):
        print("entered loop of classes")
        print(i)
        feati=feat[(label==i)]
        if np.ndim(np.array(feat)) == 2:
            P_Xv_ci = [np.average(np.histogram(item, density=True)[0]) for item in np.array(feati).T]
        else:
            P_Xv_ci = [[np.average(np.histogram(it, density=True)[0]) for it in item] for item in np.array(feati).T]
        new_adji=new_adj[label==i]
        new_adji=np.array(new_adji)
        P_Euv_XuXv = np.zeros(shape=(numNodes, numNodes,feat.T.shape[0])) + 2;
        P_Euv_XuXvCi=np.zeros(shape=(numNodes,numNodes,feat.T.shape[0]))+2;



        for it1 in range(numNodes):
            print("entered 2 loops")
            print(it1)
            for it2 in range(it1+1,numNodes):
                print("\n")
                print(it2)
                #|Xu,Xv


                p=[[np.squeeze( new_adj[ (np.array(feat)[:,it1,featIt]==iterat[it1]) & (np.array(feat)[:,it2,featIt]==iterat[it2])]) for iterat in Onedfeat.T] for Onedfeat,featIt in zip(feat.T,range(feat.T.shape[0]))]
                #|Xu,Xv,ci
                pi=[[np.squeeze( new_adji[ (np.array(feati)[:,it1,featIt]==iterat[it1]) & (np.array(feati)[:,it2,featIt]==iterat[it2]) ]) for iterat in Onedfeati.T] for Onedfeati,featIt in zip(feati.T,range(feati.T.shape[0]))]


                #*******************
                p=[[ps[np.newaxis, :, :] if len(ps.shape) == 2 else ps for ps in pit] for pit in p]
                pi=[[ps[np.newaxis, :, :] if len(ps.shape) == 2 else ps for ps in pit] for pit in pi]
                #*******************


                #Eu,v|Xu,Xv
                p=[[sum(iterat[:,it1,it2])/iterat.shape[0] for iterat in ps] for ps in p]
                #Eu,v|Xu,Xv,ci
                pi=[[sum(iterat[:,it1,it2])/iterat.shape[0] for iterat in ps] for ps in pi]

                P_Euv_XuXv[it1][it2]=[mean(ps) for ps in p]
                P_Euv_XuXvCi[it1][it2]=[mean(ps) for ps in pi]
                P_Euv_XuXv[it2][it1]=[mean(ps) for ps in p]
                P_Euv_XuXvCi[it2][it1]=[mean(ps) for ps in pi]
                # P_Euv_XuXvC1[it1][it2]=mean(p1)
        P_Euv_XuXv_Prime=1-P_Euv_XuXv
        P_Euv_XuXvCi_Prime=1-P_Euv_XuXvCi

        P_Gv_Xv=[np.prod(P_Euv_XuXvs.T.flatten())*(1e+200)*np.prod(P_Euv_XuXv_Primes.T.flatten())/(-2) for P_Euv_XuXvs,P_Euv_XuXv_Primes in zip(P_Euv_XuXv.T,P_Euv_XuXv_Prime.T)]
        P_Gv_XvCi=[np.prod(P_Euv_XuXvCis.T.flatten())*(1e+200)*np.prod(P_Euv_XuXvCi_Primes.T.flatten())/(-2) for P_Euv_XuXvCis,P_Euv_XuXvCi_Primes in zip(P_Euv_XuXvCi.T,P_Euv_XuXvCi_Prime.T)]
        numerator +=np.array([np.multiply(P_Xv_cis,P_Gv_XvCis) for P_Xv_cis,P_Gv_XvCis in zip(P_Xv_ci,P_Gv_XvCi)])*(len(feati)/len(features))#np.multiply(P_Xv_ci,P_Gv_XvCi)*(len(feati)/len(features))#+np.multiply(P_Xv_c1,P_Gv_XvC1)*(len(feat1)/len(features))
    Output=[(numerators/P_Gv_Xvs) for numerators,P_Gv_Xvs in zip(numerator,P_Gv_Xv)]
    # Output[Output >= mean(Output)] = 1
    # Output[Output !=1] = 0

    #can be changed
    Output=np.mean(np.array(Output),axis=0)

    indxs=np.argsort(Output)
    Output[indxs[Output.shape[0]-nodesize:Output.shape[0]]] = 1#200/100
    Output[indxs[0:Output.shape[0] - nodesize ]]=0

    print("end of rho prime")
    return Output
def SensitivityMeasure(Input,datasetType):

    mask=Rho(Input)
    rho=np.array(list(mask))
    print("\nRho found\n")
    Output=Rho_Prime(Input, mask)
    print("\nRho Prime found\n")
    rhop=Output
    file_object = open('rho_rhopMNIST.txt', 'a')

    file_object.write('\nDataset Type:')
    file_object.write(datasetType)
    file_object.write('\nrho=')
    file_object.write(str(rho))
    file_object.write('\nrhoPrime=')
    file_object.write(str(rhop))

    # Close the file
    file_object.close()

    return mask

