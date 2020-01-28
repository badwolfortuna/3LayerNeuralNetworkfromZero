import numpy as np
#import denemee as den
import mnist 
import firsttry as frs
##***train edilecek neural network ile test edilecek dataset uzunlugu aynı olmalı(bias weightlerden dolayı)
#import denemee as dn

x_train, t_train, x_test, t_test = mnist.load()
####################################################
labelmat=[]
setlen =30 #test datasının uzunlugu
hiddenlayer=frs.hiddenlay
hiddenlay1=frs.hiddenlay1
#np.random.seed(42)
#weighthid = np.random.rand(21,5)
#weightout = np.random.rand(5,3)




xtestin=[]
ttest=[]


for i in range (setlen):
    xtestin.append(x_test[i]/255)
    testinp=np.array(xtestin)
    
for i in range (setlen):
    ttest.append(t_test[i])
    testtest=np.array(ttest)
    
   




#lamda=0.01  #learningrate




testweight0=np.array(frs.nwweighthid1)
testweight1=np.array(frs.nweighthid)
testweight2=np.array(frs.nwweightout)
bias0=np.array(frs.nwbiashid1,order='C')
bias0.resize((setlen,hiddenlay1))

bias1=np.array(frs.nwbiashid, order='C')#resize biases
bias1.resize((setlen,hiddenlayer))

bias2=np.array(frs.nwbiasout, order='C')

bias2.resize((10,setlen))





#print(testweight2)
####################################################
labelrow=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,10):
        A.append(0)
    labelrow.append(A)  



#print("labelarray",labelarray)
#print("inputarray",inputarray)


def labelmatrix():
    x=0
    for i in range (len( testtest)):
    
        labelrow[x][testtest[i]]=1
          
            #print(labelrow[x])
        x+=1 
    labelmat.append(labelrow)
    return labelmat

labelmatrix()
nplabelx=np.array(labelrow)
nplabelT=nplabelx.T
#######################################################







def sigmoid(x):
    return np.power((np.add(1,np.exp(-x))),(-1))

ZH1 = np.dot(testweight0,testinp.T)+bias0.T
    
zh1=sigmoid(ZH1)
ZH = np.dot(testweight1,zh1)+bias1.T#hiddenweight[hiddenlay,21].input[10,21] biass=[setlen,hiddenlay]
zh=sigmoid(ZH)
    #print("zhshape",zh.shape)
ZO=np.dot(testweight2,zh)+bias2 #ouputweight[3,5].zh[5,10]+biasoutput[3,10]
zo=sigmoid(ZO)#[3,10]son katman prediction değer
    
ZOT=zo.T 
print("realclasses","\n",nplabelx)
print(zo)
#print("predicted value",zo.T)



for i in range (setlen):
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][0]):
        print("predicted class0")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][1]):
        print("predicted class1")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][2]):
        print("predicted class2")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][3]):
        print("predicted class3")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][4]):
        print("predicted class4")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][5]):
        print("predicted class5")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][6]):
        print("predicted class6")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][7]):
        print("predicted class7")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][8]):
        print("predicted class8")
    if(np.round(zo.T.max(axis=1)[i])==nplabelx[i][9]):
        print("predicted class9")
    print("endoffor",i)
    
    

    