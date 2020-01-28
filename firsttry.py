import numpy as np
import mnist 

x_train, t_train, x_test, t_test = mnist.load()
#mnist.init() ####******************İLK ÇALISTIRILIRKEN AÇILMALI
inputarray=[]
labelarray=[] 
labelmat=[]
setlen =1000
np.random.seed(1568)

hiddenlay=15
hiddenlay1=40
weighthid1 = np.random.rand(hiddenlay1,784)
weighthid = np.random.rand(hiddenlay,hiddenlay1)
weightout = np.random.rand(10,hiddenlay)
lamda=10  #learningrate



trainin=[]
traintes=[]


for i in range (setlen):
    trainin.append(x_train[i]/255)
    traininp=np.array(trainin)
    
for i in range (setlen):
    traintes.append(t_train[i])
    traintest=np.array(traintes)
    
   
#***
biasrow1=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,hiddenlay1):
        A.append(1)
    biasrow1.append(A)  
#***

biasrow=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,hiddenlay):
        A.append(1)
    biasrow.append(A)  
biasrowout=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,10):
        A.append(1)
    biasrowout.append(A)  

biashid1=np.array(biasrow1)#*****

biashid=np.array(biasrow)

biasoutt=np.array(biasrowout)
biasout=biasoutt.T
#print(biasout.shape)

#print(biashid)
#print(biasout)
mout=setlen*10




labelrow=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,10):
        A.append(0)
    labelrow.append(A)  


def labelmatrix():
    x=0
    for i in range (len(traintest)):
    
        labelrow[x][round(traintest[i])]=1
          
            #print(labelrow[x])
        x+=1 
    labelmat.append(labelrow)
    return labelmat

labelmatrix()
nplabelx=np.array(labelrow)

def sigmoid(x):
    return np.power((np.add(1,np.exp(-x))),(-1))
def sigmoder(x):
    return np.multiply( sigmoid(x),(np.subtract(1,sigmoid(x))))
def meansquare (gerçek,rastgele):
    return  np.square(np.subtract(gerçek, rastgele)).mean()
def logistic(gerçek,rastgele):
    return -np.mean(np.multiply(gerçek,np.log(rastgele))+np.multiply(np.subtract(1,gerçek),np.log(np.subtract(1,rastgele))))






for i in range(10000):
#########################################Feedforward
    ZH1 = np.dot(weighthid1,traininp.T)+biashid1.T
    
    zh1=sigmoid(ZH1)
    ZH = np.dot(weighthid,zh1)+biashid.T#hiddenweight[hiddenlay,21].input[10,21] biass=[setlen,hiddenlay]
    zh=sigmoid(ZH)
    #print("zhshape",zh.shape)
    ZO=np.dot(weightout,zh)+biasout #ouputweight[3,5].zh[5,10]+biasoutput[3,10]
    zo=sigmoid(ZO)#[3,10]son katman prediction değer
    tmp=zo
    
   # print(zo)
########################################Feedforward
    
    

    labelT=nplabelx.T
#print(labelT)
    #print(meansquare(zo,labelT))
##################################################### Error calculation
    print("cost in ",i,"epoc",logistic(labelT,zo))#loss fonksiyonu
    error=(zo-labelT)##normalde labelT-zo ama -1/m deki - isareti konulmasın diye böyle yazıldı
    errordagılım=error/mout
    errorsigmo=sigmoder(error) #son katmanda errorun nekadarının yansıtılıcagı rate
    costhidden=np.multiply(errorsigmo,errordagılım)##0.37 #yansıtılacak error
#####################################################Error calculation
#####################################################Outputbiasdeltacalculation
   # print("shshape",costhidden.shape)
    biascostrate=np.multiply(costhidden,biasout)##
   # print("biascostrate",biascostrate.shape)
    biasrowsumx=np.sum(biascostrate, axis=1)
   # print("biasrowsum",biasrowsumx)
    biasoutrowsum=biasrowsumx.T #(1,3)olmalı
    
    
    
    #print(biasoutrowsum)
    somethingout=[]
    for i in range(setlen):
        somethingout.append(biasoutrowsum)
    
    somethingoutbias=np.array(somethingout)
    somethingoutbiasT=somethingoutbias.T
    #print(somethingoutbiasT)## outputbiasta kullanılcakbiascost
    #print(biasout.shape)
##################################################### Outputbiasdeltacalculation
##################################################### Outputlayerweightlerin update edilmesi    
    weightoutrate=np.dot(costhidden,zh.T)##3,3 bu 0,024 outputlayerweightleri update ederken kullanılcak cost
##################################################### Outputlayerweightlerin update edilmesi 
#print(weightout.T.shape)
#print(costhidden.shape)
    costweight=np.dot(weightout.T,costhidden) #0,004 upside gradient
    hiddensigmo=sigmoder(zh)#0.25
    hiddenz=np.multiply(costweight,hiddensigmo)#0.001
    costweight1=np.dot(hiddenz,zh1.T)##ilklayerweightleriupdateederken kullanılcak cost
    
#print(costweight1)
    
 ##################################################******1.layer weighthler   
    costweight2=np.dot(weighthid.T,hiddenz)
    hidden1sigmo=sigmoder(zh1)#0.25
    hiddenz1=np.multiply(costweight2,hidden1sigmo)#0.001
    costweight3=np.dot(hiddenz1,traininp)
 ##################################################******1.layer weighthler    


    costbiashid=np.dot(hiddenz,biashid)
    costbiashidsum=np.sum(costbiashid,axis=1) #0.02,0.04,0.07
    
    
    somethinghid=[]
    for i in range(setlen):
        somethinghid.append(costbiashidsum)
    
    somethinghidbias=np.array(somethinghid)#hidden biasları update ederken kullanılcak biascost



    
    
    
    

###################################################*****1.layer biaslar
    costbiashid1=np.dot(hiddenz1,biashid1)
    costbiashidsum1=np.sum(costbiashid1,axis=1)
    
     
    somethinghid1=[]
    for i in range(setlen):
        somethinghid1.append(costbiashidsum1)
    
    somethinghidbias1=np.array(somethinghid1)
    
    
 ###################################################*****1.layer biaslar
    

    
    
    
    
    ######################################################### WEİGHT UPDATE
    
    
    
    
    weightout=np.subtract(weightout,np.multiply(weightoutrate,lamda))

    biasout=np.subtract(biasout,np.multiply(somethingoutbiasT,lamda))

    weighthid=np.subtract(weighthid,np.multiply(costweight1,lamda))
    
    weighthid1=np.subtract(weighthid1,np.multiply(costweight3,lamda))

    biashid=np.subtract(biashid,np.multiply(somethinghidbias,lamda))
    
    biashid1=np.subtract(biashid1,np.multiply(somethinghidbias1,lamda))

######################################################### WEİGHT UPDATE
    

nwweightout=weightout#en son çıkan weightler test sınıfında kullanılacaklar 
#print("nwweightout",nwweightout.shape)
nwbiasout=biasout
#print("nwbiasout",nwbiasout.shape)
nweighthid=weighthid
#print("nweighthid",nweighthid.shape)
nwbiashid=biashid#en son test sınıfında kullanılacak weight 
nwbiashid1=biashid1
nwweighthid1=weighthid1
#print("nwbiashid",nwbiashid.shape)
#print("biashid",biashid.shape)
#print("inputshape",npinput.shape)
#print("weightoutshape",weightout.shape)
#print("biasoutput",biasout.shape)

















 
    
    