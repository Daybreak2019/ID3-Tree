#!/usr/bin/python
import pandas as pd
import time
import numpy as np
from math import log
import matplotlib.pyplot as plt
from sklearn import tree
from decimal import Decimal


#Ten real-valued features are computed for each cell nucleus:
#	a) radius (mean of distances from center to points on the perimeter)
#	b) texture (standard deviation of gray-scale values)
#	c) perimeter
#	d) area
#	e) smoothness (local variation in radius lengths)
#	f) compactness (perimeter^2 / area - 1.0)
#	g) concavity (severity of concave portions of the contour)
#	h) concave points (number of concave portions of the contour)
#	i) symmetry 
#	j) fractal dimension ("coastline approximation" - 1)
#The mean, standard error, and "worst" or largest (mean of the three
#largest values) of these features were computed for each image,
#resulting in 30 features.  For instance, field 3 is Mean Radius, field
#13 is Radius SE, field 23 is Worst Radius.

class Tree ():
    def __init__(self, Id, Parent, Feature, Threshold):
        self.NodeID     = Id
        self.Parent     = Parent
        self.Feature    = Feature
        self.Threshold = Threshold
        self.Left       = None
        self.Right      = None

    def IsLeaf (self):
        return (self.Left == None and self.Right == None)
        

class DecisionTree ():
    
    def __init__(self, FileName):
        self.Name     = FileName
        self.TreeNode = {}
        self.FeatureCls = []
        
        if (FileName == "wdbc.data"):
            self.LoadData_wdbc (FileName)
        elif (FileName == "breast-cancer-wisconsin.data"):
            self.LoadData_bcw (FileName)
        else:
            exit (0)
            
    def LoadData_bcw (self, FileName):
        # load dataset
        Df = pd.read_table("data/" + FileName, sep=',', header=None)
        Df.drop(0, axis = 1, inplace = True)
        LastCol = Df.pop(Df.columns[-1])     
        Df.insert(0, LastCol.name, LastCol)
        self.Dataset = Df.values

        # get attributs for each Feature
        self.FeatureCls = ["Label", 
                           "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                           "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
        
    def LoadData_wdbc (self, FileName):
        # load dataset
        Df = pd.read_table("data/" + FileName, sep=',', header=None)
        Df.drop(0, axis = 1, inplace = True)
        self.Dataset = Df.values

        # get attributs for each Feature
        self.FeatureCls = ["Label",
                           "radius mean", "texture mean", "perimeter mean", "area mean", "smoothness mean", 
                           "compactness mean", "concavity mean", "concave_points mean", "symmetry mean", "fractal_dimension mean",
                           "radius stderr", "texture stderr", "perimeter stderr", "area stderr", "smoothness stderr", 
                           "compactness stderr", "concavity stderr", "concave_points stderr", "symmetry stderr", "fractal_dimension stderr",
                           "radius largest", "texture largest", "perimeter largest", "area largest", "smoothness largest", 
                           "compactness largest", "concavity largest", "concave_points largest", "symmetry largest", "fractal_dimension largest"]


    def GetIndices (self):
        IndexList = np.arange(0, self.Features.shape[0])
        TrainNum  = int(self.Features.shape[0] * self.TrainRatio)
        TrainIndices = IndexList[:TrainNum]
        ValidIndices = IndexList[TrainNum:]
        
        return TrainIndices, ValidIndices

    # compute Entropy
    def GetEntropy(self, Dataset):
        LabelCount = {}
        for Example in Dataset:
            Label = Example[0]
            if Label not in LabelCount.keys():
                LabelCount[Label] = 0
            LabelCount[Label] += 1

        ExampleNum = len(Dataset)
        Entropy = 0.0
        for Label, Count in LabelCount.items():
            Prob = 1.0*Count/ExampleNum
            Entropy -= Prob*log(Prob, 2)
        return Entropy

    # split thresholds, j = i+1, fi + (fi+1 - fi)/2
    def GetThresholds (self, Fi, Fj):
        return (Fi + (Fj - Fi)/2.0)


    #split the data set
    def Split(self, Dataset, Index, Threshod):
        SetL = []
        SetR = []
        for Example in Dataset:
            Row = []
            for Findex in range(len(Example)):
                if Findex != Index:
                    Row.append (Example[Findex])
            
            if Example[Index] <= Threshod:
                SetL.append (Row)
            else:
                SetR.append (Row)

        return SetL, SetR

    # get best feature by Entropy
    def GetBestFeature(self, Dataset):
        ExampleNum = len(Dataset)
        FeatureNum = len(Dataset[0])
        Entropy = self.GetEntropy(Dataset)
        
        BestValue     = -0.1 # maybe all grain is 0
        BestFeature   = 0
        BestThreshold = 0

        # iterate all features
        for fIndex in range(1, FeatureNum):
            
            #for a given feature, get a best threshold
            FeatureList = set([example[fIndex] for example in Dataset])
            FeatureList = list (FeatureList)
            FeatureList.sort ()

            # now compute the best threshold (smallest conditional entropy) for the given feature
            FeaEntropy    = 1.0
            FeaThreshHold = 0.0
            for Fi in range(0, len(FeatureList)-1):
                Threshold   = self.GetThresholds (FeatureList[Fi], FeatureList[Fi+1]) #Fi+1 can not be larget than len(FeatureList)
                SetL, SetR  = self.Split(Dataset, fIndex, Threshold)
                ConEntropy  = (len(SetL)/ExampleNum*1.0) * self.GetEntropy(SetL) +\
                              (len(SetR)/ExampleNum*1.0) * self.GetEntropy(SetR)
                if (ConEntropy < FeaEntropy):
                    FeaEntropy = ConEntropy
                    FeaThreshHold = Threshold

            #print ("[%2d/%d]Threshod = %f, FeaEntropy = %f" %(fIndex, FeatureNum, FeaThreshHold, FeaEntropy))            
            Gain = Entropy - FeaEntropy
            if Gain > BestValue:
                BestValue     = Gain
                BestFeature   = fIndex
                BestThreshold = FeaThreshHold

        return BestFeature, BestThreshold

    def GetMajorCls (self, ClassList):
        ClassNum = {}
        for Cls in ClassList:
            if Cls not in ClassNum.keys():
                ClassNum[Cls] = 0 
            ClassNum[Cls] += 1
            
        Majcls = 0
        MaxNum   = 0
        for Cls, Num in ClassNum.items ():
            if Num > MaxNum:
                MaxNum   = Num
                Majcls = Cls
        return Majcls

    def NewNode (self,   Parent, Feature, Threshold):
        NodeId = len (self.TreeNode)
        TNode  = Tree (NodeId, Parent, Feature, Threshold)
        self.TreeNode [NodeId] = TNode
        #print ("Add Node: parent=%d, Feature = %s, Threshold = %f" %(Parent, Feature, Threshold))
        return TNode
        
    def CreateTree (self, Parent, Dataset, FeatureCls):
        ClassList = [Example[0] for Example in Dataset]
        ClsNum = len(ClassList)   
        
        # all example have the same class 
        if (ClassList.count(ClassList[0])) == ClsNum:
            return self.NewNode (Parent, ClassList[0], 0)
            
        # all features have been processed 
        if (len (Dataset[0]) == 1):
            Cls  = self.GetMajorCls (ClassList)
            return self.NewNode (Parent, Cls , 0)
        
        BestFeature, Threshold = self.GetBestFeature(Dataset)
        BestFeatureClf = FeatureCls[BestFeature]
        
        # construct sub-tree
        Dtree = {BestFeatureClf:{}}
        SerL, SetR = self.Split(Dataset, BestFeature, Threshold)

        if (len (SerL) == 0 or len (SetR) == 0):
            Cls = self.GetMajorCls (ClassList)
            return self.NewNode (Parent, Cls , 0)
        else:   
            print ("[%d](BestFeature, Threshold) => (%s, %f), (Left, Right) = (%d, %d) " 
                   %(BestFeature, BestFeatureClf, Threshold, len (SerL), len (SetR)))
            TNode = self.NewNode (Parent, BestFeatureClf, Threshold)
            
            SubFeatureCls = FeatureCls[:] 
            SubFeatureCls.remove (BestFeatureClf)
            TNode.Left  = self.CreateTree(TNode.NodeID, SerL, SubFeatureCls)
            TNode.Right = self.CreateTree(TNode.NodeID, SetR, SubFeatureCls)
        
        return TNode

    def Predict (self, Dtree, FeatureCls, Example):
        # reach the leaf
        if (Dtree.IsLeaf ()):
            return Dtree.Feature
        
        # get the root and correlated threshold
        FeaName   = Dtree.Feature
        Threshold = Dtree.Threshold

        # get feature value in the example
        FeaIndex = FeatureCls.index(FeaName)
        FeaValue = Example[FeaIndex]

        #print ("(FeaName, Index, Value, Threshold) = (%s, %d, %f, %f)\r\n" %(FeaName, FeaIndex, FeaValue, Threshold))        
        if (FeaValue <= Threshold):
            return self.Predict (Dtree.Left, FeatureCls, Example)
        else:
            return self.Predict (Dtree.Right, FeatureCls, Example)
    

    def Classify (self, Dtree, FeatureCls, Dataset):
        Mistakes = 0
        for Example in Dataset:
            Pred = self.Predict (Dtree, FeatureCls, Example)
            if (Pred != Example[0]):
                Mistakes += 1
                
        return (1 - Mistakes * 1.0 / len (Dataset))

    def SklearnScore (self, Clf, Features, Labels):
        Pred = Clf.predict (Features)
        Mist = 0
        for i in range (len (Pred)):
            if Labels[i] != Pred[i]:
                Mist += 1
        return (1 - Mist*1.0/len(Features))
        

    def SklearnTest (self, Trainset, Validset, Testset):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(Trainset[0::, 1::], Trainset[::, 0])

        # predict Valid
        ValidAcc = self.SklearnScore (clf, Validset[0::, 1::], Validset[::, 0])
        
        # predict Testset
        TestAcc = self.SklearnScore (clf, Testset[0::, 1::], Testset[::, 0])
        print ("Sklearn => (ValidAccuracy, PrunAccuracy) = (%f, %f)" %(ValidAcc, TestAcc))
    
    def GetMajorLabel (self, Tree, Dataset):
        LabelNum = {}       
        for Example in Dataset:
            Pred = self.Predict (Tree, self.FeatureCls, Example)           
            if Pred not in LabelNum.keys():
                LabelNum[Pred] = 0
            LabelNum[Pred] += 1
        
        MajLabel = 0
        MaxNum   = 0
        for Label, Num in LabelNum.items ():
            if Num > MaxNum:
                MaxNum   = Num
                MajLabel = Label
        return MajLabel

    def GetSubtreeType (self, Parent, CName):
        FeaName = list(Parent.keys())[0]
        RTree = Parent[FeaName][RIGHT]
        if isinstance(RTree , dict) == False:
            return LEFT
        
        FeaName  = list(RTree.keys())[0]
        if FeaName == CName:
            return RIGHT
        else:
            return LEFT

    def Pruning (self, DTree, Trainset, Validset, ValidAccuracy):
        print ("\r\n\t ==> Start Pruning...")
        #bridth first visit the tree, and push all sub-tree into queue
        QIndex = 0
        Queue = []
        Queue.append (DTree)
        while QIndex < len(Queue):
            # pop the first node of queue
            TNode = Queue[QIndex]
            QIndex += 1
            
            LTree = TNode.Left
            if not LTree.IsLeaf ():
                Queue.append (LTree)

            RTree = TNode.Right
            if not RTree.IsLeaf ():
                Queue.append (RTree)

        # bottom-up visit all sub-tree, pruning with REP
        QIndex = len(Queue) - 1
        while QIndex >= 0:
            SubTree = Queue[QIndex]

            # get its majority label
            MajLabel = self.GetMajorLabel (SubTree, Trainset)

            if SubTree.NodeID == 0:
                # root
                Root = self.NewNode (0, MajLabel, 0)

                # compare accuracy after pruning
                Accuracy = self.Classify (Root, self.FeatureCls, Validset)
                if Accuracy > ValidAccuracy:
                    print ("\tPruning [%s](ValidAccuracy, PrunAccuracy) = (%f, %f)" %(SubTree.Feature, ValidAccuracy, Accuracy))
            else:
                # replace urrent sub-tree
                PNode = self.TreeNode [SubTree.Parent]
                PruneNode = self.NewNode (0, MajLabel, 0)

                if (PNode.Left == SubTree):
                    PNode.Left = PruneNode
                else:
                    PNode.Right = PruneNode

                # compare accuracy after pruning
                Accuracy = self.Classify (DTree, self.FeatureCls, Validset)
                if Accuracy <= ValidAccuracy:
                    if (PNode.Left == PruneNode):
                        PNode.Left = SubTree
                    else:
                        PNode.Right = SubTree
                else:
                    print ("\tPruning [%s](ValidAccuracy, PrunAccuracy) = (%f, %f)" %(SubTree.Feature, ValidAccuracy, Accuracy))

            QIndex -= 1
         

    def DumpTree (self, Tag=""):
        File = open("./result/" + self.Name + Tag + ".dot" , "w")

        #header
        File.write("digraph \"" + self.Name + "\"{\n")
        File.write("\tlabel=\"" + self.Name + "\";\n")

        Queue  = []
        QIndex = 0
        Queue.append (self.TreeNode[0])
        while QIndex < len(Queue):
            # pop the first node of queue
            TNode = Queue[QIndex]
            QIndex += 1

            #write node
            Label = str(TNode.Feature)
            if not TNode.IsLeaf ():
                Threshold = str(TNode.Threshold) 
                Label = Label + ", THR=" + str(Decimal(Threshold).quantize(Decimal('0.00')))
            File.write("\tN" + str(TNode.NodeID) + "[color=black, label=\"{" + Label + "}\"]\n")

            if TNode.IsLeaf ():
                continue

            #write edges
            LTree = TNode.Left
            Queue.append (LTree)
            File.write("\tN" + str(TNode.NodeID) + " -> N" + str(LTree.NodeID) + "[color=red" + ",label=\"{L}\"]\n")

            RTree = TNode.Right
            Queue.append (RTree)
            File.write("\tN" + str(TNode.NodeID) + " -> N" + str(RTree.NodeID) + "[color=red" + ",label=\"{R}\"]\n")
        File.write("}\n")
        File.close()
            
        
    def Train (self):
        print ("===========================================================")
        print ("== Process %s" %self.Name)
        print ("===========================================================")
        DataNum = self.Dataset.shape[0]
        IndexList = np.arange(0, DataNum)
        TrainNum  = int(DataNum * 0.7)
        ValidNum  = int(DataNum * 0.1)
        TrainIndices = IndexList[:TrainNum]
        ValidIndices = IndexList[TrainNum:TrainNum+ValidNum]
        TestIndices  = IndexList[TrainNum+ValidNum:]

        # create the tree by ID3
        Dtree = self.CreateTree (-1, self.Dataset[TrainIndices], self.FeatureCls)
        self.DumpTree ();

        # get accuracy on validation/test dataset
        ValidAcc = self.Classify (Dtree, self.FeatureCls, self.Dataset[ValidIndices])
        TestAcc  = self.Classify (Dtree, self.FeatureCls, self.Dataset[TestIndices])
        print ("Before Pruning => (ValidAccuracy, TestAccuracy) = (%f, %f)" %(ValidAcc, TestAcc))

        # sklearn test
        self.SklearnTest (self.Dataset[TrainIndices], self.Dataset[ValidIndices], self.Dataset[TestIndices])

        # post pruning with REP
        self.Pruning (Dtree, self.Dataset[TrainIndices], self.Dataset[ValidIndices], ValidAcc)
        ValidAcc = self.Classify (Dtree, self.FeatureCls, self.Dataset[ValidIndices])
        TestAcc  = self.Classify (Dtree, self.FeatureCls, self.Dataset[TestIndices])
        print ("After Pruning  => (ValidAccuracy, TestAccuracy) = (%f, %f)" %(ValidAcc, TestAcc))
        self.DumpTree ("Pruning");
        


