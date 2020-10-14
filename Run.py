#!/usr/bin/python

import os
import sys, getopt
import pandas as pd
from lib.DecisionTree import DecisionTree

def main(argv):
    Question = ""
    
    try:
        opts, args = getopt.getopt(argv,"hq:",["q="])
    except getopt.GetoptError:
        print ("Run.py -q <question number>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ("Run.py -q <question number>")
            sys.exit()
        elif opt in ("-q", "--question"):
            Question = arg;

    if (not os.path.exists("result")):
        os.makedirs("result")

    if (not os.path.exists("data")):
        print ("data directory does not exist!!!!")
        sys.exit(2)
  
    Dtree = DecisionTree ("wdbc.data")
    Dtree.Train ()
   

if __name__ == "__main__":
   main(sys.argv[1:])
