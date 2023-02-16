import sys
sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')
sys.path.insert(1, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\')
sys.path.insert(2, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\components')
import Layer
import pandas as pd
import Network
from run import *
from preprocessing import *

stock_aapl = pd.read_csv("./Data/datasets/AAPL.csv/AAPL.csv")

def main():
    pass

if __name__ == '__main__':
    main()
