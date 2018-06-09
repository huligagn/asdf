import pandas as pd
import numpy as np 

data = np.array(['a','b','c','d'])
data2 = {'a':0., 'b':1., 'c':2.}
s = pd.Series([1,2,3,4,5], index=['b','a','c','d','e'])
print(s[-3:])