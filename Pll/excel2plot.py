import pandas as pd

import data2Excel
xlsx_file = "test2.xlsx"
df1 = pd.read_excel(xlsx_file, sheet_name="Sheet1",usecols=[1, 2, 3, 4 , 5])
df2 = pd.read_excel(xlsx_file, sheet_name="Sheet2",usecols=[1, 2, 3, 4 , 5])
df3 = pd.read_excel(xlsx_file, sheet_name="Sheet3",usecols=[1, 2, 3, 4 , 5])

