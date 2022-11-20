import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['random','degree','betweenness','clossness','2-hop-based'])

df.loc[1,'random'] = 2.2
df.loc[2,'random'] = 2.3
df.loc[-1,'random'] = -1
print(df)