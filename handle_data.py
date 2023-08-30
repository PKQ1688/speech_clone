"""
@Author: pkq1688
@Date: 2023-08-28 17:31:01
@LastEditors: pkq1688
@LastEditTime: 2023-08-28 19:15:17
@FilePath: /speech_clone/handle_data.py
@Description: 
"""
import pandas as pd

data = pd.read_csv("kw.csv")
data = data.drop(["关键词","序号"], axis=1)
kw_list = []

print(data)

for index,row in data.iterrows():
    # kw_list.append(row["关键词"])
    for kw in row:
        if pd.isna(kw):
            continue
        else:
            kw_list.append(kw)
    
    # break

print(kw_list)

with open("kw.txt", "w") as f:
    for kw in kw_list:
        f.write(kw + "\n")