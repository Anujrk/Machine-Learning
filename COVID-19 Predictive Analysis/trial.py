
import pandas as pd
from ARIMA import arima


data = pd.read_csv("project_file_join.csv")
data1 = pd.DataFrame(data.iloc[:len(data)//2])
data2 = pd.DataFrame(data.iloc[len(data)//2:])
writer = pd.ExcelWriter('project_file_join2.xlsx', engine='xlsxwriter')

data1.to_excel(writer, sheet_name='Sheeta')

data2.to_excel(writer, sheet_name='Sheetb')
writer.save()
arima("project_file_join2.xlsx")