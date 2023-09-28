import pandas as pd

df = pd.read_csv("res/stroke_data.csv")

df['sex'].replace({1: "Male",0: "Female"}, inplace=True)
df['hypertension'].replace({0: "No",1: "Yes"}, inplace=True)
df['heart_disease'].replace({0: "No",1: "Yes"}, inplace=True)
df['ever_married'].replace({0: "No",1: "Yes"}, inplace=True)
df['work_type'].replace({0: "Never Worked",1: "Children",2:"Goverment Job",3:"Self-Employed",4:"Private"}, inplace=True)
df['Residence_type'].replace({0: "Rural",1: "Urban"}, inplace=True)
df['smoking_status'].replace({0: "No",1: "Yes"}, inplace=True)

df.to_csv("C:\Projects\GUI\stroke_data.csv")