from django.shortcuts import render
import dice_ml
from dice_ml import Dice
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from django.http import HttpResponse
from pandas import DataFrame
from django.utils.safestring import mark_safe
from IPython.display import display


# Create your views here.
def counter(request):
	return render(request,'counter.html')

def f(request):

	if request.method == 'POST':
		age = request.POST.get('age')
		age = float(age)
		bmi = request.POST.get('bmi')
		bmi = float(bmi)
		diabetes = request.POST.get('diabetes')
		diabetes = float(diabetes)
		med_conditions = request.POST.get('med_conditions')
		med_conditions = float(med_conditions)
		hypertension = request.POST.get('hypertension')
		hypertension = float(hypertension)
		hyperthyroidism = request.POST.get('hyperthyroidism')
		hyperthyroidism = float(hyperthyroidism)
		cholesterol = request.POST.get('cholesterol')
		cholesterol = float(cholesterol)
		ldl = request.POST.get('ldl')
		ldl = float(ldl)
		triglycerides = request.POST.get('triglycerides')
		triglycerides = float(triglycerides)
		creatinine = request.POST.get('creatinine')
		creatinine = float(creatinine)
		tsh = request.POST.get('tsh')
		tsh = float(tsh)
		columns =['Age Groupings','BMI Groupings','FH type 2 diabetes mellitus','Known Medical Conditions', 'hypertension','hypothyroidism','Total Cholesterol (mg/dL)','LDL (mg/dL)','Triglycerides (mg/dL)','Creatinine (mg/dL)','TSH (mg/dL)']
		row_values = [float(age),float(bmi),float(diabetes),float(med_conditions),float(hypertension),float(hyperthyroidism),float(cholesterol),float(ldl),float(triglycerides),float(creatinine),float(tsh)]
		df = pd.DataFrame([row_values], columns=columns)



		# https://machinehack.com/story/how-to-generate-counterfactual-explanations-with-dice-ml
		df_wine = pd.read_csv("chk.csv")
		column = list(df_wine['Dependent_var'])
		column = [1 if x == 0 else 0 for x in column]
		df_wine["Dependent_var"]=column
		

		
		outcome_name = "Dependent_var"
		cont_feats = df_wine.drop(outcome_name, axis=1).columns.tolist()
		cat_feats = df_wine[['Age Groupings','BMI Groupings','FH type 2 diabetes mellitus','Known Medical Conditions','hypertension','hypothyroidism']]
		target = df_wine[outcome_name]

		X_data = df_wine.drop(outcome_name, axis=1)
		x_train, x_test, y_train, y_test = train_test_split(X_data,target,test_size=0.2,random_state=0,stratify=target)

		
		clf = RandomForestClassifier(n_estimators = 100)  

		model_wine = clf.fit(x_train, y_train)
		y_test_pred = model_wine.predict(df)
		# r=y_test_pred

		if y_test_pred==0:
			input_df=df.assign(Dependent_var="Non Diseased")
		else:
			input_df=df.assign(Dependent_var="Diseased")

		input_df.to_csv("input_df.csv",index=False)
		b=pd.read_csv("input_df.csv")

		html_tables = b.to_html()
		d_wine = dice_ml.Data(dataframe=df_wine,
		continuous_features=cont_feats,
		outcome_name=outcome_name)
		m_wine = dice_ml.Model(model=model_wine, backend="sklearn", model_type='classifier')
		wine_exps = Dice(d_wine, m_wine, method="genetic")
		query_instance_wine = df
		genetic_wine = wine_exps.generate_counterfactuals(query_instance_wine, total_CFs=1, desired_class=0)
		genetic_wine.visualize_as_list()
		genetic_wine.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf='counterfactuals.csv', index=False)
		
		a=pd.read_csv("counterfactuals.csv")
		
		a["Dependent_var"]="Non Diseased"
			
		

		html_table = a.to_html()
		df1=b
		df2=a
		
		
		def highlight_changes(row):
		    df1_row = df1.iloc[0]  # get the first row of df1
		    df2_row = df2.iloc[0]  # get the first row of df2
		    styles = []
		    for col in df1.columns:
		        if df1_row[col] != df2_row[col]:
		            styles.append('color: red')
		        else:
		            styles.append('')
		    return styles

		# apply the function to the dataframe and display the styled output

		styled = df2.style.apply(highlight_changes, axis=1)
		# styled = styled.format('{:.2f}', subset=pd.IndexSlice[:, :])
		numeric_cols = df2.select_dtypes(include=['float', 'int']).columns
		styled = styled.format('{:.2f}', subset=pd.IndexSlice[:, numeric_cols])
		    
		
		   
		    
		context = {'html_tables':html_tables,'styled': styled.render()}
		
		if y_test_pred == 0:
			r=0
		else:
			r=1
		return render(request,'result.html',context)






	


	



