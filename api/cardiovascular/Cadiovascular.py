import pickle
import pandas as pd




class cadio_disease(object):
    def __init__(  self   ):
        state = 1
        self.home_path = '/home/tulio/github/Cardiovascular-Predict'
        self.transformação_var_numerica = pickle.load( open(self.home_path + '/parametros/encoding_vars_numericas_train.pkl','rb'))
        
    def data_novas_variaveis(self, df1):
        # age - days/365
        year = lambda x : x/365

        df1['age'] = df1['age'].astype( int )
        df1['age_years'] = list(map( year, df1['age'] ) )
        df1['age_years'] = df1['age_years'].astype( int )
     
        

        # Weight
        df1['weight'] = df1['weight'].astype( int )
        # Aplicando a média para os valores em outlier
        df1['height'] = df1['height'].apply(lambda x: 164 if x > 210 or x < 120 else x)
        # Aplicando a média para os valores em outlier
        df1['weight'] = df1['weight'].astype( int )
        df1['weight'] = df1['weight'].apply( lambda x: 74 if x<40 else x)
        # Aplicando a mediana para ap_lo Outliers
        df1['ap_lo'] = df1['ap_lo'].apply(lambda x: 80 if x > 120 or x < 40 else x)
        # Aplicando mediana para  ap_lo Outliers
        df1['ap_hi'] = df1['ap_hi'].apply(lambda x: 120 if x > 200 or x < 40 else x)
        # Criação da variável IMC
        df1['imc'] = (df1['weight'] / ((df1['height']/100)**2)).round(1)
        # Criação do status da variável IMCCreate a BMI Status Variable
        df1['imc_status'] = df1['imc'].apply(lambda x: 'muito magro' if x < 18.5 else 'normal' if (x >= 18.5) & (x <= 24.9)else 'sobrepeso' if (x >= 25) & (x <= 29.9) else 'obeso g-1' if (x >= 30.0) & (x <= 34.9)else 'obeso g-2' if (x >= 35.0) & (x <= 39.9)else 'obeso g-3')
        # Calculo da diferença do imc com o imc normal
        normal_imc = 21.7
        df1['imc_diff'] = df1['imc'] - normal_imc
        # Criação da variável pressão arterial 
        df1['pressão_arterial'] = ['0' if (ap_hi <= 105) & (ap_lo <= 60)else '1' if ((ap_hi > 105) & (ap_hi < 130)) & ((ap_lo > 60) & (ap_lo < 85))else '2' if ((ap_hi >= 130) & (ap_hi < 140)) & ((ap_lo >= 85) & (ap_lo < 90))else '3' if ((ap_hi >= 140) & (ap_hi < 160)) | ((ap_lo >= 90) & (ap_lo < 100))else '4' if ((ap_hi >= 160) & (ap_hi < 180)) | ((ap_lo >= 100) & (ap_lo < 110))else '5' for ap_hi, ap_lo in zip(df1['ap_hi'], df1['ap_lo'])]
        # Criação do status da variável pressão arterial
        df1['pressão_arterial_status'] = ['baixo' if (ap_hi <= 105) & (ap_lo <= 60)else 'normal' if ((ap_hi > 105) & (ap_hi < 130)) & ((ap_lo > 60) & (ap_lo < 85))else 'acima_normal' if ((ap_hi >= 130) & (ap_hi < 140)) & ((ap_lo >= 85) & (ap_lo < 90))else 'hipertenção_1' if ((ap_hi >= 140) & (ap_hi < 160)) | ((ap_lo >= 90) & (ap_lo < 100))else 'hipertenção_2' if ((ap_hi >= 160) & (ap_hi < 180)) | ((ap_lo >= 100) & (ap_lo < 110))else 'hipertenção_3' for ap_hi, ap_lo in zip(df1['ap_hi'], df1['ap_lo'])]
        # Criando variável de escala de risco variando de 0 a 1
        df1['escala_risco'] = df1['gluc'] + df1['cholesterol'] + df1['imc'] + df1['smoke'] + df1['alco'] + df1['active']
        df1.drop(['age_years'],axis=1, inplace=True)
        
        return df1
        
        
    def data_mudanca_dtypes(self, df2):
        # mudando dtypes
        df2['pressão_arterial'] = df2['pressão_arterial'].astype(int)
        
        return df2    
        
        
        
        
    def transformacao( self, df3 ):
    
        # selecionando as variaveis numericas
        
         df3['age'] = self.transformação_var_numerica.fit_transform(df3[['age']].values)
         df3['height'] = self.transformação_var_numerica.fit_transform(df3[['height']].values)
         df3['weight'] = self.transformação_var_numerica.fit_transform(df3[['weight']].values)
         df3['ap_hi'] = self.transformação_var_numerica.fit_transform(df3[['ap_hi']].values)
         df3['ap_lo'] = self.transformação_var_numerica.fit_transform(df3[['ap_lo']].values)
         df3['imc'] = self.transformação_var_numerica.fit_transform(df3[['imc']].values)
         df3['imc_diff'] = self.transformação_var_numerica.fit_transform(df3[['imc_diff']].values)
         df3['escala_risco'] = self.transformação_var_numerica.fit_transform(df3[['escala_risco']].values)
        
        
        # selecionando as variaveis categoricas
         df3 = pd.get_dummies( df3, prefix=['gender'], columns=['gender'] )
         df3 = df3[['age','weight','ap_hi','ap_lo','imc_diff','escala_risco','pressão_arterial']]
        
         return df3
    
    def get_prediction( self, model, original_data, test_data):
        
        pred = model.predict( test_data )
        original_data['predição'] = np.expml(pred)
        return original_data.to_json( orient='records', date_format='iso')
  
        
        
