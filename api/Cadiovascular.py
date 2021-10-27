import pickle
import pandas as pd

class cadio_disease(object):
    def __init__(  self   ):
        state = 1
        self.home_path = r"C:\Users\tulio.carvalho\Documents\CardiovascularDisease"
        self.escala_risco_transformação = pickle.load( open(self.home_path + 'parametros/encoding_escala_risco.pkl','rb'))
        self.transformação_var_numerica = pickle.load( open(self.home_path + 'parametros/encoding_vars_numericas_train.pkl','rb'))
        self.balancear_dados            = pickle.load( open(self.home_path + 'parametros/balancear_dados.pkl','rb'))
    
    def data_novas_variaveis(self, df):
        # age - days/365
        year = lambda x : x/365
        df['age_years'] = list(map( year, df['age'] ) )
        # Weight
        df['weight'] = df['weight'].astype( int )
        # Aplicando a média para os valores em outlier
        df['height'] = df['height'].apply(lambda x: 164 if x > 210 or x < 120 else x)
        # Aplicando a média para os valores em outlier
        df['weight'] = df['weight'].apply( lambda x: 74 if x<40 else x)
        # Aplicando a mediana para ap_lo Outliers
        df['ap_lo'] = df['ap_lo'].apply(lambda x: 80 if x > 120 or x < 40 else x)
        # Aplicando mediana para  ap_lo Outliers
        df['ap_hi'] = df['ap_hi'].apply(lambda x: 120 if x > 200 or x < 40 else x)
        # Criação da variável IMC
        df['imc'] = (df['weight'] / ((df['height']/100)**2)).round(1)
        # Criação do status da variável IMCCreate a BMI Status Variable
        df['imc_status'] = df['imc'].apply(lambda x: 'muito magro' if x < 18.5 else 'normal' if (x >= 18.5) & (x <= 24.9)else 'sobrepeso' if (x >= 25) & (x <= 29.9) else 'obeso g-1' if (x >= 30.0) & (x <= 34.9)else 'obeso g-2' if (x >= 35.0) & (x <= 39.9)else 'obeso g-3')
        # Calculo da diferença do imc com o imc normal
        normal_imc = 21.7
        df['imc_diff'] = df['imc'] - normal_imc
        # Criação da variável pressão arterial 
        df['pressão_arterial'] = ['0' if (ap_hi <= 105) & (ap_lo <= 60)else '1' if ((ap_hi > 105) & (ap_hi < 130)) & ((ap_lo > 60) & (ap_lo < 85))else '2' if ((ap_hi >= 130) & (ap_hi < 140)) & ((ap_lo >= 85) & (ap_lo < 90))else '3' if ((ap_hi >= 140) & (ap_hi < 160)) | ((ap_lo >= 90) & (ap_lo < 100))else '4' if ((ap_hi >= 160) & (ap_hi < 180)) | ((ap_lo >= 100) & (ap_lo < 110))else '5' for ap_hi, ap_lo in zip(df['ap_hi'], df['ap_lo'])]
        # Criação do status da variável pressão arterial
        df['pressão_arterial_status'] = ['baixo' if (ap_hi <= 105) & (ap_lo <= 60)else 'normal' if ((ap_hi > 105) & (ap_hi < 130)) & ((ap_lo > 60) & (ap_lo < 85))else 'acima_normal' if ((ap_hi >= 130) & (ap_hi < 140)) & ((ap_lo >= 85) & (ap_lo < 90))else 'hipertenção_1' if ((ap_hi >= 140) & (ap_hi < 160)) | ((ap_lo >= 90) & (ap_lo < 100))else 'hipertenção_2' if ((ap_hi >= 160) & (ap_hi < 180)) | ((ap_lo >= 100) & (ap_lo < 110))else 'hipertenção_3' for ap_hi, ap_lo in zip(df['ap_hi'], df['ap_lo'])]
        # Criando variável de escala de risco variando de 0 a 1
        df['escala_risco'] = df['gluc'] + df['cholesterol'] + df['imc'] + df['smoke'] + df['alco'] + df['active']
        df.drop(['id','age_years'],axis=1, inplace=True)
        df.self.balancear_dados.fit_resample(df)
        
        return df

    def data_mudança_dtypes(self, df1):
        # mudando dtypes
        df1['age_years'] = df1['age_years'].astype( int )
        df1['pressão_arterial'] = df1['pressão_arterial'].astype(int)
        return df1

    def transformação( self, df2 ):
        df3 = df2[['age','age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'imc', 'escala_risco' ]]
        df3 = self.transformação_var_numerica.fit_transform(df3)
        df4 = df2[['gender']]
        df4 = pd.get_dummies(df4, drop_first=True)
        df5 = df2[['smoke', 'alco', 'active']]
        df2 = pd.concat([df3, df4, df5])
        cols_selected = df2['age','weight','ap_hi','ap_lo','imc_diff','escala_risco','pressão_arterial']
        
        return df2['age','weight','ap_hi','ap_lo','imc_diff','escala_risco','pressão_arterial']
    
    def get_prediction( self, model, original_data, test_data):
        
        pred = model.predict( test_data )
        original_data['predição'] = np.expml(pred)
        return original_data.to_json( orient='records', date_format='iso')