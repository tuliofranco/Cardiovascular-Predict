from flask import Flask, request, Response
import pickle
import pandas as pd
from cardiovascular import Cardiovascular
# loading model
model = pickle.load( open( r"C:\Users\tulio.carvalho\Documents\CardiovascularDisease\modelo\calibrated_model_ccd.pkl", 'rb' ) )


# iniciar API
app = Flask( __name__ )
@app.route( '/cadiovascular_disease/predict', methods=['POST'] )
def cardiovascular_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance( test_json, dict ):
            df = pd.DataFrame( test_json, index=[0] )
            
        else:
            df = pd.DataFrame( test_json, columns = test_json[0].keys() )
            
            # data_novas_variaveis
            df = pipeline.data_novas_variaveis( test_json )
            
            
            # data_mudança_dtypes
            df1 = pipeline.data_mudança_dtypes(df)
            
            # transformação
            df2 = pipeline.transformação(df1)
            
            # prediction
            df_response = pipeline.get_prediction( model, test_json,  df2 )
            
            return df_response
            
            
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
    
if __name__=='__main__':
    app.run( '0.0.0.0')
