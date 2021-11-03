import pickle
import pandas as pd
from flask             import Flask, request, Response


from cardiovascular.Cadiovascular import cadio_disease



# loading model
model = pickle.load( open( '/home/tulio/github/Cardiovascular-Predict/modelo/calibrated_model_ccd.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

@app.route( '/cardiovascular/predict', methods=['POST'] )
def cardiovascular_predict():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossmann class
        pipeline = cadio_disease()
        
        
        # data cleaning
        df1 = pipeline.data_novas_variaveis( test_raw )
        
        # feature engineering
        df2 = pipeline.data_mudanca_dtypes( df1 )
        
        # data preparation
        df3 = pipeline.transformacao( df2 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
        
        
    else:
        return Reponse( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    app.run( '0.0.0.0' )

