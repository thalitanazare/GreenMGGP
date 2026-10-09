import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path, output_features, folder, filter=None, tire=None):  
    df = pd.read_csv(file_path)
    if 'Initial_Time' in df.columns and 'Final_Time' in df.columns:
        df.drop(columns=['Initial_Time', 'Final_Time'], inplace=True)
    # df = remove_window_with_noise(df, folder)
    
    if 'Ax' in df.columns:
        df.drop(columns=['Ax', 'Ay'], inplace=True)
    # df.drop(columns=["Fx_F", "Fx_R", "Fz_F", "Fz_R"], inplace=True)
    # df.drop(columns=["Fz_F", "Fz_R"], inplace=True)

    if tire is not None:
        df = df[df['Pneu'] == tire]        
        # df.drop(columns=['Pneu', 'Fx', 'Fz'], inplace=True)
        df.drop(columns=['Pneu'], inplace=True)
    
    if filter is not None:
        df = df.filter(regex=filter)

    X = df[[c for c in df.columns if c not in ['Fx', 'Fy', 'Fz']]].values
    y = df[['Fx', 'Fy', 'Fz']].values/1000 # Normalizado de N para kN
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

        
    return X, y, x_scaler, y_scaler