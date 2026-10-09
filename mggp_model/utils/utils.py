import numpy as np 
import pandas as pd
from mggp import MGGP
from sklearn.preprocessing import StandardScaler
import numpy as np 
import os
import re 
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt

def sign(X1, X2):
    return np.sign(X1 - X2)

def sub(X1, X2):
    return X1 - X2

def mcase1(u, y):
    theta = [
        1.00,
        1.35e-3,
        -4.17e-1,
        -5.09e-3,
        -1.17e-4,
        -2.08e-1,
        1.31e-1,
        -1.87e-4,
        -4.36e-4
    ]

    u = np.asarray(u)
    y = np.asarray(y)

    N = len(u)
    y_pred = np.zeros(N)

    θ1, θ2, θ3, θ4, θ5, θ6, θ7, θ8, θ9 = theta

    # maior atraso = 15
    for k in range(15, N):
        y_pred[k] = (
            θ1 * y[k-1] +
            θ2 * u[k-1] * sub(u[k-12], u[k-1]) * sign(u[k-4], u[k-1]) +
            θ3 * sub(u[k-9], u[k-13]) +
            θ4 * y[k-2] * sub(u[k-12], u[k-10]) * sign(u[k-6], u[k-1]) +
            θ5 * y[k-8] * u[k-8] * sub(u[k-1], u[k-4]) +
            θ6 * sub(u[k-10], u[k-6]) +
            θ7 * sub(u[k-1], u[k-13]) +
            θ8 * sign(u[k-15], u[k-12]) +
            θ9 * u[k-1] * u[k-12] * sub(u[k-2], u[k-1])
        )

    return y_pred

def rungeKutta(x0, h, dvFunc, *args):
    # 1a chamada
    xd=dvFunc(x0, *args)
    savex0 = x0
    phi = xd
    x0 = savex0 + 0.5*h*xd

    # 2a chamada
    xd = dvFunc(x0, *args)
    phi = phi + 2*xd
    x0 = savex0 + 0.5*h*xd

    # 3a chamada
    xd = dvFunc(x0, *args)
    phi = phi+2*xd
    x0 = savex0+h*xd

    # 4a chamada
    xd = dvFunc(x0, *args)
    k = (phi+xd)/6
    x = savex0 + k*h
    return x


def butter_lowpass_filter(data, cutoff=1, fs=1000, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def dvFunc(_h, _udot, A = 0.9, beta = 0.008, gamma = 0.008):
    return A*_udot - beta*abs(_udot)*_h - gamma*_udot*abs(_h)


def simulate(u, udot, dp = 1.6):
    ht = np.zeros((len(u)))
    y = np.zeros((len(u)))
    h = 0.001
    for k in range(1,len(u)):
        ht[k]=rungeKutta(ht[k-1],h,dvFunc,udot[k-1])
        y[k] = dp*u[k] - ht[k]
    return y


def mggp_predict(X_val, y_val, 
                 evaluationTypeTest = "FreeRun", 
                 save_model="mggp_models_hysteresis/model_1.pkl"):
    
    loaded_model = MGGP(inputs=X_val,
                    outputs=y_val,
                    filename=save_model,
                    mode="NARX",  
                    operators=['mul', 'sign', 'subtraction']
                    # kwargs={'operators': ['mul', 'subtraction', 'sign']}
                    ).load_model()
    # loaded_model.theta[0][0] = 0
    args = (y_val, X_val)
    mggp_yp, mggp_yd = loaded_model.predict(evaluationTypeTest, *args)

    return mggp_yp, mggp_yd


def plot_top_correlations(results, u_train, y_train, top_n=10):
    """
    Plota as top N correlações mais fortes
    """
    # Encontrar as top N correlações por magnitude
    sorted_results = sorted(results.items(), 
                          key=lambda x: abs(x[1]['peak_value']), 
                          reverse=True)
    
    top_results = sorted_results[:top_n]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Valores das correlações máximas
    names = [f"{key.split('_Input_')[1]}\n→{key.split('Output_')[1].split('_Input_')[0]}" 
             for key, _ in top_results]
    values = [result['peak_value'] for _, result in top_results]
    lags = [result['peak_lag'] for _, result in top_results]
    
    bars = axes[0].bar(range(len(names)), values, color='skyblue', alpha=0.7)
    axes[0].set_title(f'Top {top_n} Correlações Cruzadas (Máximos)')
    axes[0].set_ylabel('Correlação Normalizada')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for i, (bar, lag) in enumerate(zip(bars, lags)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}\n(lag:{lag})',
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Heatmap de correlações
    correlation_matrix = np.zeros((len(y_train.columns), len(u_train.columns)))
    
    for output_idx, output_col in enumerate(y_train.columns):
        for input_idx, input_col in enumerate(u_train.columns):
            key = f"Output_{output_col}_Input_{input_col}"
            if key in results:
                correlation_matrix[output_idx, input_idx] = results[key]['peak_value']
    
    im = axes[1].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', 
                       vmin=-1, vmax=1)
    axes[1].set_title('Mapa de Calor de Correlações Máximas')
    axes[1].set_xlabel('Entradas')
    axes[1].set_ylabel('Saídas')
    axes[1].set_xticks(range(len(u_train.columns)))
    axes[1].set_yticks(range(len(y_train.columns)))
    axes[1].set_xticklabels([f'In{i}' for i in range(len(u_train.columns))], fontsize=8, rotation=90)
    axes[1].set_yticklabels(y_train.columns)
    
    # Adicionar barra de cores
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


def analyze_cross_correlations(u_train, y_train, max_lag):
    """
    Analisa correlações cruzadas entre múltiplas entradas e saídas
    
    Parameters:
    u_train: DataFrame com 51 colunas de entradas
    y_train: DataFrame com 3 colunas de saídas  
    max_lag: número máximo de atrasos para calcular
    """
    
    # Garantir que são DataFrames
    u_df = pd.DataFrame(u_train)
    y_df = pd.DataFrame(y_train)
    
    print(f"Shape u_train: {u_df.shape}")
    print(f"Shape y_train: {y_df.shape}")
    print(f"Número de entradas: {u_df.shape[1]}")
    print(f"Número de saídas: {y_df.shape[1]}")
    
    results = {}
    
    # Para cada combinação entrada-saída
    for output_idx, output_col in enumerate(y_df.columns):
        print(f"\n--- Análise para Saída: {output_col} ---")
        
        output_signal = y_df[output_col].values
        
        for input_idx, input_col in enumerate(u_df.columns):
            input_signal = u_df[input_col].values
            
            # Calcular correlação cruzada normalizada
            correlation = signal.correlate(output_signal, input_signal, mode='full', method='auto')
            
            # Normalização para obter valores entre -1 e 1
            norm_factor = np.sqrt(np.sum(input_signal**2) * np.sum(output_signal**2))
            if norm_factor > 0:
                correlation_normalized = correlation / norm_factor
            else:
                correlation_normalized = correlation
            
            # Encontrar o atraso correspondente
            lags = signal.correlation_lags(len(output_signal), len(input_signal), mode='full')
            
            # Encontrar o pico de correlação
            peak_idx = np.argmax(np.abs(correlation_normalized))
            peak_lag = lags[peak_idx]
            peak_value = correlation_normalized[peak_idx]
            
            # Armazenar resultados
            key = f"Output_{output_col}_Input_{input_col}"
            results[key] = {
                'peak_lag': peak_lag,
                'peak_value': peak_value,
                'correlation': correlation_normalized,
                'lags': lags
            }
            
            # Mostrar apenas correlações fortes (ajuste o threshold conforme necessário)
            if abs(peak_value) > 0.4:  # Threshold para correlações significativas
                print(f"  Entrada {input_idx:2d} ({input_col}): "
                      f"Lag = {peak_lag:3d}, Correlação = {peak_value:6.3f}")
    
    return results


def media_movel_pandas(data, window_size=5, min_periods=1):

    if len(data.shape) == 1:
        series = pd.Series(data)
        return series.rolling(window=window_size, min_periods=min_periods).mean().values
    
    else:
        smoothed = np.zeros_like(data)
        for i in range(data.shape[1]):
            series = pd.Series(data[:, i])
            smoothed[:, i] = series.rolling(window=window_size, min_periods=min_periods).mean().values
        return smoothed


def remove_window_with_noise(df, folder):

    def indexes_to_remove(window_folder):
        arquivos = os.listdir(window_folder)

        index_to_remove = []

        for arquivo in arquivos:
            if arquivo.startswith("window_") and arquivo.endswith(".png"):
                match = re.search(r'window_(\d+)\.png', arquivo)
                if match:
                    numero = int(match.group(1))
                    index_to_remove.append(numero)

        return index_to_remove

    window_folder = rf"E:\projetos\mggp\images\tire windowed rotations\{folder}_front\window_with_noise"
    indexes_front = indexes_to_remove(window_folder)
    
    window_folder = rf"E:\projetos\mggp\images\tire windowed rotations\{folder}_rear\window_with_noise"
    indexes_rear = indexes_to_remove(window_folder)

    indexes = list(set(indexes_front + indexes_rear))
    
    df_filtered = df[~df.index.isin(indexes)]

    return df_filtered


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