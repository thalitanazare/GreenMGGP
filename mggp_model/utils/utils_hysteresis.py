import numpy as np 
import pandas as pd
from mggp import MGGP
import os
import re 
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
import pickle
# from tensorflow.keras import backend as K
# import keras
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

def sign(X1, X2):
    return np.sign(X1 - X2)[0]

def sub(X1, X2):
    res = X1 - X2
    return res[0]

def mcase_21(u, y_init):

    u = np.asarray(u)
    N = len(u)
    u = np.asarray([x[0] for x in u])
    y_pred = np.zeros(N)
    terms = np.zeros((11, N))

    y_pred[:3] = y_init[:3].reshape(-1)

    for k in range(3, N):
        y_pred[k] = (
                # -1.24900e-16 * 
                # 3.97151e-14 * sign(u[k] , u[k]) +
                -2.14570e+00 * sign(u[k] , u[k-1]) * u[k] +
                1.02547e+00 * y_pred[k-1] +
                # 1.94289e-16 * u[k] +
                -5.79036e-02 * (u[k] - u[k-1]) * u[k-2] +
                -1.11516e-01 * sign(u[k-1] , u[k]) * sign(u[k] , u[k-1]) +
                2.15428e+00 * u[k-1] * sign(u[k] , u[k-1]) +
                -2.54746e-02 * y_pred[k-2] +
                5.29324e+00 * (u[k] - u[k-1]) +
                7.67350e-01 * sign(u[k] , u[k-1]) * u[k] * (u[k] - u[k-1]) +
                7.22488e-01 * sign(u[k-2] , u[k-3]) +
                7.49283e-02 * (u[k] - u[k-1]) * (u[k] - u[k-1]) 
                # 0.00000e+00 * (u[k-1] - u[k-1]) +
                # 0.00000e+00 * (u[k] - u[k]) * u[k] 
        )

    return y_pred[3:]


def mcase30_free_run_terms(u, y_init):
    u = np.asarray(u).reshape(-1)
    y_init = np.asarray(y_init).reshape(-1)

    N = len(u)

    y_pred = np.zeros(N)

    n_terms = 12
    terms = np.zeros((n_terms, N))

    y_pred[:2] = y_init[:2]

    for k in range(2, N):

        du = u[k] - u[k-1]
        s  = np.sign(du)
        s_inv = np.sign(u[k-1] - u[k])

        # --- termos simplificados ---
        terms[0, k]  = -1.34210e+00 * s
        terms[1, k]  =  3.03195e-02 * s * y_pred[k-1] * s
        terms[2, k]  = -1.06222e+00 * y_pred[k-1] * s * du
        terms[3, k]  =  1.00000e+00 * y_pred[k-1]
        terms[4, k]  = -3.35254e-03 * du * y_pred[k-1] * y_pred[k-1]
        terms[5, k]  = -5.63146e-06 * y_pred[k-2] * u[k-1] * y_pred[k-1] * y_pred[k-1] * du
        terms[6, k]  =  9.48303e+00 * (u[k-1] - u[k]) * s
        terms[7, k]  =  7.81407e+00 * du * u[k-1] * s
        terms[8, k]  = -2.51488e-01 * y_pred[k-1] * du
        terms[9, k]  = -1.02552e-03 * s * u[k-1] * y_pred[k-1] * u[k]
        terms[10, k] =  4.96143e-01 * u[k-1] * u[k-1] * du
        terms[11, k] =  1.26599e-02 * y_pred[k-1] * u[k-1] * s

        # saída
        y_pred[k] = np.sum(terms[:, k])

    return y_pred[2:], terms[:, 2:]


def mcase_butterfly_free_run_terms(u, y_init):
    u = np.asarray(u).reshape(-1)
    y_init = np.asarray(y_init).reshape(-1)

    N = len(u)

    y_pred = np.zeros(N)

    n_terms = 14
    terms = np.zeros((n_terms, N))

    # Necessário devido aos atrasos y[k-2] e y[k-3]
    y_pred[:3] = y_init[:3]

    for k in range(3, N):

        du = u[k] - u[k-1]

        s = np.sign(du)
        s_inv = np.sign(u[k-1] - u[k])

        terms[0, k]  = 9.21945e-07 * u[k] * u[k-1] * s
        terms[1, k]  =  5.01185e-01 * y_pred[k-3]
        terms[2, k]  = -1.87658e-04 * s_inv * du * u[k]**2
        terms[3, k]  = -4.77906e-03 * u[k-3]
        terms[4, k]  = -8.31818e-01 * u[k]
        terms[5, k]  =  4.98815e-01 * y_pred[k-2]
        terms[6, k]  =  1.61398e-01 * s * s
        terms[7, k]  = -8.56080e-01 * u[k-2]
        terms[8, k]  =  2.84447e-07 * u[k-1] * u[k]**2 * du
        terms[9, k]  = -1.83973e-02 * s
        terms[10, k] = -2.43298e-02 * (u[k-1] - u[k])**2
        terms[11, k] = -2.74938e+00 * s * du
        terms[12, k] =  1.69268e+00 * u[k-1]
        terms[13, k] = -7.89230e-02 * (u[k-1] - u[k]) * u[k-1]

        y_pred[k] = np.sum(terms[:, k])

    return y_pred[3:], terms[:, 3:]

def mcase24_free_run_terms(u, y_init):
    u = np.asarray(u).reshape(-1)
    y_init = np.asarray(y_init).reshape(-1)

    N = len(u)

    y_pred = np.zeros(N)

    # número de termos da equação
    n_terms = 15
    terms = np.zeros((n_terms, N))

    # inicialização
    y_pred[:1] = y_init[:1]

    for k in range(1, N):

        du = u[k] - u[k-1]
        s  = np.sign(du)
        s_inv = np.sign(u[k-1] - u[k])  # equivalente a -s (mas mantendo explícito)

        # --- termos ---
        terms[0, k]  =  1.79890e+01 * u[k]
        terms[1, k]  =  3.75504e-01 * s * u[k-1] * u[k]
        terms[2, k]  = -1.11655e+00 * s_inv * u[k-1] * s
        terms[3, k]  = -1.79890e+01 * u[k-1]
        terms[4, k]  = -1.17417e-01 * s * y_pred[k-1] * s
        terms[5, k]  =  1.00000e+00 * y_pred[k-1]
        terms[6, k]  = -1.13020e+00 * s * s
        terms[7, k]  = -5.19738e-02 * y_pred[k-1] * s
        terms[8, k]  =  2.56990e-02 * u[k-1] * u[k] * u[k] * du
        terms[9, k]  = -5.58052e+00 * s
        terms[10, k] =  3.80298e-01 * u[k-1] * u[k-1] * (u[k-1] - u[k])
        terms[11, k] =  4.62254e-01 * (u[k-1] - u[k]) * u[k-1]
        terms[12, k] =  1.18891e-01 * du * y_pred[k-1]
        terms[13, k] = -3.02835e-02 * u[k]**3 * s
        terms[14, k] = -7.69282e-01 * du * u[k]

        # saída
        y_pred[k] = np.sum(terms[:, k])

    return y_pred[1:], terms[:, 1:]


def comparecase_free_run(u, y_init):
    u = np.asarray(u)
    N = len(u)

    y_pred = np.zeros(N)
    terms = np.zeros((6, N))

    y_pred[:3] = y_init[:3].reshape(-1)

    theta = [1,
            0.77,
            1.44e-2,
            -9.6e-3,
            3.15e-4,
            -2.46e-4
        ]
    # phi1 = sub
    # phi2 = sign

    for k in range(3, N):
        terms[0, k] = theta[0] * y_pred[k-1]
        terms[1, k] = theta[1] * sub(u[k-1], u[k-2])
        terms[2, k] = theta[2] * u[k-2][0] * sign(u[k-2], u[k-3]) * sub(u[k-2], u[k-3])
        terms[3, k] = theta[3] * y_pred[k-1] * sign(u[k-2], u[k-3]) * sub(u[k-2], u[k-3])
        terms[4, k] = theta[4] * u[k - 2][0] * u[k - 2][0] * sub(u[k-2], u[k-3])
        terms[5, k] = theta[5] * u[k - 2][0] * y_pred[k-1] * sub(u[k-2], u[k-3])

        y_pred[k] = np.sum(terms[:, k])

    return y_pred[3:], terms[:, 3:]


def mcase_free_run_terms(u, y_init):
    u = np.asarray(u)
    N = len(u)

    y_pred = np.zeros(N)
    terms = np.zeros((11, N))

    y_pred[:10] = y_init[:10].reshape(-1)

    theta = [
        -6.449701883681769e-14,
        0.300742221603808,
        1.157059550460019,
        0.03974769250889043,
        -0.09372108147028588,
        1.0571898587117923,
        -0.05383488830604312,
        0.05397338896144662,
        -3.271439267907053,
        1.0571898587352877,
        0.7530926667022311
    ]

    for k in range(10, N):
              
        terms[0, k]  = theta[0]
        terms[1, k]  = theta[1] * y_pred[k-3]
        terms[2, k]  = theta[2] * u[k-11]
        terms[3, k]  = theta[3] * sub(u[k-6], u[k-7])
        terms[4, k]  = theta[4] * sub(u[k-10], u[k-4])
        terms[5, k]  = theta[5] * u[k-1]
        terms[6, k]  = theta[6] * y_pred[k-11]
        terms[7, k]  = theta[7] * sign(u[k-2], u[k-5])
        terms[8, k]  = theta[8] * u[k-7]
        terms[9, k]  = theta[9] * u[k-1]
        terms[10, k] = theta[10] * y_pred[k-1]

        y_pred[k] = np.sum(terms[:, k])

    return y_pred[10:], terms[:, 10:]


def insert_constant_segments(signal, n_const=3, len_const=100, seed=42):
    """
    Insere segmentos constantes em posições aleatórias, mas reprodutíveis.

    Parâmetros:
    - signal: array (N,1)
    - n_const: número de segmentos constantes
    - len_const: tamanho de cada segmento
    - seed: semente do gerador aleatório

    Retorna:
    - novo sinal com trechos constantes inseridos
    """
    
    rng = np.random.default_rng(seed)
    
    signal = signal.flatten()
    N = len(signal)

    positions = np.sort(rng.choice(np.arange(1, N-1), n_const, replace=False))

    new_signal = []
    last_idx = 0

    for pos in positions:
        new_signal.extend(signal[last_idx:pos])

        const_value = signal[pos]
        new_signal.extend([const_value] * len_const)

        last_idx = pos

    new_signal.extend(signal[last_idx:])

    return np.array(new_signal).reshape(-1, 1)


def mcase(u, y_init):
    u = np.asarray(u)
    N = len(u)

    y_pred = np.zeros(N)

    y_pred[:10] = y_init[:10].reshape(-1)

    theta = [
        -6.449701883681769e-14,
        0.300742221603808,
        1.157059550460019,
        0.03974769250889043,
        -0.09372108147028588,
        1.0571898587117923,
        -0.05383488830604312,
        0.05397338896144662,
        -3.271439267907053,
        1.0571898587352877,
        0.7530926667022311
    ]

    for k in range(10, N):

        y_pred[k] = (
            theta[0] +
            theta[1] * y_pred[k-3] +
            theta[2] * u[k-11] +
            theta[3] * sub(u[k-6], u[k-7]) +
            theta[4] * sub(u[k-10], u[k-4]) +
            theta[5] * u[k-1] +
            theta[6] * y_pred[k-11] +
            theta[7] * sign(u[k-2], u[k-5]) +
            theta[8] * u[k-7] +
            theta[9] * u[k-1] +
            theta[10] * y_pred[k-1]  
        )

    return y_pred


def mcaseOSA(u, y):
    u = np.asarray(u)
    y = np.asarray(y)

    N = len(u)
    y_pred = np.zeros(N)
    
    y_pred[:10] = y[:10].reshape(-1)
    theta = [
        -6.449701883681769e-14,
        0.300742221603808,
        1.157059550460019,
        0.03974769250889043,
        -0.09372108147028588,
        1.0571898587117923,
        -0.05383488830604312,
        0.05397338896144662,
        -3.271439267907053,
        1.0571898587352877,
        0.7530926667022311
    ]

    for k in range(10, N):

        y_pred[k] = (
            theta[0] +  # bias
            theta[1] * y[k-3] +
            theta[2] * u[k-11] +
            theta[3] * sub(u[k-6], u[k-7]) +
            theta[4] * sub(u[k-10], u[k-4]) +
            theta[5] * u[k-1] +
            theta[6] * y[k-11] +
            theta[7] * sign(u[k-2], u[k-5]) +
            theta[8] * u[k-7] +
            theta[9] * u[k-1] +
            theta[10] * y[k-1]
        )

    return y_pred


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


def mcase2(u, y_init):
    theta = [0,
             2.10e-2,
             2.35e-2,
             1.52e-4,
             -1.52e-4,
             3.14,
             -3.14,
             -2.15e-2,
             1.38e+1,
             1.00]

    u = np.asarray(u)
    N = len(u)

    y_pred = np.zeros(N)
    y_pred[:10] = y_init[:10].reshape(-1)

    θ0, θ1, θ2, θ3, θ4, θ5, θ6, θ7, θ8, θ9 = theta

    for k in range(10, N):
        y_pred[k] = (
            θ0 +
            θ1 * sign(u[k-5], u[k-10]) +
            θ2 * sign(u[k-1], u[k-3]) +
            θ3 * y_pred[k-6] * u[k-10] +
            θ4 * y_pred[k-1] * u[k-4] +
            θ5 * u[k-9] +
            θ6 * u[k-7] +
            θ7 * sign(u[k-9], u[k-2]) +
            θ8 * sub(u[k-5], u[k-6]) +
            θ9 * y_pred[k-1]
        )

    return y_pred


# Caso piezoeletrico
def mcase2OSA(u, y):

    theta = [2.10e-2,
            2.35e-2,
            1.52e-4,
            -1.52e-4,
            3.14,
            -3.14,
            -2.15e-2,
            1.38e+1,
            1.00]

    u = np.asarray(u)
    y = np.asarray(y)

    N = len(u)

    y_pred = np.zeros(N)

    θ1, θ2, θ3, θ4, θ5, θ6, θ7, θ8, θ9 = theta

    # maior atraso necessário = 10
    for k in range(10, N):
        y_pred[k] = (
            θ1 * sign(u[k-5], u[k-10]) +
            θ2 * sign(u[k-1], u[k-3]) +
            θ3 * y[k-6] * u[k-10] +
            θ4 * y[k-1] * u[k-4] +
            θ5 * u[k-9] +
            θ6 * u[k-7] +
            θ7 * sign(u[k-9], u[k-2]) +
            θ8 * sub(u[k-5], u[k-6]) +
            θ9 * y[k-1]
        )

    return y_pred

def mggp_predict(X_val, y_val, 
                 evaluationTypeTest = "FreeRun", 
                 save_model="mggp_models_hysteresis/model_1.pkl"):
    
    loaded_model = MGGP(inputs=X_val,
                    outputs=y_val,
                    filename=save_model,
                    mode="NARX",
                    kwargs={'operators': ['mul', 'subtraction', 'sign']}).load_model()

    loaded_model.theta = loaded_model.hysteretic_constrained_ls(y_val, X_val)
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


def load_data(file_path, folder, filter=None, tire=None):  
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
            
    return X, y


def calculate_metrics(y_true, y_pred, model_name):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # em porcentagem
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
        
    return mape, mae, r2


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


def save(filename, dictionary):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary,f)
        f.close()


def create_sequences(u_data, y_data, time_steps=30):
    X, y = [], []

    for i in range(len(u_data) - time_steps):
        seq = np.column_stack((u_data[i:(i + time_steps)], 
                              y_data[i:(i + time_steps)]))
        X.append(seq)
        y.append(y_data[i + time_steps])

    return np.array(X), np.array(y)

# @keras.saving.register_keras_serializable()
# def r_squared(y_true, y_pred):
#     SS_res = K.sum(K.square(y_true - y_pred))
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
#     # return K.clip(1 - SS_res/(SS_tot + K.epsilon()), 0, 1)
#     return 1 - SS_res/SS_tot


def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot