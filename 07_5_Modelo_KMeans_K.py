import pandas as pd
from plot import plot_mse_k

df = pd.read_csv('out_model_kmeans.csv')
df_boya = pd.read_csv('boyas.csv')
k = [i for i in range(2, 501)]

for nombre in df_boya['Nombre']:
    df_name = df[df['loc'] == nombre].reset_index()
    for model in ['GOW', 'COP']:
        df_ = df_name[df_name['model'] == model].reset_index()

        plot_mse_k(k, 200, df_['mse'], fname=f'plot/model/05_KMeans/{nombre}_k_kmeans_{model}.png')