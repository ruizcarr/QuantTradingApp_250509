import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Get SETTINGS
from config import settings,utils
settings=settings.get_settings() #Edit Settings Dict at file config/settings.py

#Update Settings
#settings['start']='1996-01-01'

import Market_Data_Feed as mdf

#Get Data
data_ind = mdf.Data_Ind_Feed(settings).data_ind
data, _ = data_ind
data_dict = data.data_dict

tickers_returns=data.tickers_returns
cum_rets=(1+tickers_returns).cumprod()
#print('tickers_returns.describe()',tickers_returns.describe())


def compress_individual_tickers(tickers_returns, sigma_multiplier=4.0, rolling_years=4):
    """
    1. Comprime cada activo independientemente (máxima reducción de ruido).
    2. Reconstruye una matriz sincronizada de 'Escalera' para Markowitz.
    """

    # Diccionario para guardar las series comprimidas
    compressed_series = {}
    dates = tickers_returns.index
    # Volatilidad Rolling vectorizada para este activo
    window = int(252 * rolling_years)

    # --- FASE 1: PROCESAMIENTO INDIVIDUAL (Bucle por Columnas) ---
    for ticker in tickers_returns.columns:
        # Extraemos datos de un solo activo
        single_asset_rets = tickers_returns[ticker].values

        # Volatilidad Rolling vectorizada para este activo
        # Usamos Pandas para el rolling std, rellenando el inicio
        r_std = pd.Series(single_asset_rets).rolling(window=window, min_periods=30).std()
        r_std = r_std.bfill().fillna(0.01).values

        # Umbrales dinámicos
        log_thresholds = np.log1p(r_std * sigma_multiplier)

        # Lógica de Salto (Jump Logic) para 1D array
        log_price = np.log1p(single_asset_rets)
        cum_log_price = np.cumsum(log_price)

        events_idx = [0]  # Siempre incluimos el primer día
        current_idx = 0
        n = len(single_asset_rets)

        while current_idx < n - 1:
            anchor = cum_log_price[current_idx]
            future = cum_log_price[current_idx + 1:] - anchor

            # Check simple 1D
            breach = np.abs(future) >= log_thresholds[current_idx]

            if not np.any(breach):
                break

            # Salto al siguiente evento
            jump = np.argmax(breach)
            real_idx = current_idx + 1 + jump

            events_idx.append(real_idx)
            current_idx = real_idx

        # Reconstruimos la serie sparse de este activo
        # Solo guardamos los valores en los índices de evento
        # Creamos una serie con NaN en todo lo que no sea evento
        sparse_series = pd.Series(np.nan, index=dates)

        # En los puntos de evento, ponemos el precio acumulado real
        # Nota: Guardamos el PRECIO (Cumulative Return), no el retorno diario
        # Esto es vital para el paso de rellenado (ffill)
        sparse_series.iloc[events_idx] = (1 + tickers_returns[ticker]).cumprod().iloc[events_idx]

        compressed_series[ticker] = sparse_series

    # --- FASE 2: RECONSTRUCCIÓN  ---
    # Creamos un DataFrame con las columnas sparse
    df_sparse = pd.DataFrame(compressed_series)

    # 1. Forward Fill: Rellenamos los huecos con el último precio de evento conocido
    # Esto crea el efecto "Escalera" (Step Function)
    df_step_prices = df_sparse.ffill().fillna(1.0)

    # 2. Convertimos de nuevo a Retornos Diarios
    # Los días sin evento tendrán retorno 0.0
    # Los días de evento tendrán todo el retorno acumulado del periodo
    df_step_returns = df_step_prices.pct_change().fillna(0)

    return df_step_returns


def generate_final_weights(tickers_returns, sigma=3.0, rolling_yrs=4):
    """
    Create df wih crosing mean signal of compress prices
    """
    compress_mean_weights = pd.DataFrame(0.0, index=tickers_returns.index, columns=tickers_returns.columns)

    for t in tickers_returns.columns:
        # 1. Compresión de Salto
        step_rets = compress_individual_tickers(tickers_returns[[t]], sigma, rolling_yrs)

        # 2. Aislamiento de Eventos (Time-Series Pura)
        event_only_rets = step_rets[step_rets[t] != 0][t]

        if len(event_only_rets) < 20:
            continue

        event_only_prices = (1 + event_only_rets).cumprod()

        # 3. Ventanas Dinámicas Normalizadas a 200 días
        ratio = len(event_only_rets) / len(tickers_returns)

        # Normalización a 200 días según tu especificación
        slow_v = max(8, int(200 * ratio))
        fast_v = 1 #max(2, int(slow_v / 4))

        print('slow_v', t, slow_v)

        # 4. Medias y Cruce en Espacio de Eventos
        sma_f = event_only_prices.rolling(window=fast_v).mean()
        sma_s = event_only_prices.rolling(window=slow_v).mean()

        # 5. Señal Binaria y Proyección Causal
        event_signal = (sma_f > sma_s).astype(float)
        full_signal = event_signal.reindex(tickers_returns.index).ffill().fillna(1)

        # Shift(1) para evitar look-ahead bias
        compress_mean_weights[t] = full_signal.shift(1).fillna(1)

    return compress_mean_weights


# --- EJECUCIÓN ---
compress_mean_weights = generate_final_weights(tickers_returns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def backtest_total_with_cash(tickers_returns, weights_df, benchmark_ticker='ES=F'):
    # 1. Datos completos sin recortes
    rets = tickers_returns
    weights = weights_df

    # 2. Benchmark de Referencia
    bench_rets = rets[benchmark_ticker]
    cum_bench = (1 + bench_rets).cumprod()

    comparison_data = []
    # No filtramos: procesamos TODAS las columnas (incluyendo cash)
    all_tickers = rets.columns

    equity_curves = {}
    equity_curves[benchmark_ticker] = cum_bench

    for asset in all_tickers:
        # Retorno de la estrategia para el activo (o cash)
        strat_rets = rets[asset] * weights[asset]
        cum_strat = (1 + strat_rets).cumprod()
        equity_curves[asset] = cum_strat

        # Métricas
        ann_ret = (1 + strat_rets.mean()) ** 252 - 1
        vol = strat_rets.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0.0001 else 0

        # Drawdown
        dd = (cum_strat / cum_strat.cummax()) - 1
        max_dd = dd.min()

        # Alpha vs S&P 500
        bench_ann_ret = (1 + bench_rets.mean()) ** 252 - 1
        alpha = ann_ret - bench_ann_ret

        comparison_data.append({
            'Activo': asset,
            'CAGR': ann_ret,
            'Volatilidad': vol,
            'Sharpe': sharpe,
            'Max DD': max_dd,
            'Alpha vs ES=F': alpha
        })

    df_results = pd.DataFrame(comparison_data).set_index('Activo')

    # --- VISUALIZACIÓN ---
    plt.figure(figsize=(16, 9))

    # S&P 500 como referencia visual
    plt.plot(cum_bench, label=f'BENCHMARK: {benchmark_ticker}', color='black', lw=3.5, zorder=10)

    # Graficamos todos los activos
    for asset in all_tickers:
        # Resaltamos el CASH y los mejores activos
        is_cash = 'cash' in asset.lower()
        color = 'lime' if is_cash else None
        alpha = 0.8 if (is_cash or asset in df_results.sort_values('Sharpe').tail(3).index) else 0.15
        lw = 2.5 if (is_cash or alpha > 0.5) else 1

        plt.plot(equity_curves[asset], label=f'{asset}', alpha=alpha, lw=lw, color=color if is_cash else None)

    plt.title(f"Sistema Event-Time (Sigma 1.5, Win 200) - Histórico Total incluyendo Cash", fontsize=14)
    plt.yscale('log')
    plt.ylabel("Equity (Escala Log)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.grid(True, alpha=0.15)
    plt.tight_layout()

    return df_results


# Ejecución
final_metrics = backtest_total_with_cash(tickers_returns, compress_mean_weights)

import math


def plot_individual_assets(tickers_returns, weights_df, benchmark_ticker='ES=F'):
    all_tickers = tickers_returns.columns
    n_assets = len(all_tickers)
    cols = 3
    rows = math.ceil(n_assets / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), sharex=False)
    axes = axes.flatten()

    # Pre-calculamos el Benchmark (S&P 500)
    cum_bench = (1 + tickers_returns[benchmark_ticker]).cumprod()

    for i, asset in enumerate(all_tickers):
        ax = axes[i]

        # Cálculo de la estrategia para el activo
        strat_rets = tickers_returns[asset] * weights_df[asset]
        cum_strat = (1 + strat_rets).cumprod()

        # Plotting
        ax.plot(cum_bench, color='gray', alpha=0.3, label='Benchmark (ES=F)')
        ax.plot(cum_strat, color='#1f77b4', lw=2, label=f'Estrategia {asset}')

        # Formato
        ax.set_title(f"Activo: {asset}", fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.2)

        # Añadir métricas rápidas en el recuadro
        cagr = (1 + strat_rets.mean()) ** 252 - 1
        sharpe = cagr / (strat_rets.std() * np.sqrt(252)) if strat_rets.std() > 0 else 0
        ax.text(0.05, 0.9, f"Sharpe: {sharpe:.2f}\nCAGR: {cagr:.1%}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        if i == 0:
            ax.legend(loc='upper left', fontsize='small')

    # Eliminar ejes vacíos si los hay
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()


# Ejecución de la rejilla individual
plot_individual_assets(tickers_returns, compress_mean_weights)

# Tabla de resultados final
print("\n--- RANKING DE ACTIVOS (INCLUYENDO CASH) ---")
print(final_metrics.sort_values('Sharpe', ascending=False).to_string(formatters={
    'CAGR': '{:,.2%}'.format,
    'Volatilidad': '{:,.2%}'.format,
    'Sharpe': '{:,.2f}'.format,
    'Max DD': '{:,.2%}'.format,
    'Alpha vs ES=F': '{:+.2%}'.format
}))





if False:
    compress_rets=data.compressed_rets
    print('compress_rets.describe()',compress_rets.describe())
    print('compress_rets',compress_rets.tail(20))

    daily_compress_rets = compress_rets.reindex(tickers_returns.index).fillna(0)
    print('daily_compress_rets',daily_compress_rets.tail(20))
    #print('daily_compress_rets.describe()',daily_compress_rets.describe())
    daily_cum_compress_rets = (1 + daily_compress_rets).cumprod()

if False:
    cum_compress_rets=(1+compress_rets).cumprod()
    cum_compress_rets_mean_slow=cum_compress_rets.shift(1).rolling(20).mean()
    cum_compress_rets_mean_fast=cum_compress_rets.shift(1).rolling(5).mean()




    #cum_semi_compress_rets=(cum_compress_rets+cum_rets)/2
    cum_rets_mean_slow=cum_rets.shift(1).rolling(20).mean()
    #cum_semi_compress_rets_mean_slow=cum_semi_compress_rets.shift(1).rolling(20).mean()
    #cum_semi_compress_rets_mean_fast=cum_semi_compress_rets.shift(1).rolling(5).mean()
    #cum_compress_rets_mean_slow=daily_cum_compress_rets.shift(1).rolling(20).mean()

    #semi_compress_rets=(tickers_returns+daily_compress_rets)/2
    #semi_compress_rets=cum_semi_compress_rets.pct_change().fillna(0)
    #print('semi_compress_rets.describe()',semi_compress_rets.describe())

    plot_df1=pd.DataFrame()
    for ticker in cum_compress_rets.columns:
        plot_df1['cum_compress_rets']=cum_compress_rets[ticker]
        plot_df1['cum_compress_rets_mean_slow']=cum_compress_rets_mean_slow[ticker]
        plot_df1['cum_compress_rets_mean_fast'] = cum_compress_rets_mean_fast[ticker]
        plot_df1.plot(title=ticker)


    if False:
        plot_df=pd.DataFrame()
        for ticker in tickers_returns.columns:
            plot_df['daily_cum_compress_rets']=daily_cum_compress_rets[ticker]
            #plot_df['cum_semi_compress_rets'] = cum_semi_compress_rets[ticker]
            plot_df['cum_rets_mean_slow'] = cum_rets_mean_slow[ticker]
            #plot_df['cum_semi_compress_rets_mean_slow'] = cum_semi_compress_rets_mean_slow[ticker]
            plot_df['cum_compress_rets_mean_slow'] = cum_compress_rets_mean_slow[ticker]
            plot_df['cum_rets'] = cum_rets[ticker]
            #print(ticker,plot_df)
            plot_df.plot(title=ticker)




plt.show()