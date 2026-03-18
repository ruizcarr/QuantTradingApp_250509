# import libraries and functions
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from Markowitz_Vectorized import compute_optimized_markowitz_d_w
from utils import weighted_mean_of_dfs_dict, create_results_by_period_df,mean_positions
from MarkowitzWeights import MarkowitzWeights


class Strategy:
    """
    This class implements a portfolio optimization strategy based on the Markowitz model.

    Attributes:
        inputs:
            settings: A dictionary containing various settings for the strategy.
            st_tickers_returns: A DataFrame containing historical returns for the assets.
            indicators_dict: A dictionary containing technical indicators.
        outputs:
            weights_df: A DataFrame containing the calculated portfolio weights.
            positions: A DataFrame containing the positions to be taken.
            strategy_returns: A Series containing the strategy returns.

    Methods:
        calculate_markowitz_weights: Calculates the optimal portfolio weights using the Markowitz model.
        apply_strategy_weights: Applies additional strategy weights (e.g., RSI-based) to the Markowitz weights.
        calculate_returns: Calculates the strategy returns based on the final positions and ticker returns.
    """
    def __init__(self,settings,st_tickers_returns,indicators_dict):

        def trim_sources_to_common_start(sources: dict) -> dict:
            """
            Drop leading rows where any source has all-zero weights.
            Returns sources trimmed to the first date where ALL sources
            have at least one non-zero weight row.
            """
            first_valid = {}
            for name, (w, factor) in sources.items():
                # Find first row where any weight is non-zero
                nonzero_mask = (w.abs() > 1e-8).any(axis=1)
                if nonzero_mask.any():
                    first_valid[name] = nonzero_mask.idxmax()
                else:
                    raise ValueError(f"Source '{name}' has no non-zero weights at all.")

            # Use the latest of the first-valid dates across all sources
            common_start = max(first_valid.values())
            #print(f"[trim_sources] First valid dates per source: { {k: str(v.date()) for k, v in first_valid.items()} }")
            #print(f"[trim_sources] Trimming to common start: {common_start.date()}")

            return {
                name: (w.loc[common_start:], factor)
                for name, (w, factor) in sources.items()
            }

        def compute_blend_factors(sources: dict, returns_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
            """
            Compute normalized blend factors for each source based on rolling performance.

            Parameters
            ----------
            sources      : dict  name -> (weights_df, fixed_factor)
            returns_df   : pd.DataFrame  DatetimeIndex x tickers, daily returns
            settings     : dict  expects keys:
                               'blend_metric'      : 'cagr' | 'sharpe' | 'mean' | 'utility' | 'fixed'
                               'blend_window'      : int, rolling window in days             (default 60)
                               'blend_min_periods' : int, min obs before factor is active    (default window)
                               'ann_factor'        : int, annualization factor               (default 252)
                               'blend_lambda'      : float, vol penalty for utility metric   (default 1.0)

            Returns
            -------
            pd.DataFrame  DatetimeIndex x source_names, normalized factors (each row sums to 1)
            """

            metric = settings.get('blend_metric', 'fixed')
            window = settings.get('blend_window', 60)
            min_periods = settings.get('blend_min_periods', window)
            ann_factor = settings.get('ann_factor', 252)
            lam = settings.get('blend_lambda', 1.0)

            reference_index = next(iter(sources.values()))[0].index

            # ------------------------------------------------------------------
            # FIXED — use the fixed_factor stored in each source tuple
            # ------------------------------------------------------------------
            if metric == 'fixed':
                fixed = {name: factor for name, (_, factor) in sources.items()}
                total = sum(fixed.values()) or 1.0
                norm = {name: f / total for name, f in fixed.items()}
                return pd.DataFrame(norm, index=reference_index)

            # ------------------------------------------------------------------
            # Rolling metrics — compute daily portfolio returns per source first
            # ------------------------------------------------------------------
            port_returns = {}
            for name, (w, _) in sources.items():
                aligned_w = w.reindex(returns_df.index).ffill()
                port_returns[name] = (aligned_w * returns_df).sum(axis=1)

            port_ret_df = pd.DataFrame(port_returns)

            # ------------------------------------------------------------------
            # Metric helpers
            # ------------------------------------------------------------------
            def rolling_cagr(ret_series: pd.Series) -> pd.Series:
                cum = (1 + ret_series).rolling(window, min_periods=min_periods).apply(np.prod, raw=True)
                return cum ** (ann_factor / window) - 1

            def rolling_sharpe(ret_series: pd.Series) -> pd.Series:
                roll_mean = ret_series.rolling(window, min_periods=min_periods).mean()
                roll_std = ret_series.rolling(window, min_periods=min_periods).std()
                return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(ann_factor)

            def rolling_utility(ret_series: pd.Series) -> pd.Series:
                cum = (1 + ret_series).rolling(window, min_periods=min_periods).apply(np.prod, raw=True)
                cagr = cum ** (ann_factor / window) - 1
                ann_vol = ret_series.rolling(window, min_periods=min_periods).std() * np.sqrt(ann_factor)

                return cagr - lam * ann_vol


            def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
                """Clip negatives and normalize rows to sum to 1."""
                clipped = df.clip(lower=0)
                row_sums = clipped.sum(axis=1).replace(0, np.nan)
                return clipped.div(row_sums, axis=0)

            # ------------------------------------------------------------------
            # Compute perf_df according to metric
            # ------------------------------------------------------------------
            if metric == 'cagr':
                perf_df = port_ret_df.apply(rolling_cagr)

            elif metric == 'sharpe':
                perf_df = port_ret_df.apply(rolling_sharpe)

            elif metric == 'utility':
                perf_df = port_ret_df.apply(rolling_utility)

            elif metric == 'mean':
                # Normalize each metric independently before averaging
                # so they contribute equally regardless of scale
                cagr_df = port_ret_df.apply(rolling_cagr)
                sharpe_df = port_ret_df.apply(rolling_sharpe)
                utility_df = port_ret_df.apply(rolling_utility)
                perf_df = (norm_cols(cagr_df) + norm_cols(sharpe_df) + norm_cols(utility_df)) / 3

            else:
                raise ValueError(f"Unknown blend_metric '{metric}'. Use 'cagr', 'sharpe', 'utility', 'mean' or 'fixed'.")

            # ------------------------------------------------------------------
            # Clip negatives, normalize
            # ------------------------------------------------------------------
            perf_df = perf_df.clip(lower=0)
            row_sums = perf_df.sum(axis=1).replace(0, np.nan)
            norm_factors = perf_df.div(row_sums, axis=0).fillna(1 / len(sources))

            return norm_factors.reindex(reference_index).ffill().fillna(1 / len(sources))

        def combine_sources(sources: dict, returns_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
            """
            Blend source weights using normalized factors from compute_blend_factors.

            Parameters
            ----------
            sources    : dict  name -> (weights_df, fixed_factor)
            returns_df : pd.DataFrame  daily returns, used only for rolling metrics
            settings   : dict  see compute_blend_factors for keys
            """
            if len(sources) == 1:
                return next(iter(sources.values()))[0].copy()

            norm_factors = compute_blend_factors(sources, returns_df, settings)

            # Shift norm_factors by 1 — metric computed through day t-1
            # is used to set blend weights for day t, avoiding lookahead
            norm_factors = norm_factors.shift(1).fillna(1 / len(sources))


            reference_index = next(iter(sources.values()))[0].index

            return sum(
                w.reindex(reference_index).fillna(0).multiply(norm_factors[name], axis=0)
                for name, (w, _) in sources.items()
            )

        def build_benchmark_weights(returns_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
            """
            Build a volatility-scaled benchmark weight DataFrame.
            Supports custom per-ticker weights or equiweighted fallback.
            Vol scaling is applied at portfolio level.

            Parameters
            ----------
            returns_df : pd.DataFrame  DatetimeIndex x tickers
            settings   : dict  expects:
                             'benchmark_tickers'         : list of tickers e.g. ['NQ=F', 'GC=F']
                             'benchmark_ticker_weights'  : dict of custom weights e.g. {'NQ=F': 0.65, 'GC=F': 0.35}
                                                           if None, equiweighted fallback
                             'volatility_target'         : float, base vol target        (default 0.11)
                             'benchmark_vol_offset'      : float, subtracted from vol_target (default 0.07)
                             'benchmark_weight_scale'    : float, overall weight dampener (default 0.70)
                             'blend_window'              : int, rolling window            (default 22)
                             'ann_factor'                : int, annualization factor      (default 252)

            Returns
            -------
            pd.DataFrame  DatetimeIndex x tickers, vol-scaled weights
            """
            benchmark_tickers = settings.get(
                'benchmark_tickers',
                [settings.get('benchmark_ticker', 'ES=F')]
            )

            missing = [t for t in benchmark_tickers if t not in returns_df.columns]
            if missing:
                raise ValueError(f"Benchmark tickers not found in returns_df: {missing}. Available: {list(returns_df.columns)}")

            # Custom weights or equiweighted fallback
            custom_weights = settings.get('benchmark_ticker_weights', None)
            if custom_weights:
                # Normalize in case they don't sum to 1
                total = sum(custom_weights[t] for t in benchmark_tickers)
                ticker_weights = {t: custom_weights[t] / total for t in benchmark_tickers}
            else:
                ticker_weights = {t: 1.0 / len(benchmark_tickers) for t in benchmark_tickers}

            vol_target = settings.get('volatility_target', 0.11) - settings.get('benchmark_vol_offset', 0.07)
            weight_scale = settings.get('benchmark_weight_scale', 0.70)
            window = settings.get('blend_window', 22)
            ann_factor = settings.get('ann_factor', 252)

            # Rolling portfolio returns using custom weights
            bmark_ret = sum(returns_df[t] * ticker_weights[t] for t in benchmark_tickers)

            # Rolling annualized volatility of benchmark portfolio — shifted to avoid lookahead
            rolling_vol = (
                bmark_ret
                .rolling(window, min_periods=1)
                .std()
                .mul(np.sqrt(ann_factor))
                .shift(1)
            )

            # Vol scalar: clip to [0, 1] — never leverage, only scale down
            vol_scalar = (vol_target / rolling_vol.replace(0, np.nan)).clip(0, 1.0).fillna(1.0)

            # Apply scalar to each ticker's custom weight
            bmark_weights = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
            for t in benchmark_tickers:
                bmark_weights[t] = ticker_weights[t] * weight_scale * vol_scalar

            return bmark_weights

        def vol_normalize_weights(sources: dict, returns_df: pd.DataFrame, settings: dict) -> dict:
            """
            Scale each source's weights by volatility_target / rolling_vol of the implied portfolio.
            Ensures all sources target the same vol level before blending.
            Shift(1) applied to vol estimate — uses yesterday's vol to size today's weights.

            Parameters
            ----------
            sources    : dict  name -> (weights_df, fixed_factor)
            returns_df : pd.DataFrame  DatetimeIndex x tickers, daily returns
            settings   : dict  expects:
                             'volatility_target' : float  (default 0.11)
                             'blend_window'      : int    (default 22)
                             'ann_factor'        : int    (default 252)

            Returns
            -------
            dict  same structure as sources, with vol-normalized weights_df
            """
            vol_target = settings.get('volatility_target', 0.11)
            window = settings.get('blend_window', 22)
            ann_factor = settings.get('ann_factor', 252)

            normalized_sources = {}
            for name, (w, factor) in sources.items():
                # Implied daily portfolio return for this source
                port_ret = (w * returns_df.reindex(w.index)).sum(axis=1)


                # Rolling annualized vol — shift(1) to use yesterday's vol
                rolling_vol = (
                    port_ret
                    .rolling(window, min_periods=1)
                    .std()
                    .mul(np.sqrt(ann_factor))
                    .shift(1)
                )

                # Vol scalar: clip to [0, 1] — never leverage, only scale down
                vol_scalar = (vol_target / rolling_vol.replace(0, np.nan)).fillna(1.0).clip(0, 1.0)


                # Scale all ticker weights uniformly by scalar
                w_normalized = w.multiply(vol_scalar, axis=0)

                # Debug
                # raw_volat_port_ret = port_ret.shift(1).std() * 16
                # vol_scalar = vol_target / raw_volat_port_ret
                #norm_port_ret = (w_normalized * returns_df).sum(axis=1)
                #norm_volat_port_ret = norm_port_ret.std() * 16
                #if name != 'benchmark':
                #    print(name, 'Raw_Volat', raw_volat_port_ret.round(4),'Norm_Volat', norm_volat_port_ret.round(4))

                normalized_sources[name] = (w_normalized, factor)

            return normalized_sources

        def plot_source_diagnostics(sources: dict, returns_df: pd.DataFrame, settings: dict):
            """
            Plot per-source diagnostics: cumulative returns, rolling vol, rolling CAGR.
            Call after vol_normalize_weights to inspect normalized sources.

            Parameters
            ----------
            sources    : dict  name -> (weights_df, fixed_factor)
            returns_df : pd.DataFrame  DatetimeIndex x tickers, daily returns
            settings   : dict  expects 'blend_window', 'ann_factor', 'blend_min_periods'
            """
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            window = settings.get('blend_window', 22)
            min_periods = settings.get('blend_min_periods', window)
            ann_factor = settings.get('ann_factor', 252)
            lam = settings.get('blend_lambda', 0.65)

            # ------------------------------------------------------------------
            # Compute per-source portfolio returns (weights already lag-adjusted)
            # ------------------------------------------------------------------
            port_ret_df = pd.DataFrame({
                name: (w.reindex(returns_df.index) * returns_df).sum(axis=1)
                for name, (w, _) in sources.items()
            })

            # ------------------------------------------------------------------
            # Derived series
            # ------------------------------------------------------------------
            cum_ret = (1 + port_ret_df).cumprod()

            rolling_vol = (
                port_ret_df
                .rolling(window, min_periods=min_periods)
                .std()
                .mul(np.sqrt(ann_factor))
            )

            rolling_cagr = (
                (1 + port_ret_df)
                .rolling(window, min_periods=min_periods)
                .apply(np.prod, raw=True)
                .pow(ann_factor / window)
                .sub(1)
            )

            rolling_utility = rolling_cagr - lam * rolling_vol

            norm_factors = compute_blend_factors(sources, returns_df, settings)

            # ------------------------------------------------------------------
            # Plot
            # ------------------------------------------------------------------
            n_sources = len(sources)
            fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
            colors = plt.cm.tab10.colors

            # --- 1. Cumulative returns ---
            ax = axes[0]
            for i, col in enumerate(cum_ret.columns):
                ax.plot(cum_ret.index, cum_ret[col], label=col, color=colors[i % 10])
            ax.set_title('Cumulative Returns per Source')
            ax.set_ylabel('Cumulative Return')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # --- 2. Rolling volatility ---
            ax = axes[1]
            for i, col in enumerate(rolling_vol.columns):
                ax.plot(rolling_vol.index, rolling_vol[col], label=col, color=colors[i % 10])
            ax.axhline(settings.get('volatility_target', 0.11), color='red', linestyle='--',
                       linewidth=1, label='vol_target')
            ax.set_title(f'Rolling Annualized Volatility (window={window})')
            ax.set_ylabel('Annualized Vol')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # --- 3. Rolling CAGR ---
            ax = axes[2]
            for i, col in enumerate(rolling_cagr.columns):
                ax.plot(rolling_cagr.index, rolling_cagr[col], label=col, color=colors[i % 10])
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_title(f'Rolling CAGR (window={window})')
            ax.set_ylabel('Annualized CAGR')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # --- 4. Rolling utility ---
            ax = axes[3]
            for i, col in enumerate(rolling_utility.columns):
                ax.plot(rolling_utility.index, rolling_utility[col], label=col, color=colors[i % 10])
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_title(f'Rolling Utility = CAGR - {lam} * Vol (window={window})')
            ax.set_ylabel('Utility')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # --- 5. Blend factors (norm_factors after shift) ---
            ax = axes[4]
            norm_factors_shifted = norm_factors.shift(1).fillna(1 / n_sources)
            for i, col in enumerate(norm_factors_shifted.columns):
                ax.plot(norm_factors_shifted.index, norm_factors_shifted[col],
                        label=col, color=colors[i % 10])
            ax.set_title(f'Blend Factors — metric={settings.get("blend_metric", "fixed")}')
            ax.set_ylabel('Normalized Factor')
            ax.set_ylim(0, 1)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)

            plt.suptitle('Source Diagnostics', fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
        # ======================================================================
        # Main portfolio weight computation block — replaces your existing logic
        # ======================================================================

        if (not settings['mkwtz_scipy']) and (not settings['mkwtz_vectorized']) and (not settings['ddn_ltd_portfolio']):
            print("Error at settings: Any Optimize Strategy must be selected 'mkwtz_scipy' or/and 'mkwtz_vectorized' or 'ddn_ltd_portfolio'")
            return

        sources = {}  # name -> (weights_df, fixed_factor)

        if settings['mkwtz_scipy']:
            self.s_weights_df = self.PortfolioWeightsMarkowitzScipy(st_tickers_returns, indicators_dict, settings)
            sources['scipy'] = (self.s_weights_df, settings.get('scipy_blend_factor', 1.0))

        if settings['mkwtz_vectorized']:
            self.v_weights_df, *_ = compute_optimized_markowitz_d_w(st_tickers_returns, settings)
            sources['vectorized'] = (self.v_weights_df, settings.get('vectorized_blend_factor', 1.0))

        if settings['ddn_ltd_portfolio']:
            from ddn_ltd_portfolio import DDNLimitedPortfolio
            portfolio_manager = DDNLimitedPortfolio(settings)
            ddn_weights = portfolio_manager.compute_weights(st_tickers_returns)
            sources['ddn'] = (ddn_weights, settings.get('ddn_blend_factor', 1.0))

        if settings.get('use_benchmark', False):
            bmark_weights = build_benchmark_weights(st_tickers_returns, settings)
            sources['benchmark'] = (bmark_weights, settings.get('benchmark_fixed_factor', 1.0))

        #Set comon start without zero weights
        sources = trim_sources_to_common_start(sources)
        common_start = next(iter(sources.values()))[0].index[0]
        st_tickers_returns = st_tickers_returns.loc[common_start:]

        # Before vol normalization — inspect raw sources
        #plot_source_diagnostics(sources, st_tickers_returns, settings)

        settings['vol_normalize_sources'] = False  # toggle on/off

        if settings.get('vol_normalize_sources', True):
            sources = vol_normalize_weights(sources, st_tickers_returns, settings)

            # Before vol normalization — inspect raw sources
            plot_source_diagnostics(sources, st_tickers_returns, settings)

        # Combine all active sources using fixed factors or rolling performance metric
        self.weights_df = combine_sources(sources, st_tickers_returns, settings)

        """
        if (not settings['mkwtz_scipy'])  & (not settings['mkwtz_vectorized']) & (not settings['ddn_ltd_portfolio']):
            print(" Error at settings: Any Optimize Strategy must be selected 'mkwtz_scipy' or/and 'mkwtz_vectorized' or 'ddn_ltd_portfolio' ")
            return

        if settings['mkwtz_scipy']:

            #Get Porfolio Allocation Weights for this slice of tickers_returns. Mean of diferent parameters like (rebalance periods, lookback,..)
            self.s_weights_df=self.PortfolioWeightsMarkowitzScipy(st_tickers_returns,indicators_dict,settings)
            self.weights_df = self.s_weights_df.copy()

        if settings['mkwtz_vectorized']:

            #Get Markowitz_Vectorized weights
            self.v_weights_df, _, _, _, _, _ = compute_optimized_markowitz_d_w(st_tickers_returns, settings)
            self.weights_df = self.v_weights_df.copy()


        if settings['mkwtz_vectorized'] and settings['mkwtz_scipy']:

            #Make mean
            self.weights_df= mean_positions(self.s_weights_df,self.v_weights_df,settings['w_upper_lim'],
                                         sf = 1.0 ,vf = 1.0 ,overall_f = 1.0)

        #self.s_weights_df.plot(title='s_weights_df')
        #self.v_weights_df.plot(title='v_weights_df')

        if settings['ddn_ltd_portfolio']:
            # Get DDN Ltd Portfolio
            from ddn_ltd_portfolio import DDNLimitedPortfolio

            # Initialize the Class
            portfolio_manager = DDNLimitedPortfolio(settings)

            # Generate the Weights
            # This runs the math and stores the intermediates inside the object
            ddn_weights = portfolio_manager.compute_weights(st_tickers_returns)

            if settings['mkwtz_vectorized'] or settings['mkwtz_scipy']:
                ddn_weights = ddn_weights.reindex(self.weights_df.index)
                self.weights_df=ddn_weights*0.6+self.weights_df*0.4 # ddn +vectorized
                #self.weights_df = ddn_weights * 0.5 + self.weights_df * 0.5 # ddn +(vectorized+scipy)
            else:
                self.weights_df=ddn_weights
        """

        if settings['apply_strategy_weights']:
            # Apply RSI and other additional Strategy Weights on top of Markowitz weights
            positions=self.ApplyStrategyWeights(self.weights_df,indicators_dict['comb_weights'])

        else:
            positions=self.weights_df.copy()

        # Limit Upper individual position
        positions = positions.clip(upper=settings['w_upper_lim'])

         #Make Zero individual positions bellow 'w_min_lim'
        # We keep the value if |x| >= threshold, otherwise set to 0
        positions = positions.where(positions.abs() >= settings['w_min_lim'], 0)

        #positions.plot(title='positions_filtered')

        self.positions = positions.copy()

        self.strategy_returns=self.Returns(self.positions,st_tickers_returns)




    def PortfolioWeightsMarkowitzScipy(self,st_tickers_returns,indicators_dict,settings):

        #Compute Markowitz for diferent windows and save instance to a dict
        self.tsw_dict = {
            str(w): RollingMarkowitzWeights(w, p, settings['volatility_target'], st_tickers_returns,indicators_dict, settings)
            for w, p in zip(settings['mkwtz_ws'], settings['mkwtz_ps'])
        }



        def get_weights_df(tsw_dict, mkwtz_mean_fs, apply_opt_fun_predict_factor):

            if apply_opt_fun_predict_factor:

                # Get opt_fun_df with columns for each mktz window
                fun_df = pd.concat([tsw_dict[k].opt_fun_df for k in tsw_dict.keys()], axis=1)
                fun_df.columns = list(tsw_dict.keys())

                def get_fun_corr_factor(fun_df, w):

                    """Get Optimize Function factor when Prediction is good"""

                    # AutoCorrelation beetween current Optimize Function and previous values
                    fun_autocorrel = fun_df.rolling(window=w).corr(fun_df.shift(1))

                    #Factor calculation
                    fun_autocorrel_factor = fun_autocorrel.fillna(0).clip(upper=1, lower=-1) #limited values -1 to +1
                    fun_autocorrel_factor  = fun_autocorrel_factor.where(fun_df < 0, 1) #Keep value to 1 when portfolio fun_df is positive

                    # Factor making the mean and normalizing
                    fun_corr_factor = fun_autocorrel_factor.rolling(w).mean()
                    fun_corr_factor = fun_corr_factor / fun_corr_factor.rolling(250 * 4).mean().fillna(0.8).mean().mean()
                    fun_corr_factor = fun_corr_factor.fillna(1).clip(upper=1.2,lower=0.8) #limited values 0.8 to 1.2

                    return fun_corr_factor


                # Get Optimize Function factor when Prediction is good
                opt_fun_predict_factor = get_fun_corr_factor(fun_df, 20)

                # Apply opt_fun Predictibity factor
                weights_dict = {w: tsw_dict[w].weights_df.multiply(opt_fun_predict_factor[w], axis='index') for w in tsw_dict.keys()}

            else:

                weights_dict = {w: tsw_dict[w].weights_df for w in tsw_dict.keys()}
                fun_df = None
                opt_fun_predict_factor = None

            # Make weightged Mean
            weights_df = weighted_mean_of_dfs_dict(weights_dict, mkwtz_mean_fs)

            #weights_df = sum([df * weight for df, weight in zip(weights_dict.values(), mkwtz_mean_fs)])/ len(mkwtz_mean_fs)

            overall_f = 1.25  #1.25
            weights_df *=overall_f

            return weights_df, opt_fun_predict_factor, fun_df, weights_dict

        #Get weightged Mean with opt_fun Predictibity factor
        self.weights_df, self.opt_fun_predict_factor, self.opt_fun_df,self.weights_by_period_dict= (
            get_weights_df(self.tsw_dict,settings['mkwtz_mean_fs'],settings['apply_opt_fun_predict_factor']))

        return self.weights_df

    def ApplyStrategyWeights(self, weights_df , ind_weights):


        # Combined Weights
        w = ind_weights.copy()

        # Reindex as  weights_df index
        #w = w[w.index.isin(weights_df.index)]
        w = w.reindex(weights_df.index)[weights_df.columns]

        # Apply Pre Optimization with Combined Weights
        positions= weights_df * w

        # Softed Test Positions
        #positions = raw_weight_pct * weights_df + (1 - raw_weight_pct) * positions

        return positions


    def Returns(self,st_positions,st_tickers_returns):
        self.st_strategy_returns_by_ticker= st_tickers_returns * st_positions
        return self.st_strategy_returns_by_ticker.sum(axis=1)


class RollingMarkowitzWeights:
    """
    Compute MarkowitzWeigths to get Dayly Weights pd.Dataframe with columns= Tickers Names

    """
    def __init__(self,lookback,rebalance_p,volatility_target,tickers_returns,indicators_dict,settings): #Add rolling_cagr, rolling_cov_matices for this lookback
        self.rebalance_p = rebalance_p
        self.lookback=int(lookback)
        self.volatility_target=volatility_target
        self.settings=settings
        self.tickers_returns=tickers_returns
        self.tickers=tickers_returns.columns
        self.size=len(self.tickers)
        self.indicators_dict=indicators_dict



        self.compute_RollingMarkowitzWeights()



    def compute_RollingMarkowitzWeights(self):
        """
        Calculates time series of optimal weights and optimization function values.

        This function iterates through periods defined by the `rebalance_p` frequency,
        calculates optimal weights for each period using the Markowitz model, and
        upsamples the results to daily frequency.

        Attributes:
            self.rebalance_p: Rebalance frequency (e.g., 'W-FRI' for weekly Fridays,'M').
            self.lookback: Lookback window for calculating weights.  (eg. int 44,180,360 )
            self.weights_res_df: DataFrame containing weights with start, end dates,
                                 optimization function value, and weights for each asset.
            self.train_analytics_res_df: (Optional) DataFrame containing additional
                                          training period analytics (to be implemented).
            self.tickers_returns: DataFrame containing historical asset returns.

            self.volatility_target: Target volatility for the portfolio.
            self.settings: Additional settings for the Markowitz model.
            self.indicators_dict: (Optional) Dictionary of technical indicators.
            self.weights_df: DataFrame containing daily weights for each asset.
            self.opt_fun_df: DataFrame containing daily optimization function values.
            self.size: Number of assets.
        """

        #Create df to store results
        results_by_period_df = create_results_by_period_df(self.tickers_returns,self.rebalance_p,self.lookback)

        #Get data Features for this lookback: rolling_cagr, rolling_cov_matrices

        # IMPORTANT: Clear the 'Memory' of the class for a clean run
        MarkowitzWeights.reset_state()

        #weights_prev = np.zeros(self.size)
        #x0 = np.ones([self.size, 1]) / self.size * 0.1
        x0 = np.ones(self.size) / self.size * 0.1

        def get_results_by_loop(results_by_period_df, x0):

            for index,row in results_by_period_df.iterrows():
                slice_tickers_returns = self.tickers_returns.loc[row['start']:row['end']]

                #Calculate Slice Weights
                mw = MarkowitzWeights(slice_tickers_returns, self.volatility_target, self.settings, x0)
                results_by_period_df.loc[index,['opt_fun'] + self.tickers.tolist() ] = [mw.results.fun] + list(mw.results.x)

            return results_by_period_df

        #results_by_period_df = get_results_vectorized(results_by_period_df, x0)
        results_by_period_df = get_results_by_loop(results_by_period_df, x0)

        #Drop Duplicates
        results_by_period_df.drop_duplicates(subset='end', inplace=True)

        # Upsample to daily index and fill missing values
        results_by_period_df.set_index('end', inplace=True)
        #print('results_by_period_df before reindex',results_by_period_df)
        results_dayly_df = results_by_period_df.reindex(self.tickers_returns.index).shift(1).fillna(method='ffill')

        # Separate weight and opt_fun DataFrames
        self.weights_df = results_dayly_df[self.tickers]
        self.opt_fun_df = results_dayly_df[['opt_fun']]

        #print('self.weights_df at compute_RollingMarkowitzWeights',self.weights_df)



