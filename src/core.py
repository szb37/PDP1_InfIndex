import sys
import src.config as config
sys.path.append(config.path_szb_commons)
import commons_codebase.src.core as commons
import src.folders as folders
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import math
import os


class DataWrangl():
    
    @staticmethod
    def get_df_master(dir_in=folders.fromParker, csvs=config.parkers_csvs, filter=False, save=False, exclude_pids=None, transform='log'):
        """Build the master long-form dataframe of biomarkers and outcomes.

        Merges Parker CSVs, cleans values, computes log-transforms and deltas,
        creates the inflammation index, subsets markers/timepoints, and
        optionally saves to disk.

        Args:
            dir_in (str): Input directory containing Parker CSVs.
            csvs (list[str]): Filenames to read and concatenate.
            filter (bool): If True, drop outliers beyond 3 SD on log scale.
            save (dict|bool): If dict with keys `dir_out` and `fname_out`, save
                resulting CSV. If False, do not save.
            exclude_pids (list[int]|None): List of participant IDs to exclude from
                the dataset. Useful for:
                - Excluding patients with incomplete data (e.g., 1060 who only has mitokines)
                - Sensitivity analyses (e.g., excluding outliers)
                - Leave-one-out robustness checks
                If None (default), no patients are excluded.
            transform (str): Transformation applied to raw values. One of:
                - 'log' (default): natural log of (value + 1) via np.log1p
                - 'boxcox': Box-Cox transform via scipy.stats.boxcox, fit per
                  measure on (value + 1) to handle zeros.
                Output column is always named 'log_value' regardless of choice.

        Returns:
            df (pd.DataFrame): master long-form dataframe of biomarkers and outcomes
        """

        ### Merge cytokine and mitokine data from Parker
        dfs = []
        for csv in csvs:
            df = pd.read_csv(os.path.join(dir_in, csv))
            dfs.append(df)

        df = pd.concat(dfs, axis=0)
        df.dropna(subset=['trial', 'pID', 'tp', 'measure', 'value'], inplace=True)

        # Exclude specified participants (e.g., for sensitivity analyses or incomplete data)
        # This happens early in the pipeline so excluded patients are removed before any
        # transformations, index calculations, or analyses are performed.
        if exclude_pids:
            n_before = df['pID'].nunique()
            df = df[~df['pID'].isin(exclude_pids)]
            n_after = df['pID'].nunique()
            print(f"Excluded {n_before - n_after} participant(s): {exclude_pids}")

        df['tp'] = df['tp'].replace({'Baseline':'bsl', 'S':'bsl',})
        df['pID'] = df['pID'].astype(int)        
        df['measure'] = df['measure'].replace({
            'Eotaxin':'eotaxin',
            'Eotaxin-3':'eotaxin-3',
            'Tie-2': 'TIE-2',
            'Flt-1': 'FLT-1',
            'IL-12/IL-23p40': 'IL-12',
            'TNF_alpha': 'TNF-alpha',})
        
        df = df.loc[
            (df.trial=='PDP1') & 
            (df.measure.isin(config.markers)) & 
            (df.tp.isin(config.tps))]
        df.reset_index(drop=True, inplace=True)

        ### Add transformed scores (always stored in 'log_value' column regardless of transform type)
        assert transform in ['log', 'boxcox'], f"transform must be 'log' or 'boxcox', got '{transform}'"
        
        if transform == 'log':
            df['log_value'] = np.log1p(df['value'])
        elif transform == 'boxcox':
            # Box-Cox requires strictly positive values; shift by 1 to match log1p behavior
            # Fit lambda separately per measure since markers have different distributions
            df['log_value'] = math.nan
            for measure in df['measure'].unique():
                mask = df['measure'] == measure
                vals = df.loc[mask, 'value'] + 1
                transformed, _ = stats.boxcox(vals)
                df.loc[mask, 'log_value'] = transformed

        ### This patient at this tp has extremly high CRP value, hence 
        ridx_rm = df.loc[(df.tp=='A1') & (df.pID==1034)].index
        df.drop(ridx_rm, inplace=True)
        df.reset_index(drop=True, inplace=True)    

        ### Delete data > 3sd away from mean (filter acts on the log-transformed data)
        if filter:
            ridx_rm = []
            for trial in df.trial.unique():
                for measure in df.loc[(df.trial==trial)].measure.unique():
                    log_values = df.loc[(df.trial==trial) & (df.measure==measure)].log_value 
                    ridx_rm.extend(list(
                        df.loc[(
                            (df.trial==trial) & 
                            (df.measure==measure) &
                            ((df.log_value > (log_values.mean()+3*log_values.std())) | (df.log_value < (log_values.mean()-3*log_values.std())))
                            )].index))
            df.drop(ridx_rm, inplace=True)
            df.reset_index(drop=True, inplace=True)    
            
        ### Add inflammation index
        df = DataWrangl.add_composite_index(df, comps=config.markers_infind_comps, measure_type='inflammation', measure_name='infindex') 
        df = DataWrangl.add_composite_index(df, comps=config.markers_mitokines, measure_type='mitokines', measure_name='mitoindex') 

        ### Add delta scores 
        #  add_delta_scores() acts on the 'score' column and adds a 'delta_score' column, hence some temporary renaming
        df.reset_index(drop=True, inplace=True)                
        df['time'] = math.nan # add_delta_scores() expects this column

        # Delta scores of raw values
        df.rename(columns={'value': 'score',}, inplace=True) 
        df = commons.DataWrangl.add_delta_scores(df_master = df)
        df.rename(columns={'score': 'value',}, inplace=True)
        df.rename(columns={'delta_score': 'delta_value',}, inplace=True)

        # Delta scores of log-transformed values
        df.rename(columns={'log_value': 'score',}, inplace=True)
        df = commons.DataWrangl.add_delta_scores(df_master = df)
        df.rename(columns={'score': 'log_value',}, inplace=True)
        df.rename(columns={'delta_score': 'delta_log_value',}, inplace=True)

        ### Add number of days based on the timepoints
        for tp in config.tps:
            df.loc[(df.tp==tp), 'ndays'] = config.tp_to_ndays[tp]            

        ### Remove sIL-6R data as it is uninterpretable - decision approved by first auth Ellen Bradley 
        df = df.loc[df.measure!='sIL-6R']

        ### Organize and save results
        df['value'] = round(df['value'], 4)
        df['delta_value'] = round(df['delta_value'], 4)
        df['log_value'] = round(df['log_value'], 4)
        df['delta_log_value'] = round(df['delta_log_value'], 4)    

        df = df[[
            'trial', 
            'pID', 
            'tp', 
            'ndays',
            'time',
            'type', 
            'measure', 
            'is_risk', 
            'value', 
            'delta_value', 
            'log_value', 
            'delta_log_value',]]

        if save!=False:
            df.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df
        
    @staticmethod
    def add_composite_index(df_master, comps=config.markers_infind_comps, measure_type='inflammation', measure_name='infindex'):
        """Append to df_master the composite indexscores, which is the average of z-scores across selected markers.

        The mitochondria index is a composite score computed from defined markers.

        Computation:
            1. For each marker, z-score the log-transformed concentrations across all patients/timepoints
               (standardizes each marker to mean=0, std=1 so they're on comparable scales)
            2. For each patient/timepoint, average the z-scores to get a single composite index

        Args:
            df_master (pd.DataFrame): Long-form dataframe with columns including
                `trial`, `measure`, `tp`, and `log_value`.

        Returns:
            pd.DataFrame: The input dataframe with additional `infindex` rows
            appended for each patient/timepoint where computable.
        """

        # Filter to markers used in the index
        df_infind = df_master.copy()
        df_infind = df_infind.loc[(df_infind.measure.isin(comps))]
        df_infind['z_log_value'] = math.nan

        # Step 1: Z-score each marker's log-concentrations across all observations
        # This standardizes each marker (mean=0, std=1) so they contribute equally to the index
        for measure in comps:
            df_infind.loc[(df_infind.measure==measure), 'z_log_value'] = stats.zscore(
                df_infind.loc[(df_infind.measure==measure), 'log_value'], nan_policy='omit')

        # Pivot from long to wide format: one row per patient/timepoint, columns for each marker's z-score
        df_infind = df_infind[['pID', 'tp', 'measure', 'z_log_value']]
        df_infind = df_infind.pivot(index=['pID','tp'],  columns=['measure'], values=['z_log_value'],)

        # Clean up multi-index columns created by pivot
        df_infind.reset_index(inplace=True)
        df_infind.columns = [col[1] if col[0] == 'z_log_value' else col[0] for col in df_infind.columns]

        # Flag patient-timepoints with incomplete marker data (will be excluded from index)
        incomplete = df_infind[df_infind[comps].isna().any(axis=1)]
        if len(incomplete) > 0:
            print(f"Warning: {len(incomplete)} patient-timepoint(s) excluded from infindex due to missing markers:")
            for _, row in incomplete.iterrows():
                missing = [m for m in comps if pd.isna(row[m])]
                print(f"  pID {int(row['pID'])}, tp {row['tp']}: missing {missing}")

        # GDF-15 lower is better, so change score signs will be flipped, so higher is better.
        if ('GDF-15' in comps) and config.flip_GDF15:
            df_infind['GDF-15'] = -df_infind['GDF-15']

        # Step 2: Calculate inflammation index as the mean of the z-scored markers
        # Drop rows where any marker is missing
        df_infind.dropna(subset=comps, inplace=True)
        df_infind[measure_name] = df_infind[comps].mean(axis=1)

        # Format as long-form row to match df_master structure
        df_infind = df_infind[['pID', 'tp', measure_name]]
        df_infind['trial'] = 'PDP1'
        df_infind['type'] = measure_type
        df_infind['measure'] = measure_name
        df_infind = df_infind.rename(columns={measure_name: 'log_value'})

        # Append infindex rows to the master dataframe
        df_master = pd.concat([df_master, df_infind])
        df_master.reset_index(drop=True, inplace=True)

        return df_master

    @staticmethod
    def print_missing_data_summary(df_master):
        """Print a summary of all missing patient-timepoint combinations by marker group.

        Call this after loading the master dataframe to see which data points are
        missing before running analyses.

        Args:
            df_master (pd.DataFrame): Long-form dataframe with columns including
                `pID`, `tp`, and `measure`.
        """

        # Expected patient-timepoint grid (exclude patients with only partial marker coverage)
        all_pids = sorted(df_master['pID'].unique())
        all_tps = ['bsl', 'A1', 'B1', 'B30']

        marker_groups = {
            'Inflammation (index components)': config.markers_infind_comps,
            'Inflammation (other)': config.markers_inflamation_miscs,
            'Mitokines': config.markers_mitokines,
            'Exploratory': config.markers_explorat,
        }

        print("=== MISSING DATA SUMMARY ===\n")

        total_missing = 0
        for group_name, markers in marker_groups.items():
            print(f"{group_name}:")

            # Check each marker in the group
            group_missing = []
            for marker in markers:
                df_marker = df_master[df_master['measure'] == marker]
                pids_with_marker = df_marker['pID'].unique()

                for pid in pids_with_marker:
                    tps_present = df_marker[df_marker['pID'] == pid]['tp'].tolist()
                    tps_missing = [tp for tp in all_tps if tp not in tps_present]
                    for tp in tps_missing:
                        group_missing.append((pid, tp, marker))

            if group_missing:
                for pid, tp, marker in sorted(group_missing):
                    print(f"  pID {pid}, tp {tp}: {marker}")
                total_missing += len(group_missing)
            else:
                print("  (no missing data)")
            print()

        # Also report patients who are entirely missing from certain marker groups
        print("Patients missing from marker groups entirely:")
        for group_name, markers in marker_groups.items():
            df_group = df_master[df_master['measure'].isin(markers)]
            pids_in_group = df_group['pID'].unique()
            pids_missing = [pid for pid in all_pids if pid not in pids_in_group]
            if pids_missing:
                print(f"  {group_name}: pIDs {pids_missing}")

        print(f"\nTotal missing data points: {total_missing}")


class Plots():
    
    @staticmethod
    def draw_conc_trajectory(df_master, outcome='log_value', draw_inds=True, draw_group=True, **save):
        """Plot individual and/or group biomarker trajectories across time.

        Args:
            df_master (pd.DataFrame): Long-form dataframe including `measure`,
                `pID`, `ndays`, and outcome columns.
            outcome (str): One of 'value', 'delta_value', 'log_value',
                'delta_log_value'.
            draw_inds (bool): If True, draw individual subject lines.
            draw_group (bool): If True, overlay group mean line.
            **save: Optional keywords `dir_out` and `fname_prefix` for saving.

        Returns:
            None
        """

        assert outcome in ['value', 'delta_value', 'log_value', 'delta_log_value']
        df_master['tp'] = pd.Categorical(
            df_master['tp'], 
            categories=['bsl', 'A1', 'B1', 'B30',], 
            ordered=True)

        ndays = [value for value in config.tp_to_ndays.values()]

        for measure in config.markers:

            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            
            ### Draw patients' trajectories
            if draw_inds:

                alpha = 1
                palette = sns.color_palette("hls", 10)
                if draw_group: # Special settings if group mean is drawn on top
                    alpha = 0.3
                    palette = ['gray' for idx in df_master.pID.unique()]

                sns.lineplot( # individuals' lineplot
                    data = df_master.loc[(df_master.measure==measure)],
                    x = 'ndays',
                    y = outcome,
                    hue = 'pID',
                    palette = palette,
                    linewidth = 0.7,
                    alpha = alpha,
                    legend = not draw_group,  # Show legend for individuals-only plots
                    )

                # Position legend outside plot area (to the right)
                if not draw_group:
                    ax.legend(
                        title='pID',
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left',
                        borderaxespad=0,
                        fontsize=8,
                        title_fontsize=9
                    )

            ### Draw group means
            if draw_group:

                sns.lineplot(
                    data=df_master.loc[df_master.measure == measure],
                    x='ndays',
                    y=outcome,
                    color='navy',
                    marker='o',
                    markersize=10,
                    linewidth=2,
                    legend=False,
                    estimator='mean',
                    err_style='bars',        
                    errorbar=config.errorbar, 
                    err_kws=config.err_kws,   
                )

            ### Styling
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            sns.despine(top=True, right=True, left=False, bottom=False, offset=5)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.0)

            ax.set_xticks(ndays)
            ax.set_xticklabels(['Baseline', '24h post-10 mg', '24h post-25 mg', '30d post-25 mg'],
                               rotation=30, ha='center')
            ax.tick_params(axis='x', bottom=True, labelsize=16)
            ax.tick_params(axis='y', left=True, labelsize=16)  

            ax.set_xlabel('', fontdict=config.axislabel_fontdict)

            # Format measure name with Greek letters
            if outcome in ['log_value', 'delta_log_value']:
                ax.set_ylabel(f'ln(1+[{measure_formatted}])', fontdict=config.axislabel_fontdict)
            elif outcome=='infindex':
                ax.set_ylabel(f'{measure_formatted}', fontdict=config.axislabel_fontdict)
            else:
                ax.set_ylabel(f'{measure_formatted} [pg/mL]', fontdict=config.axislabel_fontdict)

            # Adjust layout to make room for legend (when showing individuals only)
            if draw_inds and not draw_group:
                fig.subplots_adjust(right=0.75)

            ### Save
            commons.Plots.save_fig(
                fig = fig,
                save_PNG = config.save_PNG,
                save_SVG = config.save_SVG,
                dir_out = save['dir_out'],
                fname_out = f'{save['fname_prefix']}_{measure}_{outcome}')

    @staticmethod
    def draw_conc_trajectory_ci95(df_master, outcome='log_value', draw_inds=True, draw_group=True, **save):
        """Plot biomarker trajectories with 95% confidence interval error bars.

        Uses 95% confidence intervals (CI) for error bars instead of standard errors.

        Args:
            df_master (pd.DataFrame): Long-form dataframe including `measure`,
                `pID`, `ndays`, and outcome columns.
            outcome (str): One of 'value', 'delta_value', 'log_value',
                'delta_log_value'.
            draw_inds (bool): If True, draw individual subject lines.
            draw_group (bool): If True, overlay group mean line.
            **save: Optional keywords `dir_out` and `fname_prefix` for saving.

        Returns:
            None
        """

        assert outcome in ['value', 'delta_value', 'log_value', 'delta_log_value']
        df_master['tp'] = pd.Categorical(
            df_master['tp'],
            categories=['bsl', 'A1', 'B1', 'B30',],
            ordered=True)

        ndays = [value for value in config.tp_to_ndays.values()]

        for measure in config.markers:

            fig, ax = plt.subplots(figsize=(4.8, 4.8))

            ### Draw patients' trajectories
            if draw_inds:

                alpha = 1
                palette = sns.color_palette("hls", 10)
                if draw_group: # Special settings if group mean is drawn on top
                    alpha = 0.3
                    palette = ['gray' for idx in df_master.pID.unique()]

                sns.lineplot( # individuals' lineplot
                    data = df_master.loc[(df_master.measure==measure)],
                    x = 'ndays',
                    y = outcome,
                    hue = 'pID',
                    palette = palette,
                    linewidth = 0.7,
                    alpha = alpha,
                    legend = not draw_group,  # Show legend for individuals-only plots
                    )

                # Position legend outside plot area (to the right)
                if not draw_group:
                    ax.legend(
                        title='pID',
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left',
                        borderaxespad=0,
                        fontsize=8,
                        title_fontsize=9
                    )

            ### Draw group means with 95% CI
            if draw_group:

                sns.lineplot(
                    data=df_master.loc[df_master.measure == measure],
                    x='ndays',
                    y=outcome,
                    color='navy',
                    marker='o',
                    markersize=10,
                    linewidth=2,
                    legend=False,
                    estimator='mean',
                    err_style='bars',
                    errorbar=('ci', 95),  # 95% confidence interval instead of SE
                    err_kws=config.err_kws,
                )

            ### Styling
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            sns.despine(top=True, right=True, left=False, bottom=False, offset=5)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.0)

            ax.set_xticks(ndays)
            ax.set_xticklabels(['Baseline', '24h post-10 mg', '24h post-25 mg', '30d post-25 mg'],
                               rotation=30, ha='center')
            ax.tick_params(axis='x', bottom=True, labelsize=16)
            ax.tick_params(axis='y', left=True, labelsize=16)

            ax.set_xlabel('', fontdict=config.axislabel_fontdict)

            # Format measure name with Greek letters
            if outcome in ['log_value', 'delta_log_value']:
                ax.set_ylabel(f'ln(1+[{measure_formatted}])', fontdict=config.axislabel_fontdict)
            elif outcome=='infindex':
                ax.set_ylabel(f'{measure_formatted}', fontdict=config.axislabel_fontdict)
            else:
                ax.set_ylabel(f'{measure_formatted} [pg/mL]', fontdict=config.axislabel_fontdict)

            # Adjust layout to make room for legend (when showing individuals only)
            if draw_inds and not draw_group:
                fig.subplots_adjust(right=0.75)

            ### Save
            commons.Plots.save_fig(
                fig = fig,
                save_PNG = config.save_PNG,
                save_SVG = config.save_SVG,
                dir_out = save['dir_out'],
                fname_out = f'{save['fname_prefix']}_{measure}_{outcome}')

    @staticmethod
    def draw_conc_trajectory_morey(df_master, outcome='log_value', draw_inds=True, draw_group=True, **save):
        """Plot biomarker trajectories with Morey (2008) within-subject error bars.

        Uses Morey's method to calculate error bars that remove between-subject
        variability, making them appropriate for repeated measures designs. This
        method normalizes data by removing subject means (Cousineau, 2005) and
        applies Morey's correction factor.

        References:
            Morey, R. D. (2008). Confidence intervals from normalized data:
                A correction to Cousineau (2005). Tutorials in Quantitative
                Methods for Psychology, 4(2), 61-64.
            Cousineau, D. (2005). Confidence intervals in within-subject designs:
                A simpler solution to Loftus and Masson's method. Tutorials in
                Quantitative Methods for Psychology, 1(1), 42-45.

        Args:
            df_master (pd.DataFrame): Long-form dataframe including `measure`,
                `pID`, `ndays`, and outcome columns.
            outcome (str): One of 'value', 'delta_value', 'log_value',
                'delta_log_value'.
            draw_inds (bool): If True, draw individual subject lines.
            draw_group (bool): If True, overlay group mean line.
            **save: Optional keywords `dir_out` and `fname_prefix` for saving.

        Returns:
            None
        """

        assert outcome in ['value', 'delta_value', 'log_value', 'delta_log_value']
        df_master['tp'] = pd.Categorical(
            df_master['tp'],
            categories=['bsl', 'A1', 'B1', 'B30',],
            ordered=True)

        ndays = [value for value in config.tp_to_ndays.values()]
        n_timepoints = len(ndays)
        morey_correction = np.sqrt(n_timepoints / (n_timepoints - 1))

        for measure in config.markers:

            # Calculate Morey normalized values
            df_measure = df_master.loc[df_master.measure == measure].copy()

            # Calculate subject means across timepoints
            subject_means = df_measure.groupby('pID')[outcome].mean()

            # Calculate grand mean
            grand_mean = df_measure[outcome].mean()

            # Normalize: subtract subject mean, add grand mean (Cousineau method)
            df_measure['normalized'] = df_measure.apply(
                lambda row: row[outcome] - subject_means[row['pID']] + grand_mean
                if pd.notna(row[outcome]) else np.nan,
                axis=1
            )

            # Calculate within-subject SE with Morey correction at each timepoint
            cm_stats = df_measure.groupby('ndays')['normalized'].agg(['mean', 'std', 'count'])
            cm_stats['se'] = (cm_stats['std'] / np.sqrt(cm_stats['count'])) * morey_correction

            fig, ax = plt.subplots(figsize=(4.8, 4.8))

            ### Draw patients' trajectories
            if draw_inds:
                alpha = 1
                palette = sns.color_palette("hls", 10)
                if draw_group:
                    alpha = 0.3
                    palette = ['gray' for idx in df_master.pID.unique()]

                sns.lineplot(
                    data=df_measure,
                    x='ndays',
                    y=outcome,
                    hue='pID',
                    palette=palette,
                    linewidth=0.7,
                    alpha=alpha,
                    legend=not draw_group,
                )

                if not draw_group:
                    ax.legend(
                        title='pID',
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left',
                        borderaxespad=0,
                        fontsize=8,
                        title_fontsize=9
                    )

            ### Draw group means with Morey error bars
            if draw_group:
                # Calculate actual means (not normalized) for plotting
                actual_means = df_measure.groupby('ndays')[outcome].mean()

                # Plot means with Morey-corrected error bars
                ax.errorbar(
                    x=cm_stats.index,
                    y=actual_means,
                    yerr=cm_stats['se'],
                    color='navy',
                    marker='o',
                    markersize=10,
                    linewidth=2,
                    capsize=config.err_kws['capsize'],
                    elinewidth=config.err_kws['elinewidth'],
                    capthick=config.err_kws['capthick'],
                )

            ### Styling
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            sns.despine(top=True, right=True, left=False, bottom=False, offset=5)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.0)

            ax.set_xticks(ndays)
            ax.set_xticklabels(['Baseline', '24h post-10 mg', '24h post-25 mg', '30d post-25 mg'],
                               rotation=30, ha='center')
            ax.tick_params(axis='x', bottom=True, labelsize=16)
            ax.tick_params(axis='y', left=True, labelsize=16)

            ax.set_xlabel('', fontdict=config.axislabel_fontdict)

            # Format measure name with Greek letters
            if outcome in ['log_value', 'delta_log_value']:
                ax.set_ylabel(f'ln(1+[{measure_formatted}])', fontdict=config.axislabel_fontdict)
            elif outcome == 'infindex':
                ax.set_ylabel(f'{measure_formatted}', fontdict=config.axislabel_fontdict)
            else:
                ax.set_ylabel(f'{measure_formatted} [pg/mL]', fontdict=config.axislabel_fontdict)

            if draw_inds and not draw_group:
                fig.subplots_adjust(right=0.75)

            ### Save
            commons.Plots.save_fig(
                fig=fig,
                save_PNG=config.save_PNG,
                save_SVG=config.save_SVG,
                dir_out=save['dir_out'],
                fname_out=f'{save['fname_prefix']}_{measure}_{outcome}')

    @staticmethod
    def draw_conc_trajectory_mixedeffects(df_master, outcome='log_value', draw_inds=True, **save):
        """Plot biomarker trajectories with confidence intervals from mixed-effects models.

        Uses the point estimates and confidence intervals from previously fitted
        mixed-effects models to display uncertainty around the mean trajectory.
        These CIs account for within-subject correlation structure.

        Args:
            df_master (pd.DataFrame): Long-form dataframe including `measure`,
                `pID`, `ndays`, and outcome columns.
            outcome (str): One of 'value', 'delta_value', 'log_value',
                'delta_log_value'.
            draw_inds (bool): If True, draw individual subject lines.
            **save: Optional keywords `dir_out` and `fname_prefix` for saving.

        Returns:
            None
        """

        assert outcome in ['value', 'delta_value', 'log_value', 'delta_log_value']
        df_master['tp'] = pd.Categorical(
            df_master['tp'],
            categories=['bsl', 'A1', 'B1', 'B30',],
            ordered=True)

        ndays = [value for value in config.tp_to_ndays.values()]

        # Load mixed-effects results
        me_files = {
            'inflamation': os.path.join(folders.exports, 'infindex_mixedeffects_inflamation.csv'),
            'exploratory': os.path.join(folders.exports, 'infindex_mixedeffects_exploratory.csv'),
            'mitokines': os.path.join(folders.exports, 'infindex_mixedeffects_mitokines.csv'),
        }

        df_me_all = pd.concat([pd.read_csv(f) for f in me_files.values()])

        # Reverse Greek letters in measure names to match with config.markers
        # df_me_all['measure'] = df_me_all['measure'].str.replace('TNF-α', 'TNF-alpha').str.replace('IFN-γ', 'IFN-gamma')

        for measure in config.markers:

            df_measure = df_master.loc[df_master.measure == measure].copy()
            df_me_measure = df_me_all.loc[df_me_all.measure == measure].copy()

            # Get baseline mean (reference point)
            baseline_mean = df_measure.loc[df_measure.tp == 'bsl', outcome].mean()

            # Create dataframe with all timepoints for plotting
            plot_data = []
            plot_data.append({
                'tp': 'bsl',
                'ndays': 0,
                'mean': baseline_mean,
                'ci_low': baseline_mean,  # No CI for baseline (reference)
                'ci_high': baseline_mean,
            })

            # Add post-baseline timepoints with estimates and CIs from mixed model
            for _, row in df_me_measure.iterrows():
                tp_ndays = config.tp_to_ndays[row['tp']]
                plot_data.append({
                    'tp': row['tp'],
                    'ndays': tp_ndays,
                    'mean': baseline_mean + row['est'],  # est is change from baseline
                    'ci_low': baseline_mean + row['ci_low'],
                    'ci_high': baseline_mean + row['ci_high'],
                })

            df_plot = pd.DataFrame(plot_data)
            df_plot = df_plot.sort_values('ndays')

            fig, ax = plt.subplots(figsize=(4.8, 4.8))

            ### Draw patients' trajectories
            if draw_inds:
                sns.lineplot(
                    data=df_measure,
                    x='ndays',
                    y=outcome,
                    hue='pID',
                    palette=['gray' for _ in df_measure.pID.unique()],
                    linewidth=0.7,
                    alpha=0.3,
                    legend=False,
                )

            ### Draw group means with mixed-effects CIs
            # Calculate error bar sizes
            yerr_low = df_plot['mean'] - df_plot['ci_low']
            yerr_high = df_plot['ci_high'] - df_plot['mean']

            ax.errorbar(
                x=df_plot['ndays'],
                y=df_plot['mean'],
                yerr=[yerr_low, yerr_high],
                color='navy',
                marker='o',
                markersize=10,
                linewidth=2,
                capsize=config.err_kws['capsize'],
                elinewidth=config.err_kws['elinewidth'],
                capthick=config.err_kws['capthick'],
            )

            ### Styling
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            sns.despine(top=True, right=True, left=False, bottom=False, offset=5)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.0)

            ax.set_xticks(ndays)
            ax.set_xticklabels(['Baseline', '24h post-10 mg', '24h post-25 mg', '30d post-25 mg'],
                               rotation=30, ha='center')
            ax.tick_params(axis='x', bottom=True, labelsize=16)
            ax.tick_params(axis='y', left=True, labelsize=16)

            ax.set_xlabel('', fontdict=config.axislabel_fontdict)

            # Format measure name with Greek letters
            # measure_formatted = measure.replace('TNF-alpha', 'TNF-α').replace('IFN-gamma', 'IFN-γ')

            if outcome in ['log_value', 'delta_log_value']:
                ax.set_ylabel(f'ln(1+[{measure_formatted}])', fontdict=config.axislabel_fontdict)
            elif outcome == 'infindex':
                ax.set_ylabel(f'{measure_formatted}', fontdict=config.axislabel_fontdict)
            else:
                ax.set_ylabel(f'{measure_formatted} [pg/mL]', fontdict=config.axislabel_fontdict)

            ### Save
            commons.Plots.save_fig(
                fig=fig,
                save_PNG=config.save_PNG,
                save_SVG=config.save_SVG,
                dir_out=save['dir_out'],
                fname_out=f'{save['fname_prefix']}_{measure}_{outcome}')

    @staticmethod
    def draw_zscore_heatmap(df_master, outcome='delta_z_log_value', **save):
        """Draw heatmaps of mean marker changes using z-scored outcomes.

        Depending on `outcome`, computes per-measure z-scores of log values and
        either differences of z-scores across timepoints or z-scores of deltas,
        then plots grouped heatmaps.

        Args:
            df_master (pd.DataFrame): Input long-form dataframe with required
                marker/timepoint/value columns.
            outcome (str): 'z_delta_log_value' or 'delta_z_log_value'.
            **save: Optional keywords `dir_out` and `fname_prefix` for saving.

        Returns:
            None
        """

        assert outcome in ['z_delta_log_value', 'delta_z_log_value']
        df = df_master.copy()

        ### Calc z-scores of Δ log conc
        if outcome=='z_delta_log_value':
            df['z_delta_log_value'] = math.nan
            for measure in config.markers:
                df.loc[(df.measure==measure), 'z_delta_log_value'] = stats.zscore(
                df.loc[(df.measure==measure), 'delta_log_value'], 
                nan_policy='omit')

        ### Calculate Δ of log conc z-scores
        #  add_delta_scores() acts on the 'score' column and adds a 'delta_score' column, hence some temporary renaming
        if outcome=='delta_z_log_value':
            df['z_log_value'] = math.nan
            for measure in config.markers:
                df.loc[(df.measure==measure), 'z_log_value'] = stats.zscore(
                    df.loc[(df.measure==measure), 'log_value'], nan_policy='omit')

            # Calc delta scores of the z of log concs
            df.rename(columns={'z_log_value': 'score',}, inplace=True)
            df = commons.DataWrangl.add_delta_scores(df_master = df)
            df.rename(columns={'score': 'z_log_value',}, inplace=True)
            df.rename(columns={'delta_score': 'delta_z_log_value',}, inplace=True)

        ### Wrangling to wide
        df = df[['pID', 'tp', 'measure', outcome]]
        df = df.groupby(['tp', 'measure'])[outcome].mean().reset_index()
        df = df.pivot(index='measure', columns='tp', values=outcome)
        df = df[['bsl','A1','B1','B30',]]

        ### Draw plot
        for group_name, group in config.heatmap_groups.items():

            fig, ax = plt.subplots(figsize=(6, len(group) * 0.5), dpi=300)
            df_tmp = df.loc[(df.index.isin(group))]
            df_tmp = df_tmp.reindex(group)

            # Create custom annotations: baseline column as "0", others with 2 decimals
            annot_labels = df_tmp.copy()
            annot_labels['bsl'] = annot_labels['bsl'].apply(lambda x: '0' if x == 0 else f'{x:.2f}')
            for col in ['A1', 'B1', 'B30']:
                annot_labels[col] = annot_labels[col].apply(lambda x: f'{x:.2f}')

            sns.heatmap(
                data = df_tmp,
                ax = ax,
                vmin=-0.8,
                vmax=0.8,
                linewidths = .05,
                cmap = 'vlag',
                square=False,
                annot=annot_labels,
                fmt='',
                annot_kws={'fontsize': 8},
                cbar_kws={'shrink': 0.8, 'aspect': 30})

            ### Styling
            if outcome=='z_delta_log_value':
                ax.set_title('Mean z-score of Δ ln(1+ln[pg/mL])', fontdict=config.title_fontdict, pad=20)
            elif outcome=='delta_z_log_value':
                ax.set_title('Mean Δ z-score of ln(1+ln[pg/mL])', fontdict=config.title_fontdict, pad=20)

            ax.set_xticklabels(['Baseline', '24h post-10 mg', '24h post-25 mg', '30d post-25 mg'],
                               rotation=30, ha='center')

            # Replace Greek letter names in y-axis labels
            yticklabels = [label.get_text() for label in ax.get_yticklabels()]
            # yticklabels_formatted = [label.replace('TNF-alpha', 'TNF-α').replace('IFN-gamma', 'IFN-γ')
            #                          for label in yticklabels]
            ax.set_yticklabels(yticklabels_formatted, rotation=0)

            ax.tick_params(axis='x', pad=2)  # Bring x-axis labels closer to plot
            ax.set_xlabel(None)
            ax.set_ylabel(None)

            if save!={}:
                    
                commons.Plots.save_fig(
                    fig = fig,
                    dir_out  = save['dir_out'],
                    fname_out = f'{save['fname_prefix']}_{group_name}',
                    save_PNG = config.save_PNG,
                    save_SVG = config.save_SVG,)


class Models():

    @staticmethod
    def fit_paired_ttests(df_master, outcome='log_value', df_res=pd.DataFrame(), **save):
        """Run paired tests comparing baseline to later timepoints for markers.

        For each primary marker and timepoint, computes paired t-test and
        Wilcoxon signed-rank test, effect sizes and CIs, applies Aoife FDR
        correction by groups, organizes and optionally saves results.

        Args:
            df_master (pd.DataFrame): Long-form dataframe with outcome column.
            outcome (str): Outcome column to analyze.
            df_res (pd.DataFrame): Optional starting results dataframe to extend.
            **save: Optional `dir_out` and `fname_prefix` to save CSVs by groups.

        Returns:
            pd.DataFrame: Summary statistics with raw and adjusted p-values and
            significance markers.
        """
  
        ### Do tests
        for measure, tp in itertools.product(config.markers, config.tps):

            if tp=='bsl':
                continue

            ### Construct wide df
            df = df_master.loc[(df_master.measure==measure) & ((df_master.tp==tp) | (df_master.tp=='bsl'))]
            df = df[['pID', 'tp', outcome,]]
            df = df.pivot(index='pID', columns='tp', values=outcome)
            df.reset_index(drop=True, inplace=True)

            ### Get paired t- Wilcoxon test results 
            t, p_ttest, w, p_wilcoxon = Helpers.get_paired_t_wilcoxon_stats(df['bsl'], df[tp])

            ### Get effect size and associated CI
            smd = pg.compute_effsize(
                df[tp],
                df['bsl'],
                paired=True,
                eftype='hedges')
            smd = round(smd, 2)

            smd_ci_low, smd_ci_high = pg.compute_esci(
                stat = smd,
                nx = df['bsl'].shape[0], 
                ny = df[tp].shape[0],
                paired=True,
                decimals=2,
                eftype='cohen') # according to docs, use eftype='cohen' even if smd is Hedges' g; https://pingouin-stats.org/build/html/generated/pingouin.compute_esci.html

            ### Concatenate results
            df_stats = pd.DataFrame(data = {
                'measure': [measure], 
                'tp': [tp], 
                'smd': [smd], 'smd_ci_low': [smd_ci_low], 'smd_ci_high': [smd_ci_high],
                't': [t], 'p_ttest': [p_ttest], 
                'w': [w], 'p_wilcoxon': [p_wilcoxon],})

            df_res = pd.concat([df_res, df_stats], ignore_index=False)

        ### Add significance marks
        df_res['sig_ttest'] = df_res['p_ttest'].apply(commons.Helpers.sig_marking)
        df_res['sig_wilcoxon'] = df_res['p_wilcoxon'].apply(commons.Helpers.sig_marking)

        ### Adjust for multiple comparisons 
        df_res = Helpers.apply_aoife_correction(df_res, col_p='p_ttest')
        df_res[f'p_ttest_adj_sig'] = df_res[f'p_ttest_adj'].apply(commons.Helpers.sig_marking)
        df_res = Helpers.apply_aoife_correction(df_res, col_p='p_wilcoxon')
        df_res[f'p_wilcoxon_adj_sig'] = df_res[f'p_wilcoxon_adj'].apply(commons.Helpers.sig_marking)

        ### Round numerics
        df_res['t'] = round(df_res['t'], 3)
        df_res['p_ttest'] = round(df_res['p_ttest'], 3)
        df_res['p_ttest_adj'] = round(df_res['p_ttest_adj'], 3)
        df_res['w'] = round(df_res['w'], 3)
        df_res['p_wilcoxon'] = round(df_res['p_wilcoxon'], 3)
        df_res['p_wilcoxon_adj'] = round(df_res['p_wilcoxon_adj'], 3)

        ### Organize columns
        df_res = df_res[[
            'measure', 'tp', 'smd', 'smd_ci_low', 'smd_ci_high',
            't', 'p_ttest',    'sig_ttest',    'p_ttest_adj',    'p_ttest_adj_sig', 
            'w', 'p_wilcoxon', 'sig_wilcoxon', 'p_wilcoxon_adj', 'p_wilcoxon_adj_sig',
            ]]

        ### Sort rows
        df_res['measure'] = pd.Categorical(
            df_res['measure'], 
            categories = config.markers, 
            ordered = True)
        df_res = df_res.sort_values(by=['measure', 'tp']).reset_index(drop=True)

        if save!={}:
            df_tmp = df_res.loc[df_res.measure.isin(config.markers_inflamation )]
            df_tmp.to_csv(os.path.join(save['dir_out'], f'{save['fname_prefix']}_inflamation.csv'), index=False)

            df_tmp = df_res.loc[df_res.measure.isin(config.markers_mitokines + ['mitoindex'])]
            df_tmp.to_csv(os.path.join(save['dir_out'], f'{save['fname_prefix']}_mitokines.csv'), index=False)

            df_tmp = df_res.loc[df_res.measure.isin(config.markers_explorat)]
            df_tmp.to_csv(os.path.join(save['dir_out'], f'{save['fname_prefix']}_exploratory.csv'), index=False)

        return df_res

    @staticmethod
    def fit_withinarm_septps(df_master, outcome='log_value', df_res=pd.DataFrame(), fdr_adj='Aoife', **save):
        """Fit within-arm mixed models separately for each timepoint vs baseline.

        Uses a random-intercept mixed linear model with `tp` as a fixed effect
        for each marker and timepoint compared to baseline, then applies Aoife
        FDR correction and returns a tidy summary.

        Args:
            df_master (pd.DataFrame): Long-form dataframe with `pID`, `tp`, and
                the outcome column.
            outcome (str): Outcome column to analyze.
            df_res (pd.DataFrame): Optional starting results dataframe to extend.
            **save: Optional `dir_out` and `fname_prefix` to save CSVs by groups.

        Returns:
            pd.DataFrame: Tidy results with estimates, CIs, and adjusted p-values.
        """

        for measure, tp in itertools.product(config.markers, config.tps):

            if (tp=='bsl'):
                continue

            ### Construct model
            df = df_master.loc[(df_master.measure==measure) & (df_master.tp.isin(['bsl', tp]))]
            df.iloc[:, df.columns.get_loc('tp')] = pd.Categorical(
                df['tp'].copy(), 
                categories=['bsl', tp,],
                ordered=True)
            df.reset_index(drop=True, inplace=True)
                
            # Random intercept model
            model = smf.mixedlm(
                f"{outcome} ~ tp",
                groups = df['pID'],
                re_formula = "1",
                data = df,)               

            ### Add stats output to model
            df_model = pd.DataFrame(model.fit().summary().tables[1])
            df_model.insert(loc=len(df_model.columns), column='formula', value=model.formula)
            df_model.insert(loc=len(df_model.columns), column='measure', value=measure)
            df_model.insert(loc=len(df_model.columns), column='did_converge', value=model.fit().converged)
            df_model.insert(loc=len(df_model.columns), column='tp', value=tp)
            df_model = Helpers.clean_df_mixedlm(df_model.reset_index())
            
            df_res = pd.concat([df_res, df_model], ignore_index=False)

        df_res = df_res.loc[(df_res.term!='Intercept')]
        df_res.reset_index(drop=True, inplace=True)

        ### Flip sign to get tp-bsl difference (instead of bsl-tp)
        df_res['est'] = -df_res['est']
        df_res['ci_low'], df_res['ci_high'] = -df_res['ci_high'], -df_res['ci_low']

        ### Adjust for multiple comparisons using FDR; use one or the other version below
        assert fdr_adj in ['Aoife', 'traditional']        
        if fdr_adj == 'Aoife':
            # This is FDR adjutment according to Aoife's groupings 
            df_res = Helpers.apply_aoife_correction(df_res)
        elif fdr_adj == 'traditional':
            # This is the traditional, adjust for all-at-once FDR
            rejected, df_res[f'p_adj'] = fdrcorrection(df_res['p']) 
        
        df_res[f'p_adj_sig'] = df_res[f'p_adj'].apply(commons.Helpers.sig_marking)
        df_res['p'] = round(df_res['p'], 3)
        df_res['p_adj'] = round(df_res['p_adj'], 3)
        df_res['p_sig'] = df_res['p'].apply(commons.Helpers.sig_marking)

        ### Sort rows
        df_res['measure'] = pd.Categorical(
            df_res['measure'], 
            categories = config.markers, # config.markers_main, 
            ordered = True)
        df_res = df_res.sort_values(by=['measure', 'tp']).reset_index(drop=True)

        df_res =  df_res[[
            'measure', 'tp', 'did_converge', 
            'est', 'se', 'ci_low', 'ci_high', 
            'p', 'p_sig', 'p_adj', 'p_adj_sig'
        ]]

        if save!={}:
            # Format measure names with Greek letters for output
            df_tmp = df_res.loc[df_res.measure.isin(config.markers_inflamation)]
            df_tmp.to_csv(os.path.join(save['dir_out'], f'{save['fname_prefix']}_inflamation.csv'), index=False)

            df_tmp = df_res.loc[df_res.measure.isin(config.markers_mitokines + ['mitoindex'])]
            df_tmp.to_csv(os.path.join(save['dir_out'], f'{save['fname_prefix']}_mitokines.csv'), index=False)

            df_tmp = df_res.loc[df_res.measure.isin(config.markers_explorat)]
            df_tmp.to_csv(os.path.join(save['dir_out'], f'{save['fname_prefix']}_exploratory.csv'), index=False)

        return df_res


class Helpers():

    @staticmethod
    def apply_aoife_correction(df, col_p='p'):
        """Apply within marker group FDR corrections as defined by Aoife O'Donovan.

        Args:
            df (pd.DataFrame): Results with a `measure` column and p-values in
                `col_p`.
            col_p (str): Name of the p-value column to adjust.

        Returns:
            pd.DataFrame: Input with added adjusted p-value columns and
            significance markings.
        """

        ### Adjust infindex
        rejected, df.loc[(df.measure=='infindex'), f'{col_p}_adj'] = fdrcorrection(df[df.measure=='infindex'][col_p])
        
        ### Adjust other inflamation markers
        inf_markers = config.markers_infind_comps + ['IL-10', 'sIL-6R']
        rejected, df.loc[(df.measure.isin(inf_markers)), f'{col_p}_adj'] = fdrcorrection(df[df.measure.isin(inf_markers)][col_p])

        ### Adjust mitokines
        rejected, df.loc[(df.measure.isin(config.markers_mitokines + ['mitoindex'])), f'{col_p}_adj'] = fdrcorrection(df[df.measure.isin(config.markers_mitokines + ['mitoindex'])][col_p])

        ### Adjust exploratory markers
        rejected, df.loc[(df.measure.isin(config.markers_explorat)), f'{col_p}_adj'] = fdrcorrection(df[df.measure.isin(config.markers_explorat)][col_p])

        ### Apply sig markings
        return df

    @staticmethod
    def get_paired_t_wilcoxon_stats(x, y):
        """Calculates t- and Wilcoxon-test results.

        Args:
            x (np.array or pd.Series): Array of measured scores (first condition).
            y (np.array or pd.Series): Array of measured scores (second condition).

        Returns:
            tuple: (t, p_ttest, w, p_wilcoxon)
                t (float): t-statistic for paired t-test
                p_ttest (float): p-value for paired t-test
                w (float): W-statistic for Wilcoxon signed-rank test
                p_wilcoxon (float): p-value for Wilcoxon test
        """

        w = math.nan
        p_wilcoxon = math.nan
        t = math.nan
        p_ttest = math.nan

        t, p_ttest = stats.ttest_rel(x, y, nan_policy='omit')

        common_indexes = (x.dropna()).index.intersection((y.dropna()).index)
        if not x.loc[common_indexes].equals(y.loc[common_indexes]):
            w, p_wilcoxon = stats.wilcoxon(x, y, nan_policy='omit', method='auto')

        return t, p_ttest, w, p_wilcoxon

    @staticmethod
    def clean_df_mixedlm(df, rm_group_var=True):
        """Tidy a MixedLM summary table into numeric, consistently named columns.

        Args:
            df (pd.DataFrame): Raw table derived from `model.fit().summary()`.
            rm_group_var (bool): If True, remove rows for variance components.

        Returns:
            pd.DataFrame: Cleaned dataframe with columns `term`, `est`, `se`,
            `ci_low`, `ci_high`, and `p` as numeric types.
        """

        df = df.rename(columns={
            'index': 'term',
            'Coef.': 'est',
            'Std.Err.': 'se',
            'z': 'z',
            'P>|z|': 'p',
            '[0.025': 'ci_low',
            '0.975]': 'ci_high',
            'formula': 'formula',})

        if rm_group_var:
            df = df.loc[~(df.term.isin(['Group Var','ndays Var','trial Var',]))]

        df.est = pd.to_numeric(df['est'])
        df.se = pd.to_numeric(df['se'])
        df.ci_low = pd.to_numeric(df['ci_low'])
        df.ci_high = pd.to_numeric(df['ci_high'])
        df.p = pd.to_numeric(df['p'])

        return df