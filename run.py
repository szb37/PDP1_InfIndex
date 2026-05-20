import src.config as config
import sys
sys.path.append(config.path_szb_commons)
import commons_codebase.src.core as commons
import src.folders as folders
import src.core as core
import pandas as pd
import sys
import os


### Data wrangling
if False: # Create master dataframe
    core.DataWrangl.get_df_master(
        filter=True,
        save={
            'dir_out': folders.exports, 
            'fname_out': 'infindex_master.csv'})

if False: # Get observed values table

    ### Observed vales
    df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv'))
    df_observed = commons.Analysis.get_df_observed(
        df_master = df_master.rename(columns={'value': 'score',}),)

    # Sort rows
    df_observed['measure'] = pd.Categorical(
        df_observed['measure'], 
        categories = config.markers, 
        ordered = True)
    df_observed = df_observed.sort_values(by=['measure', 'tp']).reset_index(drop=True)
    df_observed.to_csv(os.path.join(folders.exports, 'infindex_observed.csv'), index=False)

    # Log transformed observed values
    df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv'))
    df_observed = commons.Analysis.get_df_observed(
        df_master = df_master.rename(columns={'log_value': 'score',}),)

    ### Sort rows
    df_observed['measure'] = pd.Categorical(
        df_observed['measure'], 
        categories = config.markers, 
        ordered = True)
    df_observed = df_observed.sort_values(by=['measure', 'tp']).reset_index(drop=True)
    df_observed.to_csv(os.path.join(folders.exports, 'infindex_observed_log_value.csv'), index=False)

### Models
if False: # Calculate t-tests 
    core.Models.fit_paired_ttests(
        df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv')),
        outcome = 'log_value',
        dir_out = folders.exports,
        fname_prefix = 'infindex_pairedtwilcoxon',)        

if False: # Calculate mixed-effects models    
    core.Models.fit_withinarm_septps(
        df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv')), 
        outcome = 'delta_log_value',
        dir_out = folders.exports,
        fname_prefix = 'infindex_mixedeffects_delta_log_value',)

if False: # Calculate baseline correlations

    markers = config.markers_infind_comps + config.markers_mitokines 
    df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv'))
    df_master = df_master.rename(columns={'log_value': 'score',})
    
    for tp in config.tps:
    
        df = commons.DataWrangl.widen_master(
            df_master = df_master,
            xvars = markers,
            x_tp = tp,
            x_use_delta = False,
            yvars = markers,
            y_tp = tp,
            y_use_delta = False)

        commons.Analysis.get_corrmats(
            df = df,
            xvars = markers,
            yvars = markers,
            draw = True,
            title = f'[1+ln(pg/mL)] correlation matrix at {tp}',
            dir_out = folders.corrmats,
            fname_out = f'infindex_corrmat_{tp}',)

if True: # Correlations between biomarkers and PD-related symptoms 

    df_master = pd.read_csv(os.path.join(folders.exports, 'private', 'infindex_pdp1_master.csv'))
    # Note: we are not sharing PD outcomes data publicly
    markers = ['infindex'] + config.markers_infind_comps + config.markers_mitokines
    pd_symps = ['UPDRS_1', 'UPDRS_2', 'UPDRS_3', 'UPDRS_4', 'UPDRS_SUM']
    
    for tps in [('A1', 'A7'), ('B1', 'B7'), ('B30', 'B30')]:

        x_tp = tps[0]
        y_tp = tps[1]
        
        df = commons.DataWrangl.widen_master(
            df_master = df_master,
            xvars = markers,
            x_tp = x_tp,
            x_use_delta = True,
            yvars = pd_symps,
            y_tp = y_tp,
            y_use_delta = True)

        commons.Analysis.get_corrmats(
            df = df,
            xvars = markers,
            yvars = pd_symps,
            draw = True,
            title = f'Δ log score @{x_tp} vs. Δ PD symptoms @{y_tp} correlations',
            dir_out = os.path.join(folders.corrmats, 'markers vs PD symps'),
            fname_out = f'infindex_pdsymps_corrmat_{x_tp}_{y_tp}',)




### Visualisations 
if False: # Draw change-over-time plots
    core.Plots.draw_conc_trajectory(
        df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv')),
        outcome = 'log_value',
        draw_inds = True,
        draw_group = True,
        dir_out = folders.conc_trajectories,
        fname_prefix = 'infindex_trajectory_combined',)        

    core.Plots.draw_conc_trajectory(
        df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv')),
        outcome = 'log_value',
        draw_inds = True,
        draw_group = False,
        dir_out = folders.conc_trajectories,
        fname_prefix = 'infindex_trajectory_inds',)     

if False: # Draw Aoife's heatmap
    core.Plots.draw_zscore_heatmap(
        df_master = pd.read_csv(os.path.join(folders.exports, 'infindex_master.csv')),
        dir_out = folders.heatmaps,
        fname_prefix = 'infindex_heatmap',)

### Miscs
if False: # Merge SMDs into mixedeffect table
    df_mixed = pd.read_csv(os.path.join(folders.exports, 'infindex_mixedeffects_inflamation.csv'))
    df_pairedt = pd.read_csv(os.path.join(folders.exports, 'infindex_pairedtwilcoxon_inflamation.csv'))
    df_pairedt = df_pairedt[['measure', 'tp', 'smd', 'smd_ci_low', 'smd_ci_high',]]

    df_mixed = pd.merge(df_mixed, df_pairedt, on=['measure', 'tp'], how='left')
    df_mixed = df_mixed[[
        'measure', 'tp',
        'smd', 'smd_ci_low', 'smd_ci_high',
        'est', 'ci_low', 'ci_high', 
        'p', 'p_sig', 'p_adj', 'p_adj_sig',]]
    df_mixed.to_csv(os.path.join(folders.exports, 'infindex_Table1.csv'), index=False)