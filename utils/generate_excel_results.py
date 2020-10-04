import pandas as pd
from pandas import ExcelWriter
import os
import numpy as np
import sys

log_base_path = sys.argv[1]
main_categories = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel', 'main_mean']
novel_categories = ['bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard', 'novel_mean']
model_names = ['pcn', 'topnet', 'msn', 'cascade']
train_modes = ['cd', 'emd']
loss_cols = ['emd', 'cd_p', 'cd_p_f1', 'cd_t', 'cd_t_f1']
sheet_names = ['cd_train_main_category', 'cd_train_novel_category', 'cd_train_overview','emd_train_main_category',
               'emd_train_novel_category', 'emd_train_overview', ]


def save_xls(list_dfs, xls_path):
    assert len(list_dfs) == len(sheet_names)
    with ExcelWriter(xls_path, engine='xlsxwriter') as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, sheet_names[n])
            if n != 2 and n != 5:
                writer.sheets[sheet_names[n]].set_row(2, None, None, {'hidden': True})
        writer.save()


def generate_cat_results_row(best_emd_cat, best_cd_p_cat, best_cd_t_cat):
    main_cat_r = []
    novel_cat_r = []
    emd = [float(line.split(' ')[5]) for line in best_emd_cat]
    cd_p = [float(line.split(' ')[1][:-1]) for line in best_cd_p_cat]
    cd_p_f1 = [float(line.split(' ')[-1]) for line in best_cd_p_cat]
    cd_t = [float(line.split(' ')[3][:-1]) for line in best_cd_t_cat]
    cd_t_f1 = [float(line.split(' ')[-1]) for line in best_cd_t_cat]
    for i in range(8):
        main_cat_r.extend([emd[i], cd_p[i], cd_p_f1[i], cd_t[i], cd_t_f1[i]])
        novel_cat_r.extend([emd[i+8], cd_p[i+8], cd_p_f1[i+8], cd_t[i+8], cd_t_f1[i+8]])
    main_cat_r.extend([np.mean(emd[:8]), np.mean(cd_p[:8]), np.mean(cd_p_f1[:8]), np.mean(cd_t[:8]), np.mean(cd_t_f1[:8])])
    novel_cat_r.extend([np.mean(emd[8:]), np.mean(cd_p[8:]), np.mean(cd_p_f1[8:]), np.mean(cd_t[8:]), np.mean(cd_t_f1[8:])])
    return main_cat_r, novel_cat_r


def generate_overview_row(best_emd_overview, best_cd_p_overview, best_cd_t_overview):
    best_emd = float(best_emd_overview.split(' ')[5])
    best_cd_p = float(best_cd_p_overview.split(' ')[1][:-1])
    best_cd_p_f1 = float(best_cd_p_overview.split(' ')[-1])
    best_cd_t = float(best_cd_t_overview.split(' ')[3][:-1])
    best_cd_t_f1 = float(best_cd_t_overview.split(' ')[-1])
    return [best_emd*(10**4), best_cd_p*(10**4), best_cd_p_f1, best_cd_t*(10**4), best_cd_t_f1]


sheets = []
for mode in train_modes:
    main_cat_col = pd.MultiIndex.from_product([main_categories, loss_cols])
    main_cat_df = pd.DataFrame(columns=main_cat_col, index=model_names)
    novel_cat_col = pd.MultiIndex.from_product([novel_categories, loss_cols])
    novel_cat_df = pd.DataFrame(columns=novel_cat_col, index=model_names)
    overview_df = pd.DataFrame(columns=loss_cols, index=model_names)

    for model in model_names:
        log_file = os.path.join(log_base_path, model + '_' + mode, 'log_test.txt')
        with open(log_file) as f:
            content = f.readlines()
            main_cat_row, novel_cat_row = generate_cat_results_row(content[16:32], content[50:66], content[84:100])
            overview_row = generate_overview_row(content[33], content[67], content[101])
            main_cat_df.loc[model] = main_cat_row
            novel_cat_df.loc[model] = novel_cat_row
            overview_df.loc[model] = overview_row

    sheets.append(main_cat_df)
    sheets.append(novel_cat_df)
    sheets.append(overview_df)

save_xls(sheets, os.path.join(log_base_path, 'benchmark_results.xlsx'))





