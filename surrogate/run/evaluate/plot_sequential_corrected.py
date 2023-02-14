import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

predict_dir = pl.Path("../predict/saves/global_30min/sequential")
transform_dir = pl.Path("../transform/saves/global_30min/setup")
dir_out = pl.Path("./saves/global_30min/predict/sequential")

pred_df_files = [file for file in predict_dir.rglob("corrected_*_ts.csv")]
pred_plot_dfs = []
for pred_df_file in pred_df_files:
    pred_plot_df = pd.read_csv(pred_df_file, index_col=0)
    pred_plot_dfs.append(pred_plot_df)
pred_plot_df = pd.concat(pred_plot_dfs, axis = 0)
pred_plot_df["type"] = "Predicted"

true_df_files = [file for file in transform_dir.rglob("output_*_ts.csv")]
true_plot_dfs = []
for true_df_file in true_df_files:
    true_plot_df = pd.read_csv(true_df_file, index_col=0)
    true_plot_dfs.append(true_plot_df)
true_plot_df = pd.concat(true_plot_dfs, axis = 0)
true_plot_df["type"] = "Actual"

plot_df = pd.concat((true_plot_df, pred_plot_df))

plot_out = pl.Path("{}/corrected_ts.pdf".format(dir_out))
plot_out.parent.mkdir(parents=True, exist_ok=True)
plots = plot_ts_seperate(input_df=plot_df, plot_sensitivity=True)
pn.save_as_pdf_pages(plots = plots, filename= plot_out)

pred_df_files = [file for file in predict_dir.rglob("corrected_*_cdf.csv")]
pred_plot_dfs = []
for pred_df_file in pred_df_files:
    pred_plot_df = pd.read_csv(pred_df_file, index_col=0)
    pred_plot_dfs.append(pred_plot_df)
pred_plot_df = pd.concat(pred_plot_dfs, axis = 0)
pred_plot_df["type"] = "Predicted"

true_df_files = [file for file in transform_dir.rglob("output_*_cdf.csv")]
true_plot_dfs = []
for true_df_file in true_df_files:
    true_plot_df = pd.read_csv(true_df_file, index_col=0)
    true_plot_dfs.append(true_plot_df)
true_plot_df = pd.concat(true_plot_dfs, axis = 0)
true_plot_df["type"] = "Actual"

plot_df = pd.concat((true_plot_df, pred_plot_df))

plot_out = pl.Path("{}/corrected_cdf.pdf".format(dir_out))
plot_out.parent.mkdir(parents=True, exist_ok=True)
plots = plot_cdf_seperate(input_df=plot_df)
pn.save_as_pdf_pages(plots = plots, filename= plot_out)

pred_df_files = [file for file in predict_dir.rglob("corrected_*_sp.csv")]
pred_plot_dfs = []
for pred_df_file in pred_df_files:
    pred_plot_df = pd.read_csv(pred_df_file, index_col=0)
    pred_plot_dfs.append(pred_plot_df)
pred_plot_df = pd.concat(pred_plot_dfs, axis = 0)
pred_plot_df["type"] = "Predicted"

true_df_files = [file for file in transform_dir.rglob("output_*_sp.csv")]
true_plot_dfs = []
for true_df_file in true_df_files:
    true_plot_df = pd.read_csv(true_df_file, index_col=0)
    true_plot_dfs.append(true_plot_df)
true_plot_df = pd.concat(true_plot_dfs, axis = 0)
true_plot_df["type"] = "Actual"

plot_df = pd.concat((true_plot_df, pred_plot_df))

plot_out = pl.Path("{}/corrected_sp.pdf".format(dir_out))
plot_out.parent.mkdir(parents=True, exist_ok=True)
plots = plot_sp_seperate(input_df=plot_df)
pn.save_as_pdf_pages(plots = plots, filename= plot_out)
