import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate
from surrogate.utils.plot import plot_cdf_seperate
from surrogate.utils.plot import plot_sp_seperate

correct_dir = pl.Path("../correct/saves/global_05min")
transform_dir = pl.Path("../transform/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
trainset = "train_32"

submasks = [dir.stem for dir in correct_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    correct_submask_dir = pl.Path("{}/{}".format(correct_dir, submask))
    transform_submask_dir = pl.Path("{}/{}".format(transform_dir, submask))
    submask_out = pl.Path("{}/{}/correct".format(dir_out, submask))
    
    correct_trainset_dir = pl.Path("{}/total".format(correct_submask_dir))
    transform_trainset_dir = pl.Path("{}/{}".format(transform_submask_dir, trainset))
    trainset_out = pl.Path("{}/total".format(submask_out))
    
    df_file = pl.Path("{}/output_ts.csv".format(transform_trainset_dir))
    true_plot_df = pd.read_csv(df_file, index_col=0)
    true_plot_df["type"] = "Actual"
    
    df_file = pl.Path("{}/output_ts.csv".format(correct_trainset_dir))
    pred_plot_df = pd.read_csv(df_file, index_col=0)
    pred_plot_df["type"] = "Corrected"

    plot_df = pd.concat((true_plot_df, pred_plot_df))
    
    plot_out = pl.Path("{}/output_ts.pdf".format(trainset_out))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plots = plot_ts_seperate(input_df=plot_df, plot_sensitivity=True)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)
    
    df_file = pl.Path("{}/output_cdf.csv".format(transform_trainset_dir))
    true_plot_df = pd.read_csv(df_file, index_col=0)
    true_plot_df["type"] = "Actual"

    df_file = pl.Path("{}/output_cdf.csv".format(correct_trainset_dir))
    pred_plot_df = pd.read_csv(df_file, index_col=0)
    pred_plot_df["type"] = "Corrected"

    plot_df = pd.concat((true_plot_df, pred_plot_df))

    plot_out = pl.Path("{}/output_cdf.pdf".format(trainset_out))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plots = plot_cdf_seperate(input_df=plot_df)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)
    
    df_file = pl.Path("{}/output_sp.csv".format(transform_trainset_dir))
    true_plot_df = pd.read_csv(df_file, index_col=0)
    true_plot_df["type"] = "Actual"
    
    df_file = pl.Path("{}/output_sp.csv".format(correct_trainset_dir))
    pred_plot_df = pd.read_csv(df_file, index_col=0)
    pred_plot_df["type"] = "Corrected"

    plot_df = pd.concat((true_plot_df, pred_plot_df))

    plot_out = pl.Path("{}/output_sp.pdf".format(trainset_out))
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plots = plot_sp_seperate(input_df=plot_df)
    pn.save_as_pdf_pages(plots = plots, filename= plot_out)
    