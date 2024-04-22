import pathlib as pl
import plotnine as pn
import pandas as pd

from surrogate.utils.plot import plot_ts_seperate

save_dir = pl.Path("./saves/train-test2")
out_dir = pl.Path("./saves/train-test2")

resolution = "30min"

save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

trainsets = [dir.stem for dir in save_resolution_dir.iterdir() if dir.is_dir()]
    
trainset = trainsets[0]
for trainset in trainsets:
    print("Trainset: {}".format(trainset))

    save_trainset_dir = pl.Path("{}/{}".format(save_resolution_dir, trainset))
    out_trainset_dir = pl.Path("{}/{}".format(out_resolution_dir, trainset))

    datasets = [dir.stem for dir in save_trainset_dir.iterdir() if dir.is_dir()]
        
    dataset = datasets[0]
    for dataset in datasets:
        print("\tDataset: {}".format(dataset))
        
        save_dataset_dir = pl.Path("{}/{}".format(save_trainset_dir, dataset))
        out_dataset_dir = pl.Path("{}/{}".format(out_trainset_dir, dataset))
        
        df_files = [file for file in save_dataset_dir.rglob("*_ts.csv".format(dataset))]
        plot_dfs = []
        for df_file in df_files:
            plot_df = pd.read_csv(df_file, index_col=0)
            plot_dfs.append(plot_df)
        plot_df = pd.concat(plot_dfs)
        plot_df["type"] = "setup"

        plot_out = pl.Path("{}_ts.pdf".format(out_dataset_dir))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plots = plot_ts_seperate(input_df=plot_df,
                                plot_sensitivity=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)

        plot_out = pl.Path("{}_ts-quant.pdf".format(out_dataset_dir))
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plots = plot_ts_seperate(input_df=plot_df,
                                plot_sensitivity=True,
                                quantile_lim=True)
        pn.save_as_pdf_pages(plots = plots, filename= plot_out)
