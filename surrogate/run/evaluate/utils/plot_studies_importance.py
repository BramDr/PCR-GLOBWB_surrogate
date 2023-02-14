from typing import Sequence

import pandas as pd
import plotnine as pn

import optuna as op
import optuna.importance as im

def plot_studies_importance(studies: Sequence[op.Study],
                            subsets: Sequence[str]):
    
    importance = pd.DataFrame()
    for subset, study in zip(subsets, studies):
        importance_study = im.get_param_importances(study=study)
        importance_study = {"parameter": importance_study.keys(), 
                            "importance": importance_study.values()}
        
        best_params = [study.best_params[p] for p in importance_study["parameter"]]
        best_params = ["{:.0e}".format(p) for p in best_params]
        importance_study["best"] = best_params
        
        importance_study = pd.DataFrame(importance_study)
        importance_study["type"] = subset
        
        importance = pd.concat((importance, importance_study))
    
    plot = pn.ggplot(data = importance,
                    mapping=pn.aes(x="parameter",
                                    y="importance",
                                    fill="type",
                                    label="best"))
    plot += pn.geom_bar(stat="identity", position = "dodge")
    plot += pn.geom_text(position = pn.position_dodge(.9), size=8)
    plot += pn.theme(panel_background=pn.element_blank(),
                    panel_grid_major=pn.element_blank(),
                    panel_grid_minor=pn.element_blank(),
                    axis_text_x=pn.element_text(rotation=45))
    return plot
