import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curve(df):
    
    df_dict = {}
    for eval_set, eval_metric_dict in clf.evals_result_.items():
        for eval_metric, eval_loss in eval_metric_dict.items():
            df_dict[eval_set] = eval_loss
        
    df = pd.DataFrame(df_dict)
    ax = plot_training_curve(clf)
    ax.set_title('Training Curve - XGBoost Model')
    
    ax.set_xlabel('Number of Iterations (Number of trees)')
    ax.set_ylabel('Loss')
    ax.legend(['training loss', 'validation loss'])

    return ax