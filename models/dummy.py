# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from general.code import mae_from_maenorm_Tn, rmse_from_msenorm_Tn
from scipy.stats import spearmanr
import numpy as np
from general.code import convert_normT_2_unnormT

def get_regression_metrics(model, X, y_true, rescale=False, rescale_rule=None, dataframe=None):
    """
    Get a dictionary with regression metrics:

    model: sklearn model with predict method
    X: feature matrix
    y_true: ground truth labels (predicted values)
    """
    y_predicted = model.predict(X)

    if rescale:
        if rescale_rule == "norm_Tn":
            mae = mae_from_maenorm_Tn(mean_absolute_error(y_true, y_predicted), dataframe)
            rmse = rmse_from_msenorm_Tn(mean_squared_error(y_true, y_predicted), dataframe)

            y_true = convert_normT_2_unnormT(y_true, dataframe)
            y_predicted = convert_normT_2_unnormT(y_predicted, dataframe)
            R2 = r2_score(y_true, y_predicted)
            spear = spearmanr(y_true, y_predicted)
        elif rescale_rule == 'cbrt':
            mae = np.mean(np.abs(y_true**3 - y_predicted**3))
            rmse = np.sqrt(
                np.mean(
                    (y_true**3 - y_predicted**3)**2
                )
            )
            R2 = r2_score(y_true**3, y_predicted**3)
            spear = spearmanr(y_true**3, y_predicted**3)
        else:
            raise Exception('Dont know what rescale rule {} means'.format(rescale_rule))
    else:
        mae = mean_absolute_error(y_true, y_predicted)
        rmse = mean_squared_error(y_true, y_predicted)**0.5
        R2 = r2_score(y_true, y_predicted)
        spear = spearmanr(y_true, y_predicted)

    metrics_dict = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': R2,
        'SPEARMAN': spear
    }

    return metrics_dict

