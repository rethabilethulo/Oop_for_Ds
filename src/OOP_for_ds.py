import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tools.eval_measures import rmse
from statsmodels.tools.eval_measures import mse


class ErrorCalculator:

    def __init__(self, y_test, y_predictor):

# target is y and prediction of target is y_prediction
        self.y_test     =   np.array(y_test)       
        self.y_predictor     =   np.array(y_predictor)  

    # check that len of y_prediction is equall to len of y

    def dimension(self):

        if len(self.y_test.shape) == len(self.y_predictor.shape):
            return True

        else:
            return False



    def get_residuals(self):

        residuals = self.y_test - self.y_predictor
        return residuals

    def get_standardised_residuals(self):

        return self.get_residuals() / (self.get_residuals().std())

    def get_mse(self):
        return mse(self.y_test, self.y_predictor)

    def get_rmse(self):
        return rmse(self.y_test, self.y_predictor)
    
    def error_summary(self):
        return pd.DataFrame({"Standardised Residuals Average Mean" : [self.get_standardised_residuals().mean()],
                             "Standardised Residuals Average Min": [self.get_standardised_residuals().min()],
                             "Standardised Residuals Average Max": [self.get_standardised_residuals().max()],
                             "MSE": [self.get_mse()],
                             "RMSE": [self.get_rmse()]},
                             columns= ["Standardised Residuals Average Mean",
                                     "Standardised Residuals Average Min",
                                     "Standardised Residuals Average Max",
                                     "MSE",
                                     "RMSE"])

#Plotter class to run claculations
class Plotter():
    def __init__(self,y_test,y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
    
    def run_calculations(self):
        return self.y_test - self.y_pred
    
    def plot(self):
        plt.hist(self.y_test - self.y_pred)
        plt.title("Residuals vs model predictions")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        return plt.show()   

        
##Histogram plotter
class HistogramPlotter(Plotter):
    def __init__(self, y_test,y_pred):
        Plotter.__init__(self, y_test, y_pred)


#Scatterplot plotter
class ScatterPlotter(Plotter):
    
    def __init__(self, y_test, y_pred):
        Plotter.__init__(self,y_test, y_pred)
     
    def plot(self):
        df = pd.DataFrame({"y_test": self.y_test, "y_prediction": self.y_pred})
        df.plot.scatter( x = "y_test", y = "y_prediction",color = "indigo")
        plt.title("Model Predictions vs Actual Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Prediction")
        return  plt.show()



        
