![Prognostics - Predictor Selection](https://github.com/marcialbaptista/XAIPrognosticsSHAP/blob/master/imgs/icon.png?raw=true)

##Prognostics Metrics and Local Interpretability SHAP Values

To be capable of understanding the reasons behind the decision of a complex decision-making system is a topic of interest in many fields of study, especially those in which accountability and responsibility are crucial aspects. Maintenance decisions in domains such as aeronautics are becoming increasingly dependent on advanced and complex prognostics models. However, these models need to be better understood and more transparent. To help address this issue, we use the SHAP model (SHapley Additive exPlanations) from the field of eXplainable Artificial Intelligence (XAI) to analyze the outcome of three increasingly complex algorithms: Linear Regression, Multi-Layer Perceptron and Echo State Network. Our goal is to correlate the popular prognostics metrics of monotonicity, trendability and prognosability with the explanations of the SHAP model, i.e., the SHAP values. Our results on the C-MAPSS engine data suggest that SHAP values exhibit monotonic, trendable and prognosable trends. 

<p float="center">
  <img src="imgs/SHAP.png" width="33%">
</p>

This repository provides code to compute the monotonicity, trendability and prognosability of the predictors in C-MAPSS dataset 1 and the monotonicity,trendability and prognosability of their SHAP values
for the models Linear Regression (LR), Multi-Layer Perceptron (MLP) and Echo State Network (ESN). 

## Libraries Used

Python

- [Python Standard Library](https://docs.python.org/2/library/): Built in python modules.
- [Numpy](https://numpy.org/): Scientific computing with python.
- [Pandas](https://pandas.pydata.org/): Data analysis tools.
- [Scikit-learn](https://scikit-learn.org/stable/): Machine learning toolkit.

### Note

The code was developed for C-MAPSS datasets but it can be easily adapted to other applications. 

### Support or Contact

Having trouble with this repository? Contact me and we’ll help you sort it out.


