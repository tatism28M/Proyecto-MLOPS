Proyecto-MLOPS-Base
==============================

Proyecto Inicial: Desarrollo de un Modelo de Predicción para Integración en un Ciclo MLOps

Project Organization
------------

```
proyecto-mlops-base/
├── LICENSE     
├── README.md                  
├── environment.yml              # Conda environment
├── Makefile                     # Makefile with commands like `make data` or `make train`                   
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml              
│
├── data                         
│   ├── external                 # Data from third party sources.
│   ├── interim                  # Intermediate data that has been transformed.
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
│
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.
    │
    ├── data_eng                 # Data engineering scripts.
    │   ├── stage0_loading.py    
    │   ├── stage1_ingestion.py          
    │   ├── stage2_cleaning.py         
    │   ├── stage3_labeling.py          
    │
    ├── app_exception            # Project exceptions.    
    │   ├── exception.py         
    │
    ├── app_logging              # Project logging.      
    │   ├── logger.py        
    │   ├── __init__.py
    │
    ├── model_eng                # ML model engineering (a folder for each model).
    │   └── model1      
    │       ├── __init__.py    
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py        
        └── exploration.py       
```


--------
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>
