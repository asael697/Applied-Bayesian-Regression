# Applied Bayesian Regression

Este repositorio presenta el contenido y cuadernos para la clase de Modelos lineales para la Maestría en Matemática, tercer periodo del año 2021. 

  + [Notebooks](https://github.com/asael697/Appplied-Bayesian-Regression/tree/main/Notebooks): Guarda los cuadernos de la clase en formato `.md`  para que puedan ser visualizados de forma rápida por los alumnos.
  
  + [qmd](): Guardas los cuadernos en `quarto` para poder visualizar el archivo `.html` usado en la clase, descargar el archivo, y hacer un render desde la consola u `Rstudio`.
  
La clase se imparte los `Lunes, Martes` y `Jueves` a las `17:00` horas (GMT -6), Para acceder al enlace zoom de la clase presione [aquí](https://aalto.zoom.us/j/63860483890).

## Contenido

La segunda parte del curso se impartirá regresión Bayesiana aplicada, el objetivo principal es que el estudiante pueda resolver un problema de regresión con un conjunto de datos reales mediante un ordenador, independiente del enfoque de inferencia a utilizar. 

El contenido para el resto del curso es:

 + Regresión aplicada
  
     - Verosimilitud y función de enlace.
     - L-BFGS algorithm.
     - Intervalos de confianza, Jackniffe y Bootstrap.
     - Análisis de residuos, ANOVA  y $R^2$ ajustado.
     - selección del modelos (BIC, RMSE, MAPE, CV)
 
 + Repaso de inferencia Bayesiana y Bayesian workflow

    - Prior, likelihood, Posterior
    - MCMC
    - Predictive distribution
    - Posterior predictive checks
    - Bayes factor, ELPD, LOO-CV.
  
 + Bayesian Regression
   
   - Bayesian GLMs, normal, Binomial, Poisson and Negative Binomial regressions
   - Regularized priors (R2-D2, Horseshoe, Spike Lab)
   - Gaussian process regression.
   
## Material 
  
El material de la clase se extrae de 3 libros, varios artículos y diferentes paquetes de R y Python estos son libres y se encuentran en formato digital en la web. 

### Libros

 + **Bayes Rules!** An Introduction to Applied Bayesian Modeling. [Johnson, Ott and Dogucu, (2021)](https://www.bayesrulesbook.com/).
 
 + **Beyond Multiple Linear Regression** Applied Generalized Linear Models and Multilevel Models in R. [Roback and Legler (2021)](https://bookdown.org/roback/bookdown-BeyondMLR/).
 
 + Bayesian Modeling and Computation in Python. [Martin, Kumar, and Lao (2021)](https://bayesiancomputationbook.com/welcome.html).

### Artículos principales   
   
   + *Bayesian Regression Using a Prior on the Model Fit: The R2-D2 shrinkage Prior.* [Zhang et al. (2022)](https://doi.org/10.1080/01621459.2020.1825449)
   
   + *Handling Sparsity via the Horseshoe.* [Carvalho, Polso and Scott (2009)](https://proceedings.mlr.press/v5/carvalho09a.html)
   
   + *Bayesian Variable Selection in Linear Regression.* [Mitchell and Beauchamp (1988)](https://doi.org/10.1080/01621459.1988.10478694)

### Paquetes

Los lenguajes de programación a usar son R y Python.
 
### R core team
 
  + Probabilistic Programming Language: Stan [mc-stan](https://mc-stan.org/users/interfaces/rstan).
  
  + paquetes: 
  
    - [rstanarm](https://mc-stan.org/rstanarm/), paquete para ajustar modelos lineales.
    - [bayesplot](https://mc-stan.org/bayesplot/), visualización de posterioris.
    - [loo](https://mc-stan.org/loo/) seleccion de modelos.
  
   
### Python
 
  + Probabilistic Programming Language: [PyMC](https://www.pymc.io/welcome.html).
  
  + paquetes: 
  
    - [Bambi]( https://bambinos.github.io/bambi/main/index.html#), Ajustar modelos lineales.
    - [ArviZ](https://arviz-devs.github.io/arviz/index.html), visualización de datos y selección de modelos con LOO.
    
