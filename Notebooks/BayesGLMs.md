Una introducción al Bayesian Workflow
================
Asael Alonzo Matamoros.
2022-12-03

Los métodos Bayesianos modernos se desarrollan mayoritariamente mediante
ordendores. En la actualidad, múltiples algoritmos permiten aproximar
las densidades a posterior en tiempo real, disminuyendo la brecha de
complejidad que existía en el desarrollo y evaluación de modelos
probabilistas.

Las características mas importantes de usar métodos Bayesianos en la
práctica son:

1.  Las cantidades desconocidas se describen usando funciones de
    densidad (*parámetros*).

2.  El *teorema de Bayes* es utilizado para actualizar los parámetros
    desconocidos.

3.  Permite incorporar información adicional en el proceso de estimación
    de los parámetros mediante una densidad (*priori*).

Es importante resaltar que este Bayesian Workflow Andrew Gelman et al.
(2020), es análogo al [workflow
Básico](https://asael697.github.io/ABM/posts/GLMs/) presentado para
análisis de modelos frecuentistas. Previo a nuestra introducción de la
metodología a utilizar, es necesario establecer nuestros supuestos y
objetos de estudio.

## La historia de siempre

Sea
![Y = \\{Y_1,Y_2,\ldots,Y_n\\}](https://latex.codecogs.com/svg.latex?Y%20%3D%20%5C%7BY_1%2CY_2%2C%5Cldots%2CY_n%5C%7D "Y = \{Y_1,Y_2,\ldots,Y_n\}")
una colección de variables aleatorias[^1] intercambiables[^2]. Sea
![y = (y_1,y_2,\ldots,y_n)](https://latex.codecogs.com/svg.latex?y%20%3D%20%28y_1%2Cy_2%2C%5Cldots%2Cy_n%29 "y = (y_1,y_2,\ldots,y_n)")
el vector de datos observados
(![Y = y](https://latex.codecogs.com/svg.latex?Y%20%3D%20y "Y = y")),
cuya función de densidad es
![f(y_i\|\theta_i)](https://latex.codecogs.com/svg.latex?f%28y_i%7C%5Ctheta_i%29 "f(y_i|\theta_i)"),
y
![\theta_i](https://latex.codecogs.com/svg.latex?%5Ctheta_i "\theta_i")
son desconocidos.

En este enfoque,
![\theta_i \in \mathbb{R}^k](https://latex.codecogs.com/svg.latex?%5Ctheta_i%20%5Cin%20%5Cmathbb%7BR%7D%5Ek "\theta_i \in \mathbb{R}^k")
es un vector de parámetros considerada aleatorio, su espacio muestral es
![(\Theta,\mathcal F, P)](https://latex.codecogs.com/svg.latex?%28%5CTheta%2C%5Cmathcal%20F%2C%20P%29 "(\Theta,\mathcal F, P)"),
y su función de densidad inicial es
![f(\theta)](https://latex.codecogs.com/svg.latex?f%28%5Ctheta%29 "f(\theta)").

La función
![f(\theta)](https://latex.codecogs.com/svg.latex?f%28%5Ctheta%29 "f(\theta)")
resume todos los supuestos iniciales de los parámetros desconocidos,
resumiendo la `incertidumbre` (*mide que tan incierto es el valor del
parámetro para dichos datos*). El objetivo es actualizar la
`incertidumbre` mediante la nueva información obtenida (*datos*) del
fenómeno en estudio, y por el *teorema de Bayes*, esta se actualiza
mediante la siguiente formula:

![f(\theta_i\|y) = \frac{f(y\|\theta_i)f(\theta_i)}{\int f(y\|\theta_i)f(\theta_i)d \theta_i} \quad j = 1,2,\ldots,k.](https://latex.codecogs.com/svg.latex?f%28%5Ctheta_i%7Cy%29%20%3D%20%5Cfrac%7Bf%28y%7C%5Ctheta_i%29f%28%5Ctheta_i%29%7D%7B%5Cint%20f%28y%7C%5Ctheta_i%29f%28%5Ctheta_i%29d%20%5Ctheta_i%7D%20%5Cquad%20j%20%3D%201%2C2%2C%5Cldots%2Ck. "f(\theta_i|y) = \frac{f(y|\theta_i)f(\theta_i)}{\int f(y|\theta_i)f(\theta_i)d \theta_i} \quad j = 1,2,\ldots,k.")

Donde:

- Bajo el supuesto de intercambiabilidad,
  ![f(y\|\theta_i) = \prod\_{j=1}^n f(y_j\|\theta_i)](https://latex.codecogs.com/svg.latex?f%28y%7C%5Ctheta_i%29%20%3D%20%5Cprod_%7Bj%3D1%7D%5En%20f%28y_j%7C%5Ctheta_i%29 "f(y|\theta_i) = \prod_{j=1}^n f(y_j|\theta_i)").

- ![f(\theta_i\|y)](https://latex.codecogs.com/svg.latex?f%28%5Ctheta_i%7Cy%29 "f(\theta_i|y)")
  es la posteriori de los parámetros (`incertidumbre` “mejorada”).

<div>

> **Notar que:**
>
> - La denisdad
>   ![f(\theta_i\|y) = f(\theta_i\|Y=y)](https://latex.codecogs.com/svg.latex?f%28%5Ctheta_i%7Cy%29%20%3D%20f%28%5Ctheta_i%7CY%3Dy%29 "f(\theta_i|y) = f(\theta_i|Y=y)")
>   esta condicionada a una cantidad fija
>   (![Y=y](https://latex.codecogs.com/svg.latex?Y%3Dy "Y=y")), por lo
>   tanto, la posterior no es aleatoria ni abstracta.
>
> - ![f(y) = \int f(y\|\theta_i)f(\theta_i)d \theta_i](https://latex.codecogs.com/svg.latex?f%28y%29%20%3D%20%5Cint%20f%28y%7C%5Ctheta_i%29f%28%5Ctheta_i%29d%20%5Ctheta_i "f(y) = \int f(y|\theta_i)f(\theta_i)d \theta_i")
>   es la densidad marginal *observada* para
>   ![Y](https://latex.codecogs.com/svg.latex?Y "Y").
>
> - ![f(y)](https://latex.codecogs.com/svg.latex?f%28y%29 "f(y)") es
>   fija, conocida, y no depende de
>   ![\theta_i](https://latex.codecogs.com/svg.latex?%5Ctheta_i "\theta_i"),
>   por lo tanto, se modela como una constante
>   ![k](https://latex.codecogs.com/svg.latex?k "k").

</div>

La ecuación anterior es muy complicada de manejar y usualmente se resume
como:

![f(\theta_i\|y) \propto f(y\|\theta_i)f(\theta_i).](https://latex.codecogs.com/svg.latex?f%28%5Ctheta_i%7Cy%29%20%5Cpropto%20f%28y%7C%5Ctheta_i%29f%28%5Ctheta_i%29. "f(\theta_i|y) \propto f(y|\theta_i)f(\theta_i).")

Donde
![\propto](https://latex.codecogs.com/svg.latex?%5Cpropto "\propto")
representa la constante de proporcionalidad.

### Densidad Predictiva

Una cantidad muy importante es la función predictiva a posteriori del
modelo[^3]. Sea
![y^\*](https://latex.codecogs.com/svg.latex?y%5E%2A "y^*") una
observación nueva e independiente de la muestra
![y](https://latex.codecogs.com/svg.latex?y "y"), cuya función de
densidad real es
![f_t(y^\*)](https://latex.codecogs.com/svg.latex?f_t%28y%5E%2A%29 "f_t(y^*)").
Esta *“nueva observación”* es desconocida para los datos y se considera
aleatoria, el cual se puede cuantificar mediante la siguiente ecuación:

![f(Y^\*\|y) = \int f(Y^\*\|\theta_i)f(\theta_i\|y) d\theta_i,](https://latex.codecogs.com/svg.latex?f%28Y%5E%2A%7Cy%29%20%3D%20%5Cint%20f%28Y%5E%2A%7C%5Ctheta_i%29f%28%5Ctheta_i%7Cy%29%20d%5Ctheta_i%2C "f(Y^*|y) = \int f(Y^*|\theta_i)f(\theta_i|y) d\theta_i,")

donde:

- ![f(Y^\*\|y)](https://latex.codecogs.com/svg.latex?f%28Y%5E%2A%7Cy%29 "f(Y^*|y)")
  es la función predictiva a posteriori.

- ![Y^\*](https://latex.codecogs.com/svg.latex?Y%5E%2A "Y^*") es la
  variable aleatoria que cuantifica a
  ![y^\*](https://latex.codecogs.com/svg.latex?y%5E%2A "y^*").

- ![f(\cdot\|\theta_i):\mathbb R \to \mathbb R^+](https://latex.codecogs.com/svg.latex?f%28%5Ccdot%7C%5Ctheta_i%29%3A%5Cmathbb%20R%20%5Cto%20%5Cmathbb%20R%5E%2B "f(\cdot|\theta_i):\mathbb R \to \mathbb R^+")
  para un
  ![\theta_i](https://latex.codecogs.com/svg.latex?%5Ctheta_i "\theta_i")
  fijo, es una función medible de
  ![Y^\*](https://latex.codecogs.com/svg.latex?Y%5E%2A "Y^*").

- ![f(Y^\*\|\theta_i)](https://latex.codecogs.com/svg.latex?f%28Y%5E%2A%7C%5Ctheta_i%29 "f(Y^*|\theta_i)")
  es una transformación de
  ![Y^\*](https://latex.codecogs.com/svg.latex?Y%5E%2A "Y^*"); por lo
  tanto, es una cantidad aleatoria nueva.

Esta densidad se puede interpretar como el valor esperado a posteriori
de la función generadora de datos,

![f(Y^\*\|y) = E\_{\theta\|y}\left\[f(Y^\*\|\theta_i)\right\].](https://latex.codecogs.com/svg.latex?f%28Y%5E%2A%7Cy%29%20%3D%20E_%7B%5Ctheta%7Cy%7D%5Cleft%5Bf%28Y%5E%2A%7C%5Ctheta_i%29%5Cright%5D. "f(Y^*|y) = E_{\theta|y}\left[f(Y^*|\theta_i)\right].")

La función predictiva es de vital importancia para realizar diagnóstico
de las estimaciones obtenidas y para medir el ajuste de un modelo. El
`ajuste` de un modelo se mide al comparar
![f(Y^\*\|\theta_i)](https://latex.codecogs.com/svg.latex?f%28Y%5E%2A%7C%5Ctheta_i%29 "f(Y^*|\theta_i)")
con su valor real
![f_t(y)](https://latex.codecogs.com/svg.latex?f_t%28y%29 "f_t(y)"). En
la práctica esta comparación tiene dos limitantes:

1.  Cómo comparar funciones de densidad?

2.  ![f_t(y)](https://latex.codecogs.com/svg.latex?f_t%28y%29 "f_t(y)")
    siempre es desconocida.

Estas limitantes se pueden sobrellevar, y de esos detalles hablaremos en
las próximas secciones, por ahora, enfocarnos en la función a priori.

## Tipos de Priors

Según sea la función a priori definida, así serán las características de
la función a posteriori. Por ejemplo, en un problema de optimización,
estas densidades regularizan la verosimilitud de la muestra.

Una correcta definición de la prior es importante para un análisis de
datos objetivo e imparcial.

En la actualidad existen diferentes elecciones para la prior, las mas
comunes son:

- Priors dispersas,

- Priors Objetivas,

- Maximum entropy Priors,

- Prios débiles.

#### Prioris dispersas

Este tipo de priors se caracterizan por ser distribuciones uniformes
definidas en un subconjunto del espacio muestral
![\Theta](https://latex.codecogs.com/svg.latex?%5CTheta "\Theta") *“muy
grande”*; A. Gelman et al. (2013).

![f(\theta) \propto U(a-\varepsilon,a+\varepsilon), \\ \varepsilon \to \infty.](https://latex.codecogs.com/svg.latex?f%28%5Ctheta%29%20%5Cpropto%20U%28a-%5Cvarepsilon%2Ca%2B%5Cvarepsilon%29%2C%20%5C%20%5Cvarepsilon%20%5Cto%20%5Cinfty. "f(\theta) \propto U(a-\varepsilon,a+\varepsilon), \ \varepsilon \to \infty.")

**Características**:

- No proveen información externa.

- Son muy subjetivas.

- No exploran objetivamente el espacio muestral
  ![\Theta](https://latex.codecogs.com/svg.latex?%5CTheta "\Theta").

#### Prioris objetivas

Este tipo de priors se conocen como *“no informativas”*, y se
caracterizan por tratar de penalizar la verosimilitud mediante el
criterio de información de Fisher; Migon, Gamerman, and Louzada (2014).

![f(\theta) \propto \|I(\theta)\|^{1/2}.](https://latex.codecogs.com/svg.latex?f%28%5Ctheta%29%20%5Cpropto%20%7CI%28%5Ctheta%29%7C%5E%7B1%2F2%7D. "f(\theta) \propto |I(\theta)|^{1/2}.")

**Características**:

- Son funciones de densidad impropias (No integran 1).

- Proveen información pese se llamadas no informativas.

- Son invariantes a transformaciones de
  ![\theta](https://latex.codecogs.com/svg.latex?%5Ctheta "\theta").

#### Maximum entropy Priors

Este tipo de priors se conocen como *“priors de referencia”*, y el
objetivo es elegir la prior que sea lo mas similar posible a un
posterior de referencia elegida; Bernardo and Smith (1994).

![f(\theta) \propto \arg \max\_{f(\theta)} H(\theta \| y),](https://latex.codecogs.com/svg.latex?f%28%5Ctheta%29%20%5Cpropto%20%5Carg%20%5Cmax_%7Bf%28%5Ctheta%29%7D%20H%28%5Ctheta%20%7C%20y%29%2C "f(\theta) \propto \arg \max_{f(\theta)} H(\theta | y),")

donde,

![H(\theta \| y) =-\int f(\theta\|y)\log f(\theta\|y)d\theta.](https://latex.codecogs.com/svg.latex?H%28%5Ctheta%20%7C%20y%29%20%3D-%5Cint%20f%28%5Ctheta%7Cy%29%5Clog%20f%28%5Ctheta%7Cy%29d%5Ctheta. "H(\theta | y) =-\int f(\theta|y)\log f(\theta|y)d\theta.")

**Características**:

- Son muy complicadas de computar.

- Maximizan la selección de la posterior.

- Son muy informativas, pese a ser de la misma clase que las prioris
  objetivas.

#### Prioris conjugadas

Estas priors generan posteriors con forma analítica y que pertenecen a
la familia exponencial, las primeras aplicaciones surgieron a partir de
este tipo de distribuciones; DeGroot and Schervish (2012).

![f(\theta), \\ f(y \| \theta) \in \mathcal F\_\varepsilon, \to f(\theta\|y) \in \mathcal F\_\varepsilon.](https://latex.codecogs.com/svg.latex?f%28%5Ctheta%29%2C%20%5C%20f%28y%20%7C%20%5Ctheta%29%20%5Cin%20%5Cmathcal%20F_%5Cvarepsilon%2C%20%5Cto%20f%28%5Ctheta%7Cy%29%20%5Cin%20%5Cmathcal%20F_%5Cvarepsilon. "f(\theta), \ f(y | \theta) \in \mathcal F_\varepsilon, \to f(\theta|y) \in \mathcal F_\varepsilon.")

**Características**:

- La posterior tiene solución analítica.

- Limitan la cantidad de modelos a utilizar.

- Garantizan un análisis objetivo de los datos, pero pueden ser muy
  informativas.

#### Prioris débiles:

No existe una regla, formula o método para seleccionar este tipo de
priors, pero se basan en elegir distribuciones que no brinden mucha
información y tengan propiedades que enriquecen el análisis de modelo o
la estimación del mismo; Martin, Kumar, and Lao (2021).

**Características**:

- proveen poca información sobre
  ![\theta](https://latex.codecogs.com/svg.latex?%5Ctheta "\theta").

- regularizan la posterior.

- No tienen forma especifica, ni método de selección.

Existen múltiples estudios para cada tipo de prior estudiando los
beneficios de las posteriors en un modelo en especifico, por ejemplo ver
Fonseca et al. (2019). En la actualidad, existe una rama de inferencia
denotada prior elicitation (Mikkola et al. (2021)) que definen
algoritmos para seleccionar la mejor prior en una familia de funciones.

## Estimación de la posterior

En la actualidad existen muchos métodos para estimación de la posterior:

- **Monte Carlo Markov Chain (MCMC)**: Gibbs Sampler, Metropolis et
  al. (1953) y Metropolis-Hastings, Hastings (1970).

- **MCMC basados en gradientes**. Hamiltonean Monte Carlo (HMC) y
  Metropolis Adaptative Lavengian algorithm (MALA), ver Duane (1987),
  Hoffman and Gelman (2014), y Betancourt (2017).

- **Penalized Maximum Likelihood (P-MLE)**: encontrar
  ![MAEP(\theta) = \arg \max f(\theta \| y)](https://latex.codecogs.com/svg.latex?MAEP%28%5Ctheta%29%20%3D%20%5Carg%20%5Cmax%20f%28%5Ctheta%20%7C%20y%29 "MAEP(\theta) = \arg \max f(\theta | y)"),
  el MAPE se aproxima con métodos de Quasi-Newton, en particular L-BFGS.

- **Approximated Bayesian Computation (ABC):** Rejection Sampler.

- **Variational Inference (VI)**: Stochastic gradient Descent.

En la mayoría de los métodos de aproximación se obtiene una muestra
![\Theta_P = \\{\theta_1,\theta_2,\ldots,\theta_S\\}](https://latex.codecogs.com/svg.latex?%5CTheta_P%20%3D%20%5C%7B%5Ctheta_1%2C%5Ctheta_2%2C%5Cldots%2C%5Ctheta_S%5C%7D "\Theta_P = \{\theta_1,\theta_2,\ldots,\theta_S\}")
de la posterior, que se puede utilizar para aproximar los *estimadores
puntuales* e *intervalos de credibilidad*.

### Estimadores puntuales

Un estimador puntual es el valor que minimiza una función de perdida de
la posteriori, el estimador mas común es la Media a posteriori:

![\hat \theta_1 = E\[\theta \| y\] \approx \frac{1}{m}\sum_i^m \theta_k.](https://latex.codecogs.com/svg.latex?%5Chat%20%5Ctheta_1%20%3D%20E%5B%5Ctheta%20%7C%20y%5D%20%5Capprox%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_i%5Em%20%5Ctheta_k. "\hat \theta_1 = E[\theta | y] \approx \frac{1}{m}\sum_i^m \theta_k.")

Otro estimador muy utilizado es la mediana a posteriori, es bastante
popular en posteriors con colas pesadas.

![\hat \theta_2 = \text{median}(\theta \| y) \approx \hat q(\Theta_P)\_{0.5}.](https://latex.codecogs.com/svg.latex?%5Chat%20%5Ctheta_2%20%3D%20%5Ctext%7Bmedian%7D%28%5Ctheta%20%7C%20y%29%20%5Capprox%20%5Chat%20q%28%5CTheta_P%29_%7B0.5%7D. "\hat \theta_2 = \text{median}(\theta | y) \approx \hat q(\Theta_P)_{0.5}.")

El máximo a posterioir (MAP) es la moda de la posterior y solo se
obtiene con los métodos penalized MLE y VI.

![\hat \theta_3 = \max f(\theta \| y).](https://latex.codecogs.com/svg.latex?%5Chat%20%5Ctheta_3%20%3D%20%5Cmax%20f%28%5Ctheta%20%7C%20y%29. "\hat \theta_3 = \max f(\theta | y).")

### Incertidumbre de los estimadores

La posterior del parámetro es una medida de incertidumbre en si misma,
ventaja principal por la cual se prefiere *inferencia Bayesiana* sobre
la *frecuentista*.

La forma estándar de resumir la incertidumbre es mediante los
*intervalos de credibilidad*, estos se pueden aproximar mediante los
quantiles muestrales
![q\_\alpha](https://latex.codecogs.com/svg.latex?q_%5Calpha "q_\alpha")
de
![\Theta_P](https://latex.codecogs.com/svg.latex?%5CTheta_P "\Theta_P")[^4].

![IC\_{(1-\alpha)\*100\\%} = \[\hat q(\Theta_P)\_{\alpha/2}, \hat q(\Theta_P)\_{1-\alpha/2}\]](https://latex.codecogs.com/svg.latex?IC_%7B%281-%5Calpha%29%2A100%5C%25%7D%20%3D%20%5B%5Chat%20q%28%5CTheta_P%29_%7B%5Calpha%2F2%7D%2C%20%5Chat%20q%28%5CTheta_P%29_%7B1-%5Calpha%2F2%7D%5D "IC_{(1-\alpha)*100\%} = [\hat q(\Theta_P)_{\alpha/2}, \hat q(\Theta_P)_{1-\alpha/2}]")

## Posterior Predictive checks

Estos métodos son análogos al análisis de los residuos en inferencia
clásica. La idea es comparar la función predictiva
![f(y^\*\|y)](https://latex.codecogs.com/svg.latex?f%28y%5E%2A%7Cy%29 "f(y^*|y)")
con los datos obtenidos
![y](https://latex.codecogs.com/svg.latex?y "y"). En la mayoría de los
casos,
![f(y^\*\|y))](https://latex.codecogs.com/svg.latex?f%28y%5E%2A%7Cy%29%29 "f(y^*|y))")
se aproxima con `Monte-Carlo`.

![\hat f(y^\*\|y)  = E\_{\theta\|y} \[f(y^\*\|\theta)\] \approx \frac{1}{m}\sum\_{k=1}^m f(y^\*\|\theta_k)](https://latex.codecogs.com/svg.latex?%5Chat%20f%28y%5E%2A%7Cy%29%20%20%3D%20E_%7B%5Ctheta%7Cy%7D%20%5Bf%28y%5E%2A%7C%5Ctheta%29%5D%20%5Capprox%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bk%3D1%7D%5Em%20f%28y%5E%2A%7C%5Ctheta_k%29 "\hat f(y^*|y)  = E_{\theta|y} [f(y^*|\theta)] \approx \frac{1}{m}\sum_{k=1}^m f(y^*|\theta_k)")

Por lo tanto, se puede obtener una muestra de la predictiva para cada
uno de los ![y_i](https://latex.codecogs.com/svg.latex?y_i "y_i")
observado de la forma:

![\hat y^{(j)}\_i \sim \frac{1}{m}\sum\_{k=1}^m f(y^\*\|\theta_k), \quad f(\cdot\|\theta) \text{ conocida}.](https://latex.codecogs.com/svg.latex?%5Chat%20y%5E%7B%28j%29%7D_i%20%5Csim%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bk%3D1%7D%5Em%20f%28y%5E%2A%7C%5Ctheta_k%29%2C%20%5Cquad%20f%28%5Ccdot%7C%5Ctheta%29%20%5Ctext%7B%20conocida%7D. "\hat y^{(j)}_i \sim \frac{1}{m}\sum_{k=1}^m f(y^*|\theta_k), \quad f(\cdot|\theta) \text{ conocida}.")

Donde
![y^{(1)}\_i, y^{(2)}\_i, \ldots, y^{(m)}\_i](https://latex.codecogs.com/svg.latex?y%5E%7B%281%29%7D_i%2C%20y%5E%7B%282%29%7D_i%2C%20%5Cldots%2C%20y%5E%7B%28m%29%7D_i "y^{(1)}_i, y^{(2)}_i, \ldots, y^{(m)}_i")
es una muestra para
![f(y_i^\*\|y)](https://latex.codecogs.com/svg.latex?f%28y_i%5E%2A%7Cy%29 "f(y_i^*|y)").

Finalmente, los errores del modelo se estiman:

![\hat \varepsilon^{(j)}\_i \approx y_i - y^{(k)}\_i.](https://latex.codecogs.com/svg.latex?%5Chat%20%5Cvarepsilon%5E%7B%28j%29%7D_i%20%5Capprox%20y_i%20-%20y%5E%7B%28k%29%7D_i. "\hat \varepsilon^{(j)}_i \approx y_i - y^{(k)}_i.")

### log-Verosimilitud

Un estimador muy importante para la selección de modelos es la matriz de
log-verosimilitudes, esta se estima por métodos de `Monte-Carlo` usando
una muestra de la posterior
![\Theta_P](https://latex.codecogs.com/svg.latex?%5CTheta_P "\Theta_P"),
de la siguiente forma:

![\log f(y\|\theta) = \[\log f(y_i\|\theta_j)\] \in \mathbb R^{S \times n},](https://latex.codecogs.com/svg.latex?%5Clog%20f%28y%7C%5Ctheta%29%20%3D%20%5B%5Clog%20f%28y_i%7C%5Ctheta_j%29%5D%20%5Cin%20%5Cmathbb%20R%5E%7BS%20%5Ctimes%20n%7D%2C "\log f(y|\theta) = [\log f(y_i|\theta_j)] \in \mathbb R^{S \times n},")

Donde
![i = 1,2,\ldots,n](https://latex.codecogs.com/svg.latex?i%20%3D%201%2C2%2C%5Cldots%2Cn "i = 1,2,\ldots,n")
y
![j = 1,2,\ldots,S](https://latex.codecogs.com/svg.latex?j%20%3D%201%2C2%2C%5Cldots%2CS "j = 1,2,\ldots,S"),
para el tamaño de muestra
![n](https://latex.codecogs.com/svg.latex?n "n") y número de
simulaciones de la posterior
![S](https://latex.codecogs.com/svg.latex?S "S"). A partir de las
matrices de log-verosimilitudes se puede estimar una muestra a posterior
de la log-likelihood del modelo a partir de la siguiente ecuación:

![\text{log-lik}(M)\_j = -\sum\_{i=1}^n \log f(y_i \| \theta_j).](https://latex.codecogs.com/svg.latex?%5Ctext%7Blog-lik%7D%28M%29_j%20%3D%20-%5Csum_%7Bi%3D1%7D%5En%20%5Clog%20f%28y_i%20%7C%20%5Ctheta_j%29. "\text{log-lik}(M)_j = -\sum_{i=1}^n \log f(y_i | \theta_j).")

La muestra obtenida, estima la distribución a posteriori del modelo
![\text{log-lik}(M)](https://latex.codecogs.com/svg.latex?%5Ctext%7Blog-lik%7D%28M%29 "\text{log-lik}(M)").
Estos valores pueden utilizarse para comparación preliminar de modelos,
y se elige el modelo con criterio menor.

## Selección de modelos

Seleccionar el modelo adecuado de los datos de un conjunto de modelos
![M_1,M_2, \ldots, M_k](https://latex.codecogs.com/svg.latex?M_1%2CM_2%2C%20%5Cldots%2C%20M_k "M_1,M_2, \ldots, M_k")
es un problema muy complicado, debido a los altos costos computacionales
y complejidad de los algoritmos. En la actualidad los métodos más
utilizados son:

- Factor de Bayes

- Watanabe-Akaike Information Criteria (WAIC).

- Expected log predictive density (elpd).

#### Factores de Bayes

Los factores de Bayes, fueron propuestos por Jeffrey ([1960]()) y
re-interpretados por Kass and Raftery (1995) para selección de modelos.
Los factores de Bayes se basan en comparar las posteriors de los modelos
definidos sobre los datos:

![f(M_i \|y) \propto f(y \| M_i)f(M_i),](https://latex.codecogs.com/svg.latex?f%28M_i%20%7Cy%29%20%5Cpropto%20f%28y%20%7C%20M_i%29f%28M_i%29%2C "f(M_i |y) \propto f(y | M_i)f(M_i),")

Donde
![f(y \| M_i) = f\_{M_i}(y)](https://latex.codecogs.com/svg.latex?f%28y%20%7C%20M_i%29%20%3D%20f_%7BM_i%7D%28y%29 "f(y | M_i) = f_{M_i}(y)")
es la verosimilitud marginal de los datos, y
![f(M_i)](https://latex.codecogs.com/svg.latex?f%28M_i%29 "f(M_i)") es
la distribución a priori del modelo, o su importancia. Por lo tanto, el
factor de Bayes es:

![FB = \frac{f(M_i\|y)}{f(M_j \|y)} \propto \frac{f\_{M_i}(y)f(M_i)}{f\_{M_j}(y)f(M_j)},](https://latex.codecogs.com/svg.latex?FB%20%3D%20%5Cfrac%7Bf%28M_i%7Cy%29%7D%7Bf%28M_j%20%7Cy%29%7D%20%5Cpropto%20%5Cfrac%7Bf_%7BM_i%7D%28y%29f%28M_i%29%7D%7Bf_%7BM_j%7D%28y%29f%28M_j%29%7D%2C "FB = \frac{f(M_i|y)}{f(M_j |y)} \propto \frac{f_{M_i}(y)f(M_i)}{f_{M_j}(y)f(M_j)},")

En la práctica no tenemos importancia o favoritismo hacia un modelo
entonces elegimos las priors iguales
![f(M_i) = f(M_j)](https://latex.codecogs.com/svg.latex?f%28M_i%29%20%3D%20f%28M_j%29 "f(M_i) = f(M_j)").
Por lo tanto, el factor de Bayes es equivalente a la razón de
verosimilitudes marginales.

![FB = \frac{f\_{M_i}(y)}{f\_{M_j}(y)}.](https://latex.codecogs.com/svg.latex?FB%20%3D%20%5Cfrac%7Bf_%7BM_i%7D%28y%29%7D%7Bf_%7BM_j%7D%28y%29%7D. "FB = \frac{f_{M_i}(y)}{f_{M_j}(y)}.")

La verosimilitud marginal de los datos se puede aproximar con un método
de `Monte-Carlo` de la siguiente forma:

![f(y) = \int f(y\|\theta)f(\theta)d \theta \approx \sum\_{k=1}^mf(y\|\theta_k), \quad \theta_k \sim f(\theta).](https://latex.codecogs.com/svg.latex?f%28y%29%20%3D%20%5Cint%20f%28y%7C%5Ctheta%29f%28%5Ctheta%29d%20%5Ctheta%20%5Capprox%20%5Csum_%7Bk%3D1%7D%5Emf%28y%7C%5Ctheta_k%29%2C%20%5Cquad%20%5Ctheta_k%20%5Csim%20f%28%5Ctheta%29. "f(y) = \int f(y|\theta)f(\theta)d \theta \approx \sum_{k=1}^mf(y|\theta_k), \quad \theta_k \sim f(\theta).")

Observaciones:

- El Factor de Bayes es sensible a modelos con priors no informativas.

- Muy complicado de estimar, y los algoritmos son inestables.

- Es perfecto para encontrar el modelo real.

Aproximar las verosimilitudes marginales con `Monte Carlo` es muy
ineficiente e inestable numéricamente, otros algoritmos utilizados es
`muestreo por importancia` y el algoritmo de `Bridge-Sampling`; Gronau
et al. (2017).

#### Expected log predictive density

La elpd es una divergencia entre el modelo ajustado y la densidad real
de los datos que se calcula mediante la siguiente ecuación:

![\text{elpd}(M_k\|y) = - \int\log f(y^\*\|y) f_t(y)dy,](https://latex.codecogs.com/svg.latex?%5Ctext%7Belpd%7D%28M_k%7Cy%29%20%3D%20-%20%5Cint%5Clog%20f%28y%5E%2A%7Cy%29%20f_t%28y%29dy%2C "\text{elpd}(M_k|y) = - \int\log f(y^*|y) f_t(y)dy,")

Esta medida se puede aproximar usando un método de `Monte-Carlo`
mediante la siguiente ecuación:

![elpd(M_k\|y) \approx - \sum\_{i=1}^m\log f(y_i^\*\|y_i).](https://latex.codecogs.com/svg.latex?elpd%28M_k%7Cy%29%20%5Capprox%20-%20%5Csum_%7Bi%3D1%7D%5Em%5Clog%20f%28y_i%5E%2A%7Cy_i%29. "elpd(M_k|y) \approx - \sum_{i=1}^m\log f(y_i^*|y_i).")

El mayor problema problema es que
![\log f(y_i^\*\|y_i)](https://latex.codecogs.com/svg.latex?%5Clog%20f%28y_i%5E%2A%7Cy_i%29 "\log f(y_i^*|y_i)")
es desconocida y se calcula a partir de la predictiva:

![f(y_i^\*\|y_i) = \int f(y_i^\*\|\theta)f(\theta\|y) d\theta.](https://latex.codecogs.com/svg.latex?f%28y_i%5E%2A%7Cy_i%29%20%3D%20%5Cint%20f%28y_i%5E%2A%7C%5Ctheta%29f%28%5Ctheta%7Cy%29%20d%5Ctheta. "f(y_i^*|y_i) = \int f(y_i^*|\theta)f(\theta|y) d\theta.")

Vehtari et al. (2015) proponen hacer la estimación de la predictiva
utilizando validación cruzada, cuando la forma de la predictiva es
proporcional a la verosimilitud:

![f(y_i\|y\_{-i}) \approx \frac{1}{\frac{1}{S}\sum\_{s=1}^S\[f(y_i\|\theta_s)\]^{-1}},](https://latex.codecogs.com/svg.latex?f%28y_i%7Cy_%7B-i%7D%29%20%5Capprox%20%5Cfrac%7B1%7D%7B%5Cfrac%7B1%7D%7BS%7D%5Csum_%7Bs%3D1%7D%5ES%5Bf%28y_i%7C%5Ctheta_s%29%5D%5E%7B-1%7D%7D%2C "f(y_i|y_{-i}) \approx \frac{1}{\frac{1}{S}\sum_{s=1}^S[f(y_i|\theta_s)]^{-1}},")

donde
![\theta_1,\theta_2,\ldots,\theta_S](https://latex.codecogs.com/svg.latex?%5Ctheta_1%2C%5Ctheta_2%2C%5Cldots%2C%5Ctheta_S "\theta_1,\theta_2,\ldots,\theta_S")
es una muestra de la posterior
![\Theta_P](https://latex.codecogs.com/svg.latex?%5CTheta_P "\Theta_P");
![y\_{-i}](https://latex.codecogs.com/svg.latex?y_%7B-i%7D "y_{-i}")
representa el vector original de los datos quitando la observación
![y_i](https://latex.codecogs.com/svg.latex?y_i "y_i"), y
![f(y_i\|y\_{-i})](https://latex.codecogs.com/svg.latex?f%28y_i%7Cy_%7B-i%7D%29 "f(y_i|y_{-i})")
es la predictiva para
![y_i](https://latex.codecogs.com/svg.latex?y_i "y_i") cuando asumimos
que esta es desconocida. Por lo tanto, la elpd se aproxima con

![elpd(M_k\|y) \approx - \sum\_{i=1}^n\log \left\[ \frac{1}{\frac{1}{S}\sum\_{s=1}^S\[f(y_i\|\theta_s)\]^{-1}} \right\]](https://latex.codecogs.com/svg.latex?elpd%28M_k%7Cy%29%20%5Capprox%20-%20%5Csum_%7Bi%3D1%7D%5En%5Clog%20%5Cleft%5B%20%5Cfrac%7B1%7D%7B%5Cfrac%7B1%7D%7BS%7D%5Csum_%7Bs%3D1%7D%5ES%5Bf%28y_i%7C%5Ctheta_s%29%5D%5E%7B-1%7D%7D%20%5Cright%5D "elpd(M_k|y) \approx - \sum_{i=1}^n\log \left[ \frac{1}{\frac{1}{S}\sum_{s=1}^S[f(y_i|\theta_s)]^{-1}} \right]")

El mayor problema de esta aproximación es que es muy inestable
numéricamente, mucho mayor que las obtenidas en el factor de Bayes, y
Vehtari et al. (2015) propone resolver esta aproximación con muestreo
por importancia usando una distribución generalizada de Pareto.

Observaciones:

- Elige al modelo que más se acerque a la función real de los datos
  (![f_t](https://latex.codecogs.com/svg.latex?f_t "f_t")
  *desconocida*).

- Su estimación es con remuestreo LOO-CV y es sensible a perturbaciones
  o malos ajustes.

- Elige al modelo que predice mejor.

#### Watanabe-Akaike Information Criteria

El WAIC es un criterio de información mayormente conocido como Widely
Applicable information criteria, que penalización dela log-predictiva
del modelo es mediante su segundo momento.

![WAIC = -E\[\log f(y^\*\|y)\] -n V\[\log f(y^\*\|y)\]](https://latex.codecogs.com/svg.latex?WAIC%20%3D%20-E%5B%5Clog%20f%28y%5E%2A%7Cy%29%5D%20-n%20V%5B%5Clog%20f%28y%5E%2A%7Cy%29%5D "WAIC = -E[\log f(y^*|y)] -n V[\log f(y^*|y)]")

El criterio de información de Watanabe es asintótico al valor obtenido
por la ![elpd](https://latex.codecogs.com/svg.latex?elpd "elpd"), por lo
tanto, puede ser aproximado con validación cruzada.

Observaciones:

- se elige el modelo con menor criterio de información.

- Se puede estimar con métodos de `Monte-Carlo`.

- Problemático para modelos muy similares

## Referencias

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-Bernardo+Smith:1994" class="csl-entry">

Bernardo, José M., and Adrian F. M. Smith. 1994. *Bayesian Theory*. John
Wiley & Sons.

</div>

<div id="ref-betancourt2017" class="csl-entry">

Betancourt, Michael. 2017. “A Conceptual Introduction to Hamiltonian
Monte Carlo.” <https://arxiv.org/abs/1701.02434>.

</div>

<div id="ref-degroot2012" class="csl-entry">

DeGroot, M. H., and M. J. Schervish. 2012. *Probability and Statistics*.
Addison-Wesley. <https://books.google.es/books?id=4TlEPgAACAAJ>.

</div>

<div id="ref-Duane1987" class="csl-entry">

Duane, et al., S. 1987. “Hybrid Monte Carlo.” *Physics Letters B* 95
(2): 216–22.
https://doi.org/[https://doi.org/10.1016/0370-2693(87)91197-X"](https://doi.org/10.1016/0370-2693(87)91197-X").

</div>

<div id="ref-fonseca2019" class="csl-entry">

Fonseca, T. C. O., V. S. Cerqueira, H. S. Migon, and C. A. C. Torres.
2019. “The Effects of Degrees of Freedom Estimation in the Asymmetric
GARCH Model with Student-t Innovations.”
<https://arxiv.org/abs/1910.01398>.

</div>

<div id="ref-gelman2013" class="csl-entry">

Gelman, A., J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D.
B. Rubin. 2013. *Bayesian Data Analysis, Third Edition*. Chapman &
Hall/CRC Texts in Statistical Science. Taylor & Francis.
<https://books.google.nl/books?id=ZXL6AQAAQBAJ>.

</div>

<div id="ref-gelman2020bayesian" class="csl-entry">

Gelman, Andrew, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob
Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian
Bürkner, and Martin Modrák. 2020. “Bayesian Workflow.”
<https://arxiv.org/abs/2011.01808>.

</div>

<div id="ref-gronau2017" class="csl-entry">

Gronau, Quentin F., Alexandra Sarafoglou, Dora Matzke, Alexander Ly, Udo
Boehm, Maarten Marsman, David S. Leslie, Jonathan J. Forster, Eric-Jan
Wagenmakers, and Helen Steingroever. 2017. “A Tutorial on Bridge
Sampling.” <https://arxiv.org/abs/1703.05984>.

</div>

<div id="ref-Hasting1970" class="csl-entry">

Hastings, W. K. 1970. “Monte Carlo Sampling Methods Using Markov Chains
and Their Applications.” *Biometrika* 57 (1): 97–109.
<http://www.jstor.org/stable/2334940>.

</div>

<div id="ref-hoffman14" class="csl-entry">

Hoffman, Matthew D., and Andrew Gelman. 2014. “The No-u-Turn Sampler:
Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.” *Journal of
Machine Learning Research* 15: 1593–623.
<http://jmlr.org/papers/v15/hoffman14a.html>.

</div>

<div id="ref-Kass1995" class="csl-entry">

Kass, Robert E., and Adrian E. Raftery. 1995. “Bayes Factors.” *Journal
of the American Statistical Association* 90 (430): 773–95.
<https://doi.org/10.1080/01621459.1995.10476572>.

</div>

<div id="ref-BMCP2021" class="csl-entry">

Martin, Osvaldo A., Ravin Kumar, and Junpeng Lao. 2021. *<span
class="nocase">Bayesian Modeling and Computation in Python</span>*. Boca
Raton.

</div>

<div id="ref-metropolis1953" class="csl-entry">

Metropolis, Nicholas, Arianna W. Rosenbluth, Marshall N. Rosenbluth,
Augusta H. Teller, and Edward Teller. 1953. “Equation of State
Calculations by Fast Computing Machines.” *The Journal of Chemical
Physics* 21 (6): 1087–92. <https://doi.org/10.1063/1.1699114>.

</div>

<div id="ref-Miggon2014" class="csl-entry">

Migon, Helio, Dani Gamerman, and Francisco Louzada. 2014. *Statistical
Inference. An Integrated Approach*. Chapman and Hall CRC Texts in
Statistical Science. Chapman; Hall.

</div>

<div id="ref-Mikola-et-al:2021" class="csl-entry">

Mikkola, Petrus, Osvaldo A. Martin, Suyog Chandramouli, Marcelo
Hartmann, Oriol Abril Pla, Owen Thomas, Henri Pesonen, et al. 2021.
“Prior Knowledge Elicitation: The Past, Present, and Future.” arXiv.
<https://doi.org/10.48550/ARXIV.2112.01380>.

</div>

<div id="ref-vehtari2016" class="csl-entry">

Vehtari, Aki, Daniel Simpson, Andrew Gelman, Yuling Yao, and Jonah
Gabry. 2015. “Pareto Smoothed Importance Sampling.”
<https://arxiv.org/abs/1507.02646>.

</div>

</div>

[^1]: Estamos abusando de la notación, en este caso el objeto Y es una
    función aleatoria de dimensión d arbitraria; si d \> 1 entonces Y es
    un vector aleatorio.

[^2]: Decimos que Y1 y Y2 son intercambiables si f(Y1,Y2) = f(Y2,Y1).

[^3]: Posterior predictive density

[^4]: Si las posteriors son *uni-modales*, entonces los Intervalos de
    credibilidad son `High posterior density`.
