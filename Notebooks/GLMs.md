Modelos Lineales Generalizados Aplicados
================
Asael Alonzo Matamoros.
2022-11-16

El modelo lineal generalizado (GLM) relaciona de forma funcional un
conjunto de variables aleatorias $Z = (Y,X)$, donde la v.a.
$Y \in \mathbb{R}$ se le conoce como **variable dependiente**, y a la
v.a. $X \in \mathbb{R}^d$ son las **covariables**.

Sea $Z = \{Z_1,Z_2,\ldots,Z_n\}$ un conjunto de variables aleatorias
independientes, tal que $Z_i = (Y_i,X_i) \in \mathbb R^{d+1}$, decimos
que Z sigue un GLM si:

$$Y_i \sim \mathscr{F}_\varepsilon (\theta_i), \quad \text{y } g(\theta_i) = \beta X_i.$$
Donde:

- $\mathscr{F}_\varepsilon$ representa la familia exponencial.

- El conjunto $\theta = \{\theta_i\}$ representa una colección de
  parámetros de locación.

- $g: \mathbb R \to \mathbb R$ es una función diferenciable e
  invertible, conocida como la **función de enlace**.

- $\beta \in \mathbb R^{d+1}$ es el vector de coeficientes de regresión
  o importancias.

## Log-Verosimilitud

Los GLMs son modelos probabilistas, cuya función de probabilidad se
establece mediante la verosimilitud en escala logarítmica. Dado el
supuesto de independencia de los datos, la verosimilitud es simplemente
el producto de las densidades marginales de los datos.

$$L(y;\theta) = f(y_1,y_2,\ldots,y_n | \theta) = \prod_{i=1}^nf(y_i\theta).$$

Dado que los GLM pertenecen a la familia exponencial, la verosimilitud
se puede expresar de forma analítica como:

$$L(y;\theta) = \prod_{i=1}^n \exp \left[y_i b(\theta_i)+c(\theta_i)+d(y_i) \right].$$

Finalmente, la log-verosimilitud que es el logaritmo de $L$, también
posee forma analítica

$$l(y;\theta) = \sum_{i=1}^n y_i b(\theta_i) +\sum_{i=1}^nc(\theta_i) + \sum_{i=1}^n d(y_i).$$

## Estimación de los parámetros

El método de optimización más popular en GLMs es [Máxima
Verosimilitud]() que consiste en optimizar la log-verosimilitud, o
simplemente resolver la función de score.

$$U(y;\theta) = \frac{\partial}{\partial \theta} l(y;\theta) = 0.$$
Generalmente resolver U implica resolver un sistema de ecuaciones no
lineales, y al inicio el método mas utilizado es el algoritmo de Newton.
En la actualidad dicho algoritmo a evolucionado a un [Limited Memory
Broyden–Fletcher–Goldfarb–Shanno
algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS). El
algoritmo L-BFGS es un método de Newton que

1.  Aproxima la matriz Jacobiana usando el método de Broyden, y

2.  Aplica el algoritmo de Fletcher para corregir por estabilidad la
    aproximación de Broyden.

## Incertidumbre de los estimadores

Un estimador es cualquier estadístico $w(y)$ que se utiliza para inferir
información de $\beta$. Dado que los estimadores $\hat \beta = w(y)$ son
transformaciones de la muestra. poseen una distribución[^1]. En la
mayoría de los caso la distribución muestral no tiene forma analítica, y
esta se aproxima usando remuestreo.

Los algoritmos de remuestreo más utilizados son

1.  Jackniffe

2.  Bootstrap

De los dos algoritmos el mas popular es el algoritmo de Bootstrap que
consiste en generar una muestra de estimadores que aproxima la
distribución muestral deseada. El algoritmo es:

1.  **Elegir** el número de sub-muestras $B$

2.  **Para** $b =1,2,3,\ldots, B$ **hacer:**

    - **Extraer** una sub-muestra $Y_b$ con reemplazo de $Y$,

    - **Estimar** los parámetros del modelo $\hat \beta_b$ con la
      sub-muestra $Y_b$.

3.  La colección
    $\hat \beta_1,\hat \beta_2, \hat \beta_3 ,\ldots,\hat \beta_B$ es
    una muestra de la distribución muestral de los estimadores
    $\hat \beta$.

4.  Usar $\hat \beta_1,\hat \beta_2, \hat \beta_3 ,\ldots,\hat \beta_B$
    para aproximar los intervalos de confianza de $\hat \beta$.

El muestreo de Jackniffe es un caso particular del Bootstrap que
consiste en extraer la sub-muestra $Y_b$ eliminando la b-ésima
observación $y_b$ de la muestra original.

Para análisis de incertidumbre de los estimadores, el Bootstrap provee
mejores resultados que el método de Jackniffe, pero el segundo método
tiene su nicho al comparar diferentes modelos, [validación
cruzada](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).

## Análisis de los residuos.

Los residuos de un modelo se definen como la diferencia entre el valor
ajustado por el modelo $\hat Y$ y su valor real $Y$.

$$R_i = \hat Y_i - Y_i.$$

En GLMs solo se revisan generalidades simples de los supuestos como:

- Que estén centrados en cero $R_i \approx 0$.

- Sean de varianza homogénea $\sigma_R$

En modelos lineales generales, se pueden revisar más supuestos como
normalidad, homogeneidad, y estacionaridad. En ML los residuos estiman
los errores del modelo $R_i = \hat \varepsilon_i$.

Un estadístico importante obtenido de los residuos, es el coeficiente de
determinación $R^2$ que establece el porcentaje de varianza explicada
por el modelo.

$$R^2 = 1 - \frac{\sigma_R/(n-d-1)}{V[y]/(n-1)}$$

## Selección de modelos

En la práctica, es posible que se desarrollen múltiples modelos que
expliquen el mismo conjunto de datos $Z$, existen indicadores que
permiten describir las cualidades del modelo, los más utilizados son:

- log-verosimilitud: $\log L= -l(y;\theta)$.

- Criterio de Información de Bayes: $BIC = 2 \log L - 2\log(n^d)$

- Criterio de Información de Akaike $AIC - 2 \log L -2 d$

- Root Mean square error $RMSE = \frac{1}{\sqrt n}||R||_2$

- Mean Absolute Percentaje Error
  $MAPE =\frac{1}{n}\sum_{i=1}^n \left| \frac{R_i}{Y_i} \right|$

### Selección de variables

Una aplicación de selección de modelos es encontrar modelos reducidos,
Esto es encontrar un subconjunto de variables $X_1 \subset X$ tal que el
modelo reducido $M_r$ sea lo mas parsimonioso posible. Un modelo
parsimonioso brinda mayor explicabilidad, mayor capacidad de aprendizaje
y mayor capacidad de generalización.

Los métodos para reducción de variables se les conoce como búsquedas, y
los tipos son:

- Búsquedas hacia adelante

- Búsquedas hacia atrás.

En búsqueda hacia adelante se inicia con el modelo más pequeño posible,
se mide su criterio de información, y luego se agregan variables de
forma secuencial de tal forma que el criterio mejore.

## Aprendizaje

Los indicadores presentados miden el **ajuste del modelo**, esto es la
capacidad del modelo de explicar el conjunto de datos. En aplicaciones
mas recientes, la explicabilidad es una propiedad poco deseable, se
prefiere medir la **capacidad de aprendizaje**.

<div>

> **Aprendizaje**
>
> La **capacidad de aprendizaje** es la habilidad del modelo de explicar
> propiedades externas a partir de la información adquirida.

</div>

En términos probabilistas, es la habilidad del modelo de predecir un
nuevo conjunto de datos, a partir de los datos disponibles.

### Particiones

El aprendizaje se puede medir usando una **partición** de los datos. Sea
$Y = \{Y_1,Y_2,\ldots,Y_n\}$ una muestra aleatoria, definimos una
partición de entrenamiento y prueba como

$$Y = Y_E \bigcup Y_P$$ Donde:

- $Y_E \bigcap Y_P = \emptyset$.

- $m+k = n$ y $m >> k$.

- $Y_E = \{Y_1,Y_2,\ldots,Y_m\} \subset Y$ es el conjunto de
  entrenamiento.

- $Y_P = \{Y_1,Y_2,\ldots,Y_k\} \subset Y$ es el conjunto de prueba.

El algoritmo para medir aprendizaje es:

1.  Ajustar los modelos $M_1,M_2,\ldots,M_f$ usando el conjunto de
    entrenamiento $Y_E$.

2.  Para cada modelo $M_j$ hacer:

    - Realizar $k$ predicciones
      $\hat Y_{P,1},\hat Y_{P,2},\ldots,\hat Y_{P,k}$.

    - Comparar $\hat Y_{P}$ con $Y_P$ usando AIC, BIC log-lik, RMSE o
      MAPE.

3.  El modelo con mayor aprendizaje es el modelo con criterio menor.

La mayor limitante de las particiones es que al ser aleatorias los
resultados varian bastante según la selección de las observaciones en el
conjunto de entrenamiento. Una forma de evitar esos problemas es usando
Validación cruzada.

## Validación cruzada

Validación Cruzada consiste en realizar el proceso de partición muchas
veces, el método mas utilizados es **k-fold cross-validation** este
consisten en:

1.  **Definir** $m$ como el número de iteraciones a realizar

2.  **Para** $i = 1,2,3,\ldots, m$ **hacer**:

    - definir el conjunto de prueba $Y_p$ de tamaño $k$.

    - definir el conjunto de entrenamiento $Y_E$ como el complemento
      $Y_E = Y_P^c$.

    - Estimar los modelos con $Y_E$

    - Comparar las predicciones de cada modelo con $Y_P$ mediante algún
      criterio de información.

3.  Promediar los criterios obtenidos de cada iteración.

4.  Elegir el modelo con criterio promedio menor.

Cuando el conjunto e prueba solo posee una observación, al método se le
conoce como LOO (*Leave one out cross validation*). Otro método es
utilizar un *Bootstrap* pero es altamente costoso.

#### Notar que:

- LOO es equivalente a un muestreo de Jackniffe.

- Con validación cruzada re-utilizamos la información

- Usamos toda la muestra para validar los resultados.

- Se minimiza la variación de errores de aprendizaje.

## Referencias

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-Casella" class="csl-entry">

Casella, George, and Roger Berger. 2001. *Statistical Inference*.
Duxbury Resource Center.
<http://www.amazon.fr/exec/obidos/ASIN/0534243126/citeulike04-21>.

</div>

<div id="ref-degroot2012" class="csl-entry">

DeGroot, M. H., and M. J. Schervish. 2012. *Probability and Statistics*.
Addison-Wesley. <https://books.google.es/books?id=4TlEPgAACAAJ>.

</div>

<div id="ref-gelman2013" class="csl-entry">

Gelman, A., J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D.
B. Rubin. 2013. *Bayesian Data Analysis, Third Edition*. Chapman &
Hall/CRC Texts in Statistical Science. Taylor & Francis.
<https://books.google.nl/books?id=ZXL6AQAAQBAJ>.

</div>

<div id="ref-Miggon2014" class="csl-entry">

Migon, Helio, Dani Gamerman, and Francisco Louzada. 2014. *Statistical
Inference. An Integrated Approach*. Chapman and Hall CRC Texts in
Statistical Science. Chapman; Hall.

</div>

<div id="ref-BMLR2021" class="csl-entry">

Roback, paul., and Julie. Legler. 2021. *<span class="nocase">Beyond
Multiple Linear Regression: Applied Generalized Linear Models an
Multilevel Models in R</span>*. Boca Raton.

</div>

</div>

[^1]: La distribución muestral es la función de probabilidad de los
    estimadores.
