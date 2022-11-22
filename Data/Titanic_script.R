
setwd("~/Documents/Applied-Bayesian-Regression/Data")
## Script  para limpiar la base de datos del Titanic
Train = read.csv("Train.csv")
Test = read.csv("Test.csv")
## Data Dictionary
#
# Survival (Dependent)
# pclas: 1 = 1st, 2= 2nd, y 3 = 3rd
# Sex: Sexo
# Age: Edad en anios
# Sibsp # de parientes/esposas abordo
# parch: # de padres/hijos abordo
# ticket: Numero de ticket
# fare: tarifa del pasajero
# cabin: Numero de la cabina
# Embarke: Puerto de Embarcacion
# # C = Cerbourg, Q = Queenstown, S = Southampton

# Remove unnecesary variables"

Train = Train[,-c(4,9,11)]
Test = Test[,-c(3,8,10)]

# change data structure
str(Train)

Train$Sex = factor(Train$Sex)
Test$Sex = factor(Test$Sex)

Train$Embarked  = factor(Train$Embarked)
Test$Embarked = factor(Test$Embarked)

save.image("Titanic.RData")
 