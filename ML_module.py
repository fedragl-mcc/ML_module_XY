# modulo_ml.py
from ucimlrepo import fetch_ucirepo 
import time
import csv
import pandas as pd
import numpy as np

#machine learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

#   Open the CSV file in append mode____________________________________________________________________
config_file = open('GAforFS_ML\ML_module_SVM_KNN.csv', 'a', newline='')
#   Create a CSV writer
writer = csv.writer(config_file)
writer.writerow(["fitness"])

class ModeloML:
    def __init__(self, repository,repository_path,chromosome):
        """Inicializa la clase con el archivo de datos."""
        self.repo_ID = repository
        self.repo_Path = repository_path
        self.chromosome = chromosome

        self.df = None  # dataset DataFrame 
        self.X = None  # Features
        self.y = None  # Target
        #   splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.modelo = None  # Chosen ML model
        self.fitness = None

    #   Get dataset from computer PATH
    def dataset_PATH(self):
        self.df = pd.read_csv(self.repo_Path, header=0)

        #   agregar la linea de código que aplique al dataset dependiendo su target + seccion X/y_____
        #   WBC
        self.df.iloc[:,1] = self.df.iloc[:,1].map({'M': 1, 'B': 0}) #   Convertir la columna 'Diagnóstico' a valores numéricos
        self.y=self.df.iloc[:,1].copy() # set target column into its own df
        
        self.X=self.df.drop(self.df.columns[[0,1]], axis=1).copy() #dropping id[0] and target[1]
        #______________________________________________________________________________

        self.data_preprocessing()

    #   Get dataset from UCI Repository
    def dataset_REPO(self):
        # fetch dataset
        self.df = fetch_ucirepo(id=self.repo_ID)
        self.X = self.df.data.features.copy() 
        self.y = self.df.data.targets.copy()

        #   agregar la linea de código que aplique al dataset dependiendo su target_____
        #   WBC
        self.y['Diagnosis'] = self.y['Diagnosis'].map({'M': 1, 'B': 0}) #   Convertir la columna 'Diagnóstico' a valores numéricos
        #______________________________________________________________________________
        self.data_preprocessing()

    #Pre proccesses data and splits test/train samples
    def data_preprocessing(self):
        self.tt_split_chromosome()
        #   Treat missing values
        for col in self.X.columns:
            if self.X[col].isnull().sum() > 0:  # Verifica si hay NaN en la columna
                if self.X[col].dtype == 'object':  # Para columnas categóricas
                    self.X[col].fillna(self.X[col].mode()[0], inplace=True)  # Rellenar con la moda
            else:
                self.X[col].fillna(self.X[col].mean(), inplace=True)  # Rellenar con la media

        # Outlier detection with IQR
        Q1 = self.X.quantile(0.25)
        Q3 = self.X.quantile(0.75)
        IQR = Q3 - Q1

        # Filtrar los datos para eliminar los outliers
        self.X = self.X[~((self.X < (Q1 - 1.5 * IQR)) | (self.X > (Q3 + 1.5 * IQR))).any(axis=1)]
        self.y = self.y.loc[self.X.index] #Leave indexes needed and drop the rest

        # standard scalation
        escalador = StandardScaler()
        self.X = escalador.fit_transform(self.X)

        
    def tt_split_chromosome(self):
        #   choose the features to be used
        X_chromosome = np.array(self.chromosome)
        features=self.X.columns[X_chromosome.astype(bool)]
        self.X=self.X[features]


    def ML_model (self, model): #    chromosome stands for the number of features to be used
        # train and test samples
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # # in case values in y_train are "object" modify, agregar try_catch en "fit"
        self.y_train = self.y_train.astype(int)

        if model=='SVM':
            try:
                modelo_svm = SVC(kernel='sigmoid', C=1000, random_state=42) #   revisar aqui la hiperparametrizacion
                modelo_svm.fit(self.X_train, self.y_train)
            except ValueError:
                return 'null'
            y_pred = modelo_svm.predict(self.X_test)

        elif model=="RF":
            try:
                modelo_RF = RandomForestClassifier() #   revisar aqui la hiperparametrizacion
                modelo_RF.fit(self.X_train, self.y_train)
            except ValueError:
                return 'null'
            y_pred = modelo_RF.predict(self.X_test)
        elif model=="KNN":
            try:
                modelo_KNN = KNeighborsClassifier() #   revisar aqui la hiperparametrizacion
                modelo_KNN.fit(self.X_train, self.y_train)
            except ValueError:
                return 'null'
            y_pred = modelo_KNN.predict(self.X_test)
        
        #calculate Fitness
        self.y_test=self.y_test.astype(int)  #.to int in the case of the WBC with path, agregar un try catch mas adelante en el accuracy

        self.accuracy = accuracy_score(self.y_test, y_pred)

# ==============================
#   EJECUCIÓN DEL MÓDULO
# ==============================

if __name__ == "__main__":
    print("Iniciando módulo de Machine Learning...")
    
    # Crear una instancia de la clase con el dataset
    ml = ModeloML(None,"D:\Fedra\iCloudDrive\Mcc\Tesis\Resources\DS_breast+cancer+wisconsin+diagnostic\wdbc.csv",chromosome=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    
    # Ejecutar etapas del proceso
    ml.dataset_PATH()
    # Seleccionar modelo ('SVM','RF','KNN')
    modelo_seleccionado = "KNN"
    ml.ML_model(modelo_seleccionado)
    
    metricas = ml.accuracy
    writer.writerow([metricas])
    
    # Mostrar métricas finales
    print("Evaluación del modelo:", metricas)
