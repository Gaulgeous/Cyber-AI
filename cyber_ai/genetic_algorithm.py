import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from random import choices, randint
from statistics import mean
from copy import copy
from math import exp
from joblib import parallel, delayed

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, ConfusionMatrixDisplay, mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score as r2



classification_models = {"knn": {'n_neighbors': range(1, 101), 'weights': ["uniform", "distance"], 'p': [1, 2]},
          'svc': {'penalty': ["l2"], 'loss': ["hinge", "squared_hinge"], 'dual': [True], 'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]},
          'logistic': {'penalty': ["l2"], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 'dual': [False]},
          'rf': {'n_estimators': [100], 'criterion': ["gini", "entropy"], 'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21), 'min_samples_leaf':  range(1, 21), 'bootstrap': [True, False]},
          'decision_tree': {'criterion': ["gini", "entropy"], 'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21)}
        }

regression_models = {"rf": {'n_estimators': [100], 'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21), 'bootstrap': [True, False]},
                     "knn": {'n_neighbors': range(1, 101), 'weights': ["uniform", "distance"], 'p': [1, 2]},
                     "decision_tree": {'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21)},
                     "elastic": {'l1_ratio': np.arange(0.0, 1.01, 0.05), 'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
                     }

cleaners = {"scaler": ["minmax", "robust", "standard", "none"], "pca": ["none", "pca"]}


class GeneticAlgorithm:

    def __init__(self, model, cols, mode="classification", population=20, generations=20, cv=5, parents=4):

        self.model = model
        self.cols = cols
        self.mode = mode
        self.population_size = population
        self.generations = generations
        self.cv = cv
        self.parents = parents
        self.population = []
        self.X_train = None
        self.y_train = None
        self.best_genome = None
        self.models = None

        if mode == "classification":
            self.models = classification_models
            self.scoring = "f1"
        else:
            self.models = regression_models
            self.scoring = "r2"

        self.create_population()


    def create_genome(self):

        genome = {}
        clean = {}

        for key in self.models[self.model]:
            value = choices(self.models[self.model][key], k=1)[0]
            genome[key] = value

        for cleaner in cleaners:
            value = choices(cleaners[cleaner], k=1)[0]
            clean[cleaner] = value

        drop_margins = np.array([randint(0, 1) for _ in range(self.cols)])
        if sum(drop_margins) < 2:
            a = randint(0, len(drop_margins) - 1)
            b = randint(0, len(drop_margins) - 1)
            while b == a:
                b = randint(0, len(drop_margins) - 1)
            drop_margins[a] = 1
            drop_margins[b] = 1
            drop_margins = np.asarray(drop_margins)

        return {"drop_margins": drop_margins, "cleaners": clean, "model": genome}


    def create_model(self, genome):

        model = None

        if self.mode == "classification":
            if self.model == "knn":
                model = KNeighborsClassifier(n_neighbors=genome['n_neighbors'], weights=genome['weights'], p=genome['p'])
            elif self.model == "svc":
                model = LinearSVC(penalty=genome["penalty"], loss=genome["loss"], dual=genome["dual"], tol=genome["tol"], C=genome["C"])
            elif self.model == "logistic":
                model = LogisticRegression(penalty=genome["penalty"], C=genome["C"], dual=genome["dual"])
            elif self.model == "rf":
                model = RandomForestClassifier(n_estimators=genome["n_estimators"], criterion=genome["criterion"], max_features=genome["max_features"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"], bootstrap=genome["bootstrap"])
            elif self.model == "decision_tree":
                model = DecisionTreeClassifier(criterion=genome["criterion"], max_depth=genome["max_depth"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"])
        else:
            if self.model == "rf":
                model = RandomForestRegressor(n_estimators=genome["n_estimators"], max_features=genome["max_features"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"], bootstrap=genome["bootstrap"])
            elif self.model == "decision_tree":
                model = DecisionTreeRegressor(max_depth=genome["max_depth"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"])
            elif self.model == "elastic":
                model = ElasticNet(l1_ratio=genome["l1_ratio"], tol=genome["tol"])
            elif self.model == "knn":
                model = KNeighborsRegressor(n_neighbors=genome["n_neighbors"], weights=genome["weights"], p=genome["p"])
        return model


    def create_population(self):
        for _ in range(self.population_size):
            genome = self.create_genome()
            self.population.append(genome)


    def fit(self, X_train, y_train):

        self.X_train = X_train  
        self.y_train = y_train

        for generation in range(self.generations):

            sorted_population = self.sort_by_fitness()

            if generation != self.generations - 1:
                parents = self.create_parents(sorted_population)
                crossovers = self.create_crossovers(parents)
                parents.extend(crossovers)
                mutations = self.create_mutations(parents)
                parents.extend(mutations)
                self.population = parents
            else:
                if generation % 5 == 0:
                    print()
                    print(f"generation {generation}")
                self.best_genome = sorted_population[0]
                print("best drop_margins {0}".format(self.best_genome["drop_margins"]))
                print("best cleaners {0}".format(self.best_genome["cleaners"]))
                print("best model {0}".format(self.best_genome["model"]))

            
    def create_parents(self, population):
        parents = []

        while len(parents) < self.parents:
            new_parent = choices(population, k=1, weights=[exp(-x*0.1) for x in range(self.population_size)])[0]
            present = 0
            for parent in parents:
                if np.array_equal(new_parent["drop_margins"], parent["drop_margins"]) and new_parent["cleaners"] == parent["cleaners"] and new_parent["model"] == parent["model"]:
                    present = 1
            if not present:
                parents.append(new_parent)

        return parents
    

    def create_crossovers(self, parents):

        crossovers = []
        parent_posses = [i for i in range(len(parents))]
        pairs = list(itertools.combinations(parent_posses, 2))

        for pair in pairs:

            parent_a = parents[pair[0]]
            parent_b = parents[pair[1]]

            alternator = 0

            drop_margins_a = []
            drop_margins_b = []

            cleaners_a = {}
            cleaners_b = {}

            model_a = {}
            model_b = {}

            for i in range(len(parent_a["drop_margins"])):
                if alternator:
                    drop_margins_a.append(parent_a["drop_margins"][i])
                    drop_margins_b.append(parent_b["drop_margins"][i])
                else:
                    drop_margins_a.append(parent_b["drop_margins"][i])
                    drop_margins_b.append(parent_a["drop_margins"][i])
                alternator = not alternator

            for cleaner in parent_a["cleaners"]:
                if alternator:
                    cleaners_a[cleaner] = parent_a["cleaners"][cleaner]
                    cleaners_b[cleaner] = parent_b["cleaners"][cleaner]
                else:
                    cleaners_a[cleaner] = parent_b["cleaners"][cleaner]
                    cleaners_b[cleaner] = parent_a["cleaners"][cleaner]
                alternator = not alternator

            for key in parent_a["model"]:
                if alternator:
                    model_a[key] = parent_a["model"][key]
                    model_b[key] = parent_b["model"][key]
                else:
                    model_a[key] = parent_b["model"][key]
                    model_b[key] = parent_a["model"][key]
                alternator = not alternator

            child_a = {"drop_margins": drop_margins_a, "cleaners": cleaners_a, "model": model_a}
            child_b = {"drop_margins": drop_margins_b, "cleaners": cleaners_b, "model": model_b}

            crossovers.append(child_a)
            crossovers.append(child_b)

        return crossovers
    

    def create_mutations(self, population, n_mutations=2):

        mutations = []

        while len(population) + len(mutations) < self.population_size:
            genome = population[randint(0, len(population) - 1)].copy()

            for _ in range(n_mutations):
                segment = randint(0, len(genome)-1)

                if segment == 0:
                    index = randint(0, len(genome["drop_margins"]) - 1)
                    genome["drop_margins"][index] = not genome["drop_margins"][index]

                elif segment == 1:
                    index = randint(0, len(genome["cleaners"]) - 1)
                    key = list(genome["cleaners"].keys())[index]
                    new_value = choices(cleaners[key], k=1)[0]
                    genome["cleaners"][key] = new_value

                elif segment == 2:
                    index = randint(0, len(genome["model"]) - 1)
                    key = list(genome["model"].keys())[index]
                    new_value = choices(self.models[self.model][key], k=1)[0]
                    genome["model"][key] = new_value

                    
            mutations.append(genome)

        return mutations

                    
    def sort_by_fitness(self):
        
        fitnesses = {}
        repositioned = []

        for genome_pos in range(self.population_size):
            fitness = self.calc_fitness(self.population[genome_pos])
            fitnesses[str(genome_pos)] = fitness

        print(f"Best fitness: {max(fitnesses.values())}")

        for _ in range(self.population_size):
            max_key = max(fitnesses, key=fitnesses.get)
            repositioned.append(self.population[int(max_key)].copy())
            fitnesses.pop(max_key)
      
        return repositioned


    def calc_fitness(self, genome):
        
        X_train = np.asarray(self.X_train)
        X_train = X_train[:,np.asarray(genome["drop_margins"]).astype('bool')]
        
        if genome["cleaners"]["scaler"] == "robust":
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
        elif genome["cleaners"]["scaler"] == "standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        elif genome["cleaners"]["scaler"] == "minmax":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)

        if genome["cleaners"]["pca"] == "pca":
            pca = PCA(n_components="mle")
            X_train = pca.fit_transform(X_train)

        model = self.create_model(genome["model"])

        scores = cross_val_score(estimator=model, X=X_train, y=self.y_train, cv=self.cv, scoring=self.scoring)
        return mean(scores)
    

    def predict(self, X_test, y_test):

        X_train = np.asarray(self.X_train)
        X_test = np.asarray(X_test)
        X_train = X_train[:,np.asarray(self.best_genome["drop_margins"]).astype('bool')]
        X_test = X_test[:,np.asarray(self.best_genome["drop_margins"]).astype('bool')]
        
        if self.best_genome["cleaners"]["scaler"] == "robust":
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        elif self.best_genome["cleaners"]["scaler"] == "standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        elif self.best_genome["cleaners"]["scaler"] == "minmax":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

        if self.best_genome["cleaners"]["pca"] == "pca":
            pca = PCA(n_components="mle")
            X_train = pca.fit_transform(X_train)
            X_test = pca.fit_transform(X_test)

        model = self.create_model(self.best_genome["model"])

        model.fit(X_train, self.y_train)
        predictions = model.predict(X_test)

        if self.mode == "classification":

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            precision = precision_score(y_test, predictions)

            print()
            print()
            print(f"Final Accuracy: {accuracy}")
            print(f"Final f1: {f1}")
            print(f"Final recall: {recall}")
            print(f"Final precision: {precision}")

            cm = confusion_matrix(y_test, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.show()

        elif self.mode == "regression":

            RMSE = mse(y_test, predictions, squared=False)
            MAE = mae(y_test, predictions)
            MAPE = mape(y_test, predictions)
            R2 = r2(y_test, predictions)

            print()
            print()
            print(f"Final RMSE: {RMSE}")
            print(f"Final MAE: {MAE}")
            print(f"Final MAPE: {MAPE}")
            print(f"Final R2: {R2}")


if __name__=='__main__':

    data_path = r"/home/david/Documents/Cyber-AI/data/dataset_phishing_reduced.csv"
    df = pd.read_csv(data_path)

    # mapping = {'phishing': 1, 'legitimate': 0}
    # column = df['status'].map(mapping)
    # df['status'] = column

    labels = df["status"]
    data = df.drop("status", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    genetic_algorithm = GeneticAlgorithm("rf", cols=X_train.shape[1], mode="classification", parents=3, population=20, generations=20, cv=2)
    genetic_algorithm.fit(X_train, y_train)
    print("predicting")
    genetic_algorithm.predict(X_test, y_test)

