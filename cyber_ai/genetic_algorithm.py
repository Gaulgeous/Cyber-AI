import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from random import choices, randint
from statistics import mean
from copy import copy
from math import exp
from joblib import parallel, delayed

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, ConfusionMatrixDisplay



models = {"knn": {'n_neighbors': range(1, 101), 'weights': ["uniform", "distance"], 'p': [1, 2]},
          'svc': {'penalty': ["l2"], 'loss': ["hinge", "squared_hinge"], 'dual': [True], 'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]},
          'logistic': {'penalty': ["l2"], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 'dual': [False]},
          'rf': {'n_estimators': [100], 'criterion': ["gini", "entropy"], 'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21), 'min_samples_leaf':  range(1, 21), 'bootstrap': [True, False]},
          'decision_tree': {'criterion': ["gini", "entropy"], 'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21)}
        }


class GeneticAlgorithm:

    def __init__(self, model, population=20, generations=20, cv=5, parents=4, scoring='f1'):

        self.model = model
        self.population_size = population
        self.generations = generations
        self.cv = cv
        self.scoring = scoring
        self.parents = parents
        self.population = []
        self.X_train = None
        self.y_train = None
        self.best_genome = None

        self.create_population()


    def create_genome(self):

        genome = {}
        for key in models[self.model]:
            value = choices(models[self.model][key], k=1)[0]
            genome[key] = value

        return genome


    def create_model(self, genome):

        model = None

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
                self.best_genome = sorted_population[0]

            
    def create_parents(self, population):
        parents = []

        while len(parents) < self.parents:
            new_parent = choices(population, k=1, weights=[exp(-x*0.1) for x in range(self.population_size)])[0]
            if new_parent not in parents:
                parents.append(new_parent)

        return parents
    

    def create_crossovers(self, parents):

        crossovers = []
        parent_posses = [i for i in range(len(parents))]
        pairs = list(itertools.combinations(parent_posses, 2))

        for pair in pairs:

            parent_a = parents[pair[0]]
            parent_b = parents[pair[1]]

            child_a = {}
            child_b = {}

            alternator = 0

            for key in parent_a:
                if alternator:
                    child_a[key] = parent_a[key]
                    child_b[key] = parent_b[key]
                else:
                    child_a[key] = parent_b[key]
                    child_b[key] = parent_a[key]
                alternator = not alternator

            crossovers.append(child_a)
            crossovers.append(child_b)

        return crossovers
    

    def create_mutations(self, population, n_mutations=1):

        mutations = []

        while len(population) + len(mutations) < self.population_size:
            genome = population[randint(0, len(population) - 1)].copy()

            for n in range(n_mutations):
                index = randint(0, len(genome))

                if index != len(genome):
                    key = list(genome.keys())[index]
                    new_value = choices(models[self.model][key], k=1)[0]
                    genome[key] = new_value

                # Put in use case for margin dropping here
                    
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
        model = self.create_model(genome)
        scores = cross_val_score(estimator=model, X=self.X_train, y=self.y_train, cv=self.cv, scoring=self.scoring)
        return mean(scores)
    

    def predict(self, X_test, y_test):

        model = self.create_model(self.best_genome)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(X_test)

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


if __name__=='__main__':

    data_path = r"/home/david/Documents/Cyber-AI/data/Training Dataset.csv"
    df = pd.read_csv(data_path)

    labels = df["Result"]
    data = df.drop("Result", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    genetic_algorithm = GeneticAlgorithm("decision_tree", population=20, generations=20, cv=2)
    genetic_algorithm.fit(X_train, y_train)
    genetic_algorithm.predict(X_test, y_test)

