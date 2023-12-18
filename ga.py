import random
import numpy as np

distances = {
    "Moscow": {
        "Moscow": 0,
        "Saint Petersburg": 705,
        "Kazan": 815,
        "Ural": 1800,
        "Astrakhan": 1430,
        "Ufa": 1160
    },
    "Saint Petersburg": {
        "Moscow": 705,
        "Saint Petersburg": 0,
        "Kazan": 1500,
        "Ural": 2100,
        "Astrakhan": 2000,
        "Ufa": 1700
    },
    "Kazan": {
        "Moscow": 815,
        "Saint Petersburg": 1500,
        "Kazan": 0,
        "Ural": 960,
        "Astrakhan": 1300,
        "Ufa": 525
    },
    "Ural": {
        "Moscow": 1800,
        "Saint Petersburg": 2100,
        "Kazan": 960,
        "Ural": 0,
        "Astrakhan": 1500,
        "Ufa": 450
    },
    "Astrakhan": {
        "Moscow": 1430,
        "Saint Petersburg": 2000,
        "Kazan": 1300,
        "Ural": 1500,
        "Astrakhan": 0,
        "Ufa": 1000
    },
    "Ufa": {
        "Moscow": 1160,
        "Saint Petersburg": 1700,
        "Kazan": 525,
        "Ural": 450,
        "Astrakhan": 1000,
        "Ufa": 0
    }
}

cities = list(distances.keys())
distance_matrix = np.array([[distances[city1][city2] for city2 in cities] for city1 in cities])

population_size = 100
num_generations = 1000
mutation_rate = 0.01


def calculate_total_distance(path, distance_matrix):
    path_length = sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
    path_length += distance_matrix[path[-1], path[0]]
    return path_length


def create_initial_population(population_size, num_cities):
    paths = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]
    return paths


def crossover(parent1, parent2):
    length = len(parent1)
    start, end = sorted(random.sample(range(length), 2))
    offspring = parent1[start:end]

    for city in parent2:
        if city not in offspring:
            offspring.append(city)
    return offspring


def mutate(path, mutation_rate):
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(path)), 2))
        path[start:end] = reversed(path[start:end])
    return path


def genetic_algorithm(distance_matrix, population_size, num_generations, mutation_rate):
    num_cities = len(distance_matrix)
    population = create_initial_population(population_size, num_cities)

    for generation in range(num_generations):
        fitness_scores = [1 / calculate_total_distance(path, distance_matrix) for path in population]

        fitness_sum = sum(fitness_scores)
        selection_prob = [fitness / fitness_sum for fitness in fitness_scores]
        selected_indices = np.random.choice(range(population_size), size=population_size, p=selection_prob)
        selected_population = [population[i] for i in selected_indices]

        next_generation = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            offspring1, offspring2 = crossover(parent1, parent2), crossover(parent2, parent1)
            next_generation.extend([mutate(offspring1, mutation_rate), mutate(offspring2, mutation_rate)])

        population = next_generation

    best_path = min(population, key=lambda path: calculate_total_distance(path, distance_matrix))
    best_distance = calculate_total_distance(best_path, distance_matrix)
    best_path_cities = [cities[i] for i in best_path]

    return best_path_cities, best_distance


print(genetic_algorithm(distance_matrix, population_size, num_generations, mutation_rate))