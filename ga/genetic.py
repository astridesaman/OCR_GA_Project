import torch 
import random
from .utils import get_weights, set_weights

def initialize_population(model, pop_size):
    return [torch.randn(get_weights(model).shape) for _ in range(pop_size)]

def fitness(model, individual, batch):
    set_weights(model, individual)
    images, labels = batch
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).float().mean().item()

def select(population, fitnesses, k):
    sorted_pairs = sorted(
        zip(fitnesses, population),
        key=lambda x: x[0],   # on trie uniquement sur la fitness
        reverse=True
    )
    sorted_pop = [x[1] for x in sorted_pairs]
    return sorted_pop[:k]

def crossover(parent1, parent2):
    mask = torch.rand_like(parent1) > 0.5
    return torch.where(mask, parent1, parent2)

def mutate(individual, mutation_rate=0.05):
    mask = torch.rand_like(individual) < mutation_rate
    noise = torch.randn_like(individual) * 0.1
    individual[mask] += noise[mask]
    return individual

def train_genetic(model, train_loader, generations=20, pop_size=30, select_k=10):
    batch = next(iter(train_loader))
    population = initialize_population(model, pop_size)

    for gen in range(generations):
        fitnesses = [fitness(model, ind, batch) for ind in population]
        print(f"Génération {gen+1}, Best fitness: {max(fitnesses):.4f}")

        selected = select(population, fitnesses, k=select_k)
        new_population = selected.copy()

        while len(new_population) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    fitnesses = [fitness(model, ind, batch) for ind in population]
    best_individual = population[fitnesses.index(max(fitnesses))]
    set_weights(model, best_individual)                                            