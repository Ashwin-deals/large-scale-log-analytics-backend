"""Genetic Algorithm for Feature Selection & Anomaly Threshold Tuning.

Evolves candidate chromosomes θ = [f_1, f_2, ..., f_n, T] to maximize:
Fitness(θ) = α * Accuracy - β * FalsePositiveRate - γ * (FeatureCount / TotalFeatures)
"""

import copy
import random
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np


class Chromosome:
    """Represents a candidate solution: [f_1, ..., f_n, T]"""

    def __init__(self, n_features: int, feature_mask: Optional[np.ndarray] = None, threshold: Optional[float] = None):
        self.n_features = n_features
        if feature_mask is not None:
            self.features = feature_mask.astype(bool)
        else:
            # Initialize with at least 2 active features
            mask = np.random.rand(n_features) > 0.5
            if not np.any(mask):
                mask[np.random.choice(n_features, size=min(3, n_features), replace=False)] = True
            self.features = mask

        if threshold is not None:
            self.threshold = float(np.clip(threshold, 0.40, 0.90))
        else:
            self.threshold = float(np.random.uniform(0.50, 0.75))

        self.fitness: float = -np.inf
        self.accuracy: float = 0.0
        self.fpr: float = 0.0
        self.tpr: float = 0.0

    @property
    def selected_indices(self) -> np.ndarray:
        indices = np.where(self.features)[0]
        if len(indices) == 0:
            # Fallback to at least the first feature
            return np.array([0])
        return indices

    def clone(self) -> "Chromosome":
        c = Chromosome(self.n_features, self.features.copy(), self.threshold)
        c.fitness = self.fitness
        c.accuracy = self.accuracy
        c.fpr = self.fpr
        c.tpr = self.tpr
        return c


class GeneticOptimizer:
    """Evolutionary optimizer for cloud log anomaly detection hyperparameters."""

    def __init__(
        self,
        n_features: int,
        population_size: int = 30,
        n_generations: int = 15,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        elite_count: int = 2,
        alpha: float = 1.0,     # Weight for Accuracy/TPR
        beta: float = 1.5,      # Penalty weight for False Positives
        gamma: float = 0.1,     # Parsimony penalty for number of features
        random_seed: Optional[int] = 42
    ):
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.population: List[Chromosome] = [
            Chromosome(n_features=self.n_features) for _ in range(self.population_size)
        ]
        self.best_chromosome: Optional[Chromosome] = None
        self.generation_history: List[Dict[str, float]] = []

    def evaluate_fitness(
        self,
        chromosome: Chromosome,
        eval_fn: Callable[[np.ndarray, float], Tuple[float, float, float]]
    ) -> float:
        """Evaluates chromosome fitness using the provided evaluation callback.
        
        eval_fn returns (accuracy, fpr, tpr).
        Fitness = alpha * accuracy - beta * fpr - gamma * (selected_features / n_features)
        """
        active_indices = chromosome.selected_indices
        acc, fpr, tpr = eval_fn(active_indices, chromosome.threshold)

        chromosome.accuracy = acc
        chromosome.fpr = fpr
        chromosome.tpr = tpr

        feature_ratio = len(active_indices) / self.n_features
        fitness = (self.alpha * acc) - (self.beta * fpr) - (self.gamma * feature_ratio)
        chromosome.fitness = fitness
        return fitness

    def tournament_selection(self, k: int = 3) -> Chromosome:
        """Selects the fittest individual from a random tournament group."""
        competitors = random.sample(self.population, k)
        return max(competitors, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Performs uniform crossover between feature masks and blend crossover on threshold."""
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        # Uniform crossover for binary feature mask
        swap_mask = np.random.rand(self.n_features) > 0.5
        child1_features = np.where(swap_mask, parent1.features, parent2.features)
        child2_features = np.where(swap_mask, parent2.features, parent1.features)

        # Blend crossover for threshold
        gamma_blend = random.uniform(-0.1, 1.1)
        child1_th = np.clip(parent1.threshold + gamma_blend * (parent2.threshold - parent1.threshold), 0.40, 0.90)
        child2_th = np.clip(parent2.threshold - gamma_blend * (parent2.threshold - parent1.threshold), 0.40, 0.90)

        c1 = Chromosome(self.n_features, child1_features, child1_th)
        c2 = Chromosome(self.n_features, child2_features, child2_th)
        return c1, c2

    def mutate(self, chromosome: Chromosome) -> None:
        """Applies bit-flip mutation on features and Gaussian jitter on threshold."""
        # Feature mutation
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                chromosome.features[i] = not chromosome.features[i]

        # Ensure at least 1 feature is selected
        if not np.any(chromosome.features):
            chromosome.features[random.randint(0, self.n_features - 1)] = True

        # Threshold mutation
        if random.random() < self.mutation_rate:
            noise = np.random.normal(loc=0.0, scale=0.04)
            chromosome.threshold = float(np.clip(chromosome.threshold + noise, 0.40, 0.90))

    def evolve(
        self,
        eval_fn: Callable[[np.ndarray, float], Tuple[float, float, float]],
        verbose: bool = True
    ) -> Chromosome:
        """Executes the genetic algorithm evolutionary cycle."""
        # Evaluate initial population
        for ind in self.population:
            self.evaluate_fitness(ind, eval_fn)

        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.best_chromosome = self.population[0].clone()

        for gen in range(1, self.n_generations + 1):
            next_generation: List[Chromosome] = []

            # Elitism: retain top performers unchanged
            for elite in self.population[:self.elite_count]:
                next_generation.append(elite.clone())

            # Produce offspring
            while len(next_generation) < self.population_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)

                self.evaluate_fitness(c1, eval_fn)
                next_generation.append(c1)
                if len(next_generation) < self.population_size:
                    self.evaluate_fitness(c2, eval_fn)
                    next_generation.append(c2)

            self.population = next_generation
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)

            if self.population[0].fitness > self.best_chromosome.fitness:
                self.best_chromosome = self.population[0].clone()

            avg_fitness = float(np.mean([ind.fitness for ind in self.population]))
            best_gen_fitness = self.population[0].fitness

            self.generation_history.append({
                "generation": gen,
                "best_fitness": best_gen_fitness,
                "avg_fitness": avg_fitness,
                "best_accuracy": self.population[0].accuracy,
                "best_fpr": self.population[0].fpr,
                "threshold": self.population[0].threshold,
                "n_features_selected": len(self.population[0].selected_indices)
            })

            if verbose:
                print(f"[GA Gen {gen:02d}] Best Fitness: {best_gen_fitness:.4f} | "
                      f"Acc: {self.population[0].accuracy:.3f} | "
                      f"FPR: {self.population[0].fpr:.3f} | "
                      f"Threshold: {self.population[0].threshold:.3f} | "
                      f"Features: {len(self.population[0].selected_indices)}/{self.n_features}")

        return self.best_chromosome
