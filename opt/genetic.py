# %%
import numpy as np
import numpy.typing as npt


# A function to perform roulette wheel selection
def roulette_wheel_selection(
    population: npt.ArrayLike, fitness: npt.ArrayLike, num_parents=1
):
    """Performs roulette wheel selection given a population and parents

    Args:
        population (np.array): The population to select from
        fitness (np.array):  The fitness of the population

    Returns:
        _type_: _description_
    """
    # Calculate the fitness sum of the population
    fitness_sum = np.sum(fitness)
    # Calculate the probability of each individual
    probability = fitness / fitness_sum
    # Select the parents based on the probability
    parents = np.random.choice(
        np.arange(population.shape[0]), num_parents, p=probability, replace=False
    )
    # Return the parents
    return np.take(population, parents, axis=0)


# A function to perform rank selection given a population as an array and an array of fitness values
def rank_selection(population: npt.ArrayLike, fitness: npt.ArrayLike, num_parents=1):
    """_summary_

    Args:
        population (npt.ArrayLike): _description_
        fitness (npt.ArrayLike): _description_
        num_parents (int, optional): _description_. Defaults to 1.
    """
    fitness_rank = np.argsort(fitness)
    probability = np.arange(1, len(fitness_rank) + 1) / np.sum(
        np.arange(1, len(fitness_rank) + 1)
    )
    parents = np.random.choice(fitness_rank, num_parents, p=probability, replace=False)
    return np.take(population, parents, axis=0)


# A function to perform tournament selection given a population as an array and an array of fitness values
def tournament_selection(
    population: npt.ArrayLike, fitness: npt.ArrayLike, num_parents=2
):
    # split the population array into as many groups as there are parents
    groups = np.array_split(np.arange(population.shape[0]), num_parents)
    parents = np.array([], dtype=int)
    for group in groups:
        # for each group, select the best individual
        best_individual = np.argmax(fitness[group])
        # add the best individual to the parents array
        parents = np.append(parents, group[best_individual])
    # for each group, select the best individual
    return np.take(population, parents, axis=0)


# A function to select the top Y individuals from the population and then use roulette wheel selection to select the remaining individuals
def elitism_selection(
    population: npt.ArrayLike,
    fitness: npt.ArrayLike,
    num_elite_parents=1,
    num_remaining_parents=1,
):
    """_summary_

    Args:
        population (npt.ArrayLike): _description_
        fitness (npt.ArrayLike): _description_
        num_elite_parents (int, optional): _description_. Defaults to 1.
        num_remaining_parents (int, optional): _description_. Defaults to 1.
    """
    elite_index = np.argsort(fitness)[-num_elite_parents:]
    elite_parents = np.take(population, elite_index, axis=0)
    updated_fitness = np.delete(fitness, elite_index, axis=0)
    updated_population = np.delete(population, elite_index, axis=0)
    remaining_parents = roulette_wheel_selection(
        updated_population, updated_fitness, num_remaining_parents
    )
    return np.concatenate((elite_parents, remaining_parents), axis=0)


population = np.array(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35],
        [36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45],
        [46, 47, 48, 49, 50],
    ]
)
fitness = np.array([1, 2, 3, 4, 5, 6000, 7, 8, 9, 10])
# a unit test for the roulette wheel selection using the population and fitness values and a random seed 42
np.random.seed(42)


def test_roulette_wheel_selection():
    """Test roulette wheel selection"""
    # the expected output
    expected_output = np.array([[26, 27, 28, 29, 30], [16, 17, 18, 19, 20]])
    # the actual output
    actual_output = roulette_wheel_selection(population, fitness, 2)
    # assert statement
    assert np.all(actual_output == expected_output), "Roulette wheel selection failed"


# a unit test for rank selection using the population and fitness values and a random seed 42
def test_rank_selection():
    """Test rank selection"""
    # the expected output
    expected_output = np.array([[31, 32, 33, 34, 35], [26, 27, 28, 29, 30]])
    # the actual output
    actual_output = rank_selection(population, fitness, 2)
    # assert statement
    assert np.all(actual_output == expected_output), "Rank selection failed"


# a unit test for tournament selection using the population and fitness values and a random seed 42
def test_tournament_selection():
    """Test tournament selection"""
    # the expected output
    expected_output = np.array([[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]])
    # the actual output
    actual_output = tournament_selection(population, fitness, 2)
    # assert statement
    assert np.all(actual_output == expected_output), "Tournament selection failed"


# %%
