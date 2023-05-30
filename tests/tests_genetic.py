import numpy as np
from opt.genetic import (
    rank_selection,
    roulette_wheel_selection,
    tournament_selection,
    elitism_selection,
)

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


# a unit test for elitism selection using the population and fitness values and a random seed 42
def test_elitism_selection():
    """Test elitism selection"""
    # the expected output
    expected_output = np.array(
        [[46, 47, 48, 49, 50], [26, 27, 28, 29, 30], [16, 17, 18, 19, 20]]
    )
    # the actual output
    actual_output = elitism_selection(population, fitness, 2, 1)
    # assert statement
    assert np.all(actual_output == expected_output), "Elitism selection failed"


test_rank_selection()
test_roulette_wheel_selection()
test_tournament_selection()
test_elitism_selection()
