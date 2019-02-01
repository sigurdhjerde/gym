from gym.envs.diabetes.meal_generator import meal_generator
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Generating default meals
meals, meals_indicator = meal_generator.meal_generator()


# plotting
# fig = plt.subplot(1, 2, 1)

plt.plot(meals)

# fig = plt.subplot(1, 2, 2)

plt.plot(meals_indicator)

plt.legend(['meals', 'pump boluses'])

plt.show()
