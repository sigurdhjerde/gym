import scipy.io

todays_seed = 0

meals_time = []
meals_carb = []
meals_carbestimate = []

n_test_scenarios = 100

for i in range(n_test_scenarios):
      m = meal_generator(eating_time=1,premeal_bolus_time=0,meal_uncertainty_grams=20,seed=todays_seed+i)
      meals_time.append(np.nonzero(m[0]))
      meals_carb.append(m[0][m[0] > 0])
      meals_carbestimate.append(m[1][m[1] > 0])

meals_time = np.matrix(np.array(meals_time))
meals_carb = np.matrix(np.array(meals_carb))*180/1000
meals_carbestimate = np.matrix(np.array(meals_carbestimate))*180/1000


scipy.io.savemat('meals_time',mdict={'meals_time': meals_time})
scipy.io.savemat('meals_carb',mdict={'meals_carb': meals_carb})
scipy.io.savemat('meals_carbestimate',mdict={'meals_carbestimate': meals_carbestimate})
