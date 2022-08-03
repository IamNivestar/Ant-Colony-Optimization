import random
import numpy as np
import pandas
import matplotlib.pyplot as plt
import progress.bar as Bar
import time

start = time.time() #user time
t = time.process_time() # process time

#Nivestar

#Flags
CODE_VERBOSE = True
DEBUG = False
MINIMIZATION = True #flag to change the AG to maximization (False) or minimization (True)
GRAPH = True

#calibration variables -----------
global_best_execution = []
global_nTimes_got_optimal_solution = 0
#-----------

#  ant colony optimization for Travelling Salesman Problem  

class ACO():

	def __init__(self, n_iterations=15, ants_porcent = 1, alpha = 1, beta = 5, rho = 0.5, Q = 100, 
			tau_zero = 1*(10**-6), strategy = 'ASR', W = 5, zeta = 0.3, test_number = 1):
		self.file_reading(test_number)
		self.M = len(self.distance_matrix) #count of edges

		self.edges = [*range(0, self.M, 1)]
		self.ants = []
		self.pheromone = []

		n_ants = self.M * round(ants_porcent) #the same number of edges or porcent of the value?

		for _ in range(n_ants):
			ant = []
			self.ants.append(ant)

		self.strategy = strategy
		self.alpha = alpha #pheromone weigh
		self.beta = beta #distance weigh
		self.rho = rho  #evaporate pheromone weight
		self.Q = Q	#weight of distance used pheromone addition
		self.W = W   #Number of rank ants for ASR mode
		self.zeta = zeta #weight for EAS and ASR modes
		self.n_iterations = n_iterations

		for _ in range(self.M):
			pheromone = [tau_zero]*self.M  
			self.pheromone.append(pheromone)

		self.fits = []

	def file_reading(self, test_number):
		
		self.distance_matrix, self.solution = read_file(test_number)
	
	def run(self):
		
		global global_best_execution, global_nTimes_got_optimal_solution
		list_best = []
		list_best_ind = []
		iteration = 0

		print("Starting with ", len(self.ants), " ants...")		
		self.start_ants()

		if DEBUG:
			self.print_list(self.ants, "Ants:")

		while (iteration < self.n_iterations):

			if CODE_VERBOSE:
				print("Iteration ", iteration)

			self.trace_route()
			self.evaluation()

			if(DEBUG):
				self.print_list(self.pheromone, "Pheromone:")
				self.print_list(self.ants, "Ants:")
				print("Scores:\n", self.fits)
			iteration+=1

			temp = self.fits.copy()
			if MINIMIZATION:
				temp.sort()
			else: 
				temp.sort(reverse=True)
			best_ind_index = self.fits.index(temp[0])
			list_best_ind.append(self.ants[best_ind_index])
			list_best.append(temp[0])

			if(iteration != self.n_iterations): #last iteration?
				self.evaporate_pheromone()
				self.way_back_ants()

		if CODE_VERBOSE:
			print("END")

		if DEBUG:
			print("Final Ants:\n", self.ants)
			print("Final Score:\n", self.fits)
			print("Best Scores: ", list_best)	
	
		list_best_original = list_best.copy()
		if MINIMIZATION:
			list_best.sort()
		else:
			list_best.sort(reverse=True)
		best = list_best[0]
		best_index = list_best_original.index(best)
		solution_fitness = self.func_obj(self.solution)
		if GRAPH:
			self.make_graph(list_best_original, solution_fitness, best, best_index)
		best_ind = list_best_ind[best_index]
		print(f"The individual: { best_ind } was the best, with fitness: {self.func_obj(best_ind)}") 
		print(f"Solution:\t{self.solution}, Solution fitness: {solution_fitness}")
		
		if(best == solution_fitness):  
			print("\nBest Solution Reached")
			global_nTimes_got_optimal_solution +=1

		global_best_execution.append(best) 

	def print_list(self, list, title):

		print(title)
		for i in list:
			print(i)

	def start_ants(self):

		for i in range(len(self.ants)):
			if(len(self.ants) == self.M): # same size, so each ant with a different edge
				self.ants[i].append(self.edges[i])
			else: 
				self.ants[i].append(random.randint(0, self.M-1))

	def trace_route(self):

		for x in range(self.M -1): 
			self.ants_walk()
			
	def ants_walk(self):

		for index in range(len(self.ants)): #get next edge for each ant
			remainder_path = list(set(self.edges) - set(self.ants[index]))
			next_rote = self.probability_func(self.ants[index][-1], remainder_path)
			self.ants[index].append(next_rote)
	
	def probability_func(self, current_edge, remainder_path): 

		probability_values = []

		for edge in remainder_path:
			if(MINIMIZATION):
				neighborhood = (1 / self.distance_matrix[current_edge][edge]) ** self.beta
			else:
				neighborhood = self.distance_matrix[current_edge][edge]**self.beta
			pheromone_current = self.pheromone[current_edge][edge]**self.alpha
			
			probability_value = neighborhood * pheromone_current
			probability_values.append(probability_value)
		sum_p = sum(probability_values)
		selection_probs = [ value /sum_p for value in probability_values] 
		sorted_edge = np.random.choice(remainder_path, p=selection_probs)
		return sorted_edge

	def way_back_ants(self):

		self.launch_pheromone()
		for i in range(len(self.ants)):
			self.ants[i] = [self.ants[i][0]] #reset path

	def evaporate_pheromone(self):

		for i in range(self.M):
			for j in range(self.M):
				self.pheromone[i][j] = (1-self.rho) * self.pheromone[i][j]

	def launch_pheromone(self):
		
		if(self.strategy == 'AS'):  #ant system all ants
			for i in range(len(self.ants)):
				self.ant_system(self.ants[i], i, weight=1)
		
		elif(self.strategy == 'ASR'): #ant system for just W ants
			fits_sorted = self.fits.copy()
			fits_sorted.sort()
			best_ant = self.fits.index(fits_sorted[0])
			rank_ants_score = fits_sorted[:self.W]  # getting W best ants score
			rank_ants = [ self.fits.index(value) for value in rank_ants_score] #getting number of ants who has bests scores
			for r_ant in rank_ants:
				self.ant_system(self.ants[r_ant], r_ant, weight= ( self.W - rank_ants_score.index(self.fits[r_ant]) ) )
			self.ant_system(self.ants[best_ant], best_ant, weight=self.W) # the best ant increase twice
		
		elif(self.strategy == 'EAS'): #ant system with best so far tour
			fits_sorted = self.fits.copy()
			fits_sorted.sort()
			best_ant = self.fits.index(fits_sorted[0])  
			for i in range(len(self.ants)):
				self.ant_system(self.ants[i], i, weight=1)
			self.ant_system(self.ants[best_ant], best_ant, weight=self.zeta) #best so far tour

		else:
			print('Error strategy choose')
			exit()

	def ant_system(self, path, ant_index, weight): #causes a ant throw a certain amount of pheromone along the way

		for i in range(self.M):
			if (i == self.M -1):
				next = path[0]
			else:
				next = path[i+1]
			current = path[i]

			if MINIMIZATION:
				self.pheromone[current][next] = self.pheromone[current][next] + weight*( self.Q / self.func_obj(self.ants[ant_index]) )
			else:
				self.pheromone[current][next] = self.pheromone[current][next] + weight*( self.func_obj(self.ants[ant_index]) / self.Q )

	def func_obj(self, route):

		distance = 0
		for i, _ in enumerate(route):
			if (i < self.M-1):
				distance += self.distance_matrix[route[i]][route[i+1]]
		distance += self.distance_matrix[route[self.M-1]][route[0]]
		return distance

	def evaluation(self):

		self.fits.clear()
		for ind in self.ants:
			self.fits.append(self.func_obj(ind))

	def make_graph(self, list_best, solution_fitness, best, best_index):

		plt.plot( [*range(self.n_iterations)], list_best)
		plt.axline( (0, solution_fitness), (self.n_iterations-3, solution_fitness), color='g', linestyle='--', label="Best Solution")
		plt.annotate(f'{solution_fitness}', xy=(self.n_iterations-2, solution_fitness), fontsize=10)
		plt.annotate(f'{best}', xy=(best_index, best+0.01*(best)), xytext=(best_index, best+0.05*(best)), 
			arrowprops=dict(facecolor='black', shrink=0.01, headwidth=10, headlength=10,width=1), fontsize=10)
		plt.ylabel("Distance")
		plt.xlabel("Interations")
		plt.xticks([*range(self.n_iterations)])
		plt.legend(loc='lower left')
		plt.show() 

def read_file(test_number):

	if test_number == 1:
		print("Test Case with 15 edges (easy)")
		f_distanceMatrix = np.loadtxt("tests/lau15_dist.txt", dtype='int')
		n_fsolution= "tests/lau15_tsp.txt"
	elif test_number == 2:
		print("Test Case with 48 edges (very hard)")
		f_distanceMatrix = np.loadtxt("tests/att48_d.txt", dtype='int')
		n_fsolution= "tests/att48_s.txt"
	elif test_number == 3:
		print("Test Case with 26 edges (medium)")
		f_distanceMatrix = np.loadtxt("tests/fri26_d.txt", dtype='int')
		n_fsolution= "tests/fri26_s.txt"
	elif test_number == 4:
		f_distanceMatrix = np.loadtxt("tests/dantzig42_d.txt", dtype='int')
		print("Test Case with 42 edges (hard)")
		n_fsolution= "tests/dantzig42_s.txt" 
	elif test_number == 5:
		print("Test Case with 17 edges (hard)")
		f_distanceMatrix = np.loadtxt("tests/gr17_d.txt", dtype='int')
		n_fsolution= "tests/gr17_s.txt"

	f_solution = open(n_fsolution)
	solution = f_solution.readlines()
	solution = [int(s)-1 for s in solution]
	
	return f_distanceMatrix, solution

def calibrate():

	global global_best_execution, global_nTimes_got_optimal_solution, DEBUG, CODE_VERBOSE, GRAPH

	CODE_VERBOSE = False
	DEBUG = False
	GRAPH = False

	n_iterations_list = [15]
	strategy_method_list = ['ASR', 'EAS']  # 'AS' is most simple method
	ants_porcent_list = [1, 2]
	Q_list = [100, 10000] #weight of distance used pheromone addition
	alpha_list = [0.5, 1] #pheromone weigh (collective knowledge)
	beta_list = [2, 5] #distance weigh (local knowledge)
	rho_list = [0.2, 0.5] #evaporate pheromone weight 
	W_list = [5] #Number of rank ants (only for ASR mode)
	zeta_list = [0.3, 0.8] #weight for EAS and ASR modes

	
	df = pandas.DataFrame()
	n_iterations_column = []
	strategy_method_column = []
	ants_porcent_column = []
	alpha_column = []
	beta_column = []
	rho_column = []
	Q_column = []
	W_column = []
	zeta_column = []
	params_columns = []

	#list to saving results
	list_best_execution = [] #best reasult for execution 
	times_best_solution = 0

	#calculate mean of X times loop for each saved result
	final_mean_list_best_execution = [] #best results for execution 
	final_sum_times_best_solution = []
	times_repetition = 10

	#progress estimative ...
	total_progress = len(n_iterations_list) * len(ants_porcent_list) *	len(Q_list) * len(strategy_method_list) \
		* len(alpha_list) * len(beta_list) * len(rho_list) * len(zeta_list) * len(W_list) * times_repetition
	my_bar = Bar.ShadyBar('Calibrating...', max=total_progress,  suffix='%(percent)d%%')


	for i in n_iterations_list:
		for s in strategy_method_list:
			for ant in ants_porcent_list:
				for q in Q_list:
					for a in alpha_list:
						for b in beta_list:
							for r in rho_list:
								for w in W_list:
									for z in zeta_list:

										n_iterations_column.append(i)
										strategy_method_column.append(s)
										Q_column.append(q)
										alpha_column.append(a)
										ants_porcent_column.append(ant)
										beta_column.append(b)
										rho_column.append(r)
										W_column.append(w)
										zeta_column.append(z)				
										print("Execution with the current params: ")
										print(f'Number of interations:{i} Count_Ants_per_edges:{ant} Strategy_Pheromone_Method:{s} Q:{q} Alpha:{a} Beta:{b} Rho:{r} \
											W:{w} Zeta:{z}')
										params_columns.append(f'Number of interations:{i} Count_Ants_per_edges:{ant} Strategy_Pheromone_Method:{s} Q:{q} Alpha:{a} Beta:{b} Rho:{r} W:{w} Zeta:{z}')	
										for _ in range(times_repetition): #doing X times
											
											print('\n\n')
											my_bar.next()
											print('\n\n')

											aco = ACO(n_iterations=i, strategy=s, ants_porcent=ant, test_number=5, alpha = a, beta = b, rho = r, Q = q, W=w, zeta=z)
											aco.run()
											list_best_execution.append(global_best_execution)
											times_best_solution += global_nTimes_got_optimal_solution
											
											#clean globals
											global_best_execution = []
											global_nTimes_got_optimal_solution = 0	
									
										final_mean_list_best_execution.append(np.mean(list_best_execution))
										final_sum_times_best_solution.append(times_best_solution)
										
										#clean lists
										list_best_execution.clear()
										times_best_solution = 0
												
	df['Params'] = params_columns
	df['Ants_porcent'] = ants_porcent_column
	df["Number_Iterations"] = n_iterations_column 
	df["Strategy_Pheromone_Method"] = 	strategy_method_column
	df["Bests_Results"] = final_mean_list_best_execution
	df["Number_times_got_best_solution"] = final_sum_times_best_solution

	my_bar.finish()
	df.to_csv('results_params_ACO.csv', sep=';')
	#winner (5 optimal solutions!) = Number of interations:15 Count_Ants_per_edges:2 Strategy_Pheromone_Method:ASR Q:10000 Alpha:0.5 Beta:2 Rho:0.2 W:5 Zeta:0.8
	#(tested in test_case number 5!)

	# the most difficult test cases used ants_percent = 2, 3 ... and 20 as in case 4 

if __name__ == "__main__":
	
	#   strategy can be AS, ASR or EAS
	aco = ACO(n_iterations=10, strategy='AS', ants_porcent= 3, test_number=3, alpha = 1,
		beta = 5, rho = 0.3, Q = 10000, W=10, zeta=0.6)

	aco.run()
	#calibrate()

	end = time.time() #user
	user_time = end - start 
	elapsed_time = time.process_time() - t #process

	print("="*100) 
	print("User time: %s" % user_time)
	print("Process time: %s" % elapsed_time)
	print( time.strftime("%H hours %M minutes %S seconds", time.gmtime(user_time)) ) 
	print("="*100)
