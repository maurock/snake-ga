# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:10:29 2020

@author: mauro
"""
from snakeClass import run
from GPyOpt.methods import BayesianOptimization
import datetime

################################################
#   Set parameters for Bayesian Optimization   #
################################################

class BayesianOptimizer():
    def __init__(self, params):
        self.params = params

    def optimize_RL(self):
        def optimize(inputs):
            print("INPUT", inputs)
            inputs = inputs[0]

            # Variables to optimize
            self.params["learning_rate"] = inputs[0]
            lr_string = '{:.8f}'.format(self.params["learning_rate"])[2:]
            self.params["first_layer_size"] = int(inputs[1])
            self.params["second_layer_size"] = int(inputs[2])
            self.params["third_layer_size"] = int(inputs[3])
            self.params["epsilon_decay_linear"] = int(inputs[4])

            self.params['name_scenario'] = 'snake_lr{}_struct{}_{}_{}_eps{}'.format(lr_string,
                                                                               self.params['first_layer_size'],
                                                                               self.params['second_layer_size'],
                                                                               self.params['third_layer_size'],
                                                                               self.params['epsilon_decay_linear'])

            self.params['weights_path'] = 'weights/weights_' + self.params['name_scenario'] + '.h5'
            self.params['load_weights'] = False
            self.params['train'] = True
            print(self.params)
            score, mean, stdev = run(self.params)
            print('Total score: {}   Mean: {}   Std dev:   {}'.format(score, mean, stdev))
            with open(self.params['log_path'], 'a') as f: 
                f.write(str(self.params['name_scenario']) + '\n')
                f.write('Params: ' + str(self.params) + '\n')
            return score

        optim_params = [
            {"name": "learning_rate", "type": "continuous", "domain": (0.00005, 0.001)},
            {"name": "first_layer_size", "type": "discrete", "domain": (20,50,100,200)},
            {"name": "second_layer_size", "type": "discrete", "domain": (20,50,100,200)},
            {"name": "third_layer_size", "type": "discrete", "domain": (20,50,100,200)},
            {"name":'epsilon_decay_linear', "type": "discrete", "domain": (self.params['episodes']*0.2,
                                                                           self.params['episodes']*0.4,
                                                                           self.params['episodes']*0.6,
                                                                           self.params['episodes']*0.8,
                                                                           self.params['episodes']*1)}
        ]

        bayes_optimizer = BayesianOptimization(f=optimize,
                                               domain=optim_params,
                                               initial_design_numdata=6,
                                               acquisition_type="EI",
                                               exact_feval=True,
                                               maximize=True)

        bayes_optimizer.run_optimization(max_iter=20)
        print('Optimized learning rate: ', bayes_optimizer.x_opt[0])
        print('Optimized first layer: ', bayes_optimizer.x_opt[1])
        print('Optimized second layer: ', bayes_optimizer.x_opt[2])
        print('Optimized third layer: ', bayes_optimizer.x_opt[3])
        print('Optimized epsilon linear decay: ', bayes_optimizer.x_opt[4])
        return self.params


##################
#      Main      #
##################
if __name__ == '__main__':
    # Define optimizer
    bayesOpt = BayesianOptimizer(params)
    bayesOpt.optimize_RL()