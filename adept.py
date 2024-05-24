import scipy
from scipy.optimize import minimize
import scipy.stats
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from sksurv.nonparametric import kaplan_meier_estimator
from copy import deepcopy


class ADEPT(nn.Module):
    def __init__(self,
                 X_train, t_train, s_train,
                 X_validation, t_validation, s_validation,
                 sigmoid_temperature = 0.01, 
                 temperature_decay = True,
                 n_cutpoints = 1, 
                 iterations = 1000,
                 batch_size = 64,
                 regularization_strength = 1,
                 hidden_size = 32,
                 km_initialization = True,
                 weight_decay = 0,
                 learn_cutpoints=True,
                 seed = 1978):       
        
        super(ADEPT, self).__init__()
        torch.manual_seed(seed)
        random.seed(seed)

      
        self.X_train = torch.from_numpy(X_train).float()
        self.t_train = torch.from_numpy(t_train).float()
        self.s_train = torch.from_numpy(s_train).float()
        self.min_t = min(t_train)
        self.max_t = max(t_train)
        
        self.X_validation = torch.from_numpy(X_validation).float()
        self.t_validation = torch.from_numpy(t_validation).float()
        self.s_validation = torch.from_numpy(s_validation).float()

        # multiplier on prior
        self.regularization_strength = regularization_strength
        
        self.weight_decay = weight_decay
        
        self.sigmoid_temperature = sigmoid_temperature 
        self.temperature_decay = temperature_decay
        self.decay_iterations = [0]
        self.temperature_history = [self.sigmoid_temperature]
        
        self.n_cutpoints = n_cutpoints
        self.iterations = iterations
        self.batch_size = batch_size
        self.learn_cutpoints = learn_cutpoints
        
        self.train_loss = [0] * self.iterations
        self.validation_loss = [0] * self.iterations
            
                
        
        # set up parameters
        self.cutpoint0 =  self.cutpoint_init_percentile(self.n_cutpoints, self.t_train, self.s_train) if km_initialization else self.cutpoint_init_even(self.n_cutpoints)
        
        
        
        # if cut points are being learned add them to the list of parameters to optimize
        if self.learn_cutpoints:

            self.cutpoint_params = nn.ParameterList([nn.Parameter(torch.tensor(cutpoint), requires_grad=True) for cutpoint in self.cutpoint0])
            self.cutpoints = list(self.cutpoint_params)

        else:
            self.cutpoints = list(self.cutpoint0)


                
        p = self.X_train.shape[1]
        b = (n_cutpoints + 1)
        
        
        layer1 = torch.nn.Linear(p, hidden_size)
        layer2 = torch.nn.Linear(hidden_size, b)

        torch.nn.init.xavier_normal_(layer1.weight)
        torch.nn.init.xavier_normal_(layer2.weight)

        self.layers = torch.nn.Sequential(
                layer1,
                torch.nn.ReLU(),
                layer2,
                torch.nn.Softmax(dim=1)
            )

        self.layer_params = self.layers.parameters()
                
        self.best_layers = []
        self.best_cutpoints = []
        self.best_loss = 1e8
        self.best_iteration = 0

    
    # evenly spaced cutpoints
    def cutpoint_init_even(self, n_cutpoints):

        cutpoint0 = [ (1 + i) / (n_cutpoints + 1) for i in range(n_cutpoints)]
        
        return np.array(cutpoint0)
    
    
    def cutpoint_init_percentile(self, n_cutpoints, t, s):

        t_conv = (t - self.min_t) / (self.max_t - self.min_t)

        # get kaplan meier curve for observed outcomes
        x, y = kaplan_meier_estimator(s == 1, t_conv)

        indices = np.linspace(start=0, stop=len(y), num=n_cutpoints + 2,dtype=int)[1:-1]
        y_vals = y[indices]
        
        t_percentile = np.zeros(len(y_vals))
        
        for i in range(len(y_vals)):
            percentile = y_vals[i]
            diff_vec = np.absolute(percentile - y)
            index = diff_vec.argmin()
            t_percentile[i] = x[index]
                        
        return t_percentile
    
    
    
    
    def logit(self, x):
        return torch.log(x/ (1 - x))    
    
    
    def sigmoid(self, x, tau=1.):
        return 1 / (1 + torch.exp(-1 * x / tau))

    
    def softplus(self, x, tau=1., c=0.):
        return tau * torch.log(1 + torch.exp(x / tau)) + c
    
    
    def F(self, t, probs_tensor, cutpoints, smooth=False, temp=1e-2):
               

        t_tensor = torch.reshape(t, (len(t),1))
        
        cutpoints_tensor = torch.cat((torch.tensor([0]), torch.tensor(cutpoints), torch.tensor([1])))
    
        if smooth:
        
            progress = torch.subtract( t_tensor, cutpoints_tensor[:-1] ) / torch.diff(cutpoints_tensor)
            y_complete_bins = self.softplus(progress, tau=temp)
            
            y_complete_bins = self.softplus(-1 * (y_complete_bins - 1), tau=temp)

            y_complete_bins = 1 - y_complete_bins
        else:
    
           y_complete_bins = torch.maximum(
                    torch.minimum(
                         torch.subtract( t_tensor, cutpoints_tensor[:-1] ) / torch.diff(cutpoints_tensor),
                        torch.tensor(1)),
                    torch.tensor(0))

        val = 1 - torch.sum(y_complete_bins * probs_tensor, axis=1)
        
    
        return(val)
    
     
        
    def multinomial_loss(self, X, t, s, regularization_strength, cutpoint_bool):
        
        
        t_scaled = (t - self.min_t) / (self.max_t - self.min_t)

        likelihood = 0

        pred = self.predict(X)   
        cutpoints = self.cutpoints

        
        if all([i == 1 for i in cutpoint_bool]):
            cutpoints = self.cutpoints
        else:
            new_cutpoints = [0] * (sum(cutpoint_bool))
            counter = 0
            for i in range(self.n_cutpoints):
                if cutpoint_bool[i]:
                    new_cutpoints[counter] = self.cutpoints[i]
                    counter += 1



            new_pred = torch.zeros(pred.size()[0], sum(cutpoint_bool)+1)

            counter = 0
            new_pred[:,0] = pred[:,0]

            for i in range(len(cutpoint_bool)):
                if cutpoint_bool[i]:
                    counter += 1
                    new_pred[:,counter] = pred[:,i+1]
                else:
                    new_pred[:,counter] += pred[:,i+1]



            cutpoints = new_cutpoints
            pred = new_pred


        F_t= self.F( t_scaled, pred, cutpoints, smooth=True, temp=self.sigmoid_temperature)

        # normalize by interval length
        f_t = torch.divide(pred, 
                          torch.diff( 
                              torch.cat( ( torch.tensor([0]), torch.tensor(cutpoints), torch.tensor([1]) ) )  
                          ))


        left_boundary = torch.transpose(torch.stack([torch.sigmoid((t_scaled-lb)/ self.sigmoid_temperature) for lb in [0] + cutpoints]),0 ,1)
        right_boundary = torch.transpose(torch.stack([1-torch.sigmoid((t_scaled-rb)/ self.sigmoid_temperature) for rb in cutpoints + [1]]), 0, 1)

        ll_event = torch.log(torch.sum(f_t * left_boundary * right_boundary, axis=1))
        ll_event = torch.unsqueeze(s,0) * ll_event



        ll_censored = F_t
        ll_censored = torch.log(ll_censored)
        ll_censored =  (1 - torch.unsqueeze(s,0)) * ll_censored



        regularization = 0

        ll = ll_event + ll_censored
        nll = -1 * torch.mean(ll)
        approx_p = torch.mean(left_boundary * right_boundary, axis=0)

        tmp = [min(0, cutpoints[0])] + cutpoints + [max(1, cutpoints[-1])]
        interval_lengths = torch.Tensor([tmp[i] - tmp[i-1] for i in range(1,len(tmp))])


        for i in range(len(interval_lengths)):
            if interval_lengths[i] == 0:
                approx_p[i] = 0
            else:
                approx_p[i] = approx_p[i] / interval_lengths[i]



        # if prior
        if regularization_strength > 0:
             # prior loop
            prior = 0
            for lb, curr, rb in zip([0] + cutpoints[:-1], cutpoints, cutpoints[1:] + [1]):
                current_cutpoint = (curr - lb) / (rb - lb)

                if not self.learn_cutpoints:
                    current_cutpoint = torch.tensor(current_cutpoint)

                prior += -1 * regularization_strength * torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.5)).log_prob(current_cutpoint)
                regularization = prior

                nll += prior




        return nll, regularization

    
    # Wrapper to get just regularization from loss
    def calculate_entropy(self, X, t, s):
        cutpoint_bool = [1] * self.n_cutpoints

        return(self.multinomial_loss(X, t, s, False, 0, cutpoint_bool)[1])

    def train(self):                    
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay = self.weight_decay)

        
        loss = 0
        iteration_num = 0
        regularization = [0] * self.iterations
        cutpoint_bool = [1] * self.n_cutpoints

        
        
        ######################################
        # training iteration
        for iteration_num in range(self.iterations):
            
            
            
            if (iteration_num+1) % 10 == 0:
                print(f"iteration:\t{iteration_num+1} / {self.iterations}")
            # minibatching
            
            permutation = torch.randperm(self.X_train.shape[0])
        
            n_batch = 0
            for batch_num in range(0, self.X_train.shape[0], self.batch_size):
                n_batch += 1
                optimizer.zero_grad()

                indices = permutation[batch_num:batch_num+self.batch_size]
                
                batch_X = self.X_train[indices,:]
                batch_t = self.t_train[indices]
                batch_s = self.s_train[indices]
                
                
                    
                loss, _ = self.multinomial_loss(batch_X, batch_t, batch_s, self.regularization_strength, cutpoint_bool)
                
                loss.backward(retain_graph=True)

                optimizer.step()


                self.train_loss[iteration_num] += loss
                    
            # train loss = average train loss over minibatches
            self.train_loss[iteration_num] /= n_batch
            self.train_loss[iteration_num] = self.train_loss[iteration_num].item()
            self.validation_loss[iteration_num] = self.multinomial_loss(self.X_validation,
                                                                           self.t_validation,
                                                                           self.s_validation,
                                                                           self.regularization_strength,
                                                                           cutpoint_bool)[0].item()
            
            # if the loss from this epoch is the best loss, save the model and cut points
            if self.validation_loss[iteration_num] < self.best_loss:
                self.best_loss = deepcopy(self.validation_loss[iteration_num])
                self.best_layers = deepcopy(self.layers)
                self.best_cutpoints = deepcopy(self.cutpoints)
                self.best_iteration = iteration_num
                
                pred_validation = self.predict(self.X_validation).detach().numpy()
                bin_end_times = [x.item() for x in self.cutpoints] + [1]
                integrated = True

            if self.temperature_decay and \
                iteration_num > 4 and \
                (self.validation_loss[iteration_num - 1] < self.validation_loss[iteration_num]) and \
                (self.validation_loss[iteration_num - 2] < self.validation_loss[iteration_num - 1]) and \
                (self.validation_loss[iteration_num - 3] < self.validation_loss[iteration_num - 2]) :

                self.sigmoid_temperature /= 2
                self.decay_iterations.append(iteration_num)
                self.temperature_history.append(self.sigmoid_temperature)
                

        # end training iteration
        ######################################


        # set all parameters to best parameters
        print(f"Best validation loss achieved at iteration:\t{self.best_iteration}") 
        self.layers = deepcopy(self.best_layers)
        self.cutpoints = deepcopy(self.best_cutpoints)


    def predict(self, X_test):
        
        # if input is numpy array convert it to tensor
        is_numpy = isinstance(X_test, np.ndarray)
        if is_numpy:
            X_test = torch.from_numpy(X_test).float()
        
        pred = self.layers(X_test)

        
        
        # if input is numpy array convert output back to numpy array
        if is_numpy:
            pred = pred.detach().numpy()
        
        return pred
    
    
    
    
    
    