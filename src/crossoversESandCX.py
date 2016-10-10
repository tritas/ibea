

    def recombinationESsearchPath(x,sigma,z,n,lamda,E):
        
        '''
        sigma = vector of step-sizes and/or standard deviations
        n =
        lamda = number of offspring, offspring population size
        E = global step-size = N(0,1)
        z = N(0,I) = mutation steps
        u = number of parents, parental population size
        s = search path or evolution path
        
        '''
        c= np.sqrt(u/(n+4))
        u= lamda/4
        d= 1+np.sqrt(u/n)
        di=3*n
        s=0
        s=(1-c)*s+np.sqrt(c*(2-c))*(np.sqrt(u)/u)*np.sum(z)
        sigma = np.dot(sigma,np.power(np.exp((np.norm(s)/np.mean(E))-1),1/di))*np.power(np.exp(np.norm(s)/np.mean(z) -1),c/d)
        xRes = (1/u)*np.sum(x)
        return xRes




    def cxSimulatedBinary(ind1, ind2, eta):
        """Executes a simulated binary crossover that modify in-place the input
         individuals. The simulated binary crossover expects :term:`sequence`
         individuals of floating point numbers.
         
         :param ind1: The first individual participating in the crossover.
         :param ind2: The second individual participating in the crossover.
         :param eta: Crowding degree of the crossover. A high eta will produce
         children resembling to their parents, while a small eta will
         produce solutions much more different.
         :returns: A tuple of two individuals.
         
         This function uses the :func:`~random.random` function from the python base
         :mod:`random` module.
         """
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            rand = random.random()
            if rand <= 0.5:
                beta = 2. * rand
            else:
                beta = 1. / (2. * (1. - rand))
                beta **= 1. / (eta + 1.)
                ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
                ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))
        
        return ind1, ind2




