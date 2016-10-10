def DerandomizedMutation(x,sigma,E,z,n):
        # E = global step-size
        # z = N(0,I) = mutation steps
        # d = sqrt(n)
        # sigma = vector of step-sizes and/or standard deviations

        d=np.sqrt(n)
        xRes=x+np.exp(E)*np.dot(sigma,z,out=None)
        sigmaRes=np.dot(sigma , np.power(np.exp(np.norm(z)/np.mean(E*3) -1), 1/n),out=None)*np.power(np.exp(E),1/d)
        
        return xRes, sigmaRes