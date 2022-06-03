#Hengrui Luo
#hrluo@lbl.gov
#2021-11-25
import GPy
import numpy as np
from paramz.transformations import Logexp
from typing import Union
#This is the cluster-based kernel with overlapping penalty.
class PenalizedClusterKernel(GPy.kern.Kern):
    def __init__(self, input_dim, encoder=None,variance=1.0, active_dims=None,
                 name='PCK', overlap_penalty=0.0, label_mapping='None', num_clusters=None):
        super().__init__(input_dim, active_dims=active_dims, name=name)
        self.variance = GPy.core.parameterization.Param('variance',variance, Logexp())
        self.overlap_penalty = overlap_penalty                                                       
        self.link_parameter(self.variance)
        #Just link those trainable parameters.
        self.num_clusters = num_clusters
        #self.label_X = label_X
        #self.label_Y = label_Y
        self.data_dim = input_dim
        self.encoder = encoder
        #self.label_formatter = label_formatter
        if label_mapping == 'None':
            raise AttributeError('You need to supply a label mapping option(NO default), that takes categorical variables as input, and assign single value label as outputs.')
        else:
            self.label_mapping = label_mapping
            
    def K_label_mapping(self, X):
        return self.label_mapping.predict(X)
    
    def K(self, X, X2=None):
        X = np.asarray(X).reshape(-1,self.input_dim)   
        #
        if X2 is None:
            X2 = X
        #
        if self.encoder is not None:
            from dirty_cat import SimilarityEncoder, GapEncoder, MinHashEncoder
            #enc = SimilarityEncoder(similarity='levenshtein-ratio',hashing_dim=X.shape[1])
            #enc = GapEncoder(n_components=X.shape[1], random_state=42)
            #enc.partial_fit(X)
            #enc.partial_fit(X2)
            enc = self.encoder
            X = enc.transform(X.astype(str))[:,range(X.shape[1])]
            X2 = enc.transform(X2.astype(str))[:,range(X.shape[1])]
            #print('?dirty_cat',X.shape,':',X2.shape)
            
        X2 = np.asarray(X2).reshape(-1,self.input_dim)
        #X = np.asarray(X)
        X_label = self.label_mapping.predict(X)
        X2_label = self.label_mapping.predict(X2)
        #print('PCK',self.label_mapping,X_label,X2_label)
        # broadcasting approach
        diff = X_label[:, None] - X2_label[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats
        diff1 = np.logical_not(diff)
        #X12 = np.asarray(X_label!=0).astype(float)-np.asarray(X2_label!=0).astype(float)
        #X12 = np.abs(diff1)
        #print(np.sum(X12),'<<')
        k_cat = self.variance * np.sum(diff1, -1) / self.input_dim     
        return k_cat
        
    def Kdiag(self,X):
        return self.variance*np.ones(X.shape[0]) 

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance
        
class Lanczos(GPy.kern.src.stationary.Stationary):
    """
    Lanczos Kernel 
    https://en.wikipedia.org/wiki/Lanczos_resampling
    sinc[x-b]*sinc[x/a] inside -a<=x-b<=a
    """
    def __init__(self, input_dim, variance = 1.0, lengthscale=1.0, ARD=False, active_dims=None,
                 name='Lanczos'):
        super(GPy.kern.src.stationary.Stationary, self).__init__(input_dim, active_dims, name)
        self.ARD = ARD
        if not ARD:
            if lengthscale is None:
                lengthscale = np.ones(1)
            else:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size in [1, input_dim], "Bad number of lengthscales"
                if lengthscale.size != input_dim:
                    lengthscale = np.ones(input_dim)*lengthscale
            else:
                lengthscale = np.ones(self.input_dim)
        self.lengthscale = GPy.core.parameterization.Param('lengthscale', lengthscale, Logexp())
        self.variance = GPy.core.parameterization.Param('variance', variance, Logexp())
        assert self.variance.size==1
        self.link_parameters(self.variance, self.lengthscale)

    def K_of_r(self, r):  
        # Compute the distance between two 
        dist1 = r
        part1 = np.sinc( dist1 )
        part2 = np.sinc( dist1/self.lengthscale )
        part3 = dist1 
        part3[np.where(np.abs(part3)>=self.lengthscale)] = 0
        part3[np.where(np.abs(part3)<self.lengthscale)] = 1        
        #print(part1,part2,part3)
        k_val = part1*part2*part3*self.variance
        #k_val = np.sum(k_val)
        #print('vee')
        #if (X2 == X).all():
        #    k_val = 1.
        #return np.asarray(k_val).reshape(-1,self.input_dim)
        return k_val
        
    def dK_dr(self, r):
        return self.variance*(np.sinc(r)*(r*np.cos(r/self.lengthscale)-self.lengthscale*np.sin(r/self.lengthscale)) / (r*2) + np.sinc(r/self.lengthscale)*(r*np.cos(r)-np.sin(r))/(r*r) )
    
class Cosine(GPy.kern.src.stationary.Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Cosine'):
        super(Cosine, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.cos(r)


    def dK_dr(self, r):
        return -self.variance * np.sin(r)
        
#This is the kernel used by Kondor-Lafferty https://www.ml.cmu.edu/research/dap-papers/kondor-diffusion-kernels.pdf
class KLDiffusionKernel(GPy.kern.Kern):
    """
    Kernel that counts the number of categories that are the same
    between inputs and returns the normalised similarity score:
    k = variance * 1/N_c * (degree of overlap)
    """

    def __init__(self, input_dim, categorical_list,variance=1.0,lengthscale=0.1, active_dims=None,
                 name='KondorLaffertyDiffusion'):
        super().__init__(input_dim, active_dims=active_dims, name=name)
        self.variance = GPy.core.parameterization.Param('variance',
                                                        variance, Logexp())
        self.lengthscale = GPy.core.parameterization.Param('lengthscale',
                                                        lengthscale, Logexp())
        self.link_parameter(self.variance,self.lengthscale)
        #self.lengthscale = 1
        self.categorical_list = categorical_list
        self.catCount = [len(k) for k in self.categorical_list]
        self.catCount = np.asarray(self.catCount)
        
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # Counting the number of categories that are the same using GPy's
        # broadcasting approach
        diff = X[:, None] - X2[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats, this is also the Hamming distance 
        diff1 = np.logical_not(diff) #Hamming distance
        diff1 = np.asarray(diff1)
        diff2 = np.logical_not(diff1) #(n-Hamming) distance, K_H in section 5.
        # dividing by number of cat variables to keep this term in range [0,1]
        #lengthscale = 2
        numerator = 1-np.exp(-self.catCount*self.lengthscale)
        denominator = 1+(self.catCount-1)*np.exp(-self.catCount*self.lengthscale)
        k_cat = np.prod( (numerator/denominator)**diff1 )
        #k_cat = self.variance * np.sum(diff1, -1) / self.input_dim
        return k_cat

    def Kdiag(self,X):
        return self.variance*np.ones(X.shape[0]) 

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance
        #derivative of lengthscale \beta.
        diff = X[:, None] - X2[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats, this is also the Hamming distance 
        diff1 = np.logical_not(diff) #Hamming distance
        diff1 = np.asarray(diff1)
        diff2 = np.logical_not(diff1) #(n-Hamming) distance, K_H in section 5.
        # dividing by number of cat variables to keep this term in range [0,1]
        #lengthscale = 2
        numerator = 1-np.exp(-self.catCount*self.lengthscale)
        denominator = 1+(self.catCount-1)*np.exp(-self.catCount*self.lengthscale)
        #k_cat = self.variance * np.sum(diff1, -1) / self.input_dim
        
        prod1 = diff1*(numerator/denominator)**(diff1-1)
        prod2 = self.catCount*np.exp(-self.catCount*self.lengthscale)*denominator -\
                (1-self.catCount)*self.catCount*np.exp(-self.catCount*self.lengthscale)*numerator
        prod3 = (1/denominator)**2 #denominator

        #print(prod1*prod2*prod3)
        #print(prod1,prod2,prod3)

        diag1 = prod1*prod2*prod3
        #print(diag1.shape)
        pre_mat = np.tile((numerator/denominator)**diff1,reps=(len(self.catCount),1))
        #print(pre_mat)
        np.fill_diagonal(pre_mat,diag1)
        #print(pre_mat)
        post_mat = np.prod(pre_mat,axis=1)
        #print(post_mat)
        deri = np.sum(post_mat)
        #print(deri)
        self.lengthscale.gradient = np.sum(deri* dL_dK) 
        
#Below are the kernels used by CoCaBO https://github.com/rubinxin/CoCaBO_code/blob/b7af3102397945bdba7d5568c91cac1151ec90c0/utils/ml_utils/models/additive_gp.py#L299
class CategoryOverlapKernel(GPy.kern.Kern):
    """
    Kernel that counts the number of categories that are the same
    between inputs and returns the normalised similarity score:
    k = variance * 1/N_c * (degree of overlap)
    """

    def __init__(self, input_dim, encoder=None, variance=1.0, active_dims=None,
                 name='catoverlap'):
        super().__init__(input_dim, active_dims=active_dims, name=name)
        self.variance = GPy.core.parameterization.Param('variance',
                                                        variance, Logexp())
        self.link_parameter(self.variance)
        self.encoder = encoder

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #print('!COK',X.shape,':',X2.shape)
        if self.encoder == 'dirty_cat':
            from dirty_cat import SimilarityEncoder, GapEncoder, MinHashEncoder
            enc = SimilarityEncoder(similarity='levenshtein-ratio',hashing_dim=X.shape[1])
            #enc = GapEncoder(n_components=X.shape[1], random_state=42)
            #enc.partial_fit(X)
            #enc.partial_fit(X2)
            X = enc.fit_transform(X.astype(str))[:,range(X.shape[1])]
            X2 = enc.fit_transform(X2.astype(str))[:,range(X.shape[1])]
            #print('?dirty_cat',X.shape,':',X2.shape)
        # Counting the number of categories that are the same using GPy's
        # broadcasting approach
        diff = X[:, None] - X2[None, :]
        # nonzero location = different cat
        diff[np.where(np.abs(diff))] = 1
        # invert, to now count same cats
        diff1 = np.logical_not(diff)
        # dividing by number of cat variables to keep this term in range [0,1]
        k_cat = self.variance * np.sum(diff1, -1) / self.input_dim
        return k_cat
        
    def Kdiag(self,X):
        return self.variance*np.ones(X.shape[0]) 

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance
 
class MixtureViaSumAndProduct(GPy.kern.Kern):
    """
    Kernel of the form

    k = (1-mix)*(k1 + k2) + mix*k1*k2


    Parameters
    ----------
    input_dim
        number of all dims (for k1 and k2 together)
    k1
        First kernel
    k2
        Second kernel
    active_dims
        active dims of this kernel
    mix
        see equation above
    fix_variances
        unlinks the variance parameters if set to True
    fix_mix
        Does not register mix as a parameter that can be learned

    """

    def __init__(self, input_dim: int, k1: GPy.kern.Kern, k2: GPy.kern.Kern,
                 active_dims: Union[list, np.ndarray] = None, variance=1.0,
                 mix: float = 0.5,
                 fix_inner_variances: bool = False, fix_mix=True,
                 fix_variance=True):

        super().__init__(input_dim, active_dims, 'MixtureViaSumAndProduct')

        self.acceptable_kernels = (GPy.kern.RBF, GPy.kern.Matern52,
                                   CategoryOverlapKernel
                                   )

        assert isinstance(k1, self.acceptable_kernels)
        assert isinstance(k2, self.acceptable_kernels)

        self.mix = GPy.core.parameterization.Param('mix', mix, Logexp())
        self.variance = GPy.core.parameterization.Param('variance', variance,
                                                        Logexp())

        self.fix_variance = fix_variance
        if not self.fix_variance:
            self.link_parameter(self.variance)

        # If we are learning the mix, then add it as a visible param
        self.fix_mix = fix_mix
        if not self.fix_mix:
            self.link_parameter(self.mix)

        self.k1 = k1
        self.k2 = k2

        self.fix_inner_variances = fix_inner_variances
        if self.fix_inner_variances:
            self.k1.unlink_parameter(self.k1.variance)
            self.k2.unlink_parameter(self.k2.variance)

        self.link_parameters(self.k1, self.k2)

    def get_dk_dtheta(self, k: GPy.kern.Kern, X, X2=None):
        assert isinstance(k, self.acceptable_kernels)

        if X2 is None:
            X2 = X
        X_sliced, X2_sliced = X[:, k.active_dims], X2[:, k.active_dims]

        if isinstance(k, (GPy.kern.RBF, GPy.kern.Matern52)):
            dk_dr = k.dK_dr_via_X(X_sliced, X2_sliced)

            # dr/dl
            if k.ARD:
                tmp = k._inv_dist(X_sliced, X2_sliced)
                dr_dl = -np.dstack([tmp * np.square(
                    X_sliced[:, q:q + 1] - X2_sliced[:, q:q + 1].T) /
                                    k.lengthscale[q] ** 3
                                    for q in range(k.input_dim)])
                dk_dl = dk_dr[..., None] * dr_dl
            else:
                r = k._scaled_dist(X_sliced, X2_sliced)
                dr_dl = - r / k.lengthscale
                dk_dl = dk_dr * dr_dl

            # # For testing the broadcast multiplication
            # dk_dl_slow = []
            # for ii in range(dr_dl.shape[-1]):
            #     dr_dlj = dr_dl[...,ii]
            #     dk_dlj = dk_dr * dr_dlj
            #     dk_dl_slow.append(dk_dlj)
            #
            # dk_dl_slow = np.dstack(dk_dl_slow)

        elif isinstance(k, CategoryOverlapKernel):
            dk_dl = None

        else:
            raise NotImplementedError

        # Return variance grad as well, if not fixed
        if not self.fix_inner_variances:
            return k.K(X, X2) / k.variance, dk_dl
        else:
            return dk_dl

    def update_gradients_full(self, dL_dK, X, X2=None):

        # This gets the values of dk/dtheta as a NxN matrix (no summations)
        if X2 is None:
            X2 = X
        dk1_dtheta1 = self.get_dk_dtheta(self.k1, X, X2)  # N x N
        dk2_dtheta2 = self.get_dk_dtheta(self.k2, X, X2)  # N x N

        # Separate the variance and lengthscale grads (for ARD purposes)
        if self.fix_inner_variances:
            dk1_dl1 = dk1_dtheta1
            dk2_dl2 = dk2_dtheta2
            dk1_dvar1 = []
            dk2_dvar2 = []
        else:
            dk1_dvar1, dk1_dl1 = dk1_dtheta1
            dk2_dvar2, dk2_dl2 = dk2_dtheta2

        # Evaluate each kernel over its own subspace
        k1_xx = self.k1.K(X, X2)  # N x N
        k2_xx = self.k2.K(X, X2)  # N x N

        # dk/dl for l1 and l2
        # If gradient is None, then vars other than lengthscale don't exist.
        # This is relevant for the CategoryOverlapKernel
        if dk1_dl1 is not None:
            # ARD requires a summation along last axis for each lengthscale
            if hasattr(self.k1, 'ARD') and self.k1.ARD:
                dk_dl1 = np.sum(
                    dL_dK[..., None] * (
                            0.5 * dk1_dl1 * (1 - self.mix) * self.variance
                            + self.mix * self.variance * dk1_dl1 *
                            k2_xx[..., None]),
                    (0, 1))
            else:
                dk_dl1 = np.sum(
                    dL_dK * (0.5 * dk1_dl1 * (1 - self.mix) * self.variance
                             + self.mix * self.variance * dk1_dl1 * k2_xx))
        else:
            dk_dl1 = []

        if dk2_dl2 is not None:
            if hasattr(self.k2, 'ARD') and self.k2.ARD:
                dk_dl2 = np.sum(
                    dL_dK[..., None] * (
                            0.5 * dk2_dl2 * (1 - self.mix) * self.variance
                            + self.mix * self.variance * dk2_dl2 *
                            k1_xx[..., None]),
                    (0, 1))
            else:
                dk_dl2 = np.sum(
                    dL_dK * (0.5 * dk2_dl2 * (1 - self.mix) * self.variance
                             + self.mix * self.variance * dk2_dl2 * k1_xx))
        else:
            dk_dl2 = []

        # dk/dvar for var1 and var 2
        if self.fix_inner_variances:
            dk_dvar1 = []
            dk_dvar2 = []
        else:
            dk_dvar1 = np.sum(
                dL_dK * (0.5 * dk1_dvar1 * (1 - self.mix) * self.variance
                         + self.mix * self.variance * dk1_dvar1 * k2_xx))
            dk_dvar2 = np.sum(
                dL_dK * (0.5 * dk2_dvar2 * (1 - self.mix) * self.variance
                         + self.mix * self.variance * dk2_dvar2 * k1_xx))

        # Combining the gradients into one vector and updating
        dk_dtheta1 = np.hstack((dk_dvar1, dk_dl1))
        dk_dtheta2 = np.hstack((dk_dvar2, dk_dl2))
        self.k1.gradient = dk_dtheta1
        self.k2.gradient = dk_dtheta2

        # if not self.fix_mix:
        self.mix.gradient = np.sum(dL_dK *
                                   (-0.5 * (k1_xx + k2_xx) +
                                    (k1_xx * k2_xx))) * self.variance

        # if not self.fix_variance:
        self.variance.gradient = \
            np.sum(self.K(X, X2) * dL_dK) / self.variance

    def K(self, X, X2=None):
        k1_xx = self.k1.K(X, X2)
        k2_xx = self.k2.K(X, X2)
        return self.variance * ((1 - self.mix) * 0.5 * (k1_xx + k2_xx)
                                + self.mix * k1_xx * k2_xx)

    def gradients_X(self, dL_dK, X, X2, which_k=2):
        """
        This function evaluates the gradients w.r.t. the kernel's inputs.
        Default is set to the second kernel, due to this function's
        use in categorical+continuous BO requiring gradients w.r.t.
        the continuous space, which is generally the second kernel.

        which_k = 1  # derivative w.r.t. k1 space
        which_k = 2  # derivative w.r.t. k2 space
        """
        active_kern, other_kern = self.get_active_kernel(which_k)

        # Evaluate the kernel grads in a loop, as the function internally
        # sums up results, which is something we want to avoid until
        # the last step
        active_kern_grads = np.zeros((len(X), len(X2), self.input_dim))
        for ii in range(len(X)):
            for jj in range(len(X2)):
                active_kern_grads[ii, jj, :] = \
                    active_kern.gradients_X(
                        np.atleast_2d(dL_dK[ii, jj]),
                        np.atleast_2d(X[ii]),
                        np.atleast_2d(X2[jj]))

        other_kern_vals = other_kern.K(X, X2)

        out = np.sum(active_kern_grads *
                     (1 - self.mix + self.mix * other_kern_vals[..., None]),
                     axis=1)
        return out

    def gradients_X_diag(self, dL_dKdiag, X, which_k=2):
        active_kern, other_kern = self.get_active_kernel(which_k)
        if isinstance(active_kern, GPy.kern.src.stationary.Stationary):
            return np.zeros(X.shape)
        else:
            raise NotImplementedError("gradients_X_diag not implemented "
                                      "for this type of kernel")

    def get_active_kernel(self, which_k):
        if which_k == 1:
            active_kern = self.k1
            other_kern = self.k2
        elif which_k == 2:
            active_kern = self.k2
            other_kern = self.k1
        else:
            raise NotImplementedError(f"Bad selection of which_k = {which_k}")
        return active_kern, other_kern 
        
    def Kdiag(self,X):
        return self.variance * ((1 - self.mix) * 0.5 * (self.k1.Kdiag(X) + self.k2.Kdiag(X))
                                + self.mix * self.k1.Kdiag(X) * self.k2.Kdiag(X))

