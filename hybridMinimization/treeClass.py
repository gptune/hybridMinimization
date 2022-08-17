# Setting up the MCTS part, the ultimate goal is to 
# Python program to for tree traversals
from scipy.stats import rankdata
import numpy as np
np.seterr(divide='ignore')
np.seterr(invalid='ignore')
#Used for flattening lists
import itertools
flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]

import matplotlib.pyplot as plt
from  matplotlib.lines import Line2D
print('Python treeClassPlus, by Hengrui Luo. Version: 2022-Jan-19')
# A class that represents an individual node in a
# general tree

def setupTree(categorical_list,policy_list=None,update_list=None,default_reward=0,exploration_probability=0.05,print_tree=False, VERBOSE=False):
    #Ideally, we do not have to generate the whole tree but just span as needed. Currently, such a data structure is work-in-progress.
    if policy_list == None:
        policy_list = ['UCTS']*len(categorical_list)
    if update_list == None:
        update_list = ['UCTS']*len(categorical_list)
    if len(policy_list)!= len(categorical_list) or len(update_list)!= len(categorical_list):
        raise ValueError('The lengths of categorical list and policy_list and update_list must be equal.')
    myroot = Node(key=0,reward=default_reward,depth=0,n_visit=0,search_policy=policy_list[0],update_strategy=update_list[0],random_prob=exploration_probability)
    node_counter = 1
    layer_all = [[]]*len(categorical_list)
    #print( layer_all )
    for l in range(len(categorical_list)):
        #layer_all = []
        if l==0:
            for current_layer_node in categorical_list[l]:
                current_node1 = Node(key=node_counter,reward=default_reward,depth=l,n_visit=0,search_policy=policy_list[l],update_strategy=update_list[l],random_prob=exploration_probability)
                current_node1.setWord([current_layer_node])
                myroot.appendChildNode(current_node1)
                node_counter = node_counter + 1
                layer_all[l].append(current_node1)
            #myroot.printTree()
            if VERBOSE: print('layer 0',layer_all)
        else:
            if VERBOSE: print('layer ',l)
            tmp_layer_all = []
            for previous_layer_node in layer_all[l-1]:
                if VERBOSE: print('>Handling parent node',previous_layer_node.word)
                for current_layer_node in categorical_list[l]:
                    current_node2 = Node(key=node_counter,reward=default_reward,depth=l,n_visit=0,search_policy=policy_list[l],update_strategy=update_list[l],random_prob=exploration_probability)
                    tmp_word = previous_layer_node.word.copy()
                    tmp_word.append(current_layer_node)
                    if VERBOSE: print('>>Handling parent node with key ',previous_layer_node.key,'>>>',tmp_word,'<-',previous_layer_node.word,'+',current_layer_node)
                    current_node2.setWord(tmp_word)
                    previous_layer_node.appendChildNode(current_node2)
                    node_counter = node_counter + 1
                    tmp_layer_all.append(current_node2) 
            layer_all[l] = tmp_layer_all.copy()
    size_list = [len(layer_all[l]) for l in range(len(categorical_list))]
    print(myroot.maxDepth(),'Layers, with size list: ',size_list)
    if print_tree:myroot.printTree()
    return myroot
    
def treeFig(myroot,terminalkey=1,largestCatN=2):
    fig1, ax1 = plt.subplots(figsize=(5,5))
    myroot.plotTree(ax1)
    myroot.plotPath(terminalkey,ax1)
    width = largestCatN**(myroot.maxDepth()+myroot.depth)/2
    plt.xlim(myroot.plotx-1*width,myroot.plotx+1*width)
    plt.ylim(myroot.maxDepth()+myroot.depth+1,myroot.depth-1)
    plt.ylabel('depth')
    plt.xticks([], [])
    plt.title('from node with key '+str(myroot.key)+' to node with key '+str(terminalkey))
    plt.show()
    return ax1

# A class that represents an individual node in a
# Binary Tree
class Node:
    def __init__(self,key,reward = -np.inf,depth=0,n_visit=0,
                 search_policy='UCTS',update_strategy='UCTS',random_prob=0.05):
        self.parent = None
        self.children = []
        
        self.key = key
        self.word = []
        self.depth = depth
        
        self.search_policy = search_policy
        self.update_strategy = update_strategy
        
        self.n_visit = n_visit
        self.dist_param = []
        self.reward_history = []
        self.reward_history.append(reward)
        self.random_prob = random_prob
        self.pooled = False
        
        self.plotx = 0
        
        self.VERBOSE = False
            
    def searchKey(self,mykey):
        if self.key==mykey:
            return [self]
        else:
            res_list = flatten_list([cs.searchKey(mykey) for cs in self.children])
            return res_list
            
    def searchWord(self,myword):
        if self.word==myword or myword in self.word:
            return [self]
        else:
            res_list = flatten_list([cs.searchWord(myword) for cs in self.children])
            return res_list
        
    def setDistParam(self,param_list):
        self.dist_param = param_list
        
    def appendChildNode(self,node):
        node.parent = self
        node.updateDepth()
        self.children.append(node)
        #Initialize the DIrichlet base measure
        if self.search_policy=='Multinomial':
            self.dist_param = np.ones((1,len(self.children) ))#*1e-12
        if self.search_policy=='EXP3':
            self.dist_param = [1]*len(self.children)
        
    def get_reward_history(self):
        full_list = [s for s in self.reward_history if not np.isinf(s)]
        if len(full_list)==0:
            full_list = self.reward_history
        return full_list
        
    def siblings(self):
        #private interior function to get siblings.
        tmp_list = self.parent.children
        tmp_idx = [x!=self for x in tmp_list]
        filtered_list = [tmp_list[i] for i in tmp_idx if i]
        return filtered_list
        
    def printTree(self,VERBOSE=False):
        print('---'*self.depth,self.key, ' visited:',self.n_visit,'(',self.search_policy,';', sep = '',end='')
        print(self.dist_param,')(coin=',self.random_prob,')', sep = ''),
        if VERBOSE: 
            print('---'*self.depth,self.key, ' visited:',self.n_visit,' reward_history:',self.get_reward_history(),'(',self.search_policy,';',self.dist_param,')', sep = ''),
        for children_node in self.children:
            children_node.printTree(VERBOSE)
    
    def update_plotx(self):
        #my_x = self.parent.children.index(self)
        layer_w = len(self.parent.children)
        layer_w_max = np.log(self.getRoot().countLeaf()).astype(int) +1
        #layer_w = np.arange(layer_w)
        #if len(layer_w) % 2 == 0:
        #    layer_w = np.array(layer_w)+1
        padding = [ch.plotx for ch in self.parent.children]
        my_m = np.mean(padding)
        #my_x = ( (my_x-np.mean(padding) ) )*2**(self.getRoot().maxDepth()+1)/2**(self.depth) + self.parent.plotx
        my_x = self.parent.children.index(self)
        my_x = (my_x-1)*layer_w_max**(self.getRoot().maxDepth()+1)/layer_w_max**(self.depth+1)+ self.parent.plotx
        self.plotx = my_x
        
    def plotTree(self,ax):
        #if self.isLeaf():
        #    return
        if self.isRoot():
            self.plotx = 0
            ax.annotate(self.key,(self.plotx,self.depth))
        else:
            self.update_plotx()
            ax.annotate(self.key,(self.plotx,self.depth))
        for ch in self.children:
            ch.update_plotx()
            x1, y1 = self.plotx, self.depth
            x2, y2 = ch.plotx, ch.depth
            l = Line2D([x1, x2], [y1, y2])
            ax.add_line(l)
            #ax.plot(x1, y1, x2, y2, marker = 'o')
            ch.plotTree(ax)
            
    def plotPath(self,targetKey,ax):
        mynode = self.getRoot().searchKey(targetKey)
        if len(mynode)<=0:
            print('No such path from node ',self.key,' to node ',targetKey,' exists!')
            return None
        myL = self.pathTo(mynode[0],pathL=[])
        #(myL)
        for ct in range(len(myL)-1):
            node1 = myL[ct]
            node2 = myL[ct+1]
            if node1 is not None:
                if not node1.isRoot(): 
                    node1.update_plotx()
            if node2 is not None: 
                if not node2.isRoot(): 
                    node2.update_plotx()
            x1, y1 = node1.plotx, node1.depth
            x2, y2 = node2.plotx, node2.depth
            l = Line2D(xdata=[x1, x2], ydata=[y1, y2],c='r')
            ax.add_line(l)
        
    def maxDepth(self):
        if self.isLeaf():
            return 0
        else:
            tmp_list = [s.maxDepth() for s in self.children]
            return np.max(tmp_list)+1
    
    def updateDepth(self):
        if self.isRoot():
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        for children_node in self.children:
            children_node.updateDepth()
            
    def appendReward(self,new_value):
        self.reward_history.append(new_value)
        
    def backPropagate(self):
        #Conduct an update strategy from here.
        self.n_visit = self.n_visit + 1
        if self.isRoot():
            #print('Backprpagation reaches the root Node.')
            return
        else:

            if self.parent.update_strategy=='UCTS' or self.parent.update_strategy=='UCTS_var':
                #UCTS update
                self.parent.appendReward(self.reward_history[-1])
                #self.parent.appendReward(1) #0-1 test
            elif self.parent.update_strategy=='EXP3':
                #EXP3 update
                self.parent.appendReward(self.reward_history[-1])
                #self.parent.appendReward(1) #0-1 test
                observed_reward = self.reward_history[-1]
                parent_weights = np.asarray(self.parent.dist_param)#.reshape(1,-1)
                parent_gamma = self.parent.random_prob
                #print('Parent Check',parent_weights,parent_gamma)
                parent_pis = (1-parent_gamma)*parent_weights/np.sum(parent_weights) + parent_gamma*(1/len(parent_weights))
                #print(parent_pis)
                parents_childrenkey_list = [s.key for s in self.parent.children]
                self_idx_aschild = np.where(np.asarray(parents_childrenkey_list) == self.key)
                self_idx_aschild = int(self_idx_aschild[0])
                #print('Format Check',self_idx_aschild,parents_childrenkey_list,self.key)
                #print('Key Check', self.parent.children[self_idx_aschild].key,self.key,self_idx_aschild)
                self_pis = parent_pis[self_idx_aschild]
                estimate_reward = observed_reward/self_pis
                exp3_multiplier = np.exp(parent_gamma*estimate_reward/len(parent_weights))
                self.parent.dist_param[self_idx_aschild] = self.parent.dist_param[self_idx_aschild]*exp3_multiplier
                #print('EXP3 push:',self.dist_param)
            elif self.parent.update_strategy=='Multinomial' :
                #Dirichlet-Multinomial update
                self.parent.appendReward(self.reward_history[-1])
                #self.parent.appendReward(1) #0-1 test
                tmp_list = [np.mean(s.get_reward_history()) for s in self.parent.children]
                tmp_vist = [s.totalVisit()+1 for s in self.parent.children]
                #
                #tmp_sample = len(tmp_list) - rankdata(tmp_list).astype(int)
                #tmp_sample = rankdata(tmp_list).astype(int) - 1
                tmp_sample = (tmp_list == np.max(tmp_list))
                tmp_sample = tmp_sample.astype(float)
                #print('???',self.parent.key,tmp_list,'>>',np.max(tmp_list),tmp_sample)
                if len(self.parent.dist_param)<=0:
                    self.parent.dist_param = ( (np.asarray(tmp_sample))*0.+1e-12)/(np.asarray(tmp_vist))
                else:
                    self.parent.dist_param = np.asarray(self.parent.dist_param) + \
                         (np.asarray(tmp_sample))/(np.asarray(tmp_vist))
                         #(np.asarray(tmp_sample)+2)/np.log(2+self.parent.n_visit)
                         #(np.asarray(tmp_sample)+1)/(1+self.parent.n_visit)
                         #(np.asarray(tmp_sample)+1)/np.log(2+self.parent.n_visit)
            self.parent.backPropagate()
        return
        
    def pathToRoot(self):
        #Conduct an path-to-root update strategy from here.
        pathL = []
        if self.isRoot():
            pathL = [self]
        else:
            pathL = pathL + self.parent.pathToRoot()
        return pathL
        
    def totalVisit(self):
        if self.isRoot():
            return self.n_visit
        else:
            return self.parent.totalVisit()
    
    def getWord(self):
        return self.word
    
    def setWord(self,new_word):
        self.word = new_word
    
    def getUCT(self,variance=False):
        if not variance:
            return np.mean(self.reward_history) + np.sqrt(2*np.log( self.totalVisit()+1 )/self.n_visit)
        else:
            return np.mean(self.reward_history) + np.sqrt(2*np.log( self.totalVisit()+1 )*np.var(self.reward_history))
    
    def countLeaf(self):
        if self.isLeaf():
            return 1
        else:
            num_leaf = 0
            for children_node in self.children:
                num_leaf = num_leaf + children_node.countLeaf()
            return num_leaf
        
    def countNode(self):
        if self.isLeaf():
            return 1
        else:
            num_node = 1
            for children_node in self.children:
                num_node = num_node + children_node.countNode()
            return num_node
    
    def isLeaf(self):
        return len(self.children)==0
    
    def isRoot(self):
        return self.parent is None
        
    def getRoot(self):
        if self.isRoot():
            return self
        else:
            return self.parent.getRoot()
    
    def getLeafKeys(self):
        Leaf_list = []
        if self.isLeaf():
            Leaf_list = [self.key]
        else:
            Leaf_list = [s.getLeafKeys() for s in self.children]
        return flatten_list(Leaf_list)    
        
    def pathTo(self,TargetNode,pathL=[]):
        #Note that the path is directed, we cannot back-trace to parent.
        if len(pathL)==0:
            pathL = [self]
        if self==TargetNode and self.isLeaf():
            if pathL[-1] is not self:
                pathL.append(self)
            return pathL
        elif not self.isLeaf():
            #pathL.append(self)
            for ch in self.children:
                pathL_ch = ch.pathTo(TargetNode,[])
                if TargetNode in pathL_ch:
                    pathL = pathL + pathL_ch
            return pathL               
        res_list = flatten_list([cs.pathTo(TargetNode,pathL) for cs in self.children])
        return res_list
        
    def get_depthk_nodes(self,k=0):
        res_list = []
        if k==0:
            res_list = [self]
        elif k==1:
            res_list = self.children
        else:
            for ch in self.children:
                res_list.append(ch.get_depthk_nodes(k-1))
        return flatten_list(res_list)
            
    def removeMyself(self):
        #if this node is a leaf
        if self.isLeaf():
            my_ind = self.parent.children.index(self)
            self.parent.children.pop(my_ind)
            #After removal, if the parent becomes a new leaf, recursively remove the parent as well.
            if self.parent.isLeaf():
                self.parent.removeMyself()
            self.getRoot().updateDepth()
            del self
        else:
        #if this node is an internal
            #disconnect this node
            my_ind = self.parent.children.index(self)
            self.parent.children.pop(my_ind)
            #connect all its children to its parent
            myself_parent = self.parent
            ch_ind = my_ind
            for ch in self.children:
                ch.depth = self.depth
                ch.parent = myself_parent
                self.parent.children.insert(ch_ind,ch)
                ch_ind = ch_ind + 1
            self.getRoot().updateDepth()
            del self
        
            
    def findBestLeaf(self):
        #print(self.key)
        #print([k.key for k in self.children],self.isLeaf(),self.key,self.search_policy)
        if self.isLeaf():
            return self
        else:
            coin = np.random.uniform()
            #print('?????',coin)
            if coin < self.random_prob:
                children_idx = np.random.randint(low=0, high=len(self.children),size=None)
                if self.VERBOSE==True and self.isRoot():
                    print('Random selection ',children_idx,'-th child, root node -> node ',self.key,'.')
                elif self.VERBOSE==True:
                    print('Random selection ',children_idx,'-th child, node',self.parent.key,'-> node ',self.key,'.')
            else:
                if self.search_policy == 'UCTS':
                    children_UCT = [s.getUCT(variance=False) for s in self.children]
                    children_idx = np.argmax(children_UCT)
                    if self.VERBOSE:print('treeClass:UCTS>>>Children UCT: ',children_UCT)
                if self.search_policy == 'UCTS_var':
                    children_UCT = [s.getUCT(variance=True) for s in self.children]
                    children_idx = np.argmax(children_UCT)
                    if self.VERBOSE:print('treeClass:UCTS_var>>>Children UCT: ',children_UCT)
                if self.search_policy == 'EXP3':
                    #variant of exponential weighting.
                    
                    weights = np.asarray(self.dist_param)#.reshape(1,-1)
                    gamma = self.random_prob
                    pis = (1-gamma)*weights/np.sum(weights) + gamma*(1/len(weights))
                    if self.VERBOSE: print(pis)
                    pis = pis/np.sum(pis)
                    children_idx = np.random.multinomial(n=1,pvals=pis, size=1) 
                    if self.VERBOSE:print(children_idx,self.dist_param)
                    children_idx = np.argmax(children_idx)
                    #print('?????',[k.key for k in self.children],self.isLeaf(),self.key,children_idx)
                    if self.VERBOSE:print('treeClass:EXP3>>>Children probabilities, sample: ',np.round(pis,3))
                if self.search_policy == 'Multinomial':
                    alpha_measure = np.asarray(self.dist_param).reshape(1,-1)
                    alpha_measure = alpha_measure[0]
                    #print('Multinomial',alpha_measure)
                    a_sample_of_children_prob = np.random.dirichlet(alpha_measure/np.sum(alpha_measure),1)
                    a_sample_of_children_prob = a_sample_of_children_prob[0]

                    children_idx = np.random.multinomial(n=1,pvals=a_sample_of_children_prob, size=1) 
                    #size=1 means we observe 1 multi-nomial sample.
                    children_idx = np.argmax(children_idx)
                    #alpha_updater = np.asarray(alpha_measure)*0.
                    #alpha_updater[children_idx] = 1.
                    #alpha_measure = alpha_measure + alpha_updater 
                    if self.VERBOSE:print('treeClass:Multinomial>>>Children probabilities, sample: ',np.round(a_sample_of_children_prob,3))
                if self.isRoot():
                    if self.VERBOSE:print('treeClass:',self.search_policy,' selection ',children_idx,'-th child, root node -> node ',self.key,'.')
                else:
                    if self.VERBOSE:print('treeClass:',self.search_policy,' selection ',children_idx,'-th child, node',self.parent.key,'-> node ',self.key,'.')
            return self.children[children_idx].findBestLeaf()
