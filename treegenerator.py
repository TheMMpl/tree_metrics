import numpy as np
from sklearn import datasets
from scipy import sparse
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque
from scipy.sparse import dok_matrix
from scipy.sparse import linalg
from scipy.sparse import csgraph

class classicNode:
    def __init__(self,set_id,point_id,parent_set,parent_point):
        self.set_id=set_id
        self.point_id=point_id
        self.parent_set=parent_set
        self.parent_point=parent_point
        self.children=None

# ok lukier syntaktyczny, jeśli działa
@dataclass
class Node:
    set_id : int
    point_id :  int | None
    parent_set : int | None
    parent_point : int | None
    point_class : int | None
    dist : float
    children : list
    members : set

class Tree:
    def __init__(self,points,labels,delta,k):
        self.points=points
        self.p = np.random.permutation(points)
        self.radius=np.random.uniform(0.5,1)
        self.labels=labels
        self.depth=k#labels[self.p[0]]
        self.root=Node(0,None,None,None,None,self.radius*delta,[],{i for i in range(self.points)})
        self.nodes=[[] for i in range(self.depth)]



    def node_generation(self,X : np.ndarray):
        # we operate only on point indices - and find them in X
        self.nodes[0].append(self.root)
        #precomputing the values which don't require O(n^2) space
        currid=1
        squares=np.sum(X*X,axis=1)
        for layer in range(self.depth):
            i=self.depth-layer
            radius=(2**i)*self.radius
            #tych będzie liniowo - a nie logarytmicznie
            #w subsets każdy punkt pojawi się raz
            # a poziomów jest logn
            # czyli duże O(n^2logn)
            for node in self.nodes[layer]:
                
                #to jest część kwadratowa
                for index in tqdm(self.p):
                    #liczymy wiele razy te same odległości, ale cała macierz odległości nie mieści się w pamięci dla mnista
                    #+- reshape (teraz już k - wsszystkk ppieienny byc 700)- tu  zadziala brtoadcasting
                    ##n*dim dla tedo np.dot
                    dists=np.sqrt(squares-2*np.dot(X,X[index])+squares[index])
                    #slow python - for thios we probably need to swich language or look for optimizations
                    #liniowe
                    ball={j if dists[j]<radius else index for j in range(self.points)}
                    #spr jak działania na setach chodzą
                    intersection=node.members & ball
                    if intersection and layer<self.depth-2:
                        node.members-=intersection
                        self.nodes[layer+1].append(Node(currid,None,node.set_id,None,None,radius,[],intersection))
                        node.children.append(currid)
                        currid+=1
                        #ew 1 więcej layer
                    elif intersection and layer==self.depth-2:
                        node.members-=intersection
                        self.nodes[layer+1].append(Node(currid,index,node.set_id,None,None,radius,[],intersection))
                        node.children.append(currid)
                        currid+=1

    def testing_printout(self):
        for nodelist in self.nodes:
            for node in nodelist:
                print(node)
            print('################################')

    def naiive_generation(self,X : np.ndarray):
        # we operate only on point indices - and find them in X
        sets=[[] for i in range(self.depth)]
        #self.nodes[0].append(self.root)
        #sets.append({X[i] for i in self.p})
        #precomputing the values which don't require O(n^2) space
        squares=np.sum(X*X,axis=1)
        for layer in tqdm(range(self.depth)):
            i=self.depth-layer
            radius=i*self.radius
            #tych będzie liniowo - a nie logarytmicznie
            #w subsets każdy punkt pojawi się raz
            # a poziomów jest logn
            # czyli duże O(n^2logn)
            for subset in sets[layer]:
                
                #to jest część kwadratowa
                for index in self.p:
                    #liczymy wiele razy te same odległości, ale cała macierz odległości nie mieści się w pamięci dla mnista
                    #+- reshape (teraz już k - wsszystkk ppieienny byc 700)- tu  zadziala brtoadcasting
                    dists=np.sqrt(squares-2*np.dot(X,X[index])+squares[index])
                    #slow python - for thios we probably need to swich language or look for optimizations
                    ball={j if dists[j]<radius else index for j in range(self.points)}
                    intersection=subset & ball
                    subset-=intersection
                    sets[layer+1].append(intersection)

    def benchmark_generation(self,X : np.ndarray):

        squares=np.sum(X*X,axis=1)
        for layer in tqdm(range(self.depth)):
            i=self.depth-layer
            radius=i*self.radius

            for index in tqdm(self.p):
                dists=np.sqrt(squares-2*np.dot(X,X[index])+squares[index])

            


    def transform(self):
        pass

@dataclass
class np_Node:
    set_id : int
    point_id :  int | None
    parent_set : int | None
    parent_point : int | None
    point_class : int | None
    dist : float
    children : list
    #if we are to build up children may not be necessary
    #children : np.ndarray
    members : np.ndarray

@dataclass
class PointNode:
    #set_id : int
    point_id :  int
    #parent_set : int | None
    parent_point : int
    children : np.ndarray
    #not now
    #point_class : int | None
    dist : float


class np_Tree:
    def __init__(self,points,labels,delta,k):
        self.points=points
        self.p = np.random.permutation(points)
        self.radius=np.random.uniform(0.5,1)
        self.labels=labels
        self.depth=k#labels[self.p[0]]
        self.root=np_Node(0,None,None,None,None,self.radius*delta,[],np.array([i for i in range(points)]))
        self.nodes=[[] for i in range(self.depth)]

    def node_generation(self,X : np.ndarray):
        # we operate only on point indices - and find them in X
        self.nodes[0].append(self.root)
        #precomputing the values which don't require O(n^2) space
        
        squares=np.sum(X*X,axis=1)
        for layer in range(self.depth):
            i=self.depth-layer
            #for now heuristic while distances are not worked out, not to waste time
            radius=(2**(i-2))*self.radius
            currid=0
            for node in self.nodes[layer]:
                #this will break indexing unless another array is used to translate the indices
                if node.members.size==1 and layer<=self.depth-2:
                    self.nodes[layer+1].append(np_Node(currid,node.members[0],node.set_id,None,None,radius,[],node.members))
                    currid+=1
                else:
                    localsquares=squares[node.members]
                    localX=X[node.members]
                    print(localsquares.shape)
                    print(localX.shape)
                    #to jest część kwadratowa
                    for index in tqdm(self.p):
                        #some numerical errors cause negative valuses when the true distance is 0 (so abs)
                        # we can view oly the distances from members, though we still need to iterate or te points
                        dists=np.sqrt(np.abs(localsquares-2*np.dot(localX,X[index])+squares[index]))
                        ball=np.flatnonzero(dists<radius)
                        intersection=np.intersect1d(node.members,ball,assume_unique=True)
                        
                        if intersection.size>0 and layer<self.depth-2:
                            node.members=np.setdiff1d(node.members,intersection,assume_unique=True)
                            #we could append nodes themselves to children
                            self.nodes[layer+1].append(np_Node(currid,None,node.set_id,None,None,radius,[],intersection))
                            node.children.append(currid)
                            currid+=1
                            #ew 1 więcej layer
                        elif intersection.size>0 and layer==self.depth-2:
                            node.members=np.setdiff1d(node.members,intersection,assume_unique=True)
                            self.nodes[layer+1].append(np_Node(currid,index,node.set_id,None,None,radius,[],intersection))
                            node.children.append(currid)
                            currid+=1

    def node_generation_with_children(self,X : np.ndarray):
        # we operate only on point indices - and find them in X
        self.nodes[0].append(self.root)
        #precomputing the values which don't require O(n^2) space
        
        squares=np.sum(X*X,axis=1)
        for layer in range(self.depth):
            i=self.depth-layer
            #for now heuristic while distances are not worked out, not to waste time
            radius=(2**(i-2))*self.radius
            currid=0
            print(layer)
            for node in self.nodes[layer]:
                # print(node)
                #this will break indexing unless another array is used to translate the indices
                if node.members.size==1 and layer<=self.depth-2:
                    self.nodes[layer+1].append(np_Node(currid,node.members[0],node.set_id,None,None,radius,[],node.members))
                    #adding children here
                    currid+=1
                else:
                    original_members=node.members
                    localsquares=squares[original_members]
                    localX=X[original_members]
                    print(localsquares.shape)
                    print(localX.shape)
                    #to jest część kwadratowa
                    for index in tqdm(self.p):
                        #this wasn't an issue before as empty members would just give empty nonzero
                        #now we have to ensure node is nonempty
                        # but the members also change shape, so we would need to  view new arrays for each calculation
                        #by maintaining old_members this becomes unecessary
                        #if node.members.size>0:
                        #some numerical errors cause negative valuses when the true distance is 0 (so abs)
                        # we can view oly the distances from members, though we still need to iterate or te points
                        dists=np.sqrt(np.abs(localsquares-2*np.dot(localX,X[index])+squares[index]))
                        #ball=np.flatnonzero(dists<radius)
                        # print('###########')
                        # print(layer)
                        # print((dists<radius).shape)
                        # print(node.members.shape)
                        # print('#############')
                        ball=original_members[dists<radius]
                        #print(ball)
                        intersection=np.intersect1d(node.members,ball,assume_unique=True)
                        #print(intersection)
                        if intersection.size>0 and layer<self.depth-2:
                            node.members=np.setdiff1d(node.members,intersection,assume_unique=True)
                            #we could append nodes themselves to children
                            w=np_Node(currid,None,node.set_id,None,None,radius,[],intersection)
                            self.nodes[layer+1].append(w)
                            node.children.append(w)
                            currid+=1
                            #ew 1 więcej layer
                        elif intersection.size>0 and layer==self.depth-2:
                            #as everything is a pointer, w will be yupdated in both children and list
                            node.members=np.setdiff1d(node.members,intersection,assume_unique=True)
                            w=np_Node(currid,index,node.set_id,None,None,radius,[],intersection)
                            self.nodes[layer+1].append(w)
                            node.children.append(w)
                            currid+=1

    def testing_printout(self):
        for nodelist in self.nodes:
            for node in nodelist:
                print(node)
            print('################################')

    def transform(self):
        #+-index
        for layer in range(self.depth-2,-1,-1):
            #maintaining children makes this easier
            #we start one layer up - iffthe numbering is coreect, tiihu work
            for node in self.nodes[layer]:
                #this assumes children are nodes
                #if id were o reset every level, we could reconstruct the inheritance that way
                new_parent=node.children.pop()
                for child in node.children:
                    #some distance adjustment would be necessary - in the initial transform different children may have different distancces (it remains to be seen which version preforms better)
                    #what if we now think of dist as dist to parent instead of to all children - tat would work
                    child.parent_point=new_parent.point_id
                    new_parent.children.append(child)
                #new_parent.children=node.children
                #should be fine overall
                node=new_parent
            # parent=self.nodes[layer].pop()
            # if layer>0:
            #     self.nodes[layer-1][parent.parent_set]=parent
            # for node in self.nodes[layer]:
            #     node.parent_point=parent.point_id
    
    def create_matrix(self):
        G=dok_matrix((self.points,self.points),dtype=np.float32)
        #instead of BFS, we will travel layer by layer - no need to maintain pointers to children
        #this may chcnge if ittwould be more elegant
        for layer in range(self.depth):
            for node in self.nodes[layer]:
                for child in node.children:
                    #applying the dist fix here for now
                    G[node.point_id,child]=node.dist*2
                    G[child,node.point_id]=node.dist*2
        
        L=csgraph.laplacian(G,symmetrized=True)
        vals,vectors=linalg.eigs(L,k=3)
        # tovisit=deque()
        # tovisit.append(self.nodes[0][0])
        # while tovisit:
        #     node=tovisit.popleft()
        #     for child in node.children:
                



mnist = datasets.fetch_openml('mnist_784', version=1)
X, y = mnist.data.to_numpy(), mnist.target.astype('float32')
moduli=np.sum(X*X,axis=1)
diameter=2*np.sqrt(np.max(moduli))
delta=1
k=1
while delta<diameter:
    delta=delta*2
    k+=1
print('preprocessing complete')
# testtree=Tree(70000,y,delta,k)
# #testtree.benchmark_generation(X.astype('float32'))
# testtree.node_generation(X.astype('float32'))
# testtree.testing_printout()

testtree=np_Tree(70000,y,delta,k)
#testtree.benchmark_generation(X.astype('float32'))
testtree.node_generation_with_children(X.astype('float32'))
testtree.testing_printout()
testtree.transform()