import numpy as np
from sklearn import datasets
from scipy import sparse
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque
from scipy.sparse import dok_matrix
from scipy.sparse import linalg
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import pickle

@dataclass
class np_Node:
    set_id : int
    point_id :  int | None
    parent_set : int | None
    parent_point : int | None
    point_class : int | None
    dist : float
    children : list
    members : np.ndarray


class np_Tree:
    def __init__(self,points,labels,delta,k,scaling):
        self.points=points
        self.p = np.random.permutation(points)
        self.radius=np.random.uniform(0.5,1)
        self.labels=labels
        self.depth=k#labels[self.p[0]]
        self.root=np_Node(0,None,None,None,None,self.radius*delta,[],np.array([i for i in range(points)]))
        self.nodes=[[] for i in range(self.depth)]
        self.scaling=scaling
        self.state='default'
        print(self.radius)

    def node_generation_with_children(self,X : np.ndarray):
        # we operate only on point indices - and find them in X
        self.nodes[0].append(self.root)
        #precomputing the values which don't require O(n^2) space
        
        squares=np.sum(X*X,axis=1)
        #big oof
        for layer in range(self.depth-1):
            i=self.depth-layer
            #to be seen if this gives good approximations
            radius=(2**(i-1))*self.radius
            currid=0
            #print(layer)
            completed=0
            for node in self.nodes[layer]:
                # print(node)
                print(layer)
                completed+=node.members.size
                print(completed)
                #this will break indexing unless another array is used to translate the indices
                if node.members.size==1 and layer<=self.depth-2:
                    w=np_Node(currid,node.members[0],node.set_id,None,None,radius,[],node.members)
                    self.nodes[layer+1].append(w)
                    node.children.append(w)
                    currid+=1
                else:
                    original_members=node.members
                    localsquares=squares[original_members]
                    localX=X[original_members]
                    print(localsquares.shape)

                    for index in tqdm(self.p):

                        dists=np.sqrt(np.abs(localsquares-2*np.dot(localX,X[index])+squares[index]))/self.scaling
                        ball=original_members[dists<radius]
                        intersection=np.intersect1d(node.members,ball,assume_unique=True)

                        if intersection.size>0 and layer<self.depth-2:
                            node.members=np.setdiff1d(node.members,intersection,assume_unique=True)
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

        if self.state=='default':
            name='tree.txt'
        else:
            name='transformed_tree.txt'

        with open(name, mode="w") as file:

            for nodelist in self.nodes:
                for node in nodelist:
                    file.write(str(node))
                file.write('\n')
        print('printout complete')

    def transform(self):
        #+-index
        print("transforming tree")
        for layer in range(self.depth-2,-1,-1):
            print(layer)
            #maintaining children makes this easier
            #we start one layer up - iffthe numbering is coreect, tiihu work
            for node in self.nodes[layer]:
                #this assumes children are nodes
                #if id were o reset every level, we could reconstruct the inheritance that way
                print(node)
                new_parent=node.children.pop()
                for child in node.children:
                    #some distance adjustment would be necessary - in the initial transform different children may have different distancces (it remains to be seen which version preforms better)
                    #what if we now think of dist as dist to parent instead of to all children - tat would work
                    child.parent_point=new_parent.point_id
                    new_parent.children.append(child)
                    #now this gives information about thr distance to parents
                    child.dist=child.dist*2
                    new_parent.dist=new_parent.dist*2
                #new_parent.children=node.children
                #should be fine overall
                node=new_parent
        self.state='transformed'

    def create_matrix(self):
        G=dok_matrix((self.points,self.points),dtype=np.float32)
        #instead of BFS, we will travel layer by layer - no need to maintain pointers to children
        #this may chcnge if ittwould be more elegant
        print("generating matrix")
        #this being 1 larger shouldn't cause issues, but this is enough
        for layer in range(self.depth-1):
            for node in self.nodes[layer]:
                for child in node.children:
                    #applying the dist fix here for now
                    print(node.point_id,child.point_id)
                    G[node.point_id, child.point_id]=child.dist*2
                    G[child.point_id, node.point_id]=child.dist*2

        sparse.save_npz('graph',G) 
        L=csgraph.laplacian(G,symmetrized=True)
        sparse.save_npz('laplacian',L) 
        self.vals,self.vectors=linalg.eigs(L,k=3,which='SM')
        print(self.vals,self.vectors)


with open('transformed_tree.pkl','rb') as result:
    tree=pickle.load(result)

print(tree.nodes[0])

tree.transform()