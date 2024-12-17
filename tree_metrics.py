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
                    #print(localX.shape)
                    #to jest część kwadratowa
                    for index in tqdm(self.p):
                        #this wasn't an issue before as empty members would just give empty nonzero
                        #now we have to ensure node is nonempty
                        # but the members also change shape, so we would need to  view new arrays for each calculation
                        #by maintaining old_members this becomes unecessary
                        #if node.members.size>0:
                        #some numerical errors cause negative valuses when the true distance is 0 (so abs)
                        # we can view oly the distances from members, though we still need to iterate or te points
                        dists=np.sqrt(np.abs(localsquares-2*np.dot(localX,X[index])+squares[index]))/self.scaling
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
        print("transforming tree")
        for layer in range(self.depth-2,-1,-1):
            print(layer)
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
                    #now this gives information about thr distance to parents
                    child.dist=child.dist*2
                    new_parent.dist=new_parent.dist*2
                #new_parent.children=node.children
                #should be fine overall
                node=new_parent

    
    def create_matrix(self):
        G=dok_matrix((self.points,self.points),dtype=np.float32)
        #instead of BFS, we will travel layer by layer - no need to maintain pointers to children
        #this may chcnge if ittwould be more elegant
        print("generating matrix")
        for layer in range(self.depth):
            for node in self.nodes[layer]:
                for child in node.children:
                    #applying the dist fix here for now
                    G[node.point_id,child.point_id]=child.dist*2
                    G[child.point_id,node.point_id]=child.dist*2
        
        L=csgraph.laplacian(G,symmetrized=True)
        self.vals,self.vectors=linalg.eigs(L,k=3,which='SM')
        print(self.vals,self.vectors)

mnist = datasets.fetch_openml('mnist_784', version=1)
X, y = mnist.data.to_numpy(), mnist.target.astype('float32')
moduli=np.sum(X*X,axis=1)
diameter=2*np.sqrt(np.max(moduli))
#poentially watch out for this modlili buisness - although they all should be in the positive quadfrant, so i is to be expected that the max sistance will be smaller than them

def compute_dists(X, test_X, diameter, batch_size=800):
    mindist=diameter
    maxdist=0
    for i in tqdm(range(0, test_X.shape[0], batch_size)):
        batch_X = test_X[i:i + batch_size]
        #this wa we avoid compting  the distanes betweenthhemselves, but the disttances bettwween  thee batch aarere also absent
        print(X[:i].shape)
        print(X[i+batch_size:].shape)
        if i>0:
            train_X=np.vstack([X[:i],X[i+batch_size:]])
        else:
            train_X=X[i+batch_size:]
        #print(train_X.shape)
        dists =  np.sqrt(np.sum(train_X**2, axis=1).reshape(1,-1).T + np.sum(batch_X**2,axis=1) - 2*np.dot(train_X, batch_X.T))
        #print(dists)
        if np.min(dists)<mindist:
            print(mindist)
            mindist=np.min(dists)
        if np.max(dists)>maxdist:
            print(maxdist)
            maxdist=np.max(dists)

    return maxdist,mindist

diameter,scaling=compute_dists(X.astype('float32'),X.astype('float32'),diameter)
print(diameter,scaling)
diameter=diameter/scaling
print(diameter)
delta=1
k=1
while delta<diameter:
    delta=delta*2
    k+=1
print(delta,k)
print('preprocessing complete')
# testtree=Tree(70000,y,delta,k)
# #testtree.benchmark_generation(X.astype('float32'))
# testtree.node_generation(X.astype('float32'))
# testtree.testing_printout()

testtree=np_Tree(70000,y,delta,k,scaling)
#testtree.benchmark_generation(X.astype('float32'))
testtree.node_generation_with_children(X.astype('float32'))
testtree.testing_printout()
testtree.transform()
testtree.create_matrix()


# Example NumPy arrays
x_coords = testtree.vectrs[1]
y_coords = testtree.vectors[2]
labels = y # Labels signify the class of each point

# Create a scatter plot with points colored based on their labels
unique_labels = np.unique(labels)  # Get unique labels
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Generate distinct colors

for label, color in zip(unique_labels, colors):
    # Mask for points belonging to the current label
    mask = labels == label
    plt.scatter(x_coords[mask], y_coords[mask], color=color, label=f'Class {label}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mnist projection')
plt.legend()
plt.savefig("Mnist_projection.png", dpi=300, bbox_inches='tight')