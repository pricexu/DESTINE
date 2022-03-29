import random
import networkx as nx
import scipy as sp
from scipy import sparse
from scipy.sparse import find
from scipy.linalg import block_diag
from scipy.sparse.linalg import norm
import numpy as np
import random
import time
import argparse

parser = argparse.ArgumentParser(description='Hyperparameters.')
parser.add_argument('-p', type=float, help='the penalty weight of missing links')
parser.add_argument('-g', type=float, help='trade off parameters')

args = parser.parse_args()
ppp = args.p
gamma = args.g

# UseCR: from which iteration the cross-layer dependency regularization works
if gamma == 0:
	UseCR = 9999
else:
	UseCR = -1

Prob = 0.05 # regular cross-layer link probability
block_num = 10 # the total numebr of blocks
ProbN = 0.001 # cross-layer link probability between unmatched subgraphs

random.seed =123
insertedPosition = 0 # the index from which to insert the synthetic cliques
Asizes = [1800,2400,3000] # size of networks from three layers
SizeA = sum(Asizes)
cumsum_Asizes = np.cumsum([0]+Asizes)
cliqueSizes = [int(i/block_num) for i in Asizes] # in each layer, select one of the block as the clique

# update settings
iteration = 300
startArmijo = -1
learningRate = 10**(-5)
beta = 0.9
sigma = 0.01

As = []
groundTruths = []
ER = False # if ER, generate synthetic ER graphs, otherwise, generate synthetic SF graphs
if ER:
	ps = [0.1,0.2,0.6]
	for i,p_ in enumerate(ps):
		As.append(sparse.lil_matrix(nx.to_scipy_sparse_matrix(nx.erdos_renyi_graph(n=Asizes[i],p=p_,seed=1234))))
else:
	ms = [20,40,60]
	for i,m_ in enumerate(ms):
		As.append(sparse.lil_matrix(nx.to_scipy_sparse_matrix(nx.barabasi_albert_graph(n=Asizes[i],m=m_,seed=1234))))

def EvaluateF1(minedResult,groundTruth):
	# Input are two sets of nodes.
	tmp = groundTruth&minedResult

	if len(minedResult) == 0:
		precision = 0
	else:
		precision = len(tmp)/len(minedResult)
	recall = len(tmp)/len(groundTruth)

	if precision+recall == 0:
		f1 = 0
	else:
		f1 = 2*precision*recall/(precision+recall)
	return precision,recall,f1


def Destine(iteration,p,learningRate,A,CR,CRR,groundTruths,As):

	# iteration: number of update iterations
	# p: the penalty weight of missing links
	# learningRate: the learning rate
	# A: the aggregated diagnoal block adjacency matrix
	# CR: Hadamard between the C matrix and R matrix (Check Eq. 10)
	# CRR: Hadamard between the CR matrix and R matrix (Check Eq. 10)
	# groundTruths: position of inserted cliques
	# As: the list of adj matrix from every layer

	# the project function used in PGD
	def projectetBack(v,lowb,upperb):
		tmp1 = np.where(v>upperb,upperb,v)
		tmp2 = np.where(tmp1<lowb,lowb,tmp1)
		return tmp2

	ss = []
	mu, sigmaa = 0.5, 0.01
	for G in As:

		################### Three intializations of the selection vector from every layer #################

		# gaussian initialization
		# s = np.random.normal(mu, sigmaa, (G.shape[0],1))

		# uniform initialization
		# s = 0.01*np.ones((G.shape[0],1))

		# degree initialization
		s = G.sum(axis=1)/G.sum(axis=1).max()

		ss.append(s)


	# The concatenated selection vector
	S = np.concatenate(ss,axis=0)
	one = np.ones(S.shape)

	bestF1 = 0
	for it in range(iteration):
		ss = np.split(S,np.cumsum(Asizes)[:-1],axis=0)

		# the following is a fast computation of Eq. 9
		tmpss = []
		for s in ss:
			tmpss.append(np.linalg.norm(s,1)*np.ones(s.shape))
		HatS = np.concatenate(tmpss,axis=0)
		gradient_Ld = 2*p*(HatS-S) - 2*(1+p)*A.dot(S)
		gradient_total = gradient_Ld


		# Compute the existing loss for Armijo search of the learning rate
		tmp_loss = 0
		for s in ss:
			tmp_loss += np.linalg.norm(s, 1)**2-np.linalg.norm(s, 2)**2
		Within_loss = p*tmp_loss - (1+p)*S.transpose().dot(A.dot(S))
		Loss_old = Within_loss
		gradient_Lr = 0

		if it > UseCR:
			# if we adopt the cross-layer dependency regularization
			gradient_Lr = np.multiply(S,CRR.dot(8*np.multiply(S,S)+2*one-8*S)) + CRR.dot(2*S-4*np.multiply(S,S))
			gradient_total += gradient_Lr

			row_index, col_index, data_ = find(CR)
			oneS = one-S
			CR_loss = np.linalg.norm(data_ - np.multiply(data_, (np.multiply(S[row_index], S[col_index])+np.multiply(oneS[row_index], oneS[col_index])).flatten()), 2)**2
			Loss_old += CR_loss

		if it > startArmijo:
			ArmijoCondition = 0
			while(ArmijoCondition == 0):
				S_n = S - learningRate*gradient_total
				S_n = projectetBack(S_n, 0, 1)
				ss_n = np.split(S_n,np.cumsum(Asizes)[:-1],axis=0)

				tmp_loss = 0
				for s in ss_n:
					tmp_loss += np.linalg.norm(s, 1)**2-np.linalg.norm(s, 2)**2
				LossWithinLayer = p*tmp_loss - (1+p)*S_n.transpose().dot(A.dot(S_n))
				Loss_new = LossWithinLayer
				LossCrossLayer = 0

				if it > UseCR:
					row_index, col_index, data_ = find(CR)
					oneS_n = one-S_n
					LossCrossLayer = np.linalg.norm(data_ - np.multiply(data_, (np.multiply(S_n[row_index], S_n[col_index])+np.multiply(oneS_n[row_index], oneS_n[col_index])).flatten()), 2)**2

				Loss_new += LossCrossLayer

				ArmijoCondition = ((Loss_new - Loss_old) <= sigma*gradient_total.transpose().dot(S_n-S))[0,0]
				if ArmijoCondition == 0:
					learningRate *= beta
				else:
					learningRate /= np.sqrt(beta)

		# projected gradient descent
		S -= learningRate*gradient_total
		S = projectetBack(S, 0, 1)

		if it % 10 == 0:
			print('Iteration: '+str(it))

			ss = np.split(S,np.cumsum(Asizes)[:-1],axis=0)
			avgF1 = 0
			for i in range(len(ss)):
				si = ss[i]
				gdi = groundTruths[i]
				minedResult = set()
				for j in range(len(si)):
					if si[j][0]>=1/2:
						minedResult.add(j)
				prec,recal,f1 = EvaluateF1(minedResult, gdi)
				print('Precison: ',end='')
				print(prec)
				print('Recall: ',end='')
				print(recal)
				print('F1: ',end='')
				print(f1)
				print('------------------------------')
				avgF1 += f1
			avgF1 /= 2.0
			if avgF1 > bestF1:
				bestF1 = avgF1
	return ss,bestF1

GG = np.ones((len(As),len(As)))-np.identity(len(As))

for i,graph_ in enumerate(As):

	graph_[insertedPosition:insertedPosition+cliqueSizes[i],insertedPosition:insertedPosition+cliqueSizes[i]] = sparse.lil_matrix(np.ones((cliqueSizes[i],cliqueSizes[i]),dtype='float')-np.identity(cliqueSizes[i],dtype='float'))
	groundTruths.append(frozenset(np.array(range(insertedPosition,insertedPosition+cliqueSizes[i]),dtype='int')))

######################################### Make Cs ##############################################

def random_matrix(size, prob):
	a = np.random.random(size)
	return np.where(a<prob,1,0)

Cs = dict()
for i in range(len(As)):
	block_size_i = int(As[i].shape[0]/block_num)
	for j in range(len(As)):
		block_size_j = int(As[j].shape[0]/block_num)
		if i == j:continue
		tmp = random_matrix((As[i].shape[0],As[j].shape[0]),ProbN)
		for k in range(block_num):
			tmp[block_size_i*k:block_size_i*(k+1),block_size_j*k:block_size_j*(k+1)] = random_matrix((block_size_i,block_size_j),Prob)
		Cs[(i,j)] = tmp
################################### Make graph pair-specific gamma ################################

gammas = {}
for i in range(len(As)):
	di = As[i].sum()/(As[i].shape[0]*(As[i].shape[0]-1))
	for j in range(len(As)):
		dj = As[j].sum()/(As[j].shape[0]*(As[j].shape[0]-1))
		gammas[(i,j)] = (di/dj)**2*gamma
# print(gammas)

################################### Make Aggregated Matrix A ######################################

A = sparse.lil_matrix((SizeA,SizeA), dtype=np.float64)
for i in range(len(As)):
	A[cumsum_Asizes[i]:cumsum_Asizes[i+1], cumsum_Asizes[i]:cumsum_Asizes[i+1]] = As[i]

################################### Make Aggregated Matrix C ######################################

C = sparse.lil_matrix((SizeA,SizeA))
for i in range(len(As)):
	for j in range(len(As)):
		if GG[i,j] == 0:continue
		C[cumsum_Asizes[i]:cumsum_Asizes[i+1], cumsum_Asizes[j]:cumsum_Asizes[j+1]] = Cs[(i,j)]

################################### Make Aggregated Matrix R ######################################

CR = C.copy()
CRR = C.copy()
for i in range(len(As)):
	for j in range(len(As)):
		if GG[i,j] == 0:continue
		CR[cumsum_Asizes[i]:cumsum_Asizes[i+1], cumsum_Asizes[j]:cumsum_Asizes[j+1]] *= gammas[(i,j)]**(0.5)
		CRR[cumsum_Asizes[i]:cumsum_Asizes[i+1], cumsum_Asizes[j]:cumsum_Asizes[j+1]] *= gammas[(i,j)]

################################### Start calculation #############################################

time_start = time.time()
ss, F1 = Destine(iteration,ppp,learningRate,A,CR,CRR,groundTruths,As)
time_end = time.time()
print('F1 average: '+str(F1))
