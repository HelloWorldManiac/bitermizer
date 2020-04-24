import numpy as np
from itertools import combinations, chain



class GibbsSampler:
    """ Biterm Topic Model

        Code and naming is based on this paper http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf
        Thanks to jcapde for providing the code on https://github.com/jcapde/Biterm
    """

    def __init__(self, num_topics, V, alpha=1., beta=0.01, l=0.5):
        self.num_topics = num_topics
        self.V = V
        self.alpha = np.full(self.num_topics, alpha)
        self.beta = np.full((len(self.V), self.num_topics), beta)
        self.l = l


    def _gibbs(self, iterations):

        Z = np.zeros(len(self.B), dtype=np.int32)
        n_wz = np.zeros((len(self.V), self.num_topics), dtype=int)
        n_z = np.zeros(self.num_topics, dtype=int)
        theta = np.random.dirichlet([self.alpha] * self.num_topics, 1)
        
        for i, b_i in enumerate(self.B):
            #topic = np.random.choice(self.K, 1)[0]
            topic = np.random.choice(self.num_topics, 1, p=theta[0,:])[0]
	    print(topic)
            n_wz[b_i[0], topic] += 1
            n_wz[b_i[1], topic] += 1
            n_z[topic] += 1
            Z[i] = topic
            #print(Z)

        for _ in range(iterations):
            for i, b_i in enumerate(self.B):
                n_wz[b_i[0], Z[i]] -= 1
                n_wz[b_i[1], Z[i]] -= 1
                n_z[Z[i]] -= 1
                P_w0z = (n_wz[b_i[0], :] + self.beta[b_i[0], :]) / (2 * n_z + self.beta.sum(axis=0))
                P_w1z = (n_wz[b_i[1], :] + self.beta[b_i[1], :]) / (2 * n_z + 1 + self.beta.sum(axis=0))
                P_z = (n_z + self.alpha) * P_w0z * P_w1z
                P_z = P_z / P_z.sum()
                Z[i] = np.random.choice(self.num_topics, 1, p=P_z)
                n_wz[b_i[0], Z[i]] += 1
                n_wz[b_i[1], Z[i]] += 1
                n_z[Z[i]] += 1


        return n_z, n_wz

    def fit_transform(self, B_d, iterations):
       self.fit(B_d, iterations)
       return self.transform(B_d)

    def fit(self, B_d, iterations):
        self.B = list(chain(*B_d))
        n_z, self.nwz = self._gibbs(iterations)

        self.phi_wz = (self.nwz + self.beta) / np.array([(self.nwz + self.beta).sum(axis=0)] * len(self.V))
        self.theta_z = (n_z + self.alpha) / (n_z + self.alpha).sum()

        self.alpha += self.l * n_z
        self.beta += self.l * self.nwz


    def transform(self, B_d):

        P_zd = np.zeros([len(B_d), self.num_topics])
        for i, d in enumerate(B_d):
            P_zb = np.zeros([len(d), self.num_topics])
            for j, b in enumerate(d):
                P_zbi = self.theta_z * self.phi_wz[b[0], :] * self.phi_wz[b[1], :]
                P_zb[j] = P_zbi / P_zbi.sum()
            P_zd[i] = P_zb.sum(axis=0) / P_zb.sum(axis=0).sum()

        return P_zd

    def summarize(self, bi_vecs, n_words, summary = False):
        res = {
            
            'top_words': [[None]] * len(self.phi_wz.T)
        }
        for z, P_wzi in enumerate(self.phi_wz.T):
            V_z = np.argsort(P_wzi)[:-(n_words + 1):-1]
            W_z = self.V[V_z]
            C_z = 0
            for m in range(1, n_words):
                for l in range(m):
                    D_vmvl = np.in1d(np.nonzero(bi_vecs[:,V_z[l]]), np.nonzero(bi_vecs[:,V_z[m]])).sum(dtype=int) + 1
                    D_vl = np.count_nonzero(bi_vecs[:,V_z[l]])
                    
            res['top_words'][z] = W_z
            print('Topic {} :: {}'.format(z, ' '.join(W_z)))
        if summary:
            return res
