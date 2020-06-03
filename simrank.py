import numpy as np


class SimRank:

    def __init__(self):
        self.queries = None
        self.ads = None
        self.graph = None
        self.query_sim = None
        self.ad_sim = None

    def load_data(self, path='test/sample1.txt'):
        with open(path, 'r') as log_fp:
            logs_tuple = [tuple(log.strip().split(",")) for log in log_fp.readlines()]
        self.queries = list(set([log[0] for log in logs_tuple ]))
        self.ads = list(set([log[1] for log in logs_tuple ]))
        # Graph means the relations number
        self.graph = np.matrix(np.zeros([len(self.queries), len(self.ads)]))
        for log in logs_tuple:
            query = log[0]
            ad = log[1]
            q_i = self.queries.index(query)
            a_j = self.ads.index(ad)
            self.graph[q_i, a_j] += 1
        print(self.graph)
        
        self.query_sim = np.matrix(np.identity(len(self.queries)))
        self.ad_sim = np.matrix(np.identity(len(self.ads)))

    def get_ads_num(self, query):
        q_i = self.queries.index(query)
        return self.graph[q_i]
    
    def get_queries_num(self, ad):
        a_j = self.ads.index(ad)
        return self.graph.transpose()[a_j]
    
    def get_ads(self, query):
        series = self.get_ads_num(query).tolist()[0]
        return [ self.ads[x] for x in range(len(series)) if series[x] > 0 ]
    
    def get_queries(self, ad):
        series = self.get_queries_num(ad).tolist()[0]
        return [self.queries[x] for x in range(len(series)) if series[x] > 0]

    def query_simrank(self, q1, q2, C):
        """
        in this, graph[q_i] -> connected ads
        """
        """
        print "q1.ads"
        print get_ads_num(q1).tolist()
        print "q2.ads"
        print get_ads_num(q2).tolist()
        """
        if q1 == q2 : return 1
        prefix = C / (self.get_ads_num(q1).sum() * self.get_ads_num(q2).sum())
        postfix = 0
        for ad_i in self.get_ads(q1):
            for ad_j in self.get_ads(q2):
                i = self.ads.index(ad_i)
                j = self.ads.index(ad_j)
                postfix += self.ad_sim[i, j]
        return prefix * postfix

    def ad_simrank(self, a1, a2, C):
        """
        in this, graph need to be transposed to make ad to be the index
        """
        """
        print "a1.queries"
        print get_queries_num(a1)
        print "a2.queries"
        print get_queries_num(a2)
        """
        if a1 == a2 : return 1
        prefix = C / (self.get_queries_num(a1).sum() * self.get_queries_num(a2).sum())
        postfix = 0
        for query_i in self.get_queries(a1):
            for query_j in self.get_queries(a2):
                i = self.queries.index(query_i)
                j = self.queries.index(query_j)
                postfix += self.query_sim[i,j]
        return prefix * postfix

    def simrank(self, C=0.8, times=1):
        # global query_sim, ad_sim
        for run in range(times):
            # queries simrank
            new_query_sim = np.matrix(np.identity(len(self.queries)))
            for qi in self.queries:
                for qj in self.queries:
                    i = self.queries.index(qi)
                    j = self.queries.index(qj)
                    new_query_sim[i,j] = self.query_simrank(qi, qj, C)
            # ads simrank
            new_ad_sim = np.matrix(np.identity(len(self.ads)))
            for ai in self.ads:
                for aj in self.ads:
                    i = self.ads.index(ai)
                    j = self.ads.index(aj)
                    new_ad_sim[i,j] = self.ad_simrank(ai, aj, C)
            self.query_sim = new_query_sim
            self.ad_sim = new_ad_sim


if __name__ == '__main__':
    simrank = SimRank()
    simrank.load_data()
    simrank.simrank()
#    print(queries)
#    print(ads)
#    simrank()
#    print(query_sim)
#    print(ad_sim)
