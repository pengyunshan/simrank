import numpy as np


class SimRank:

    def __init__(self):
        self.users = []
        self.items = []
        self.graph = None
        self.user_sim = None
        self.item_sim = None

    def load_data(self, path='data/sample1.txt'):
        with open(path, 'r') as log_fp:
            logs_tuple = [tuple(log.strip().split(",")) for log in log_fp.readlines()]
        self.users = list(set([log[0] for log in logs_tuple]))
        self.items = list(set([log[1] for log in logs_tuple]))
        # Graph means the relations number
        self.graph = np.array(np.zeros([len(self.users), len(self.items)]))
        for log in logs_tuple:
            query = log[0]
            ad = log[1]
            q_i = self.users.index(query)
            a_j = self.items.index(ad)
            self.graph[q_i, a_j] += 1
        self.user_sim = np.array(np.identity(len(self.users)))
        self.item_sim = np.array(np.identity(len(self.items)))

    def load_serials(self, path="data/20200602"):
        with open(path, "r", encoding="utf-8") as f:
            logs_tuple = [(line.strip("\n").split()[0], line.strip("\n").split()[1:]) for line in f.readlines()]
            self.users = list(set([log[0] for log in logs_tuple]))
            click_serials = list([log[1] for log in logs_tuple])
            self.items = list(set([item for serials in click_serials for item in serials]))
            self.graph = np.array(np.zeros([len(self.users), len(self.items)]))
            for log in logs_tuple:
                user, items = log[0], log[1]
                u_i = self.users.index(user)
                for item in items:
                    i_j = self.items.index(item)
                    self.graph[u_i, i_j] += 1
            self.user_sim = np.array(np.identity(len(self.users)))
            self.item_sim = np.array(np.identity(len(self.items)))
            print("user length: ", len(self.users))
            print("item length: ", len(self.items))

    def get_ads_num(self, query):
        q_i = self.users.index(query)
        return self.graph[q_i]
    
    def get_queries_num(self, ad):
        a_j = self.items.index(ad)
        return self.graph.transpose()[a_j]
    
    def get_ads(self, query):
        series = self.get_ads_num(query).tolist()
        return [self.items[i] for i in range(len(series)) if series[i] > 0]
    
    def get_queries(self, ad):
        series = self.get_queries_num(ad).tolist()
        return [self.users[i] for i in range(len(series)) if series[i] > 0]

    def user_simrank(self, q1, q2, C):
        """
        in this, graph[q_i] -> connected ads
        """
        if q1 == q2:
            return 1
        prefix = C / (self.get_ads_num(q1).sum() * self.get_ads_num(q2).sum())
        postfix = 0
        for ad_i in self.get_ads(q1):
            for ad_j in self.get_ads(q2):
                i = self.items.index(ad_i)
                j = self.items.index(ad_j)
                postfix += self.item_sim[i, j]
        return prefix * postfix

    def item_simrank(self, a1, a2, C):
        """
        in this, graph need to be transposed to make ad to be the index
        """
        if a1 == a2 : return 1
        prefix = C / (self.get_queries_num(a1).sum() * self.get_queries_num(a2).sum())
        postfix = 0
        for query_i in self.get_queries(a1):
            for query_j in self.get_queries(a2):
                i = self.users.index(query_i)
                j = self.users.index(query_j)
                postfix += self.user_sim[i, j]
        return prefix * postfix

    def simrank(self, C=0.8, times=1):
        # global query_sim, ad_sim
        for run in range(times):
            # queries simrank
            new_user_sim = np.array(np.identity(len(self.users)))
            for qi in self.users:
                for qj in self.users:
                    i = self.users.index(qi)
                    j = self.users.index(qj)
                    new_user_sim[i, j] = self.user_simrank(qi, qj, C)
            # ads simrank
            new_item_sim = np.array(np.identity(len(self.items)))
            for ai in self.items:
                for aj in self.items:
                    i = self.items.index(ai)
                    j = self.items.index(aj)
                    new_item_sim[i, j] = self.item_simrank(ai, aj, C)
            self.user_sim = new_user_sim
            self.item_sim = new_item_sim


def save_data(data, path):
    if isinstance(data, list):
        with open(path, "w", encoding="utf-8") as f:
            for i in data:
                f.write(i + "\n")
    elif isinstance(data, np.ndarray):
        np.savetxt(path, data, delimiter="\t")


if __name__ == '__main__':
    simrank = SimRank()
    simrank.load_serials("data/tmp")
    print("load finished")
    simrank.simrank()
    save_data(simrank.users, "data/users.txt")
    save_data(simrank.items, "data/items.txt")
    save_data(simrank.item_sim, "data/item_sim.txt")
    save_data(simrank.user_sim, "data/user_sim.txt")