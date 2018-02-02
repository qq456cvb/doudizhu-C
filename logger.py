class Logger:
    def __init__(self, moving_avg=False):
        self.dic = {}
        self.moving_avg = moving_avg
    
    def updateAcc(self, key, val):
        if not key in self.dic:
            self.dic[key] = []
        
        self.dic[key].append(val)
        if self.moving_avg and len(self.dic[key]) > 1000:
            self.dic[key].pop(0)
        # self.dic[key]['cnt'] += 1
        # self.dic[key]['acc'] += (val - self.dic[key]['acc']) / self.dic[key]['cnt']
    
    def __getitem__(self, item):
        if item in self.dic:
            # return self.dic[item]['acc']
            if len(self.dic[item]) == 0:
                return 0
            return sum(self.dic[item]) / len(self.dic[item])
        return 0

