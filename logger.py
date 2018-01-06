class Logger:
    def __init__(self):
        self.dic = {}
    
    def updateAcc(self, key, val):
        if not key in self.dic:
            self.dic[key] = {"cnt" : 0, "acc" : 0}
        
        self.dic[key]['cnt'] += 1
        self.dic[key]['acc'] += (val - self.dic[key]['acc']) / self.dic[key]['cnt']
    
    def __getitem__(self, item):
        if item in self.dic:
            return self.dic[item]['acc']
        return 0

