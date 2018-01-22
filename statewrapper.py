class WrappedState:
    def __init__(self):
        self.dic = {}

    def __getitem__(self, item):
        return self.dic[item]

    def __contains__(self, item):
    	return item in self.dic

    def is_intermediate(self):
        return self['stage'] != 'p_decision' and self['stage'] != 'a_decision'

    def __setitem__(self, key, value):
        self.dic[key] = value
