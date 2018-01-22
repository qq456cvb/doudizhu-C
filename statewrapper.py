class WrappedState:
    reserved_key = {'idx', 'last_category_idx', 'last_cards', 'control_idx',
                    'land_score', 'lord_idx', 'extra_cards', 'player_cards', 'histories', 'stage'}

    def __init__(self):
        self.dic = {}

    def __getitem__(self, item):
        return self.dic[item]

    def __contains__(self, item):
        return item in self.dic

    def is_intermediate(self):
        return self['stage'] != 'p_decision' and self['stage'] != 'a_decision'

    def clear(self):
        for k in self.dic:
            if k not in WrappedState.reserved_key:
                self.dic.pop(k, None)

    def __setitem__(self, key, value):
        self.dic[key] = value
