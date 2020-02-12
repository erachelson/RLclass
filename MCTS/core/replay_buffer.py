import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def _add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def add(self, transitions):
        # transform dict of lists into lists of dicts
        num_transitions = len(transitions['observations'])
        transitions_list = [{key: value[i] for key, value in transitions.items()} for i in range(num_transitions)]
        for transition in transitions_list:
            self._add(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        batch = {}
        for key in self.storage[0].keys():
            batch[key] = []

        for i in ind:
            for key, value in self.storage[i].items():
                batch[key].append(value)

        for key, value in batch.items():
            batch[key] = np.stack(value)

        return batch