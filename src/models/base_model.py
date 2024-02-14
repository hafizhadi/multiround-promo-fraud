from utils_func import verPrint
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'GCN:__init__ | {kwargs}')

    def forward(self, blocks, x,):
        verPrint(self.verbose, 3, f'GCN:forward | {blocks} {x}')
        return x, None # No loss returned
    
    def postBackprop(self, **kwargs):
        return None

    @staticmethod
    def prepare_graph(graph):
        return graph