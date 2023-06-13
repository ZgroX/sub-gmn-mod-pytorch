from torch_geometric.data import Data


class PairData(Data):
    def __init__(
            self,
            d_id=None,
            edge_features_d=None,
            edge_index_d=None,
            x_d=None,
            pos_d=None,
            edge_features_q=None,
            edge_index_q=None,
            x_q=None,
            pos_q=None,
            category=None,
            truth_matrix=None
    ):
        super().__init__()

        self.d_id = d_id,
        self.edge_features_d = edge_features_d,
        self.edge_index_d = edge_index_d,
        self.x_d = x_d,
        self.pos_d = pos_d,
        self.edge_features_q = edge_features_q,
        self.edge_index_q = edge_index_q,
        self.x_q = x_q,
        self.pos_q = pos_q,
        self.category = category,
        self.truth_matrix = truth_matrix

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_d":
            return self.x_d[0].shape[0]
        if key == "edge_index_q":
            return self.x_q[0].shape[0]
        else:
            return super().__inc__(key, value, *args, **kwargs)
