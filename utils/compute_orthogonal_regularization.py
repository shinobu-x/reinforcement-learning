from weight_regularization import WeightRegularization

# Neural Photo Editing with Introspective Adversarial Networks
# https://arxiv.org/abs/1609.07093
def compute_orthogonal_regularization(model):
    weight_regularization = WeightRegularization('orthogonal')
    return weight_regularization.compute_regularization(model)
