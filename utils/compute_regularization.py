from weight_regularization import WeightRegularization

# Neural Photo Editing with Introspective Adversarial Networks
# https://arxiv.org/abs/1609.07093
def compute_regularization(model, regularization_type):
    weight_regularization = WeightRegularization(regularization_type)
    return weight_regularization.compute_regularization(model)
