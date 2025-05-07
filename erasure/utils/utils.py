
from erasure.utils.config.local_ctx import Local


### THIS ENSURES COMPATIBILITY WITH KAN MODELS
def clone_predictor_from_config(global_ctx, data_manager):

    new_current = Local(global_ctx.config.predictor)
    new_current.dataset = data_manager

    new_predictor = global_ctx.factory.get_object(new_current)

    new_predictor.load_state_dict(global_ctx.predictor.state_dict())

    return new_predictor
