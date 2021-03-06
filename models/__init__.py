from .vit import vit_base_patch16_224

def create_model(cfg):
    model_discard_keys = []
    model = eval(cfg['model_name'])
    if 'kargs' not in model.__code__.co_varnames:
        for k in list(cfg.keys()):
            if k not in model.__code__.co_varnames:
                model_discard_keys.append(k)
                cfg.pop(k)
    print(k)
    return model(**cfg)
