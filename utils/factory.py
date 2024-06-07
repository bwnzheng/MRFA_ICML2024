def get_model(model_name, args):
    name = model_name.lower()
    if name == "replay":
        from models.replay import Replay
        return Replay(args)
    elif name == "replay_mrfa":
        from models.replay_mrfa import Replaymrfa
        return Replaymrfa(args)
    elif name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "icarl_mrfa":
        from models.icarl_mrfa import iCaRLmrfa
        return iCaRLmrfa(args)
    else:
        assert 0
