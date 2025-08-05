import numpy as np

def make_state_from_vars(vars, max_layers=20, n_materials=3):
    state = np.zeros((max_layers, n_materials+1))
    layer_thickness = vars[:max_layers]
    materials_inds = np.floor(vars[max_layers:]).astype(int)  
    for i in range(max_layers):
        #state[i,0] = vars[f"layer_{i}"]
        #state[i,vars[f"layer_{i}_material"]+1] = 1
        state[i,0] = layer_thickness[i]
        state[i,materials_inds[i]+2] = 1
    return state