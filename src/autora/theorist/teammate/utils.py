USE_VAR_NAME_STRUCTURE = 'short'

def get_variable_name(idx):
    var_name_structs = {'short': short_name,
                        'idx': x_idx_name}
    naming_func = var_name_structs[USE_VAR_NAME_STRUCTURE]
    return naming_func(idx)

def short_name(idx):
    return f"x{idx}"

def x_idx_name(idx):
    return f"x_{idx}"