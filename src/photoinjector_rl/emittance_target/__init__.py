"""
First-iteration narrow-scope surrogate: MLP from 11 XOPT-sampled knob values
-> scalar norm_emit_4d at PR10241.
"""

# Authoritative ordering of the 11 XOPT-sampled variables. Frozen here so
# every script (preprocess, plot, train, inference) uses the same column
# order in the input tensor.
SETTING_KEYS = [
    "SOL10111:solenoid_field_scale",
    "CQ10121:b1_gradient",
    "SQ10122:b1_gradient",
    "GUNF:rf_field_scale",
    "GUNF:theta0_deg",
    "distgen:r_dist:sigma_xy:value",
    "distgen:r_dist:truncation_radius:value",
    "distgen:start:MTE:value",
    "distgen:t_dist:sigma_t:value",
    "distgen:transforms:s1:scale",
    "distgen:transforms:r1:angle:value",
]

# Bounds from configs/sweep/lhs_train_hifi.yaml -- used for min-max normalization
# of the settings vector. Keep in sync with the YAML.
SETTING_BOUNDS = {
    "SOL10111:solenoid_field_scale": (-0.32, -0.2),
    "CQ10121:b1_gradient": (-0.2, 0.2),
    "SQ10122:b1_gradient": (-0.2, 0.2),
    "GUNF:rf_field_scale": (46960818.3433, 53086142.475),
    "GUNF:theta0_deg": (-70.0, -58.0),
    "distgen:r_dist:sigma_xy:value": (1.0, 2.5),
    "distgen:r_dist:truncation_radius:value": (2.0, 3.5),
    "distgen:start:MTE:value": (100.0, 1000.0),
    "distgen:t_dist:sigma_t:value": (0.8, 1.5),
    "distgen:transforms:s1:scale": (0.6, 1.4),
    "distgen:transforms:r1:angle:value": (-90.0, 90.0),
}

N_INPUT = len(SETTING_KEYS)   # 11
