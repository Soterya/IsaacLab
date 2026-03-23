DEFAULT_STIFFNESS = 10000.0
DEFAULT_DAMPING   = 500.0
DEFAULT_EFFORT    = 20000.0

high_stiffness  = 50000 
high_damping    = 5000
high_effort     = 20000 

low_stiffness   = 1000
low_damping     = 200 
low_effort      = 2000 

STIFFNESS_PER_JOINT = {
    "L_Hip_x"     : low_stiffness, "L_Hip_y"     : low_stiffness, "L_Hip_z"     : low_stiffness,
    "R_Hip_x"     : low_stiffness, "R_Hip_y"     : low_stiffness, "R_Hip_z"     : low_stiffness,
    "Torso_x"     : high_stiffness, "Torso_y"     : high_stiffness, "Torso_z"     : high_stiffness,
    "L_Knee_x"    : DEFAULT_STIFFNESS, "L_Knee_y"    : DEFAULT_STIFFNESS, "L_Knee_z"    : DEFAULT_STIFFNESS,
    "R_Knee_x"    : DEFAULT_STIFFNESS, "R_Knee_y"    : DEFAULT_STIFFNESS, "R_Knee_z"    : DEFAULT_STIFFNESS,
    "Spine_x"     : high_stiffness, "Spine_y"     : high_stiffness, "Spine_z"     : high_stiffness,
    "L_Ankle_x"   : DEFAULT_STIFFNESS, "L_Ankle_y"   : DEFAULT_STIFFNESS, "L_Ankle_z"   : DEFAULT_STIFFNESS,
    "R_Ankle_x"   : DEFAULT_STIFFNESS, "R_Ankle_y"   : DEFAULT_STIFFNESS, "R_Ankle_z"   : DEFAULT_STIFFNESS,
    "Chest_x"     : high_stiffness, "Chest_y"     : high_stiffness, "Chest_z"     : high_stiffness,
    "L_Toe_x"     : DEFAULT_STIFFNESS, "L_Toe_y"     : DEFAULT_STIFFNESS, "L_Toe_z"     : DEFAULT_STIFFNESS,
    "R_Toe_x"     : DEFAULT_STIFFNESS, "R_Toe_y"     : DEFAULT_STIFFNESS, "R_Toe_z"     : DEFAULT_STIFFNESS,
    "Neck_x"      : low_stiffness, "Neck_y"      : low_stiffness, "Neck_z"      : low_stiffness,
    "L_Thorax_x"  : DEFAULT_STIFFNESS, "L_Thorax_y"  : DEFAULT_STIFFNESS, "L_Thorax_z"  : DEFAULT_STIFFNESS,
    "R_Thorax_x"  : DEFAULT_STIFFNESS, "R_Thorax_y"  : DEFAULT_STIFFNESS, "R_Thorax_z"  : DEFAULT_STIFFNESS,
    "Head_x"      : low_stiffness, "Head_y"      : low_stiffness, "Head_z"      : low_stiffness,
    "L_Shoulder_x": low_stiffness, "L_Shoulder_y": low_stiffness, "L_Shoulder_z": low_stiffness,
    "R_Shoulder_x": low_stiffness, "R_Shoulder_y": low_stiffness, "R_Shoulder_z": low_stiffness,
    "L_Elbow_x"   : DEFAULT_STIFFNESS, "L_Elbow_y"   : DEFAULT_STIFFNESS, "L_Elbow_z"   : DEFAULT_STIFFNESS,
    "R_Elbow_x"   : DEFAULT_STIFFNESS, "R_Elbow_y"   : DEFAULT_STIFFNESS, "R_Elbow_z"   : DEFAULT_STIFFNESS,
    "L_Wrist_x"   : DEFAULT_STIFFNESS, "L_Wrist_y"   : DEFAULT_STIFFNESS, "L_Wrist_z"   : DEFAULT_STIFFNESS,
    "R_Wrist_x"   : DEFAULT_STIFFNESS, "R_Wrist_y"   : DEFAULT_STIFFNESS, "R_Wrist_z"   : DEFAULT_STIFFNESS,
    "L_Hand_x"    : DEFAULT_STIFFNESS, "L_Hand_y"    : DEFAULT_STIFFNESS, "L_Hand_z"    : DEFAULT_STIFFNESS,
    "R_Hand_x"    : DEFAULT_STIFFNESS, "R_Hand_y"    : DEFAULT_STIFFNESS, "R_Hand_z"    : DEFAULT_STIFFNESS,
}

DAMPING_PER_JOINT = {
    "L_Hip_x"     : DEFAULT_DAMPING, "L_Hip_y"     : DEFAULT_DAMPING, "L_Hip_z"     : DEFAULT_DAMPING,
    "R_Hip_x"     : DEFAULT_DAMPING, "R_Hip_y"     : DEFAULT_DAMPING, "R_Hip_z"     : DEFAULT_DAMPING,
    "Torso_x"     : DEFAULT_DAMPING, "Torso_y"     : DEFAULT_DAMPING, "Torso_z"     : DEFAULT_DAMPING,
    "L_Knee_x"    : DEFAULT_DAMPING, "L_Knee_y"    : DEFAULT_DAMPING, "L_Knee_z"    : DEFAULT_DAMPING,
    "R_Knee_x"    : DEFAULT_DAMPING, "R_Knee_y"    : DEFAULT_DAMPING, "R_Knee_z"    : DEFAULT_DAMPING,
    "Spine_x"     : DEFAULT_DAMPING, "Spine_y"     : DEFAULT_DAMPING, "Spine_z"     : DEFAULT_DAMPING,
    "L_Ankle_x"   : DEFAULT_DAMPING, "L_Ankle_y"   : DEFAULT_DAMPING, "L_Ankle_z"   : DEFAULT_DAMPING,
    "R_Ankle_x"   : DEFAULT_DAMPING, "R_Ankle_y"   : DEFAULT_DAMPING, "R_Ankle_z"   : DEFAULT_DAMPING,
    "Chest_x"     : DEFAULT_DAMPING, "Chest_y"     : DEFAULT_DAMPING, "Chest_z"     : DEFAULT_DAMPING,
    "L_Toe_x"     : DEFAULT_DAMPING, "L_Toe_y"     : DEFAULT_DAMPING, "L_Toe_z"     : DEFAULT_DAMPING,
    "R_Toe_x"     : DEFAULT_DAMPING, "R_Toe_y"     : DEFAULT_DAMPING, "R_Toe_z"     : DEFAULT_DAMPING,
    "Neck_x"      : DEFAULT_DAMPING, "Neck_y"      : DEFAULT_DAMPING, "Neck_z"      : DEFAULT_DAMPING,
    "L_Thorax_x"  : DEFAULT_DAMPING, "L_Thorax_y"  : DEFAULT_DAMPING, "L_Thorax_z"  : DEFAULT_DAMPING,
    "R_Thorax_x"  : DEFAULT_DAMPING, "R_Thorax_y"  : DEFAULT_DAMPING, "R_Thorax_z"  : DEFAULT_DAMPING,
    "Head_x"      : DEFAULT_DAMPING, "Head_y"      : DEFAULT_DAMPING, "Head_z"      : DEFAULT_DAMPING,
    "L_Shoulder_x": DEFAULT_DAMPING, "L_Shoulder_y": DEFAULT_DAMPING, "L_Shoulder_z": DEFAULT_DAMPING,
    "R_Shoulder_x": DEFAULT_DAMPING, "R_Shoulder_y": DEFAULT_DAMPING, "R_Shoulder_z": DEFAULT_DAMPING,
    "L_Elbow_x"   : DEFAULT_DAMPING, "L_Elbow_y"   : DEFAULT_DAMPING, "L_Elbow_z"   : DEFAULT_DAMPING,
    "R_Elbow_x"   : DEFAULT_DAMPING, "R_Elbow_y"   : DEFAULT_DAMPING, "R_Elbow_z"   : DEFAULT_DAMPING,
    "L_Wrist_x"   : DEFAULT_DAMPING, "L_Wrist_y"   : DEFAULT_DAMPING, "L_Wrist_z"   : DEFAULT_DAMPING,
    "R_Wrist_x"   : DEFAULT_DAMPING, "R_Wrist_y"   : DEFAULT_DAMPING, "R_Wrist_z"   : DEFAULT_DAMPING,
    "L_Hand_x"    : DEFAULT_DAMPING, "L_Hand_y"    : DEFAULT_DAMPING, "L_Hand_z"    : DEFAULT_DAMPING,
    "R_Hand_x"    : DEFAULT_DAMPING, "R_Hand_y"    : DEFAULT_DAMPING, "R_Hand_z"    : DEFAULT_DAMPING,
}

EFFORT_PER_JOINT = {
    "L_Hip_x"     : DEFAULT_EFFORT, "L_Hip_y"     : DEFAULT_EFFORT, "L_Hip_z"     : DEFAULT_EFFORT,
    "R_Hip_x"     : DEFAULT_EFFORT, "R_Hip_y"     : DEFAULT_EFFORT, "R_Hip_z"     : DEFAULT_EFFORT,
    "Torso_x"     : DEFAULT_EFFORT, "Torso_y"     : DEFAULT_EFFORT, "Torso_z"     : DEFAULT_EFFORT,
    "L_Knee_x"    : DEFAULT_EFFORT, "L_Knee_y"    : DEFAULT_EFFORT, "L_Knee_z"    : DEFAULT_EFFORT,
    "R_Knee_x"    : DEFAULT_EFFORT, "R_Knee_y"    : DEFAULT_EFFORT, "R_Knee_z"    : DEFAULT_EFFORT,
    "Spine_x"     : DEFAULT_EFFORT, "Spine_y"     : DEFAULT_EFFORT, "Spine_z"     : DEFAULT_EFFORT,
    "L_Ankle_x"   : DEFAULT_EFFORT, "L_Ankle_y"   : DEFAULT_EFFORT, "L_Ankle_z"   : DEFAULT_EFFORT,
    "R_Ankle_x"   : DEFAULT_EFFORT, "R_Ankle_y"   : DEFAULT_EFFORT, "R_Ankle_z"   : DEFAULT_EFFORT,
    "Chest_x"     : DEFAULT_EFFORT, "Chest_y"     : DEFAULT_EFFORT, "Chest_z"     : DEFAULT_EFFORT,
    "L_Toe_x"     : DEFAULT_EFFORT, "L_Toe_y"     : DEFAULT_EFFORT, "L_Toe_z"     : DEFAULT_EFFORT,
    "R_Toe_x"     : DEFAULT_EFFORT, "R_Toe_y"     : DEFAULT_EFFORT, "R_Toe_z"     : DEFAULT_EFFORT,
    "Neck_x"      : DEFAULT_EFFORT, "Neck_y"      : DEFAULT_EFFORT, "Neck_z"      : DEFAULT_EFFORT,
    "L_Thorax_x"  : DEFAULT_EFFORT, "L_Thorax_y"  : DEFAULT_EFFORT, "L_Thorax_z"  : DEFAULT_EFFORT,
    "R_Thorax_x"  : DEFAULT_EFFORT, "R_Thorax_y"  : DEFAULT_EFFORT, "R_Thorax_z"  : DEFAULT_EFFORT,
    "Head_x"      : DEFAULT_EFFORT, "Head_y"      : DEFAULT_EFFORT, "Head_z"      : DEFAULT_EFFORT,
    "L_Shoulder_x": DEFAULT_EFFORT, "L_Shoulder_y": DEFAULT_EFFORT, "L_Shoulder_z": DEFAULT_EFFORT,
    "R_Shoulder_x": DEFAULT_EFFORT, "R_Shoulder_y": DEFAULT_EFFORT, "R_Shoulder_z": DEFAULT_EFFORT,
    "L_Elbow_x"   : DEFAULT_EFFORT, "L_Elbow_y"   : DEFAULT_EFFORT, "L_Elbow_z"   : DEFAULT_EFFORT,
    "R_Elbow_x"   : DEFAULT_EFFORT, "R_Elbow_y"   : DEFAULT_EFFORT, "R_Elbow_z"   : DEFAULT_EFFORT,
    "L_Wrist_x"   : DEFAULT_EFFORT, "L_Wrist_y"   : DEFAULT_EFFORT, "L_Wrist_z"   : DEFAULT_EFFORT,
    "R_Wrist_x"   : DEFAULT_EFFORT, "R_Wrist_y"   : DEFAULT_EFFORT, "R_Wrist_z"   : DEFAULT_EFFORT,
    "L_Hand_x"    : DEFAULT_EFFORT, "L_Hand_y"    : DEFAULT_EFFORT, "L_Hand_z"    : DEFAULT_EFFORT,
    "R_Hand_x"    : DEFAULT_EFFORT, "R_Hand_y"    : DEFAULT_EFFORT, "R_Hand_z"    : DEFAULT_EFFORT,
}

ALL_HUMANOID_JOINTS = [
    "L_Hip_x"     , "L_Hip_y"     , "L_Hip_z"     ,
    "R_Hip_x"     , "R_Hip_y"     , "R_Hip_z"     ,
    "Torso_x"     , "Torso_y"     , "Torso_z"     ,
    "L_Knee_x"    , "L_Knee_y"    , "L_Knee_z"    ,
    "R_Knee_x"    , "R_Knee_y"    , "R_Knee_z"    ,
    "Spine_x"     , "Spine_y"     , "Spine_z"     ,
    "L_Ankle_x"   , "L_Ankle_y"   , "L_Ankle_z"   ,
    "R_Ankle_x"   , "R_Ankle_y"   , "R_Ankle_z"   ,
    "Chest_x"     , "Chest_y"     , "Chest_z"     ,
    "L_Toe_x"     , "L_Toe_y"     , "L_Toe_z"     ,
    "R_Toe_x"     , "R_Toe_y"     , "R_Toe_z"     ,
    "Neck_x"      , "Neck_y"      , "Neck_z"      ,
    "L_Thorax_x"  , "L_Thorax_y"  , "L_Thorax_z"  ,
    "R_Thorax_x"  , "R_Thorax_y"  , "R_Thorax_z"  ,
    "Head_x"      , "Head_y"      , "Head_z"      ,
    "L_Shoulder_x", "L_Shoulder_y", "L_Shoulder_z",
    "R_Shoulder_x", "R_Shoulder_y", "R_Shoulder_z",
    "L_Elbow_x"   , "L_Elbow_y"   , "L_Elbow_z"   ,
    "R_Elbow_x"   , "R_Elbow_y"   , "R_Elbow_z"   ,
    "L_Wrist_x"   , "L_Wrist_y"   , "L_Wrist_z"   ,
    "R_Wrist_x"   , "R_Wrist_y"   , "R_Wrist_z"   ,
    "L_Hand_x"    , "L_Hand_y"    , "L_Hand_z"    ,
    "R_Hand_x"    , "R_Hand_y"    , "R_Hand_z"    ,
    ]

