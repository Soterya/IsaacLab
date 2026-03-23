#!/usr/bin/env python3
# ---------------------------------------
# 0) SMPL joint order for pose_body  
# ---------------------------------------
SMPL_BODY_JOINT_ORDER = [
    "L_Hip"     ,
    "R_Hip"     ,
    "Torso"     ,
    "L_Knee"    ,
    "R_Knee"    ,
    "Spine"     ,
    "L_Ankle"   ,
    "R_Ankle"   ,
    "Chest"     ,
    "L_Toe"     ,
    "R_Toe"     ,
    "Neck"      ,
    "L_Thorax"  ,
    "R_Thorax"  ,
    "Head"      ,
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow"   ,
    "R_Elbow"   ,
    "L_Wrist"   ,
    "R_Wrist"   ,
    "L_Hand"    ,
    "R_Hand"    ,
]

# ----------------------------
# 1) SMPL to ISAAC Mapping ---
# ----------------------------
SMPL_TO_ISAAC = {
    "L_Hip"     :("L_Hip_x"     , "L_Hip_y"     , "L_Hip_z"     ),
    "R_Hip"     :("R_Hip_x"     , "R_Hip_y"     , "R_Hip_z"     ),
    "Torso"     :("Torso_x"     , "Torso_y"     , "Torso_z"     ),
    "L_Knee"    :("L_Knee_x"    , "L_Knee_y"    , "L_Knee_z"    ),
    "R_Knee"    :("R_Knee_x"    , "R_Knee_y"    , "R_Knee_z"    ),
    "Spine"     :("Spine_x"     , "Spine_y"     , "Spine_z"     ),
    "L_Ankle"   :("L_Ankle_x"   , "L_Ankle_y"   , "L_Ankle_z"   ),
    "R_Ankle"   :("R_Ankle_x"   , "R_Ankle_y"   , "R_Ankle_z"   ),
    "Chest"     :("Chest_x"     , "Chest_y"     , "Chest_z"     ),
    "L_Toe"     :("L_Toe_x"     , "L_Toe_y"     , "L_Toe_z"     ),
    "R_Toe"     :("R_Toe_x"     , "R_Toe_y"     , "R_Toe_z"     ),
    "Neck"      :("Neck_x"      , "Neck_y"      , "Neck_z"      ),
    "L_Thorax"  :("L_Thorax_x"  , "L_Thorax_y"  , "L_Thorax_z"  ),
    "R_Thorax"  :("R_Thorax_x"  , "R_Thorax_y"  , "R_Thorax_z"  ),
    "Head"      :("Head_x"      , "Head_y"      , "Head_z"      ),
    "L_Shoulder":("L_Shoulder_x", "L_Shoulder_y", "L_Shoulder_z"),
    "R_Shoulder":("R_Shoulder_x", "R_Shoulder_y", "R_Shoulder_z"),
    "L_Elbow"   :("L_Elbow_x"   , "L_Elbow_y"   , "L_Elbow_z"   ),
    "R_Elbow"   :("R_Elbow_x"   , "R_Elbow_y"   , "R_Elbow_z"   ),
    "L_Wrist"   :("L_Wrist_x"   , "L_Wrist_y"   , "L_Wrist_z"   ),
    "R_Wrist"   :("R_Wrist_x"   , "R_Wrist_y"   , "R_Wrist_z"   ),
    "L_Hand"    :("L_Hand_x"    , "L_Hand_y"    , "L_Hand_z"    ),
    "R_Hand"    :("R_Hand_x"    , "R_Hand_y"    , "R_Hand_z"    ),
}