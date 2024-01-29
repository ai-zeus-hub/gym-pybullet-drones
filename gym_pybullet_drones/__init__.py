from gymnasium.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:CtrlAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VelocityAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:HoverAviary',
)

register(
    id='single-tracking-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:SingleTrackingAviary',
)

register(
    id='multihover-aviary-v0',
    entry_point='gym_pybullet_drones.envs:MultiHoverAviary',
)

register(
    id='multi-tracking-aviary-v0',
    entry_point='gym_pybullet_drones.envs.multi_agent_rl:MultiTrackingAviary',
)
