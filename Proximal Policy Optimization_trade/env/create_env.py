
# create environment
# OPTIONS
# create environment for train and test
PATH_TRAIN = "./data/train/"
PATH_TEST = "./data/test/"
TIMESTEP = 30  # window size
environment = create_btc_env(window_size=TIMESTEP, path=PATH_TRAIN, train=True)
test_environment = create_btc_env(window_size=TIMESTEP, path=PATH_TEST, train=False)

# create spec for network and baseline
network_spec = create_network_spec() # json format
baseline_spec = create_baseline_spec()

# create agent
agent = PPOAgent(
    discount=0.9999,
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='timesteps',  # 'episodes',
        # 10 episodes per update
        batch_size=32,
        # # Every 10 episodes
        frequency=10
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=50000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.0,  # None
    # PGModel

    baseline_mode='states',
    baseline=dict(type='custom', network=baseline_spec),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=(1e-4)  # 3e-4
        ),
        num_steps=5
    ),
    gae_lambda=0,  # 0
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=(1e-4)  # 1e-4
    ),
    subsampling_fraction=0.2,  # 0.1
    optimization_steps=10,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)
