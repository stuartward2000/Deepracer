def response(agent, env, env_response):
    reward_params = {}
    episode_status = {}
    for env_agent in env.agent_list:
        agent_id = (
            'agent' if len(agent.name.split('_')) <= 1 
            else agent.name.split('_')[-1]
        )
        racecar_id = (
            'racecar' if len(env_agent.ctrl._agent_name_.split('_')) <= 1 
            else env_agent.ctrl._agent_name_.split('_')[-1]
        )
        if (
            (agent_id == racecar_id)
            or
            (agent_id == 'agent' and racecar_id == 'racecar')
        ):
            reward_params = env_agent.ctrl._reward_params_
            episode_status = env_agent.ctrl._reset_rules_manager.get_dones()
    env_response.info['reward_params'] = reward_params
    env_response.info['episode_status'] = episode_status
    return env_response
