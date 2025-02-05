import tensorflow as tf

from partial_rollout import PartialRollout

def env_runner(env, policy, num_local_steps, summary_writer, render, predictor,
                envWrap, noReward):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()  # reset lstm memory
    length = 0
    rewards = 0
    values = 0
    if predictor is not None:
        ep_bonus = 0
        life_bonus = 0

    while True:
        terminal_end = False
        rollout = PartialRollout(predictor is not None)

        for _ in range(num_local_steps):
            # run policy
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = action.argmax()
            state, reward, terminal, info = env.step(stepAct)
            if noReward:
                reward = 0.
            if render:
                env.render()

            curr_tuple = [last_state, action, reward, value_, terminal, last_features]
            if predictor is not None:
                bonus = predictor.pred_bonus(last_state, state, action)
                curr_tuple += [bonus, state]
                life_bonus += bonus
                ep_bonus += bonus

            # collect the experience
            rollout.add(*curr_tuple)
            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_features = features

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            if terminal or length >= timestep_limit:
                # prints summary of each life if envWrap==True else each game
                if predictor is not None:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d. Bonus: %.4f." % (rewards, length, life_bonus))
                    life_bonus = 0
                else:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, length))
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])
                length = 0
                rewards = 0
                terminal_end = True
                last_features = policy.get_initial_features()  # reset lstm memory
                # TODO: don't reset when gym timestep_limit increases, bootstrap -- doesn't matter for atari?
                # reset only if it hasn't already reseted
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()

            if info:
                # summarize full game including all lives (even if envWrap=True)
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                if terminal:
                    summary.value.add(tag='global/episode_value', simple_value=float(values))
                    values = 0
                    if predictor is not None:
                        summary.value.add(tag='global/episode_bonus', simple_value=float(ep_bonus))
                        ep_bonus = 0
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            if terminal_end:
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout