
import numpy as np
import tensorflow as tf

from src.models.helpers.network_models import (
    ValueNetwork,
    PolicyNetworkSoft,
)
from src.models.helpers.experience_buffer import (
    ExperienceBuffer,
)
from src.models.helpers.bound_action import (
    bound_actions,
    bound_actions_and_log_prob_densities_graph,
)


class SoftActorCritic:
    """A Soft Actor Critic learning algorithm. TODO: More"""

    def __init__(
            self,
            rng: np.random.Generator,
            future_reward_discount_gamma: float,
            entropy_scale_alpha_initial: float,
            target_entropy: float,
            entropy_scale_optimizer,
            entropy_scale_optimizer_args: dict,
            training_minimum_experiences: int,
            training_batch_size: int,
            training_target_update_momentum_tau: float,
            training_l2_norm_scale_value: float,
            training_l2_norm_scale_policy: float,
            experience_buffer_args: dict,
            network_args: dict,
            action_bound_mode: str or None,
    ) -> None:

        self.rng = rng

        self.future_reward_discount_gamma = future_reward_discount_gamma

        # Gradients are applied on the log value. This way, entropy_scale_alpha is restricted to positive range
        self.log_entropy_scale_alpha = tf.Variable(np.log(entropy_scale_alpha_initial),
                                                   trainable=True, dtype=tf.float32)
        self.target_entropy = target_entropy
        self.entropy_scale_alpha_optimizer = entropy_scale_optimizer(**entropy_scale_optimizer_args)

        self.training_minimum_experiences = training_minimum_experiences
        self.training_batch_size = training_batch_size
        self.training_target_update_momentum_tau = training_target_update_momentum_tau

        self.l2_norm_scale_value = training_l2_norm_scale_value
        self.l2_norm_scale_policy = training_l2_norm_scale_policy

        self.experience_buffer = ExperienceBuffer(rng=rng, **experience_buffer_args)

        self.bound_actions = lambda actions: bound_actions(
            actions,
            mode=action_bound_mode,
        )
        self.bound_actions_and_log_prob_densities = lambda actions, action_log_prob_densities: bound_actions_and_log_prob_densities_graph(
            actions,
            action_log_prob_densities,
            mode=action_bound_mode,
        )

        self.networks: dict = {
            'value': [],
            'policy': [],
        }
        self._initialize_networks(**network_args)

    def _initialize_networks(
            self,
            value_network_args: dict,
            policy_network_args: dict,
            value_network_optimizer,
            value_network_optimizer_args: dict,
            # value_network_loss,
            policy_network_optimizer,
            policy_network_optimizer_args: dict,
            # policy_network_loss,
            size_state: int,
            num_actions: int,
    ) -> None:

        # create networks
        # TODO: The number of value networks should probably be part of config
        #  although tf methods make it annoying to do more than 2

        for _ in range(2):
            self.networks['value'].append(
                {
                    'primary': ValueNetwork(**value_network_args),
                    'target': ValueNetwork(**value_network_args),
                }
            )

        for _ in range(1):
            self.networks['policy'].append(
                {
                    'primary': PolicyNetworkSoft(num_actions=num_actions, **policy_network_args),
                    'target': PolicyNetworkSoft(num_actions=num_actions, **policy_network_args),
                }
            )

        # compile networks
        dummy_state = self.rng.random((1, size_state), dtype='float32')
        dummy_action = self.rng.random((1, num_actions), dtype='float32')
        for network_type, network_list in self.networks.items():
            # create dummy input
            if network_type == 'policy':
                dummy_input = dummy_state
                optimizer = policy_network_optimizer
                optimizer_args = policy_network_optimizer_args
                # loss = policy_network_loss
            elif network_type == 'value':
                dummy_input = np.concatenate([dummy_state, dummy_action], axis=1)
                optimizer = value_network_optimizer
                optimizer_args = value_network_optimizer_args
                # loss = value_network_loss
            # feed dummy input, compile primary
            for network_pair in network_list:
                for network_rank, network in network_pair.items():
                    network.initialize_inputs(dummy_input)
                network_pair['primary'].compile(
                    optimizer=optimizer(**optimizer_args),
                    # loss=loss,  # TODO: doesnt work anymore when saving/loading currently, probably fixable
                    jit_compile=True,
                )
        self.update_target_networks(tau_target_update_momentum=1.0)

    @tf.function
    def update_target_networks(
            self,
            tau_target_update_momentum: float,
    ) -> None:

        if tau_target_update_momentum == 0:
            return

        for network_list in self.networks.values():
            for network_pair in network_list:
                for v_primary, v_target in zip(network_pair['primary'].trainable_variables,
                                               network_pair['target'].trainable_variables):
                    v_target.assign(tau_target_update_momentum * v_primary
                                    + (1 - tau_target_update_momentum) * v_target)

    def get_action(
            self,
            state: np.ndarray,
    ) -> np.ndarray:
        """
        Get action from neural net and convert to flat numpy array.
        """

        actions, _ = self.networks['policy'][0]['primary'].get_action_and_log_prob_density(state=state,
                                                                                           print_stds=False)

        actions_bounded = self.bound_actions(actions.numpy().flatten())

        return actions_bounded

    def add_experience(
            self,
            experience: dict,
    ) -> None:
        self.experience_buffer.add_experience(experience=experience)

    def train(
            self,
            toggle_train_value_networks: bool = True,
            toggle_train_policy_network: bool = True,
            toggle_train_entropy_scale_alpha: bool = True,
    ) -> tuple:

        if self.experience_buffer.get_len() < self.training_minimum_experiences:
            return np.nan, np.nan

        (
            sample_experiences,
            experience_ids,
            sample_importance_weights,
        ) = self.experience_buffer.sample(batch_size=self.training_batch_size)

        (
            mean_log_prob_density,
            value_loss,
            td_error,
        ) = self.train_graph(
            states=np.array(
                [experience['state'] for experience in sample_experiences], dtype='float32'),
            actions=np.array(
                [experience['action'] for experience in sample_experiences], dtype='float32'),
            rewards=np.array(
                [experience['reward'] for experience in sample_experiences], dtype='float32')[np.newaxis].transpose(),
            next_states=np.array(
                [experience['next_state'] for experience in sample_experiences], dtype='float32'),
            sample_importance_weights=np.array(
                sample_importance_weights, dtype='float32')[np.newaxis].transpose(),
            toggle_train_value_networks=toggle_train_value_networks,
            toggle_train_policy_network=toggle_train_policy_network,
            toggle_train_entropy_scale_alpha=toggle_train_entropy_scale_alpha,
        )

        self.experience_buffer.adjust_priorities(
            experience_ids=experience_ids,
            new_priorities=abs(td_error.numpy().flatten())
        )

        return mean_log_prob_density, value_loss

    @tf.function
    def train_graph(
            self,
            states,
            actions,
            rewards,
            next_states,
            sample_importance_weights,
            toggle_train_value_networks,
            toggle_train_policy_network,
            toggle_train_entropy_scale_alpha,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        if toggle_train_value_networks:
            # Construct target: r(s, a) + \gamma * (Q_hat(s', a') - \alpha * log prob(a'|s'))
            # In the mean sense, the entropy reward term increases the value of s+a that lead to a' with high
            #  variance (because we really look at pdfs, not probabilities).
            target_q = rewards
            if self.future_reward_discount_gamma > 0.0:
                (
                    next_actions,
                    next_action_log_prob_densities,
                ) = self.networks['policy'][0]['primary'].get_action_and_log_prob_density(state=next_states)
                next_actions, next_action_log_prob_densities = self.bound_actions_and_log_prob_densities(next_actions, next_action_log_prob_densities)
                value_network_input = tf.concat([next_states, next_actions], axis=1)
                next_states_value_estimates_1 = self.networks['value'][0]['target'].call(value_network_input)
                next_states_value_estimates_2 = self.networks['value'][1]['target'].call(value_network_input)
                next_states_conservative_value_estimates = tf.reduce_min(
                    [next_states_value_estimates_1, next_states_value_estimates_2], axis=0)
                target_q = target_q + self.future_reward_discount_gamma * (
                    next_states_conservative_value_estimates
                    - tf.exp(self.log_entropy_scale_alpha) * next_action_log_prob_densities
                )

            value_network_input_batch = tf.concat([states, actions], axis=1)
            for network in [self.networks['value'][0]['primary'], self.networks['value'][1]['primary']]:
                with tf.GradientTape() as tape:  # Autograd
                    estimated_q = network.call(value_network_input_batch, training=True)
                    td_error = estimated_q - target_q

                    # from bengio deep learning book: we note
                    # that for neural networks, we typically choose to use a parameter norm penalty
                    # Ω that only penalizes the interaction weights, i.e we leave the offsets unregular-
                    # ized. The offsets typically require less data to fit accurately than the weights.
                    # Each weight specifies how two variables interact. Fitting the weigth well requires
                    # observing both variables in a variety of conditions. Each offset controls only a
                    # single variable. This means that we do not induce too much variance by leaving
                    # the offsets unregularized. Also, regularizing the offsets can introduce a significant
                    # amount of underfitting.

                    l2_norm_loss = 0.0
                    if self.l2_norm_scale_value != 0:
                        for weights_layer in network.trainable_variables[0::2]:  # skip bias layers
                            l2_norm_loss += tf.reduce_sum(tf.square(weights_layer))  # sum instead of mean is fine because every weight affects only itself
                        l2_norm_loss = self.l2_norm_scale_value * l2_norm_loss  # w_new = w_old - lr * d Loss / d_w - lr * norm_scale * w_old

                    value_loss = (
                        tf.reduce_mean(sample_importance_weights * td_error ** 2)
                        + l2_norm_loss
                    )
                gradients = tape.gradient(target=value_loss, sources=network.trainable_variables)
                network.optimizer.apply_gradients(zip(gradients, network.trainable_variables))

            self.update_target_networks(tau_target_update_momentum=self.training_target_update_momentum_tau)

        if toggle_train_policy_network:
            policy_network_input_batch = states
            network = self.networks['policy'][0]['primary']
            with tf.GradientTape() as tape:
                (
                    policy_actions,
                    policy_action_log_prob_densities,
                ) = network.get_action_and_log_prob_density(state=policy_network_input_batch, training=True)
                policy_actions, policy_action_log_prob_densities = self.bound_actions_and_log_prob_densities(policy_actions, policy_action_log_prob_densities)
                value_network_input_batch = tf.concat([states, policy_actions], axis=1)
                # target or primary? primary -> faster updates, target -> stable but delayed
                value_estimate_1 = self.networks['value'][0]['primary'].call(value_network_input_batch)
                value_estimate_2 = self.networks['value'][1]['primary'].call(value_network_input_batch)
                value_estimate_min = tf.reduce_min([value_estimate_1, value_estimate_2], axis=0)

                l2_norm_loss = 0.0
                if self.l2_norm_scale_policy != 0:
                    for weights_layer in network.trainable_variables[0::2]:  # skip bias layers
                        l2_norm_loss += tf.reduce_sum(tf.square(weights_layer))  # sum instead of mean is fine because every weight affects only itself
                    l2_norm_loss = self.l2_norm_scale_policy * l2_norm_loss  # w_new = w_old - lr * d Loss / d_w - lr * norm_scale * w_old

                policy_loss = tf.reduce_mean(
                    # pull towards high value:
                    sample_importance_weights * -value_estimate_min
                    # pulls towards high variance - we want to minimize mean log probs -> more uncertainty:
                    + tf.exp(self.log_entropy_scale_alpha) * policy_action_log_prob_densities
                    + l2_norm_loss  # todo: we can probably move this out of the reduce_mean
                )
            gradients = tape.gradient(target=policy_loss, sources=network.trainable_variables)
            network.optimizer.apply_gradients(zip(gradients, network.trainable_variables))

        if toggle_train_entropy_scale_alpha:
            with tf.GradientTape() as tape:
                # if (logprobs (negative) + target entropy) > 0, increase weight of variance
                #  to encourage higher variance, thus bringing logprobs + target entropy closer to zero
                alpha_loss = -self.log_entropy_scale_alpha * tf.reduce_mean(
                    tf.add(policy_action_log_prob_densities, self.target_entropy))
            alpha_gradients = tape.gradient(target=alpha_loss, sources=[self.log_entropy_scale_alpha])
            self.entropy_scale_alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_entropy_scale_alpha]))

        return tf.reduce_mean(policy_action_log_prob_densities), value_loss, td_error
