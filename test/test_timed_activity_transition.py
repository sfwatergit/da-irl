import pickle
from unittest import TestCase

from impl.timed_activities.timed_activity_mdp import TimedActivityTransition


class TestTimedActivityTransition(TestCase):

    @classmethod
    def setUp(cls):
        with open('/home/sfeygin/python/da-irl/notebooks/states.pkl',
                  'rb') as f:
            cls.states = pickle.load(f)

        with open('/home/sfeygin/python/da-irl/notebooks/actions.pkl',
                  'rb') as f:
            cls.actions = pickle.load(f)

        cls.activity_states = cls.states['activity']
        cls.travel_states = cls.states['travel']
        cls.activity_actions = cls.actions['activity']
        cls.travel_actions = cls.actions['travel']

        all_states = {}
        all_states.update(cls.activity_states)
        all_states.update(cls.travel_states)
        cls.states = all_states

        all_actions = {}
        all_actions.update(cls.activity_actions)
        all_actions.update(cls.travel_actions)
        cls.actions = all_actions

    def _mockTransition(self):
        return TimedActivityTransition(self.states, self.actions, 1440)

    def get_state(self, symbol, time_index, done):
        return self.states[(symbol, done)][time_index]

    def get_action(self, symbol, duration):
        return self.actions[symbol][duration]

    def assertWrongState(self, p, state):
        self.assertEqual(state.symbol, 'F H')
        self.assertEqual(p, 0)

    def test_start_state(self):
        test_trans = self._mockTransition()
        start_state = self.get_state('S H', 0, 0)

        actions = [self.get_action('H', t) for t in range(0, 1440, 15)]

        for action in actions:
            p, next_state = test_trans(start_state.state_id, action.action_id)[
                0]
            if action.duration == 0:
                self.assertWrongState(p,next_state)
            elif 0 < action.duration < 1440:
                self.assertEqual(next_state.symbol, 'S H')
                self.assertTrue(next_state.is_done)
                self.assertEqual(p, 1)
            else:
                self.assertEqual(next_state.symbol, 'F H')
                self.assertTrue(next_state.is_done)
                self.assertEqual(p, 1)

    def test_transition_activity_to_trip(self):
        test_trans = self._mockTransition()

        # Transition from home to trip to work is possible if duration == 0:
        home_states = [self.get_state('S H', t, 1).state_id for t in range(
            15, 1440, 15)]

        action = self.get_action('Trip => W', 0).action_id

        for home_state in home_states:
            p,next_state = test_trans(home_state,action)[0]
            self.assertEqual(p,1)
            self.assertEqual(next_state.symbol,'Trip => 2 W')

        # Transition from home to trip to home:
        action = self.get_action('Trip => H', 0).action_id
        p, next_state = test_trans(home_states[0], action)[0]
        self.assertEqual(p,1)
        self.assertEqual(next_state.symbol, 'Trip => 2 H')

        # Transition from home to trip should not be possible if not done
        start_state = self.get_state('S H', 0, 0).state_id
        action = self.get_action('Trip => o', 0).action_id
        self.assertWrongState(*test_trans(start_state, action)[0])

        # Transition from episode 2 W to trip to home not possible if not
        # duration == 0
        work_ep_2 = self.get_state('2 W', 45, 1)
        action = self.get_action('Trip => H', 30)
        self.assertWrongState(*test_trans(work_ep_2.state_id,
                                         action.action_id)[0])


    def test_transition_trip_to_activity(self):
        test_trans = self._mockTransition()

        # There should never be a done indicator for a travel state

        trip_2W1_state = self.get_state('Trip => 2 W', 30, 0).state_id
        good_duration_good_activity = self.get_action('Trip => W',45).action_id
        p, next_state = test_trans(trip_2W1_state,
                                   good_duration_good_activity)[0]
        self.assertEqual(p,1)
        self.assertEqual(next_state.symbol, '2 W')
        self.assertEqual(next_state.time_index, 75)

        # The following are the unhappy path actions
        bad_duration_good_activity = self.get_action('Trip => W', 0).action_id
        good_duration_bad_activity = self.get_action('H', 30).action_id
        bad_duration_bad_activity = self.get_action('Trip => H', 0).action_id

        self.assertWrongState(*test_trans(trip_2W1_state,
                                          bad_duration_good_activity)[0])
        self.assertWrongState(*test_trans(trip_2W1_state,
                                          good_duration_bad_activity)[0])
        self.assertWrongState(*test_trans(trip_2W1_state,
                                          bad_duration_bad_activity)[0])

    def test_transition_between_activities(self):
        test_trans = self._mockTransition()

        # If at activity and not done can go to same activity

        test_3W_state = self.get_state('3 W', 30, 0).state_id
        good_duration_good_activity = self.get_action('W', 45).action_id
        p, next_state = test_trans(test_3W_state,
                                   good_duration_good_activity)[0]
        self.assertEqual(p, 1)
        self.assertEqual(next_state.symbol, '3 W')
        self.assertEqual(next_state.time_index, 75)  # should add duration to
        #  current time.

        # The following are the unhappy path actions
        good_duration_bad_activity = self.get_action('H', 90).action_id
        bad_duration_bad_activity = self.get_action('Trip => W', 0).action_id
        bad_duration_good_activity = self.get_action('W', 0).action_id

        self.assertWrongState(
            *test_trans(test_3W_state, good_duration_bad_activity)[0])
        self.assertWrongState(*test_trans(test_3W_state,
                                          bad_duration_good_activity)[0])
        self.assertWrongState(*test_trans(test_3W_state,
                                          bad_duration_bad_activity)[0])


