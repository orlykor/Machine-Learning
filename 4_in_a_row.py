from policies import base_policy as bp
import numpy as np
import pickle
import tensorflow as tf

EMPTY_VAL = 0
STATE_DIM = 15
ACTIONS = 7
ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
LAYERS = [50, 50]


class NeuralNetwork:

    @staticmethod
    def run_op_in_batches(session, op, batch_dict={}, batch_size=None,
                          extra_dict={}):

        """Return the result of op by running the network on small batches of
        batch_dict."""

        if batch_size is None:
            return session.run(op, feed_dict={**batch_dict, **extra_dict})

        n = len(next(iter(batch_dict.values())))

        s = []
        for i in range(0, n, batch_size):
            bd = {k: b[i: i + batch_size] for (k, b) in batch_dict.items()}
            s.append(session.run(op, feed_dict={**bd, **extra_dict}))

        if s[0] is not None:
            if np.ndim(s[0]):
                return np.concatenate(s)
            else:
                return np.asarray(s)

    def __init__(self, input_dim, output_dim, hidden_layers, weights, biases, name_prefix=""):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        self.weights = []
        self.biases = []
        self.session = tf.Session()
        self.input = tf.placeholder(tf.float32,
                                    shape=(None, self.input_dim),
                                    name="{}input".format(self.name_prefix)
                                    )
        self.layers = [self.input]
        # create the layers for the NN
        for i, width in enumerate(hidden_layers):
            a = self.affine("{}hidden{}".format(self.name_prefix, i),
                self.layers[-1], width, weights[i], biases[i])
            self.layers.append(a)
        self.output = self.affine("{}output".format(self.name_prefix),
                                  self.layers[-1], self.output_dim, weights[-1], biases[-1], relu=False)
        self.output_max = tf.reduce_max(self.output, axis=1)
        self.output_argmax = tf.argmax(self.output, axis=1)

    def take(self, indices):
        """Return an operation that takes values from network outputs.
        e.g. NN.predict_max() == NN.take(NN.predict_argmax())
        """

        mask = tf.one_hot(indices=indices, depth=self.output_dim,
                          dtype=tf.bool,
                          on_value=True, off_value=False, axis=-1)
        return tf.boolean_mask(self.output, mask)

    def affine(self, name_scope, input_tensor, out_channels, weight, bias, relu=True):
        input_shape = input_tensor.get_shape().as_list()
        input_channels = input_shape[-1]
        if weight is None:
            weight = tf.truncated_normal(
                                    [input_channels, out_channels],
                                    stddev=1.0 / np.sqrt(float(input_channels)))
            bias = tf.zeros([out_channels])

        with tf.variable_scope(name_scope):

            W = tf.get_variable("weights", initializer=weight)
            b = tf.get_variable("biases", initializer=bias)

            self.weights.append(W)
            self.biases.append(b)

            A = tf.matmul(input_tensor, W) + b

            if relu:
                R = tf.nn.relu(A)
                return R
            else:
                return A

    def train_in_batches(self, train_op, feed_dict, n_batches, batch_size):
        """Train the network by randomly sub-sampling feed_dict."""

        keys = tuple(feed_dict.keys())
        ds = DataSet(*[feed_dict[k] for k in keys])

        for i in range(n_batches):
            batch = ds.next_batch(batch_size)
            d = {k : b for (k, b) in zip(keys, batch)}
            self.session.run(train_op, d)

    def predict_argmax(self, inputs_feed, batch_size=None):
        """Return argmax on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_argmax,
                                      feed_dict, batch_size)

    def predict_max(self, inputs_feed, batch_size=None):
        """Return max on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_max,
                                      feed_dict, batch_size)


class DataSet:
    """A class for datasets (labeled data). Supports random batches."""

    def __init__(self, *args):
        """Create a new dataset."""

        self.X = [a.copy() for a in args]
        self.n = self.X[0].shape[0]
        self.ind = 0
        self.p = np.random.permutation(self.n)

    def next_batch(self, batch_size):
        """Get the next batch of size batch_size."""

        if batch_size > self.n:
            batch_size = self.n

        if self.ind + batch_size > self.n:
            # we reached end of epoch, so we shuffle the data
            self.p = np.random.permutation(self.n)
            self.ind = 0

        batch = self.p[self.ind: self.ind + batch_size]
        self.ind += batch_size

        return tuple(a[batch] for a in self.X)


class ReplayDB:
    """Holds previous games and allows sampling random combinations of
        (state, action, new state, reward)
    """

    def __init__(self, state_dim, db_size):
        """Create new DB of size db_size."""

        self.state_dim = state_dim
        self.db_size = db_size
        self._empty_state = np.zeros((1, self.state_dim))

        self.DB = np.rec.recarray(self.db_size, dtype=[
            ("s1", np.float32, self.state_dim),
            ("s2", np.float32, self.state_dim),
            ("a", np.int32),
            ("r", np.float32),
            ("done", np.bool)])
        self.clear()

    def clear(self):
        """Remove all entries from the DB."""

        self.index = 0
        self.n_items = 0
        self.full = False

    def store(self, s1, s2, a, r, done):
        """Store new samples in the DB."""
        self.DB[self.index] = (s1, s2, a, r, done)
        self.index += 1
        if self.index == self.db_size:
            self.index = 0

        self.n_items = min(self.n_items + 1, self.db_size)

    def sample(self, sample_size=None):
        """Get a random sample from the DB."""

        if self.full:
            db = self.DB
        else:
            db = self.DB[:self.index]

        if (sample_size is None) or (sample_size > self.n_items):
            return db
        else:
            return np.rec.array(np.random.choice(db, sample_size, False))

    def iter_samples(self, sample_size, n_samples):
        """Iterate over random samples from the DB."""

        if sample_size == 0:
            sample_size = self.n_items

        ind = self.n_items
        for i in range(n_samples):
            end = ind + sample_size
            if end > self.n_items:
                ind = 0
                end = sample_size
                p = np.random.permutation(self.n_items)
                db = np.rec.array(self.DB[p])
            yield db[ind: end]
            ind = end

def check_for_win(board, player_id, col):
    """
    check the board to see if last move was a winning move.
    :param board: the new board
    :param player_id: the player who made the move
    :param col: his action
    :return: True iff the player won with his move
    """

    row = 0
    # check which row was inserted last:
    for i in range(ROWS):
        if board[ROWS - 1 - i, col] == EMPTY_VAL:
            row = ROWS - i
            break

    # check horizontal:
    vec = board[row, :] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check vertical:
    vec = board[:, col] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check diagonals:
    vec = np.diagonal(board, col - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True
    vec = np.diagonal(np.fliplr(board), COLS - col - 1 - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    return False

def make_move(board, action, player_id):
    """
    return a new board with after performing the given move.
    :param board: original board
    :param action: move to make (column)
    :param player_id: player that made the move
    :return: new board after move was made
    """
    row = np.max(np.where(board[:, action] == EMPTY_VAL))
    new_board = np.copy(board)
    new_board[row, action] = player_id
    return new_board


class Policy203595541(bp.Policy):

    def cast_string_args(self, policy_args):

        policy_args['db_size'] = int(policy_args['db_size']) if 'db_size' in policy_args else 10000
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if 'learning_rate' in policy_args else 0.001
        policy_args['q_sample_size'] = int(policy_args['q_sample_size']) if 'q_sample_size' in policy_args else 0
        policy_args['batch_size'] = int(policy_args['batch_size']) if 'batch_size' in policy_args else 50
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else 0.5
        policy_args['num_batches'] = int(policy_args['num_batches']) if 'num_batches' in policy_args else 100
        policy_args['q_learning_iters'] = int(policy_args['q_learning_iters']) if 'q_learning_iters' in policy_args else 1
        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'policy203595541.model.pkl'
        policy_args['load_from'] = str(policy_args['load_from']) if 'load_from' in policy_args else 'models/policy203595541.model.pkl'


        return policy_args

    def init_run(self):
        weights = [None]*(len(LAYERS)+1)
        biases = [None]*(len(LAYERS)+1)
        try:
            model = pickle.load(open(self.load_from, 'rb'))
            self.epsilon = model[2]
            self.db = model[3]
            for i in range(len(LAYERS)+1):
                weights[i] = tf.constant(model[0][i])
                biases[i] = tf.constant(model[1][i])

        except:
            self.log("Model not found, initializing random weights.", 'STATUS')
            self.epsilon = 1
            self.db = ReplayDB(STATE_DIM, self.db_size)

        self.nn = NeuralNetwork(STATE_DIM, ACTIONS, LAYERS, weights, biases)
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_values = self.nn.take(self.actions)
        self.q_estimation = tf.placeholder(tf.float32, (None,),
                                           name="q_estimation")

        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.nn.session.run(tf.global_variables_initializer())

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if too_slow:
            if self.num_batches > 20:
                self.num_batches -= 10
                print("in learn: num_batches" + str(self.num_batches))

            elif self.batch_size > 25:
                self.batch_size -= 5
                print("in learn: batch_size" + str(self.batch_size))

        transform_board, win_move = self.transform_board(new_state)
        transform_board_flatten = np.append(transform_board.flatten(), np.array([win_move]))
        if prev_state is not None and self.mode == 'train':
            transform_board_prev, win_move_prev = self.transform_board(prev_state)
            transform_board_flatten_prev = np.append(transform_board_prev.flatten(), np.array([win_move_prev]))
            self.db.store(transform_board_flatten_prev,
                          transform_board_flatten, prev_action, reward, False)
        action = self.nn.predict_argmax(transform_board_flatten[None,:])[0]
        legal_actions = self.get_legal_moves(new_state)

        p = np.random.random()

        if (p < self.epsilon and self.mode == 'train') or action not in legal_actions:
            action = np.random.choice(legal_actions)

        return action

    def transform_board(self, state):
        legal_actions = self.get_legal_moves(state)
        win_move = 0.0

        transformed_board = np.zeros((2, COLS))
        for action in legal_actions:
            # for the first player
            new_board = make_move(state, action, self.id)
            if check_for_win(new_board, self.id, action):
                transformed_board[0,action] = 1.0
                win_move = 1.0

            # for the second player
            new_board = make_move(state, action, 3-self.id)
            if check_for_win(new_board, 3-self.id, action):
                transformed_board[1,action] = -1.0

        return transformed_board, win_move

    def learn(self, round, prev_state, prev_action, reward, new_state,
              too_slow):

        if too_slow:
            if self.num_batches > 20:
                self.num_batches -= 10
                print("in learn: num_batches" + str(self.num_batches))

            elif self.batch_size > 25:
                self.batch_size -= 5
                print("in learn: batch_size" + str(self.batch_size))

        if round % (self.game_duration // 40) == 0:
            self.epsilon /= 2
            print("EPSILON: " + str(self.epsilon))

        # the last act that won the round

        if prev_state is not None:
            transform_board, win_move = self.transform_board(new_state)
            transform_board_flatten = np.append(transform_board.flatten(), np.array([win_move]))
            transform_board_prev, win_move_prev = self.transform_board(prev_state)
            transform_board_flatten_prev = np.append(transform_board_prev, np.array([win_move_prev]))
            self.db.store(transform_board_flatten_prev,
                          transform_board_flatten, prev_action, reward, True)

        samples = self.db.iter_samples(self.q_sample_size,
                                       self.q_learning_iters)
        for sample in samples:
            v = self.nn.predict_max(sample.s2, self.batch_size)
            q = sample.r + (~sample.done * self.gamma * v)

            feed_dict = {
                self.nn.input: sample.s1,
                self.actions: sample.a,
                self.q_estimation: q
            }

            self.nn.train_in_batches(self.train_op, feed_dict,
                                     self.num_batches, self.batch_size)



    def save_model(self):

        return [self.nn.session.run(self.nn.weights), self.nn.session.run(self.nn.biases), self.epsilon, self.db], self.save_to
