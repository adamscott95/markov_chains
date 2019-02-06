# markov_chains.py
"""Volume II: Markov Chains.
Adam Robertson
Math 321
November 7, 2018
"""

import numpy as np
from sklearn.preprocessing import normalize
from scipy import linalg as la

# Problem 1
def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    #create an n x n matrix of values between zero and 1
    transition = np.random.rand(n,n)
    #normalize each column of the transition matrix
    #store sums of columns in vector
    col_sums = transition.sum(axis=0)
    #divide each column of transition matrix by the length of its column
    new_transition = transition / col_sums[np.newaxis, :]
    return new_transition


# Problem 2
def forecast(days):
    """Forecast the weather for a given number of future days given that the today day is hot."""
    #transition matrix for hot or cold day
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])

    #list of whether the weather will be hot or cold
    weather = []
    #assume that today is hot
    current_day = 0
    for _ in range(days):
        # Sample from a binomial distribution to choose a new state.
        if current_day == 0:
            #current day is hot, draw from probability for hot day
            current_day = np.random.binomial(1, transition[1, 0])
        else:
            #current day is cold, draw from probability for cold day
            current_day = np.random.binomial(1, transition[1, 1])
        weather.append(current_day)
    return weather


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    #transition matrix
    transition = np.array([[0.5, 0.3, 0.1, 0.],[0.3, 0.3, 0.3, 0.3],[0.2, 0.3, 0.4, 0.5],[0., 0.1, 0.2, 0.2]])
    #forecast of days
    fcast = []
    #today is mild
    current_day = 1
    for _ in range(days):
        # select next day based on the weather of the current day
        if current_day == 0:
            # Draw from multinomial distribution with n = 1 (choose a single day)
            prob = np.random.multinomial(1, transition[:,0])
            # get index where value is 1
            current_day = np.argmax(prob)
        elif current_day == 1:
            # Draw from multinomial distribution with n = 1 (choose a single day)
            prob = np.random.multinomial(1, transition[:,1])
            current_day = np.argmax(prob)
        elif current_day == 2:
            # Draw from multinomial distribution with n = 1 (choose a single day)
            prob = np.random.multinomial(1, transition[:,2])
            current_day = np.argmax(prob)
        else:
            # Draw from multinomial distribution with n = 1 (choose a single day)
            prob = np.random.multinomial(1, transition[:,3])
            current_day = np.argmax(prob)
        #add current day to forecast
        fcast.append(current_day)
    return fcast

# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    #get size of A
    n = len(A)
    #create an normalize a random state distrubtion vector
    x = np.random.rand(n,1)
    x = x / np.linalg.norm(x, ord=1)
    k = 0
    while k < N:
        #compute Ax
        x_k_1 = A @ x
        #if the new vector is within epsilon of the previous vector, return the new vector
        if la.norm(x - x_k_1) < tol:
            # the vector is the steady state distribution
            return x_k_1
        else:
            # continue to multiply Ax
            x = x_k_1
        k += 1
    #after N iterations, x has not converged to a consistent matrix
    raise ValueError("Matrix A does not converge")


# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        # Count number of unique words in training set
        # create set to store unique words in
        set_of_words = set()
        myfile = open(filename)
        for line in myfile:
            # divide line into words, add words to set
            for x in line.strip().split():
                set_of_words.add(x)
        myfile.close()
        # size of set of words = number of unique words
        num_unique_words = len(set_of_words)

        # Initialize a square array of zeros with side length equal to the number of unique words + 2 (for $tart and $top
        self.transition = np.zeros((num_unique_words + 2, num_unique_words + 2))
        # The transition matrix will be column-stochastic
        
        # Initialize list of states, beginning with "$tart"
        self.states = ["$tart"]
        # read in file again
        myfile = open(filename)
        for line in myfile:
            #split sentence into words
            words = line.strip().split()
            #add new words to list of states
            for x in words:
                if x not in self.states:
                    self.states.append(x)
            
            #convert list of words into a list of indices indiating which row and column of transition matrix each word corresponds to
            word_indices = []
            for x in words:
                #assign word to index that it corresponds to in the transition matrix
                word_indices.append(self.states.index(x))
            #add 1 to entry of transition matrix that links $tart with the first word
            #transition matrix is column-stochastic
            self.transition[word_indices[0], 0] += 1
            
            # for each consecutive pair of words, 
            # add 1 to the entry of the transition matrix 
            # that transitions from the first word to the second word
            i = 1
            while i < len(word_indices):
                #transition from previous word to current word
                #transition matrix is column stochastic
                self.transition[word_indices[i], word_indices[i-1]] += 1
                i += 1
            
            #add 1 to entry of transition matrix corresponding to transitioning from the last word of the sentence to the stop state
            #transition matrix is column-stochastic
            self.transition[num_unique_words + 1, word_indices[len(word_indices) - 1]] += 1
        # make $top state transition to itself
        self.transition[num_unique_words + 1, num_unique_words + 1] += 1 
        myfile.close()
        self.states.append("$top")

        #normalize each column by dividing by column sums
        #store sums of columns in vector
        col_sums = self.transition.sum(axis=0)
        #divide each column of transition matrix by the length of its column
        self.transition = self.transition / col_sums[np.newaxis, :]

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        #initalize array of words and word indices
        words = []
        word_indices = []

        #take a draw of the first column of the transition matrix
        index = np.random.multinomial(1, self.transition[:,0])
        
        #get the index of the first randomly drawn word
        current_ind = np.argmax(index)
        word_indices.append(current_ind)

        #draw through the matrix repeatedly until arriving at the '$top' column
        while current_ind != len(self.transition) - 1:
            #take a draw of the current_indth column of the transition matrix
            index = np.random.multinomial(1, self.transition[:,current_ind])
            current_ind = np.argmax(index)
            #append index of new word to list of indices
            word_indices.append(current_ind)

        #collect words from word_indices
        i = 0
        while i < len(word_indices) - 1:
            #get word from list of states using word index
            word = self.states[word_indices[i]]
            words.append(word)
            i += 1

        #join and return array as a string
        return " ".join(words)








