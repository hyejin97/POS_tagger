POS tagging with Hidden Markov Model
written by Hyejin Kim(hk3342@nyu.edu)

1. How to run the system

- Run 'python main.py'. The program runs in the following order;

1)Load train files
2)Preprocessing data with OOV strategies
3)Train the transition and emission
4)Convert the counts into probabilities 
5)Load the test file
6)Create a transducer for test data 
7)Execute viterbi algorithm 
8)Choose the highest POS tags for each observed word
9)Write tags to the output file
    

2. Strategies to improve the score

- OOV strategies
  1) Treated words occurring once in training collectively as unknown words.
  2) Used suffixes, punctuations, digits, and uppercase to predict the pos tags of oov words. 
  Added additional 'unknown' classes and assigned unknown words to each class by detecting their corresponding morphology.

- Smoothing
   Applied laplace smoothing when calculating transition and emission probabilities.
   
  