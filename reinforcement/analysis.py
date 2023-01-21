# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.015
    # Less noise, less chance of falling on the bridge
    # More probability going right is worth it vs keeping left
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.2
    answerNoise = 0.01
    answerLivingReward = -1
    # Close exit: Low discount
    # Avoid cliff: Low noise
    # Reward: Less than 0, so it prefers the close exit
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.35
    answerNoise = 0.3
    answerLivingReward = -1
    # Avoid the cliff: High enough noise
    # Close exit: Low discount and negative rewards so it prefers less steps
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 0
    # Risk the cliff: Very low noise
    # Distant exit: Very high discount (value long term) and no negative reward
    # so it prefers long walk
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 0.95
    answerNoise = 0.5
    answerLivingReward = 0
    # Avoid the cliff: High noise
    # Prefer the distant exit: High discount and no negative reward!
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = -10
    # 0 discount: No point in long term analysis
    # Extremely high negative living reward: No matter what you do it won't be worth it
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
