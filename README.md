# financial_portfolio_optimization
IT426 Fundamentals of Artificial Intelligence


In this python financial portfolio optimization project, the goal is to leverage artificial intelligence (AI) techniques to enhance decision-making processes in finance. Specifically, the focus is on optimizing a portfolio by maximizing expected returns while effectively managing associated risks. 

The optimization formula aims to maximize the sum of weighted expected returns while minimizing the portfolio's risk, subject to constraints such as a budget constraint and non-negativity constraints on the weights of individual assets. The solution approach involves creating an initial population, defining a repairing method to ensure adherence to constraints, and computing fitness scores. Two optimization techniques are then employed: local search and genetic algorithm.


Local search is an optimization technique that explores neighboring solutions to find the best possible outcome in a specific search space. It improves solutions by making small adjustments 
then evaluating the results, and repeating this process until an optimal solution is found, or a termination criterion is reached, in this case reaching 10,000 fitness evaluations.

And in the case of getting stuck in a local maximum, the local maximum will be recorded, and 
the search will start again randomly.

The genetic algorithm is an optimization technique that solves both constrained and unconstrained optimization problems, it’s based on natural selection as it repeatedly modifies a population of individual solutions using crossover, mutation or both together. It evolves 
solutions towards an optimal solution. 

The genetic algorithm implemented as follows:
1. Generate an initial population of p randomly generated solutions, and evaluate the fitness of every 
individual in the population. 
2. Use binary tournament selection (with replacement) twice to select two parents a and b. 
3. Run single-point crossover on these parents to give 2 children, e and f.
4. Run mutation on e and f to give two new solutions u and v. Evaluate the fitness of u and v. 
5. Run weakest replacement, firstly for u, then v. 
6. If a termination criterion has been reached, then stop. Otherwise return to step 2.

Termination Criterion: reaching a maximum number of fitness evaluations, which is 10,000 fitness 
evaluations.

Binary Tournament Selection: Randomly choose a chromosome from the population; call it a. Randomly 
choose another chromosome from the population; call this b. The fittest of these two (breaking ties 
randomly) becomes the selected parent. 

Single-Point Crossover: Randomly select a ‘crossover point’ which should be smaller than the total length 
of the chromosome. Take the two parents, and swap the gene values between them only for those genes 
which appear after the crossover point to create two new children.

Multi-Gene Mutation: You should use the following parameterised mutation operator, Mk, where k is an 
integer. The Mk operator simply repeats the following k times: choose a gene (locus) at random, and change 
it to a random new value within the permissible range. Hence M1 changes just one gene, while M15 might 
change 15 genes (maybe fewer than 15, because there is a chance that the same gene might be chosen more 
than once).

Weakest Replacement: If the new solution is fitter than the worst in the population, then overwrite the 
worst (breaking ties randomly) with the new solution.

The results obtained from both local search and the genetic algorithm are compared, and the best solution is selected based on the optimization criteria. The comprehensive approach outlined in this project integrates AI methodologies to tackle the complex task of financial portfolio optimization, contributing to the advancement of decision-making processes in the field of finance.
