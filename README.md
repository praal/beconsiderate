
## Installation instructions

The code has the following requirements: 

- Python3
- numpy
- matplotlib


## How to run the code

To run the code, first move to the *src* folder. There are multiple scripts to run in *tests* folder, and each one reproduces one of the experiments in the paper. 

- **Baselines (Table 1)**
To run this experiment you need to execute the lines below (each line is for one column). 
```
python3 -m tests.kitchen_salad
```
```
python3 -m tests.kitchen_peanut
```
```
python3 -m tests.kitchen_salt
```
```
python3 -m tests.kitchen_cookie
```

- **Effect of Caring Coefficient (Figure 2)** 
To run this experiment you need to execute the line below. foldername specifies the name of folder to store data that is needed to draw the plot, the folder will be created at *datasets* folder, and the plot will be stored at *fig.png* at the specified folder. The parameters of alpha_start and alpha_end specify the range of caring coefficient. In the plot of the paper, we set alpha2_start to 0 and alpha2_end to 700. 

```
python3 -m tests.craftgen <foldername> <alpha2_start> <alpha2_end> 
```

- **Using different notions of reward - difference between Equation 1, 2, and 3. (Figure 3)**
To run this experiment you need to execute the line below. The parameter func defines the reward function to use. To reproduce Equation 1 set func to "sum", for Equation 2 set it to "min", for Equation 3 set it to "neg", for the Krakovna et al baseline set it to "baseline", and for the q-learning baseline set it to "nonaug".

```
python3 -m tests.craftfunc <func>
```

- **Treating agents differently (Figure 4)**
To run this experiment you need to execute the line below. The parameter alpha2 specifies how much we care about agent 2 (owner of the garden). To reproduce the policies in the paper, set alpha2 to 0, 1, and 10. 

```
python3 -m tests.craftdiff <alpha2>
```

- **Effect of options (Figure 5)**
To run this experiment you need to execute the line below. The parameter alpha2 specifies how much we care about other agents. To reproduce the policies in the paper set alpha2 to 1, 5 and 25.

```
python3 -m tests.craftoption <alpha2>
```
