# Orbital Mechanics
Set of Orbital Mechanics functions transcribed from MATLAB or textbooks in Python to support larger Orbital Mechanics projects and classes.

## Structure
The repository is broken up into 5 main categories:

- Basic Orbital Mechanics
- Rendezvous/Relative Motion Mechanics 
- Perturbational Effects 
- Observations/Filtering
- Optimal Transfer

This structure closely follows the breakdown of the Orbital Mechanics path at California Polytechnic State University - San Luis Obispo: AERO 351 (Basic Orbital Mechanics), AERO 452 (Rendezvous/Relative Motion, Perturbational Effects), and AERO 557 (Observations/Filtering, Optimal Transfer).

## Approach
The approach in building this repository was to avoid using unnecessary classes and containers to minimize the learning curve and to provide the simplest form of usability for students and faculty. That being said, NumPy and MatPlotLib are heavily used throughout the repository to enable the math to be simply executed. Many optimization opportunities were not taken to minimize confusion and enable users to peek into the source code and understand what is happening. 

## Extending the Package
I hope students will continue to add on to this package and improve usability for the curriculum. My intentions were to provide resources (e.g., pointing to specific textbooks/algorithms) for people to refer back to and ensure the code is very readable.

Extensions/additional functions should always include type-hinting in the arguments and return statement and should include docstrings at the top of each function for context. The function docstring should include the purpose, where the function is adapted from (e.g., "Orbital Mechanics for Engineering Students", Curtis), and the explanations for each input and return argument.
