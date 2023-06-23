"""
perturbations
Module concerned with perturbations in orbit 

NOTE:
Many of the concepts encapsulated in this module will not work with the 
    previous ODE's supplied.
Users should write a new ODE function and call the corresponding perturbations.
This is due to the permutations of different perturbations users can implement 
    and the variety of inputs required.
Users should be able to use the basic two-body function as a stepping stone.

Each perturbation method will note how to implement it within an ODE function 

"""

from typing import List, Union

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from . import astroconsts as ast
from .orbitalcore import ivp_result_to_dataframe


def drag_event_listener():
    raise NotImplementedError
    return


def cowells_method():
    raise NotImplementedError
    return


def variation_of_parameters():
    raise NotImplementedError
    return


def enckes_method():
    raise NotImplementedError
    return


def drag_perturbation():
    raise NotImplementedError
    return


def solar_position():
    raise NotImplementedError
    return


def solar_radiation_perturbation():
    raise NotImplementedError
    return


def oblateness_perturbation():
    raise NotImplementedError
    return


def n_body_perturbation():
    raise NotImplementedError
    return
