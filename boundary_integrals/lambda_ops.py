import numpy as np


def add(f, g):
    return lambda t: f(t) + g(t)


def mult(f, g):
    return lambda t: f(t)*g(t)


def scalmult(f, c):
    return lambda t: c * f(t)

def diff(f, g):
    return add(f, scalmult(g, -1))

def scaldiff(f, c):
    return lambda t: f(t) - c

def pow(f, p):
    return lambda t: f(t) ** p


def div(f, g):
    return mult(f, pow(g, -1))


def dot(f1,f2,g1,g2):
    return add(mult(f1,g1), mult(f2,g2))


def magn(f1,f2):
    return dot(f1,f2,f1,f2)


def cross(f1,f2,g1,g2):
    return dot(f1,f2,g2,scalmult(g1, -1))

