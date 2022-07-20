from devito import Grid, Function, Eq,Operator
grid = Grid(shape=(4, 4))
f = Function(name='f', grid=grid)
g = Function(name='g', grid=grid, space_order=8)

eq = Eq(f,g)

op = Operator(eq)

print(op.ccode)