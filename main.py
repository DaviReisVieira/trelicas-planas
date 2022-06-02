from funcoesTermosol import *
from solver import *

input = input('Selecione o arquivo (1,2): ')
entrada = f'entrada_{input}.xlsx'

[nn,N,nm,Inc,nc,F,nr,R] = importa(entrada)

# plota(N, Inc)

solver = Solver()

# Add nodes
nodes_transposed = N.T
for x, y in nodes_transposed:
    solver.add_node(x,y)

# Add elements
for n1, n2, elasticity, area in Inc:
    solver.create_element(n1, n2, elasticity, area)

# Add constraints
for node, direction in R:
    solver.add_constraint(node,direction)

# Solve
reactions, strain, displacement = solver.solver(F,1e-8,1000)
# Tensoes internas [Pa]
stress_element = np.array([[el.elasticity] for el in solver.elements])*strain
# Forças internas [N]
internal_forces = np.array([[el.area] for el in solver.elements])*stress_element
geraSaida("saida", reactions, displacement, strain, internal_forces, stress_element)

nn_new = solver.get_nodes()

nn_new = (nn_new.T.reshape(displacement.shape) + (displacement*1e4)).reshape(nn_new.T.shape).T
# plota(nn_new, Inc)
