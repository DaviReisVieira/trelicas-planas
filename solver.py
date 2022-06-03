import math
import numpy as np
from funcoesTermosol import insert_constraints

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Node(Point):
    def __init__(self, x,y, id):
        super().__init__(x,y)
        self.id = id

    def __str__(self):
        return "Node {} - ({}, {})".format(self.id, self.x, self.y)


class Element():
    def __init__(self, n1, n2, id, elasticity, area):
        self.n1 = n1
        self.n2 = n2
        self.id = id
        self.elasticity = elasticity
        self.area = area
        self.length = self.element_length()
        self.coords_difference = np.array([[self.n2.x - self.n1.x],[self.n2.y - self.n1.y]])
        self.stiff = (elasticity*area/self.length) * (self.coords_difference.dot(self.coords_difference.T)/(np.linalg.norm(self.coords_difference)**2))
        self.cos = (self.n2.x - self.n1.x)/self.length
        self.sin = (self.n2.y - self.n1.y)/self.length

    def element_length(self):
        """
        Calculates the length of the element
        Input: None
        Output: Length of the element
        """
        return math.sqrt((self.n1.x - self.n2.x)**2 + (self.n1.y - self.n2.y)**2)
    
    def connectivity_matrix(self, total_nodes):
        """
        Calculates the connectivity matrix of the element
        Input: total_nodes
            - total_nodes: Total number of nodes
        Output: Connectivity matrix of the element
        """
        connectivity = [0]*total_nodes
        connectivity[self.n1.id - 1] = -1
        connectivity[self.n2.id - 1] = 1
        return np.array(connectivity)

    def strain(self, u1, v1, u2, v2):
        """
        Calculates the strain of the element
        Input: u1, v1, u2, v2
            - u1, v1: Displacements of the first node
            - u2, v2: Displacements of the second node
        Output: Strain of the element
        """
        self.strain_value = np.array([-self.cos, -self.sin,self.cos, self.sin]).dot(np.array([[u1], [v1], [u2], [v2]]))/self.length

    def stiffness(self,total_nodes):
        """
        Calculates the stiffness matrix of the element
        Input: total_nodes
            - total_nodes: Total number of nodes
        Output: Stiffness matrix of the element
        """
        array = np.array([self.connectivity_matrix(total_nodes)])
        return np.kron(array.T.dot(array), self.stiff)
    
    def __str__(self):
        return "Element: {} | {} | E={}Pa | A={}m²".format(self.n1, self.n2, self.elasticity, self.area)


class Solver():
    def __init__(self):
        print('------- Initializing solver -------\n')
        self.nodes = []
        self.elements = []
        self.constraints = []

    def add_node(self, x,y):
        """
        Function to add a node to the list of nodes
        Input: x,y
            - x,y: coordinates of the node
        Output: None
        """
        new_node = Node(x,y, len(self.nodes)+1)
        self.nodes.append(new_node)
        print(f'- Added {new_node.__str__()}\n')

    def create_element(self, n1, n2, E, A):
        """
        Function to create an element
        Input: n1, n2, E, A
            - n1, n2: nodes of the element
            - E: elasticity
            - A: area
        Output: None
        """
        n1 = self.nodes[int(n1) - 1]
        n2 = self.nodes[int(n2) - 1]
        new_element = Element(n1, n2, len(self.elements)+1, E, A)
        self.elements.append(new_element)
        print(f'- Added {new_element.__str__()}\n')

    def add_constraint(self,node,direction):
        """
        Function to add a constraint to the list of constraints
        Input: node, direction
            - node: node to be constrained
            - direction: direction of the constraint
        Output: None
        """
        node = int(node)
        direction = int(direction)
        self.constraints = insert_constraints(self.constraints, node, direction)
        print(f"- Added constraint to node {node} in {'x' if direction == 1 else 'y'} direction\n")

    def strain(self):
        """
        Function to calculate the strain of the elements
        Input: None
        Output: None
        """
        return np.array([el.strain_value for el in self.elements])

    def stiffness_sum(self):
        """
        Function to calculate the stiffness matrix
        Input: None
        Output: None
        """
        return sum([el.stiffness(len(self.nodes)) for el in self.elements])

    def get_nodes(self):
        """
        Function to get the nodes
        Input: None
        Output: nodes
            - nodes: list of nodes
        """
        return np.array([[n.x, n.y] for n in self.nodes]).T

    def gauss_seidel(self, K, F, n_iterations, tolerance):
        """
        Function to solve the system of equations using the Gauss-Seidel method
        Input: K, F, n_iterations, tolerance
            - K: stiffness matrix
            - F: force vector
            - n_iterations: number of iterations
            - tolerance: tolerance of the method
        Output: u
            - u: displacements
        """
        x = np.zeros_like(F, dtype=np.double)
        
        for k in range(n_iterations):
            x_old  = x.copy()
            for i in range(K.shape[0]):
                x[i] = (F[i] - np.dot(K[i, :i], x[:i]) - np.dot(K[i, (i + 1):], x_old[(i + 1):])) / K[i ,i]
                
            calc_tol = np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
            if calc_tol < tolerance:
                print(f"Max number of iteration reached: {k} with Relative Difference {calc_tol}")
                return x
        
        print(f"Max number of iteration reached: {k} with Relative Difference {calc_tol}")
            
        return x

    def solver(self, F, tolerance, max_iterations):
        """
        Function to solve the system of equations
        Input: F, tolerance, max_iterations
            - F: force vector
            - tolerance: tolerance of the method
            - max_iterations: maximum number of iterations
        Output: reactions, strain, displacement
            - reactions: reactions of the nodes
            - strain: strain of the elements
            - displacement: displacement of the nodes
        """
        stiffness_sum = self.stiffness_sum()

        cutted_lines_collumns = []

        for node, direction in self.constraints:
            cutted_line_collumn = 2*(int(node) - 1) + int(direction)//2
            
            F = np.delete(F, cutted_line_collumn, 0)
            stiffness_sum = np.delete(stiffness_sum, cutted_line_collumn, 0)
            stiffness_sum = np.delete(stiffness_sum, cutted_line_collumn, 1)
            
            cutted_lines_collumns.append(cutted_line_collumn)
        
        solved_values = self.gauss_seidel(stiffness_sum, F, max_iterations, tolerance)

        displacement = np.zeros((len(self.nodes)*2, 1))

        counter = 0
        for i in range(2*len(self.nodes)):
            if (i not in cutted_lines_collumns):
                displacement[i] = solved_values[counter]
                counter += 1

        reactions = self.stiffness_sum().dot(displacement)

        reactions[np.abs(reactions) < tolerance] = 0

        reactions = np.around(reactions, decimals=2)

        final_reactions = []
        for node, direction in self.constraints:
            cutted_line_collumn = 2*(int(node) - 1) + int(direction)//2
            final_reactions.append([reactions[cutted_line_collumn][0]])
          
        final_reactions = np.array(final_reactions)
        final_reactions = np.flip(final_reactions, axis=0)
        final_reactions = np.flip(final_reactions, axis=1)

        for el in self.elements:
            ids = displacement[[2*(el.n1.id - 1), 2*(el.n1.id - 1) + 1,2*(el.n2.id - 1), 2*(el.n2.id - 1) + 1], 0]
            el.strain(*ids)

        return final_reactions, self.strain(), displacement