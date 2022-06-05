<div align="center">
<h1>
  <strong>Análise de Treliças Planas</strong>
</h1>

# Membros do Time

<table>
  <tr>
    <td align="center"><a href="https://github.com/DaviReisVieira"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/36394034?v=4" width="100px;" alt=""/><br /><sub><b>Davi Reis Vieira</b></sub></a></td>
    <td align="center"><a href="https://github.com/fran-janela"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/21694400?v=4" width="100px;" alt=""/><br /><sub><b>Francisco Pinheiro Janela</b></sub></a></td>
    <td align="center"><a href="https://github.com/lucamelao"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/63018319?v=4" width="100px;" alt=""/><br /><sub><b>Luca Coutinho Melão</b></sub></a></td>
    <td align="center"><a href="https://github.com/NicolasQueiroga"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/62630822?v=4" width="100px;" alt=""/><br /><sub><b>Nicolas Maciel Queiroga</b></sub></a></td>
  </tr>
</table>

# Objetivo

<div align='left'>
  <p>
    O objetivo do projeto é desenvolver um software para análise de treliças planas a partir do input dos <strong>nós</strong>, <strong>incidências</strong>, <strong>carregamento</strong> e <strong>restrições</strong>, conforme o input do modelo em Excel.
  </p>
</div>

# Funções e Classes

Abaixo, estão listadas as classes e funções que serão utilizadas no projeto. Para mais informações, consulte o arquivo solver para saber sobre os métodos (funções) utilizadas em cada uma.

<div align='left'>
Classe de Ponto:

```python
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

Classe de Nó:

```python
class Node(Point):
    def __init__(self, x,y, id):
        super().__init__(x,y)
        self.id = id
```

Classe de Elemento:

```python
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
```

Classe do Solver:

```python
class Solver():
    def __init__(self):
        print('------- Initializing solver -------\n')
        self.nodes = []
        self.elements = []
        self.constraints = []
```

Função de Gauß-Seidel:

```python
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
```

</div>
</div>
