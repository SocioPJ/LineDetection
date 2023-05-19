from sympy import symbols, diff, Eq, solve, sqrt, atan,pi
import math



def f(x):
    return -0.00021*x**2 + 0.2152*x - 221.2021






# Definindo resolução fov/pixels
fov = 48.8
pixel_height = fov/590
print(f'Resolução: {pixel_height}')
# Definindo angulo O
Re = 6378
Rs = 430

cat = math.sqrt((Re+Rs)**2-Re**2)
O = atan(abs(cat)/abs(Re))*(180/math.pi)
# Definindo a variável simbólica x
x = symbols('x')
# Definindo o ponto
a = 525
b = -295

# Definindo a derivada de f(x)
# dif = diff(f(x), x)
dif = 0.21520117825704 - 0.000411205753899529*x
print(dif)

# Substituindo f(x) e f'(x) na equação original
equation = Eq(1 / dif, -(f(x)-b) / (x - a))

# Resolvendo a equação
solutions = solve(equation, x)
x_value = solutions[0]

# Avaliando a função f(x) no valor de x
y_value = f(x_value)
print(f'')
print(f'')
print(f'P({x_value},{y_value})')
# Avaliando a derivada dif no valor de x = 2
f_prime_value = dif.subs(x, x_value).evalf()
f_value = y_value

# Calculando distancia entre os dois pontos
d = abs((f_prime_value*a-(b)+(f_value-f_prime_value*x_value))/(sqrt((f_prime_value)**2+1)))
print(f'Distancia: {d}')

# Calculando angulo do roll
roll = atan(abs(x_value-a)/abs(y_value-b))*(180/math.pi)
print(f'Angulo do Roll: {roll}°')

# Calculando angulo pitch

pitch = d*pixel_height
print(pitch)
print(f'equation: {equation}')