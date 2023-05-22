from cmath import pi
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import os




# Analisa a entrada do dado ( imagem )
def analyze_data(file: str):
     if os.path.exists(file):
          return file
     else:
          print('Not found')
     
 
     
# Obtem a imagem em Canny Edge e salva imagens na pasta    
def get_edges(image):
     image = cv2.convertScaleAbs(image,20,1)
     median = cv2.medianBlur(image,35) # Aplica desfoque na imagem
     gray = cv2.cvtColor(median,cv2.COLOR_BGR2GRAY) # Convert image to grayscale
     edges = cv2.Canny(gray,50,150,apertureSize=3)  # Use canny edge detection

     cv2.imwrite('./stellarium/01/DetectedLines.png',image) # Salva imagem original
     cv2.imwrite('./stellarium/01/DetectedEdges.png',edges) # Salva versão canny edge da imagem
     cv2.imwrite('./stellarium/01/DetectedMedianFilter.png',median) # Salva versão borrada da imagem
     return edges



# Aplica Transformada de Hough na imagem Canny Edge e obter linhas
def hough_transformation(edges):
     lines = cv2.HoughLinesP(
               edges, # Input imagem edge
               1, # Resolução da distancia em pixels
               pi/180, # Angulo da resolução em radianos
               threshold=50, # Numero minimo de votos para criar linha válida
               minLineLength=20, # Tamanho minimo permitido da linha 
               maxLineGap=5 # Gap maximo permitido entre linhas para elas se juntarem
               )
     return lines


# Obtem pontos dada as linhas da Transformada de Hough
def get_x_and_y(image,edges):
     lines = hough_transformation(edges) # Chamada da função
     x = [] 
     y = []
     for points in lines: # Pegar pontos de todas linhas
     
          x1,y1,x2,y2=points[0]
          
          x.append(x1) # Add lista de pontos x
          
          y.append(y1*(-1)) # Add lista de pontos y (É preciso inverter o sinal de y pois o plano cartesiano do OpenCV é diferente do Matplotlib)
          
          x.append(x2) # Add lista de pontos x
          
          y.append(y2*(-1)) # Add lista de pontos y (É preciso inverter o sinal de y pois o plano cartesiano do OpenCV é diferente do Matplotlib)
          
          cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2) # Cria linhas na imagem original dado dois pontos de coordenadas.
         
          # print(x)
          
     # print(f'x = {x}') # Verifica valores x
     # print(f'y = {y}') # Verifica valores y  
     cv2.imshow('image', image) # Exibe imagem com as linhas
     return x,y # Retorna array com os pontos x e y


# Cria linha de tendencia dado os pontos x e y
def tredline_function(x,y):
     eq_trendline_array = np.polyfit(x,y,2) # Equação da linha de tendencia em array
     eq_trendline = np.poly1d(eq_trendline_array)# Equação da linha de tendencia.
     return eq_trendline,eq_trendline_array


# Obtem os valores medios da função    
def median_values(function_array):
     # Testar para todas as extremidades do quadrante
     # Depende do tamanho da imagem 
     # No caso do OpenCV, geralmente a imagem esta em uma resolução de 1050x590
     # Cuidado com os eixos do plano cartesiano (ponto de origem está em cima na esquerda)
     x_width_max = 1050 # Tamanho máximo
     x_width_min = 0
     y_height_max = -590
     y_height_min = 0
     a = function_array[0]
     b = function_array[1]
     c = function_array[2]
     roots = []
     x = sp.symbols('x')
     function_y = (a*x**2)+(b*x)+c
     result_x_plus_max = (-b + (b**2 - 4*a*(c+(y_height_max*(-1))))**(1/2)) / (2*a) # Bhaskara x1 com -590
     result_x_plus_min = (-b + (b**2 - 4*a*(c+(y_height_min*(-1))))**(1/2)) / (2*a) # Bhaskara x1 com 0
     result_x_minus_max = (-b - (b**2 - 4*a*((c+y_height_max*(-1))))**(1/2)) / (2*a) # Bhaskara x2 com -590
     result_x_minus_min = (-b - (b**2 - 4*a*((c+y_height_min*(-1))))**(1/2)) / (2*a) # Bhaskara x2 com 0
     roots.append(result_x_plus_max)
     roots.append(result_x_minus_max)
     roots.append(result_x_plus_min)
     roots.append(result_x_minus_min)
     result_y_max_x = function_y.subs(x,x_width_max)
     result_y_min_x = function_y.subs(x,x_width_min)
     if (result_y_max_x <= 0 and result_y_max_x >=-590 and result_y_min_x <= 0 and result_y_min_x >= -590):
          # print('A curva intersecta em x = 0 e x = 1050(1)')   
          x_initial = 0
          x_final = 1050
     elif (result_x_plus_min >= 0 and result_x_plus_min <= 1050 and result_x_plus_max >= 0 and result_x_plus_max <= 1050) :
          # print('A curva intersecta em x = 0 e y = -590(2)')
          x_initial = result_x_plus_max
          x_final = result_x_plus_min
     elif (result_x_minus_min >= 0 and result_x_minus_min <= 1050 and result_x_minus_max >= 0 and result_x_minus_max <= 1050):
          # print('A curva intersecta em x = 0 e y = -590(3)')
          x_initial = result_x_minus_min
          x_final = result_x_minus_max
     elif (result_y_max_x <= 0 and result_y_max_x >=-590 and result_x_plus_max >= 0 and result_x_plus_max <=1050):
          # print('A curva intersecta em x = 10intel xeon prata50 e y = -590(4)')
          x_initial = result_x_plus_max
          x_final = 1050
     elif (result_y_min_x <= 0 and result_y_min_x >=-590 and result_x_minus_max >= 0 and result_x_minus_max <=1050):
          # print('A curva intersecta em x = 0 e y = -590(5)')
          x_initial = 0
          x_final = result_x_minus_max
     elif (result_y_min_x <= 0 and result_y_min_x >=-590 and result_x_plus_min >= 0 and result_x_plus_min <=1050):
          # print('A curva intersecta em x = 0 e y = 0(6)')
          x_initial = 0
          x_final = result_x_plus_min
     elif (result_y_max_x <= 0 and result_y_max_x >=-590 and result_x_minus_min >= 0 and result_x_minus_min <=1050):
          # print('A curva intersecta em x = 1050 e y = 0(7)')
          x_initial = result_x_minus_min
          x_final = 1050
     else:
          x_initial = 0
          x_final = 0 
     return x_initial,x_final
     
     
def f(x):
     return a*x**2+b*x+c

def tangent_coefficients():
     x_points = np.array([float(x_in_curve), xm])
     y_points = np.array([float(y_in_curve), ym])
     function = np.polyfit(x_points,y_points,1)# Equação da linha de tendencia 
     slope = function[0]
     intercept = function[1]
     a = -(1/slope)
     b = y_in_curve - (a)*(x_in_curve)
     return a,b

def tangent_equation(x):
     a,b = tangent_coefficients()
     return a*x+b


def plot_graph():
     plt.gca().xaxis.tick_top()
     plt.title("Curvatura da Terra")
     plt.plot(x,y, label='f(x)')
     plt.plot(x_in_curve,y_in_curve,'yo', label='Ponto de menor distância')
     plt.plot(xm,ym,'r',label='Centro da Imagem')
     plt.plot(x,reta_tangente,'r',label='Derivada',linestyle='-')
     plt.plot((x_in_curve,xm),(y_in_curve,ym),'go',linestyle='-')
     plt.xlim(0, 1050)
     plt.ylim(-590, 0)
     text_function = f"$f(x)={a:0.5f}\;x²{b:+0.4f}\;x\;{c:+0.4f}$"
     text_derivative = f"$f(x)'={tangent_slope:0.5f}\;x{tangent_intercept:+0.5f}$"
     plt.gca().text(0.45, 0.40, text_function,transform=plt.gca().transAxes,fontsize=8, verticalalignment='top')
     plt.gca().text(0.45, 0.35, text_derivative,transform=plt.gca().transAxes,fontsize=8, verticalalignment='top')

     # # Definir os rótulos dos eixos x e y
     plt.xlabel('x')
     plt.ylabel('y')
     plt.grid()
     # # Adicionar uma legenda
     plt.legend()
     
# ===================== Programa Principal ===================== #

src = './stellarium/01.png'
file = analyze_data(src)
image = cv2.imread(file,cv2.IMREAD_UNCHANGED)

width_image = 1050
height_image = 590
xm = width_image/2
ym = -height_image/2
dsize = (width_image,height_image)
image = cv2.resize(image,dsize)
fov = 48.8
pixel_height = fov/height_image
earth_radius = 6378
orbit_distance = 430



    
edges = get_edges(image)
x_points_array,y_points_array = get_x_and_y(image,edges)
print(x_points_array)
function,function_array = tredline_function(x_points_array,y_points_array)
a = function_array[0]
b = function_array[1]
c = function_array[2]
x_initial,x_final = median_values(function_array)

x = sp.symbols('x')
dif = sp.diff(f(x), x)
equation = sp.Eq(1 / dif, -(f(x)-ym) / (x - xm))

# Resolvendo a equação
solutions = sp.solve(equation, x)
x_in_curve = solutions[0]
y_in_curve = f(x_in_curve)

print(f'P({x_in_curve},{y_in_curve})')
f_prime_value = dif.subs(x, x_in_curve).evalf()
f_value = y_in_curve

d = abs((f_prime_value*xm-(ym)+(f_value-f_prime_value*x_in_curve))/(sp.sqrt((f_prime_value)**2+1)))
print(f'Distancia: {d}')

tangent_slope,tangent_intercept = tangent_coefficients()



x = np.linspace(0, 1200, 1000)
reta_tangente = tangent_equation(x)
y = f(x)

plot_graph()
plt.show()
