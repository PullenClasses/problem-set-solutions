import taichi as ti
from taichi import exp, cos, sin
from taichi.math import pi

# Start talking to the GPU
ti.init(arch = ti.gpu) 

# Define some useful parameters
dim     = 500
steps   = 5
dx      = 1/dim
dt      = 2e-1 * (2*dx*dx)
wave    = ti.Vector.field(2, ti.f32, (dim, dim))
wavenew = ti.Vector.field(2, ti.f32, (dim, dim))
V       = ti.field(ti.f32, (dim, dim))
R       = 3e-1
V_max   = -1e4
pixels  = ti.Vector.field(3, ti.f32, (dim, dim))
window  = ti.ui.Window("Recitation", res=(dim,dim))
canvas  = window.get_canvas() 

@ti.kernel
def step(h:ti.f32):
    for s in ti.static(range(steps)):
        for i,j in ti.ndrange((1,dim - 1),(1, dim-1)):
            wavenew[i,j][0] = wave[i,j][0] + V[i,j]*h*wave[i,j][1] - (wave[i+1,j][1] + wave[i-1,j][1] + wave[i,j+1][1] + wave[i,j-1][1]  - 4*wave[i,j][1])*h/(2*dx*dx)
        
        for i,j in ti.ndrange((1,dim - 1),(1, dim-1)):
            wavenew[i,j][1] = wave[i,j][1] - V[i,j]*h*wavenew[i,j][0] + (wavenew[i+1,j][0] + wavenew[i-1,j][0] + wavenew[i,j+1][0] + wavenew[i,j-1][0]  - 4*wavenew[i,j][0])*h/(2*dx*dx)
            wave[i,j]       = wavenew[i,j]

    for i,j in ti.ndrange((1,dim - 1),(1, dim-1)):
        pixels[i,j] = wave[i,j].norm()/10


@ti.kernel
def intialize(h:float, x:float, y:float, px:float, py:float, sx:float, sy:float):
    for i,j in ti.ndrange((1,dim - 1),(1, dim-1)):
        psi = exp(-((2*i-dim)/dim - x)**2/(2*sx**2) - ((2*j-dim)/dim - y)**2/(2*sy**2))/(2*pi*sx*sy)
        wave[i,j][0] = psi * cos((2*i-dim)/dim*px + (2*j-dim)/dim*py)
        wave[i,j][1] = psi * sin((2*i-dim)/dim*px + (2*j-dim)/dim*py)
    
    H = h/(4*dx*dx)
    for i,j in ti.ndrange((1,dim - 1),(1, dim-1)):
        # if i!=0 and j!=0 and i!=n-1 and j!=n-1: 
        wave[i,j][1] += (- 4*H - h*V[i,j]/2)*wave[i,j][0] + H*(wave[i+1,j][0] + wave[i-1,j][0] + wave[i,j+1][0] + wave[i,j-1][0])

    V.fill(0)
    for i,j in V:
        X = (2*i - dim)/dim
        Y = (2*j - dim)/dim

        if X**2 + Y**2 <= R**2:
            V[i,j] = -V_max

if __name__=="__main__":
    intialize(dt, 0.6, 0, -80, 40, 1e-1,1e-1)
    
    while window.running:
        step(dt)

        canvas.set_image(pixels)
        window.show()